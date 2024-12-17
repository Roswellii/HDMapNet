import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString

from .const import CLASS2LABEL

class VectorizedLocalMap(object):
    def __init__(self,
                 dataroot,
                 patch_size,
                 canvas_size,
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 normalize=False,
                 fixed_num=-1):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num

    def gen_vectorized_samples(self, location, ego2global_translation, ego2global_rotation):
        """
        生成指定区域内的地图元素向量表示。

        该函数将车辆周围区域的地图元素（如车道线、人行横道、建筑物轮廓）转换为向量表示形式，这些向量用于后续处理和分析，
        例如在自动驾驶算法中。

        参数:
        - location: 感兴趣区域的位置，用于获取该区域的地图信息。
        - ego2global_translation: 车辆在全局坐标系中的位置平移部分，用于确定感兴趣区域的位置。
        - ego2global_rotation: 车辆在全局坐标系中的旋转部分，用于调整地图元素的方向以匹配车辆的方向。

        返回:
        - filtered_vectors: 包含所有地图元素向量表示的列表，每个元素包括点集、点的数量和元素类型。
        """
        # 提取车辆在全局坐标系中的2D位置
        map_pose = ego2global_translation[:2]
        # 将车辆方向的四元数表示转换为Quaternion对象，以便进行后续操作
        rotation = Quaternion(ego2global_rotation)

        # 定义以车辆位置为中心的兴趣区域（patch），大小由self.patch_size指定
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        # 计算车辆方向的偏航角，用于调整地图元素的方向
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        # 获取并矢量化指定区域内的线型地图元素（如车道线）
        line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location) #? 怎么获得的？
        line_vector_dict = self.line_geoms_to_vectors(line_geom)

        # 获取并矢量化指定区域内的行人过街地图元素
        ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location) #? 怎么获得的？
        # 使用line_geoms_to_vectors方法矢化行人过街，仅选择与行人过街相关的部分
        ped_vector_list = self.line_geoms_to_vectors(ped_geom)['ped_crossing']

        # 获取并矢量化指定区域内的多边形地图元素（如建筑物）
        polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location) #? 怎么获得的？
        poly_bound_list = self.poly_geoms_to_vectors(polygon_geom)

        # 初始化一个列表，用于存储所有矢化的地图元素
        vectors = []
        # 将线型地图元素添加到列表中，并分配相应的类型标签  #? 保存提取的地图结果
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append((line.astype(float), length, CLASS2LABEL.get(line_type, -1)))

        # 将行人过街地图元素添加到列表中，并分配相应的类型标签
        for ped_line, length in ped_vector_list:
            vectors.append((ped_line.astype(float), length, CLASS2LABEL.get('ped_crossing', -1)))

        # 将多边形地图元素添加到列表中，并分配相应的类型标签
        for contour, length in poly_bound_list:
            vectors.append((contour.astype(float), length, CLASS2LABEL.get('contours', -1)))

        # 过滤掉没有对应类型标签的地图元素
        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({
                    'pts': pts,
                    'pts_num': pts_num,
                    'type': type
                })

        # 返回过滤后的矢化地图元素列表
        return filtered_vectors

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
            points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            line = line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].extract_polygon(record['polygon_token'])
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
            add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)

        return line_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

        if self.normalize:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        return sampled_points, num_valid
