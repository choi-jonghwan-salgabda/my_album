'''
*****************************************************************************************
* CRAFT: Character-Region Awareness For Text detection
* 참고 논문: Character Region Awareness for Text Detection (CRAFT)
* https://arxiv.org/abs/1904.01941
*****************************************************************************************
'''

import cv2
import numpy as np
import torch
from collections import OrderedDict
from scipy.spatial import Delaunay


class CRAFTCollateFN:
    def __init__(self, text_threshold=0.7, link_threshold=0.4, low_text=0.4, gaussian_kernel_size=7):
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.gaussian_kernel_size = gaussian_kernel_size
        self.inference_mode = False
        
    def __call__(self, batch):
        images = [item['image'] for item in batch]
        filenames = [item['image_filename'] for item in batch]
        inverse_matrix = [item['inverse_matrix'] for item in batch]
        
        collated_batch = OrderedDict(
            images=torch.stack(images, dim=0),
            image_filename=filenames,
            inverse_matrix=inverse_matrix,
        )
        
        if self.inference_mode:
            return collated_batch
            
        polygons = [item['polygons'] for item in batch]
        
        # 각 이미지에 대한 영역 맵과 어피니티 맵 생성
        region_maps = []
        affinity_maps = []
        
        for i, image in enumerate(images):
            # 문자 영역 맵과 어피니티 맵 생성
            _, h, w = image.shape
            targets = self.generate_target_maps(image, polygons[i], filenames[i])
            
            # 텐서로 변환
            region_map_tensor = torch.tensor(targets['region_map']).unsqueeze(0)
            affinity_map_tensor = torch.tensor(targets['affinity_map']).unsqueeze(0)
            
            region_maps.append(region_map_tensor)
            affinity_maps.append(affinity_map_tensor)
            
        # 배치에 결과 추가
        collated_batch.update(
            polygons=polygons,
            region_maps=torch.stack(region_maps, dim=0),
            affinity_maps=torch.stack(affinity_maps, dim=0),
        )
        
        return collated_batch
        
    def generate_target_maps(self, image, polygons, filename):
        _, h, w = image.shape
        region_map = np.zeros((h, w), dtype=np.float32)
        affinity_map = np.zeros((h, w), dtype=np.float32)
        
        # 각 다각형에 대해 처리
        for poly in polygons:
            if poly.size < 3:  # 다각형 점이 3개 미만이면 건너뜀
                continue
                
            poly = poly.astype(np.int32)
            
            # 1. 다각형 내의 문자 센터 포인트 추정
            # (실제 CRAFT 구현에서는 word-level polygons를 
            # 문자 단위로 나누는 방법이 필요할 수 있음)
            char_points = self._estimate_character_points(poly[0])
            
            # 2. 각 문자 포인트에 대한 가우시안 히트맵 생성
            for point in char_points:
                region_map = self._add_gaussian(region_map, point, self.gaussian_kernel_size)
                
            # 3. 연속된 문자 포인트 사이의 어피니티 맵 생성
            if len(char_points) > 1:
                for i in range(len(char_points) - 1):
                    mid_point = (char_points[i] + char_points[i+1]) / 2
                    affinity_map = self._add_gaussian(affinity_map, mid_point, self.gaussian_kernel_size)
                    
        return OrderedDict(region_map=region_map, affinity_map=affinity_map)
        
    def _estimate_character_points(self, polygon):
        num_chars = max(2, int(cv2.arcLength(polygon, True) / 20))
        
        if len(polygon) >= 4:
            n = len(polygon) // 2
            top_line = polygon[:n]
            bottom_line = np.flip(polygon[n:], axis=0)
            
            char_points = []
            for i in range(num_chars):
                ratio = i / (num_chars - 1) if num_chars > 1 else 0.5
                top_point_idx = int(ratio * (len(top_line) - 1))
                bottom_point_idx = int(ratio * (len(bottom_line) - 1))
                
                top_point = top_line[top_point_idx]
                bottom_point = bottom_line[bottom_point_idx]
                
                center_point = (top_point + bottom_point) / 2
                char_points.append(center_point)
                
            return np.array(char_points)
        else:
            center = np.mean(polygon, axis=0)
            return np.array([center])
            
    def _add_gaussian(self, heat_map, center, kernel_size):
        center_x, center_y = int(center[0]), int(center[1])
        height, width = heat_map.shape
        
        if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
            return heat_map
            
        size = kernel_size * 2 + 1
        x, y = np.meshgrid(np.arange(0, size, 1), np.arange(0, size, 1))
        mean_x = size // 2
        mean_y = size // 2
        sigma = kernel_size / 3  # 커널 크기에 비례하는 표준편차
        
        gaussian = np.exp(-((x - mean_x)**2 + (y - mean_y)**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.max()
        
        x_min = max(0, center_x - kernel_size)
        y_min = max(0, center_y - kernel_size)
        x_max = min(width, center_x + kernel_size + 1)
        y_max = min(height, center_y + kernel_size + 1)
        
        gau_x_min = max(0, -center_x + kernel_size)
        gau_y_min = max(0, -center_y + kernel_size)
        gau_x_max = gau_x_min + (x_max - x_min)
        gau_y_max = gau_y_min + (y_max - y_min)
        
        heat_map[y_min:y_max, x_min:x_max] = np.maximum(
            heat_map[y_min:y_max, x_min:x_max],
            gaussian[gau_y_min:gau_y_max, gau_x_min:gau_x_max]
        )
        
        return heat_map

    def set_inference_mode(self, mode=True):
        self.inference_mode = mode

class DBCollateFN:
    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.inference_mode = False

    def __call__(self, batch):
        images = [item['image'] for item in batch]
        filenames = [item['image_filename'] for item in batch]
        inverse_matrix = [item['inverse_matrix'] for item in batch]

        collated_batch = OrderedDict(images=torch.stack(images, dim=0),
                                     image_filename=filenames,
                                     inverse_matrix=inverse_matrix,
                                     )

        if self.inference_mode:
            return collated_batch

        polygons = [item['polygons'] for item in batch]

        prob_maps = []
        thresh_maps = []
        for i, image in enumerate(images):
            # Probability map / Threshold map 생성
            segmentations = self.make_prob_thresh_map(image, polygons[i], filenames[i])
            prob_map_tensor = torch.tensor(segmentations['prob_map']).unsqueeze(0)
            thresh_map_tensor = torch.tensor(segmentations['thresh_map']).unsqueeze(0)
            prob_maps.append(prob_map_tensor)
            thresh_maps.append(thresh_map_tensor)

        collated_batch.update(polygons=polygons,
                              prob_maps=torch.stack(prob_maps, dim=0),
                              thresh_maps=torch.stack(thresh_maps, dim=0),
                              )

        return collated_batch

    def make_prob_thresh_map(self, image, polygons, filename):
        _, h, w = image.shape
        prob_map = np.zeros((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)

        for poly in polygons:
            # Calculate the distance and polygons
            poly = poly.astype(np.int32)
            # Polygon point가 3개 미만이라면 skip
            if poly.size < 3:
                continue

            # https://arxiv.org/pdf/1911.08947.pdf 참고
            L = cv2.arcLength(poly, True) + np.finfo(float).eps
            D = cv2.contourArea(poly) * (1 - self.shrink_ratio ** 2) / L
            pco = pyclipper.PyclipperOffset()
            pco.AddPaths(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

            # Probability map 생성
            shrinked = pco.Execute(-D)
            for s in shrinked:
                shrinked_poly = np.array(s)
                cv2.fillPoly(prob_map, [shrinked_poly], 1.0)

            # Threshold map 생성
            dilated = pco.Execute(D)
            for d in dilated:
                dilated_poly = np.array(d)

                xmin = dilated_poly[:, 0].min()
                xmax = dilated_poly[:, 0].max()
                ymin = dilated_poly[:, 1].min()
                ymax = dilated_poly[:, 1].max()
                width = xmax - xmin + 1
                height = ymax - ymin + 1

                polygon = poly[0].copy()
                polygon[:, 0] = polygon[:, 0] - xmin
                polygon[:, 1] = polygon[:, 1] - ymin

                xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width),
                                     (height, width))
                ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1),
                                     (height, width))

                distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
                for i in range(polygon.shape[0]):
                    j = (i + 1) % polygon.shape[0]
                    absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
                    distance_map[i] = np.clip(absolute_distance / D, 0, 1)
                distance_map = distance_map.min(axis=0)

                xmin_valid = min(max(0, xmin), thresh_map.shape[1] - 1)
                xmax_valid = min(max(0, xmax), thresh_map.shape[1] - 1)
                ymin_valid = min(max(0, ymin), thresh_map.shape[0] - 1)
                ymax_valid = min(max(0, ymax), thresh_map.shape[0] - 1)

                thresh_map[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
                    1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height,
                        xmin_valid - xmin:xmax_valid - xmax + width],                        # noqa
                    thresh_map[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

        # Normalize the threshold map
        thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min

        return OrderedDict(prob_map=prob_map, thresh_map=thresh_map)

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1]) + np.finfo(float).eps

        denom = 2 * np.sqrt(square_distance_1 * square_distance_2) + np.finfo(float).eps
        cosin = (square_distance - square_distance_1 - square_distance_2) / denom
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result
