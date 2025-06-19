import os
import cv2
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from util.misc import to_device
from util.misc import fill_hole
from dataset.data_util import pil_load_img
from dataset.dataload import TextInstance
from util.io import read_lines
from util.misc import norm2


class CustomDataset(Dataset):
    def __init__(self, data_root, is_training=False, transform=None):
        super(CustomDataset, self).__init__()
        self.data_root = data_root
        self.transform = transform
        self.is_training = is_training
        
        # 지원하는 이미지 확장자
        self.img_exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        #self.max_images = 5
        # 이미지 리스트 생성
        self.image_list = []
        for f in os.listdir(data_root):
            ext = os.path.splitext(f)[1].lower()
            if ext in self.img_exts:
                self.image_list.append(f)
                # if len(self.image_list) >= self.max_images:
                #     break
        
        print(f"총 {len(self.image_list)}개 이미지를 찾았습니다.")
    
    def __len__(self):
        return len(self.image_list)
        
    def __getitem__(self, index):
        img_name = self.image_list[index]
        img_path = os.path.join(self.data_root, img_name)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지를 로드할 수 없습니다: {img_path}")
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 변환 적용 (이미지는 NumPy 배열 형식이어야 함)
        if self.transform is not None:
            img, _ = self.transform(img)
        
        # 메타데이터 생성
        meta = {
            'image_id': img_name,
            'Height': torch.tensor(h),
            'Width': torch.tensor(w),
            'annotation': [torch.tensor([])],  # 빈 주석
            'n_annotation': [torch.tensor(0)],
            'label_tag': torch.tensor([])
        }
        
        return img, meta

def rescale_img(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    h = int(h * scale)
    w = int(w * scale)
    img = cv2.resize(img, (w, h))
    return img

class CustomTextDataset(Dataset):
    def __init__(self, data_root, json_path, is_training=True, load_memory=False, transform=None):
        super().__init__()
        
        self.data_root = data_root
        self.is_training = is_training
        self.transform = transform
        self.image_root = data_root
        
        # JSON 데이터 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
            
        print(f"Loaded {len(self.annotations)} images from {json_path}")
        
        # 메모리에 이미지 미리 로드 (옵션)
        self.load_memory = load_memory
        self.images = {}
        if self.load_memory:
            print("Pre-loading images into memory...")
            for anno in self.annotations:
                img_path = os.path.join(self.image_root, anno['image_path'])
                img = cv2.imread(img_path)
                if img is not None:
                    self.images[anno['image_path']] = img
            print(f"Loaded {len(self.images)} images into memory")

    def parse_annotation(self, annotation):
        """다각형 주석을 TextInstance 형식으로 변환"""
        polygons = annotation['polygons']
        
        instances = []
        
        for polygon in polygons:
            # 다각형 좌표를 numpy 배열로 변환
            polygon = np.array(polygon, dtype=np.float32)
            
            # 포인트 수를 정확히 20개로 조정
            if len(polygon) != 20:  # config.num_points
                # 포인트 수가 너무 많으면 다운샘플링
                if len(polygon) > 20:
                    indices = np.linspace(0, len(polygon)-1, 20, dtype=int)
                    polygon = polygon[indices]
                # 포인트 수가 부족하면 보간
                else:
                    new_polygon = np.zeros((20, 2), dtype=np.float32)
                    for i in range(20):
                        idx = (i * len(polygon)) // 20
                        next_idx = (idx + 1) % len(polygon)
                        ratio = (i * len(polygon) / 20) - idx
                        new_polygon[i] = (1 - ratio) * polygon[idx] + ratio * polygon[next_idx]
                    polygon = new_polygon
            
            # TextInstance 객체 생성
            instances.append(TextInstance(polygon, 'Polygon', ""))
            
        return instances

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        annotation = self.annotations[item]
        
        # 이미지 로드
        if self.load_memory and annotation['image_path'] in self.images:
            img = self.images[annotation['image_path']].copy()
        else:
            img_path = os.path.join(self.image_root, annotation['image_path'])
            img = cv2.imread(img_path)
            if img is None:
                print(f"Cannot find image: {img_path}")
                # 이미지를 찾을 수 없는 경우 빈 이미지 반환
                img = np.zeros((640, 640, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 텍스트 인스턴스 가져오기
        instances = self.parse_annotation(annotation)
        
        # 훈련 중이고 변환이 지정된 경우 랜덤 변환 적용
        if self.is_training and self.transform:
            img, instances = self.transform(img, instances)
        
        # 감지 기준 데이터 생성
        train_mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
        tr_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)

        if instances and self.is_training:
            # 각 인스턴스에 대해 bottom과 sideline 찾기
            for instance in instances:
                instance.find_bottom_and_sideline()
                
            # 텍스트 영역 마스크 생성
            for instance in instances:
                cv2.fillPoly(tr_mask, [np.array(instance.points, np.int32)], color=(1,))
            
            # 중간 에지 완성
            if instances and hasattr(instances[0], 'mid_points'):
                for instance in instances:
                    cv2.fillPoly(tr_mask, [np.array(instance.mid_points, np.int32)], color=(1,))
            
            # 구멍 채우기
            train_mask = fill_hole(tr_mask)
        
        # 히트맵, 방향 필드, 가중치 행렬 초기화
        distance_field = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        direction_field = np.zeros((2, img.shape[0], img.shape[1]), dtype=np.float32)
        weight_matrix = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
        
        # CustomTextDataset.__getitem__ 함수에 추가
        if np.sum(tr_mask) == 0:
            print(f"경고: 이미지 {item}에 텍스트 영역이 없습니다!")
            # 가짜 텍스트 영역 생성
            h, w = tr_mask.shape
            tr_mask[h//3:2*h//3, w//3:2*w//3] = 1
            train_mask = tr_mask.copy()
            
            # 거리 필드 생성
            distance_field[h//3:2*h//3, w//3:2*w//3] = 1.0

        # 훈련 중이고 인스턴스가 있는 경우 필드 계산
        edge_field = None
        if self.is_training and instances:
            # 거리 필드 생성
            for instance in instances:
                cv2.fillPoly(distance_field, [np.array(instance.points, np.int32)], color=(1,))
                
            # 중간점이 있는 경우 에지 필드 생성
            if hasattr(instances[0], 'mid_points'):
                edge_field = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                for instance in instances:
                    for i in range(len(instance.points)):
                        delta = instance.mid_points[i] - instance.points[i]
                        distance = norm2(delta)
                        direction = delta / (distance + 1e-6)
                        
                        # 점 사이에 선 그리기
                        for j in range(int(distance) + 1):
                            point = instance.points[i] + j * direction
                            point_int = np.round(point).astype(np.int32)
                            if 0 <= point_int[0] < img.shape[1] and 0 <= point_int[1] < img.shape[0]:
                                edge_field[point_int[1], point_int[0]] = 1.0
        
        # 텍스트 중심선 및 제안 포인트 수집
        pts = []
        pos_proposals = []
        for instance in instances:
            pts.append(instance.points)
            pos_proposals.append(instance.points)
        
        # 무시 태그 설정 - 다각형 필터링 규칙
        ignore_tags = []
        for instance in instances:
            # 너무 작은 다각형 무시
            if len(instance.points) < 3:  # 유효한 다각형이 아님
                ignore_tags.append(True)
                continue
                
            polygon_area = cv2.contourArea(instance.points.astype(np.int32))
            if polygon_area < 100:  # 최소 면적 기준
                ignore_tags.append(True)
            # 너무 복잡한 다각형 무시
            elif len(instance.points) > 20:  # 최대 점 수 기준
                ignore_tags.append(True)
            # 극단적인 비율의 다각형 무시
            else:
                x, y, w, h = cv2.boundingRect(instance.points.astype(np.int32))
                aspect_ratio = max(w/h, h/w) if h > 0 and w > 0 else 0
                if aspect_ratio > 10:  # 너무 길쭉한 다각형 무시
                    ignore_tags.append(True)
                else:
                    ignore_tags.append(False)
        
        # 출력을 위한 이미지 준비
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        
        # 훈련 중인 경우에만 ground truth points 설정
        gt_points = pts if self.is_training else None
        
        # 출력 딕셔너리 구성
        output = {
            'img': img,
            'train_mask': train_mask, 
            'tr_mask': tr_mask,
            'distance_field': distance_field, 
            'direction_field': direction_field,
            'weight_matrix': weight_matrix,
            'gt_points': gt_points,
            'proposal_points': pos_proposals,
            'ignore_tags': ignore_tags,
        }
        
        # 에지 필드가 있는 경우 추가
        if edge_field is not None:
            output['edge_field'] = edge_field
        
        # 중간점이 있는 경우 추가
        if instances and hasattr(instances[0], 'mid_points'):
            mid_pts = [instance.mid_points for instance in instances]
            output['gt_mid_points'] = mid_pts
        
        return output

def collate_fn(batch):
    """
    커스텀 collate 함수: 크기가 다른 배치 항목들을 적절히 처리
    """
    # 빈 배치 제거
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        print("Warning: Empty batch detected. Skipping...")
        return None
    
    # img만 stack하고 나머지는 그대로 리스트로 반환
    imgs = []
    train_masks = []
    tr_masks = []
    distance_fields = []
    direction_fields = []
    weight_matrices = []
    gt_points = []
    proposal_points = []
    ignore_tags = []
    edge_fields = []
    
    for i, sample in enumerate(batch):
        # 유효한 샘플인지 확인
        required_keys = ['img', 'train_mask', 'tr_mask', 'distance_field', 
                         'direction_field', 'weight_matrix', 'proposal_points', 'ignore_tags']
        if not all(key in sample for key in required_keys):
            print(f"Warning: Sample {i} is missing required keys. Skipping...")
            continue
            
        # 각 필드가 유효한 값(None이 아님)을 가지고 있는지 확인
        if any(sample[key] is None for key in required_keys if key != 'gt_points'):
            print(f"Warning: Sample {i} has None values. Skipping...")
            continue
        
        imgs.append(torch.from_numpy(sample['img']).float())
        train_masks.append(torch.from_numpy(sample['train_mask']).float())
        tr_masks.append(torch.from_numpy(sample['tr_mask']).float())
        distance_fields.append(torch.from_numpy(sample['distance_field']).float())
        direction_field_np = sample['direction_field']
        # (H, W, 2) -> (2, H, W)로 변환
        direction_field_np = np.transpose(direction_field_np, (2, 0, 1))
        direction_fields.append(torch.from_numpy(direction_field_np).float())
        weight_matrices.append(torch.from_numpy(sample['weight_matrix']).float())
        
        if 'gt_points' in sample and sample['gt_points'] is not None:
            gt_points.append(sample['gt_points'])
        else:
            gt_points.append([])
            
        proposal_points.append(sample['proposal_points'])
        
        # 각 항목마다 False이면 0, True이면 1로 변환하여 텐서로 만듦
        sample_ignore_tags = [0 if not tag else 1 for tag in sample['ignore_tags']]
        ignore_tags.append(torch.tensor(sample_ignore_tags, dtype=torch.int64))
        
        if 'edge_field' in sample:
            edge_fields.append(torch.from_numpy(sample['edge_field']).float())
    
    # 유효한 샘플이 없는 경우
    if not imgs:
        print("Warning: No valid samples in batch after filtering. Skipping...")
        return None
    
    # 결과 딕셔너리 구성
    result = {
        'img': torch.stack(imgs, 0),
        'train_mask': torch.stack(train_masks, 0),
        'tr_mask': torch.stack(tr_masks, 0),
        'distance_field': torch.stack(distance_fields, 0),
        'direction_field': torch.stack(direction_fields, 0),
        'weight_matrix': torch.stack(weight_matrices, 0),
        'gt_points': gt_points,
        'proposal_points': proposal_points,
        'ignore_tags': ignore_tags,  # 이제 텐서들의 리스트
    }
    
    if edge_fields and len(edge_fields) == len(imgs):
        result['edge_field'] = torch.stack(edge_fields, 0)
    
    # gt_mid_points가 있다면 추가
    if batch and 'gt_mid_points' in batch[0]:
        gt_mid_points = []
        for sample in batch:
            if 'gt_mid_points' in sample:
                gt_mid_points.append(sample['gt_mid_points'])
            else:
                gt_mid_points.append([])
        result['gt_mid_points'] = gt_mid_points
    
    return result
    