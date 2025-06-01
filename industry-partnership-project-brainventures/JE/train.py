"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import csv

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection Training')
parser.add_argument('--train_image_dir', default='/data/ephemeral/home/data/images/train', type=str, help='directory containing training images')
parser.add_argument('--train_json_path', default='/data/ephemeral/home/data/jsons/train.json', type=str, help='path to training json file')
parser.add_argument('--val_image_dir', default='/data/ephemeral/home/data/images/val', type=str, help='directory containing validation images')
parser.add_argument('--val_json_path', default='/data/ephemeral/home/data/jsons/val.json', type=str, help='path to validation json file')
parser.add_argument('--pretrained_model', default='/data/ephemeral/home/CRAFT-pytorch/weight/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--batch_size', default=4, type=int, help='batch size for training')
parser.add_argument('--num_epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
parser.add_argument('--save_dir', default='./weights', type=str, help='directory to save weights')
parser.add_argument('--output_dir', default='./output', type=str, help='directory to save output results')

args = parser.parse_args()

# custom_collate 함수 추가 (DataLoader 생성 전에 정의)
def custom_collate(batch):
    """모든 항목이 동일한 크기를 갖도록 확인하는 collate 함수"""
    # 배치가 비어있는 경우 확인
    if len(batch) == 0:
        return [], []
    
    # 첫 번째 항목 크기 가져오기
    first_x, first_target = batch[0]
    x_shape = first_x.shape
    target_shape = first_target.shape
    
    # 모든 항목이 같은 크기인지 확인
    for x, target in batch:
        if x.shape != x_shape or target.shape != target_shape:
            # 다른 크기의 항목 제외 (또는 리사이징할 수 있음)
            print(f"다른 크기의 이미지 발견: {x.shape} vs {x_shape}")
            return torch.utils.data._utils.collate.default_collate([batch[0]] * len(batch))
    
    # 모든 크기가 같으면 기본 collate 함수 사용
    return torch.utils.data._utils.collate.default_collate(batch)

# 학습 데이터셋 클래스 정의
class CRAFTDataset(Dataset):
    def __init__(self, image_dir, json_path, target_size=768, canvas_size=1280, mag_ratio=1.5):
        self.image_dir = image_dir
        self.target_size = target_size
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        
        # JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 이미지 이름 추출
        if isinstance(self.annotations, dict) and 'images' in self.annotations:
            all_image_names = list(self.annotations['images'].keys())
        else:
            all_image_names = []
            print("지원하지 않는 JSON 형식입니다.")
        
        # 실제 존재하는 이미지만 필터링
        self.image_names = []
        for img_name in all_image_names:
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                self.image_names.append(img_name)
            else:
                print(f"이미지 파일을 찾을 수 없어 건너뜁니다: {img_path}")
        
        print(f"이미지 총 개수: {len(all_image_names)}, 사용 가능한 이미지: {len(self.image_names)}")
        if len(self.image_names) > 0:
            print(f"첫 번째 이미지: {self.image_names[0]}")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # 이미지 로드
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 이미지 로드 - 이미 존재하는 파일만 처리하므로 확인 불필요
        image = imgproc.loadImage(img_path)
        
        # JSON에서 폴리곤 정보 가져오기
        polygons = []
        if img_name in self.annotations['images']:
            img_anno = self.annotations['images'][img_name]
            if 'words' in img_anno:
                for word_id, word_info in img_anno['words'].items():
                    if 'points' in word_info:
                        points = word_info['points']
                        polygons.append(np.array(points, dtype=np.float32))
        
        # 이미지 리사이징 및 타겟 생성
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio
        )
        
        # 폴리곤 좌표 조정
        for i in range(len(polygons)):
            polygons[i] = polygons[i] * target_ratio
        
        # 타겟 맵 생성 (히트맵 크기에 맞춤)
        target = self.generate_target(img_resized.shape[0], img_resized.shape[1], size_heatmap, polygons)
        
        # 이미지 전처리
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        
        # 타겟 텐서 형식으로 변환
        target = torch.from_numpy(target).float()
        # 채널 차원 이동 (만약 필요하다면)
        if target.dim() == 3 and target.shape[2] == 2:  # [h, w, 2] -> [2, h, w]
            target = target.permute(2, 0, 1)
        
        return x, target
    
    def generate_target(self, height, width, size_heatmap, polygons):
        """타겟 맵 생성 - CRAFT 출력과 동일한 형식"""
        # 리사이징 비율 계산
        height_ratio = size_heatmap[0] / height
        width_ratio = size_heatmap[1] / width
        
        # 히트맵 크기에 맞는 지역 및 링크 맵 생성
        target_height, target_width = size_heatmap
        region_score = np.zeros((target_height, target_width), dtype=np.float32)
        affinity_score = np.zeros((target_height, target_width), dtype=np.float32)
        
        # 간소화된 예시: 폴리곤을 이용한 마스크 생성
        for polygon in polygons:
            # 폴리곤 리사이징
            resized_polygon = polygon.copy()
            resized_polygon[:, 0] = resized_polygon[:, 0] * width_ratio
            resized_polygon[:, 1] = resized_polygon[:, 1] * height_ratio
            resized_polygon = resized_polygon.astype(np.int32)
            
            # 리전 스코어맵 생성
            cv2.fillPoly(region_score, [resized_polygon], 1.0)
            
            # 어피니티 스코어맵을 단순화(실제로는 더 복잡한 로직 필요)
            cv2.polylines(affinity_score, [resized_polygon], True, 1.0, 2)
        
        # CRAFT 출력 형식에 맞게 [h, w, 2] 형태로 반환
        target = np.dstack([region_score, affinity_score])
        return target

# 손실 함수 정의
class CRAFTLoss(nn.Module):
    def __init__(self):
        super(CRAFTLoss, self).__init__()
    
    def forward(self, pred, gt):
        # 텐서 크기 출력
        print(f"pred 크기: {pred.shape}, gt 크기: {gt.shape}")
        
        # pred 차원 순서 변경: [배치, 높이, 너비, 채널] -> [배치, 채널, 높이, 너비]
        if pred.dim() == 4 and pred.shape[3] == 2:
            pred = pred.permute(0, 3, 1, 2)
            print(f"pred 차원 변경 후: {pred.shape}")
        
        # 높이/너비 크기가 다른 경우 gt 리사이징
        if pred.shape[2] != gt.shape[2] or pred.shape[3] != gt.shape[3]:
            print(f"gt 크기 조정: {gt.shape} -> [B, C, {pred.shape[2]}, {pred.shape[3]}]")
            gt = torch.nn.functional.interpolate(
                gt, 
                size=(pred.shape[2], pred.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
        
        # MSE 손실 계산
        loss = torch.mean((pred - gt) ** 2)
        return loss

# 학습 함수
def train(net, train_loader, criterion, optimizer, device, epoch):
    net.train()
    total_loss = 0
    batch_count = 0
    
    for inputs, targets in train_loader:
        # 데이터 크기 확인
        print(f"입력 배치 크기: {inputs.shape}, 타겟 배치 크기: {targets.shape}")
        
        # 차원 변환 (필요한 경우)
        if targets.dim() == 3:  # [batch, height, width] 형태인 경우
            targets = targets.unsqueeze(1)  # [batch, 1, height, width]로 변환
        elif targets.dim() == 4 and targets.shape[1] == 3:  # [batch, 3, height, width] 형태인 경우
            # 첫 2개 채널만 사용
            targets = targets[:, :2, :, :]
        
        # 데이터를 GPU로 이동
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        try:
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs, _ = net(inputs)
            
            # 손실 계산
            loss = criterion(outputs, targets)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 로그
            total_loss += loss.item()
            batch_count += 1
            if batch_count % 5 == 0:
                print(f'Epoch {epoch}, Batch {batch_count}, Loss: {loss.item():.4f}')
        
        except Exception as e:
            print(f"배치 처리 중 오류 발생: {e}")
            continue
    
    avg_loss = total_loss / max(batch_count, 1)
    print(f'Epoch {epoch} - Average Loss: {avg_loss:.4f}')
    return avg_loss

def validate(net, val_loader, criterion, device):
    net.eval()
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            # 데이터를 GPU로 이동
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 순전파
            outputs, _ = net(inputs)
            
            # 손실 계산
            loss = criterion(outputs, targets)
            
            # 로그
            total_loss += loss.item()
            batch_count += 1
    
    avg_loss = total_loss / batch_count
    print(f'Validation - Average Loss: {avg_loss:.4f}')
    return avg_loss

# JSON을 CSV로 변환하는 함수
def json_to_csv(json_path, output_dir):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # CSV 파일 경로
    csv_path = os.path.join(output_dir, 'annotations.csv')
    
    # CSV 파일 작성
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_path', 'polygons']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # JSON 데이터에서 이미지 정보 추출
        if isinstance(json_data, dict) and 'images' in json_data:
            for img_name, img_info in json_data['images'].items():
                if 'words' in img_info:
                    # 이미지별로 모든 폴리곤 수집
                    all_polygons = []
                    
                    for word_id, word_info in img_info['words'].items():
                        if 'points' in word_info and len(word_info['points']) == 4:
                            points = word_info['points']
                            # 폴리곤 좌표를 문자열로 변환
                            polygon_str = f"{points[0][0]},{points[0][1]},{points[1][0]},{points[1][1]},{points[2][0]},{points[2][1]},{points[3][0]},{points[3][1]}"
                            all_polygons.append(polygon_str)
                    
                    # 폴리곤을 '|'로 구분하여 하나의 행으로 저장
                    if all_polygons:
                        row = {
                            'image_path': img_name,
                            'polygons': '|'.join(all_polygons)
                        }
                        writer.writerow(row)
    
    print(f'CSV 파일이 저장되었습니다: {csv_path}')
    return csv_path

if __name__ == '__main__':
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # JSON 파일 구조 확인 중...
    print("JSON 파일 구조 확인 중...")
    with open(args.train_json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"JSON 데이터 타입: {type(json_data)}")
    if isinstance(json_data, dict) and 'images' in json_data:
        if isinstance(json_data['images'], dict):
            # 딕셔너리인 경우 첫 번째 키 사용
            image_keys = list(json_data['images'].keys())
            print(f"images는 딕셔너리입니다. 키 개수: {len(image_keys)}")
            if image_keys:
                print(f"첫 번째 이미지: {image_keys[0]}")
                print(f"첫 번째 이미지 구조: {json_data['images'][image_keys[0]].keys()}")
        else:
            # 리스트인 경우(일반적인 COCO 포맷)
            print(f"images 항목 샘플: {json_data['images'][0] if json_data['images'] else 'images 항목 비어있음'}")
    
    print(f"이미지 디렉토리 파일 목록: {os.listdir(args.train_image_dir)}")
    
    # 모델 초기화
    net = CRAFT()
    
    # 사전 훈련된 가중치 로드
    if os.path.exists(args.pretrained_model):
        print(f'Loading weights from checkpoint ({args.pretrained_model})')
        net.load_state_dict(copyStateDict(torch.load(args.pretrained_model)))
    
    # GPU 설정
    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    
    # 손실 함수 및 옵티마이저 설정
    criterion = CRAFTLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    
    # 학습 데이터셋 준비
    train_dataset = CRAFTDataset(
        image_dir=args.train_image_dir,
        json_path=args.train_json_path,
        canvas_size=args.canvas_size,
        mag_ratio=args.mag_ratio
    )
    
    # 검증 데이터셋 준비 (제공된 경우)
    if args.val_image_dir and args.val_json_path and os.path.exists(args.val_json_path):
        val_dataset = CRAFTDataset(
            image_dir=args.val_image_dir,
            json_path=args.val_json_path,
            canvas_size=args.canvas_size,
            mag_ratio=args.mag_ratio
        )
        print(f"검증 데이터셋 크기: {len(val_dataset)}")
    else:
        # 검증 데이터셋이 제공되지 않은 경우 학습 데이터셋을 사용
        print("검증 데이터셋이 제공되지 않아 학습 데이터셋을 사용합니다.")
        val_dataset = train_dataset
    
    # 데이터셋 크기 확인
    train_dataset_size = len(train_dataset)
    if train_dataset_size == 0:
        print("학습 데이터셋이 비어 있습니다. 프로그램을 종료합니다.")
        sys.exit(1)
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate
    )
    
    # 학습 시작
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        print(f'Epoch {epoch}/{args.num_epochs}')
        
        # 학습
        train_loss = train(net, train_loader, criterion, optimizer, 'cuda' if args.cuda else 'cpu', epoch)
        
        # 검증
        val_loss = validate(net, val_loader, criterion, 'cuda' if args.cuda else 'cpu')
        
        # 최종 에폭에서만 가중치 저장
        if epoch == args.num_epochs:
            save_path = os.path.join(args.save_dir, f'craft_final.pth')
            
            # 모델이 DataParallel로 래핑되어 있는 경우
            if isinstance(net, nn.DataParallel):
                torch.save(net.module.state_dict(), save_path)
            else:
                torch.save(net.state_dict(), save_path)
            
            print(f'최종 모델 저장됨: {save_path}')
    
    # 마지막 에폭에만 체크포인트 저장
    checkpoint_path = os.path.join(args.save_dir, f'checkpoint_final.pth')
    if isinstance(net, nn.DataParallel):
        torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': net.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, checkpoint_path)
    else:
        torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, checkpoint_path)
    print(f'최종 체크포인트 저장됨: {checkpoint_path}')
    
    # JSON을 CSV로 변환하여 저장
    csv_path = json_to_csv(args.train_json_path, args.output_dir)
    print(f'학습 결과 CSV 파일 저장됨: {csv_path}')
    
    print(f'Training completed in {time.time() - start_time:.2f} seconds')