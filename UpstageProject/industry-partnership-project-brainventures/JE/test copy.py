"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

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

parser = argparse.ArgumentParser(description='CRAFT 텍스트 감지 학습')
parser.add_argument('--train_image_dir', default='/data/ephemeral/home/test', type=str, help='학습 이미지가 있는 디렉토리')
parser.add_argument('--train_json_path', default='/data/ephemeral/home/test.json', type=str, help='학습 JSON 파일 경로')
parser.add_argument('--batch_size', default=8, type=int, help='배치 크기')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='학습률')
parser.add_argument('--num_epochs', default=50, type=int, help='학습 에폭 수')
parser.add_argument('--cuda', default=True, type=str2bool, help='CUDA 사용 여부')
parser.add_argument('--canvas_size', default=1280, type=int, help='이미지 크기')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='이미지 확대 비율')
parser.add_argument('--pretrained_model', default='', type=str, help='사전 학습된 모델 경로')
parser.add_argument('--save_interval', default=5, type=int, help='모델 저장 간격(에폭)')

args = parser.parse_args()

# 결과 폴더 생성
result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

# 모델 저장 폴더 생성
model_folder = './weights/'
if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

# 데이터셋 클래스 정의
class CRAFTDataset(Dataset):
    def __init__(self, image_dir, json_path, canvas_size=1280, mag_ratio=1.5):
        self.image_dir = image_dir
        self.json_path = json_path
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        
        # JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 이미지 파일 목록 생성
        self.image_files = list(self.annotations.keys())
        print(f"총 {len(self.image_files)}개의 이미지를 로드했습니다.")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 이미지 파일명 및 경로
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        
        # 이미지 로드
        image = imgproc.loadImage(img_path)
        
        # 어노테이션 가져오기
        img_anno = self.annotations[img_filename]
        polygons = []
        
        # 단어 폴리곤 추출
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
        
        # 데이터를 디바이스로 이동
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 그래디언트 초기화
        optimizer.zero_grad()
        
        # 순전파
        outputs, _ = net(inputs)
        
        # 손실 계산
        loss = criterion(outputs, targets)
        
        # 역전파 및 최적화
        loss.backward()
        optimizer.step()
        
        # 통계 업데이트
        total_loss += loss.item()
        batch_count += 1
        
        # 배치 진행 상황 출력
        if batch_count % 10 == 0:
            print(f"에폭 {epoch}, 배치 {batch_count}, 손실: {loss.item():.4f}")
    
    # 에폭 평균 손실 반환
    return total_loss / batch_count

if __name__ == '__main__':
    # CUDA 설정
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = CRAFTDataset(
        args.train_image_dir, 
        args.train_json_path, 
        canvas_size=args.canvas_size, 
        mag_ratio=args.mag_ratio
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    # 모델 초기화
    net = CRAFT()
    
    # 사전 학습된 모델 로드 (있는 경우)
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"사전 학습된 모델 로드 중: {args.pretrained_model}")
        net.load_state_dict(copyStateDict(torch.load(args.pretrained_model)))
    
    # 모델을 디바이스로 이동
    net = net.to(device)
    if args.cuda and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    # 손실 함수 및 옵티마이저 정의
    criterion = CRAFTLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    
    # 학습 시작
    print(f"총 {args.num_epochs}개의 에폭으로 학습을 시작합니다.")
    for epoch in range(1, args.num_epochs + 1):
        # 에폭 시작 시간
        epoch_start = time.time()
        
        # 학습 수행
        avg_loss = train(net, train_loader, criterion, optimizer, device, epoch)
        
        # 에폭 종료 시간 및 소요 시간
        epoch_time = time.time() - epoch_start
        
        # 에폭 결과 출력
        print(f"에폭 {epoch}/{args.num_epochs} 완료, 평균 손실: {avg_loss:.4f}, 소요 시간: {epoch_time:.2f}초")
        
        # 모델 저장 (지정된 간격마다)
        if epoch % args.save_interval == 0 or epoch == args.num_epochs:
            save_path = os.path.join(model_folder, f'craft_epoch_{epoch}.pth')
            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module.state_dict(), save_path)
            else:
                torch.save(net.state_dict(), save_path)
            print(f"모델이 저장되었습니다: {save_path}")
    
    print("학습이 완료되었습니다!")