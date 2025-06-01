'''
*****************************************************************************************
* CRAFT: Character Region Awareness For Text detection
*
* 참고 논문:
* Character Region Awareness for Text Detection
* https://arxiv.org/abs/1904.01941
*
* 참고 Repository:
* https://github.com/clovaai/CRAFT-pytorch
*****************************************************************************************
'''

import torch
import torch.nn as nn
from collections import OrderedDict
import cv2
import numpy as np

class CRAFTHead(nn.Module):
    def __init__(self, in_channels=128, out_channels=2, upscale=1, 
                 text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                 postprocess=None):
        assert postprocess is not None, "postprocess should not be None for CRAFTHead"

        super(CRAFTHead, self).__init__()
        self.postprocess = CRAFTPostProcessor(**postprocess)
        self.in_channels = in_channels
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()  # 바이어스는 0으로 초기화
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()  # 0으로 수정
            
    def forward(self, features, return_loss=True):
        """
        순전파: 특징 맵을 받아 문자 영역 및 연결성 맵 생성
        
        Args:
            features: 디코더의 특징 맵 리스트 또는 최종 특징 맵
            return_loss: 손실 계산을 위한 모든 출력 반환 여부
            
        Returns:
            문자 영역 맵과 문자 간 연결성 맵을 포함한 OrderedDict
        """
        # 입력이 리스트인 경우 마지막 특징 맵만 사용
        # (CRAFT는 일반적으로 가장 해상도가 높은 마지막 특징 맵 사용)
        if isinstance(features, list):
            feature = features[-1]
        else:
            feature = features
            
        # 문자 영역 맵과 문자 간 연결성 맵 생성 (채널 0: 영역, 채널 1: 연결성)
        outputs = self.conv_out(feature)
        region_score = outputs[:, 0:1, :, :]  # 첫 번째 채널: 문자 영역 점수
        affinity_score = outputs[:, 1:2, :, :]  # 두 번째 채널: 문자 간 연결성 점수
            
        result = OrderedDict(
            region_score=region_score,
            affinity_score=affinity_score
        )
            
        return result
    
    def get_polygons_from_maps(self, gt, pred):        
        print(f"Using thresholds - text: {self.text_threshold}, link: {self.link_threshold}, low_text: {self.low_text}")

        if self.postprocess is not None:
            return self.postprocess.represent(gt, pred)
        else:
            raise ValueError("Postprocess is not defined in CRAFTHead.")


class CRAFTPostProcessor:
    def __init__(self, text_threshold=0.7, link_threshold=0.4, 
                 low_text=0.4, min_size=10, max_candidates=1000):
        """
        CRAFT 모델의 출력을 처리하기 위한 후처리 클래스
        
        Args:
            text_threshold: 텍스트 영역으로 간주할 점수 임계값
            link_threshold: 연결성으로 간주할 점수 임계값
            low_text: 낮은 신뢰도 텍스트 영역 임계값
            min_size: 최소 텍스트 영역 크기
            max_candidates: 최대 텍스트 영역 후보 수
        """
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.min_size = min_size
        self.max_candidates = max_candidates
        
    def represent(self, batch, pred):
        # 필수 항목 확인
        assert 'images' in batch, "images is required in batch"
        assert 'inverse_matrix' in batch, "inverse_matrix is required in batch"
        
        # 예측에서 region_score와 affinity_score 추출
        assert 'region_score' in pred, "region_score is required in pred"
        assert 'affinity_score' in pred, "affinity_score is required in pred"
        
        region_scores = pred['region_score']
        affinity_scores = pred['affinity_score']
        
        batch_size = region_scores.size(0)
        boxes_batch = []
        scores_batch = []
        
        for b in range(batch_size):
            region_score = region_scores[b, 0].cpu().data.numpy()
            affinity_score = affinity_scores[b, 0].cpu().data.numpy()
            inverse_matrix = batch['inverse_matrix'][b]
            
            # 텍스트 영역과 연결성 결합하여 텍스트 박스 추출
            boxes, scores = self._get_text_boxes(
                region_score, 
                affinity_score, 
                inverse_matrix
            )
            
            boxes_batch.append(boxes)
            scores_batch.append(scores)
            
        return boxes_batch, scores_batch
    
    def _get_text_boxes(self, region_score, affinity_score, inverse_matrix):
        # 텍스트 영역 맵에 임계값 적용
        text_score = region_score.copy()
        text_score_binary = text_score >= self.text_threshold
        nonzero_count = np.count_nonzero(text_score_binary)
       
        # 연결성 맵에 임계값 적용
        link_score = affinity_score.copy()
        link_score_binary = link_score >= self.link_threshold
        nonzero_link_count = np.count_nonzero(link_score_binary)
       
        # 임계값 적용 (디버깅용 로그 추가)
        text_score[text_score < self.text_threshold] = 0
        link_score[link_score < self.link_threshold] = 0
        
        # 두 맵을 결합하여 텍스트 세그멘테이션 맵 생성
        text_score_comb = np.clip(text_score + link_score, 0, 1)
        
        # 이진화 및 윤곽선 추출
        text_mask = (text_score_comb * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            text_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        scores = []
        
        for contour in contours[:self.max_candidates]:
            # 최소 크기 필터링
            if cv2.contourArea(contour) < self.min_size:
                continue
                
            # 텍스트 영역을 포함하는 최소 회전 사각형 구하기
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            
            # 신뢰도 점수 계산 (영역 내 평균 점수)
            mask = np.zeros_like(region_score, dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(box)], 1)
            score = float(np.mean(region_score * mask))
            
            # 낮은 신뢰도 필터링
            if score < self.low_text:
                continue
                
            # 좌표 변환 (필요시)
            if inverse_matrix is not None:
                box = self._transform_coordinates(box, inverse_matrix)
                
            boxes.append(np.round(box).astype(np.int16).tolist())
            scores.append(score)
            
        return boxes, scores
        
    def _transform_coordinates(self, coords, matrix):
        coords = np.array(coords)
        coords = np.dot(matrix, np.vstack([coords.T, np.ones(coords.shape[0])]))
        coords /= coords[2, :]
        return coords.T[:, :2]