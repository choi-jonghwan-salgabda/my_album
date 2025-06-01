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

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRAFTLoss(nn.Module):
    def __init__(self, region_weight=1.0, affinity_weight=1.0, ohem_ratio=3.0, eps=1e-6):
        """
        CRAFT 모델을 위한 손실 함수
        
        Args:
            region_weight: 문자 영역 맵 손실 가중치
            affinity_weight: 문자 간 연결성 맵 손실 가중치
            ohem_ratio: Online Hard Example Mining 비율 (음수 대 양수 비율)
            eps: 수치 안정성을 위한 작은 값
        """
        super(CRAFTLoss, self).__init__()
        self.region_weight = region_weight
        self.affinity_weight = affinity_weight
        self.ohem_ratio = ohem_ratio
        self.eps = eps
        
    def ohem_single(self, score_map, gt_map, mask):
        """
        Online Hard Example Mining 알고리즘
        
        Args:
            score_map: 예측한 점수 맵 (B, 1, H, W)
            gt_map: 정답 맵 (B, 1, H, W)
            mask: 유효 영역 마스크 (B, 1, H, W)
        """
        # 입력 텐서들의 크기 확인
        if score_map.shape != gt_map.shape or score_map.shape != mask.shape:
            print(f"Shape mismatch: score_map={score_map.shape}, gt_map={gt_map.shape}, mask={mask.shape}")
            
            # 크기가 다른 경우 보간하여 크기 맞추기
            target_size = gt_map.shape[2:]  # 정답 맵의 높이, 너비 사용
            if score_map.shape[2:] != target_size:
                score_map = F.interpolate(score_map, size=target_size, mode='bilinear', align_corners=False)
                print(f"Resized score_map to {score_map.shape}")
            
            if mask.shape[2:] != target_size:
                mask = F.interpolate(mask, size=target_size, mode='nearest')
                print(f"Resized mask to {mask.shape}")
        
        # 양성 샘플과 음성 샘플 마스크 생성
        pos_mask = (gt_map * mask) > 0.5
        neg_mask = ((gt_map * mask) <= 0.5) & (mask > 0)
        
        # 양성 샘플 수 계산
        n_pos = pos_mask.float().sum().item()
        if n_pos == 0:
            return mask  # 양성 샘플이 없으면 전체 마스크 반환
        
        # 선택할 음성 샘플 수 계산
        n_neg = min(int(n_pos * self.ohem_ratio), neg_mask.float().sum().int().item())
        if n_neg == 0:
            return pos_mask.float()  # 음성 샘플이 없으면 양성 마스크만 반환
        
        # 배치별 처리
        batch_size = score_map.size(0)
        final_selected_mask = torch.zeros_like(mask)
        
        for b in range(batch_size):
            # 배치 내 단일 이미지에 대한 마스크
            pos_mask_b = pos_mask[b]
            neg_mask_b = neg_mask[b]
            score_map_b = score_map[b]
            
            # 양성 샘플 수 계산
            n_pos_b = pos_mask_b.float().sum().item()
            
            # 음성 샘플에서 점수가 높은 것들 선택
            neg_score_b = score_map_b[neg_mask_b]
            
            if len(neg_score_b) == 0:
                # 음성 샘플이 없으면 양성 마스크만 사용
                final_selected_mask[b] = pos_mask_b.float()
                continue
                
            # 선택할 음성 샘플 수 계산
            n_neg_b = min(int(n_pos_b * self.ohem_ratio), len(neg_score_b))
            if n_neg_b == 0:
                final_selected_mask[b] = pos_mask_b.float()
                continue
                
            # 점수에 따라 상위 n_neg_b개 음성 샘플 선택
            values, _ = torch.topk(neg_score_b, k=n_neg_b)
            threshold_b = values[-1]  # n_neg_b번째로 높은 점수
            
            # 임계값보다 높은 점수를 가진 음성 샘플 마스크 생성
            selected_neg_mask_b = (score_map_b >= threshold_b) & neg_mask_b
            
            # 양성 마스크와 선택된 음성 마스크 결합
            final_selected_mask[b] = (pos_mask_b | selected_neg_mask_b).float()
        
        return final_selected_mask
        
    def forward(self, pred, **kwargs):
      # 예측값 추출
        pred_region = pred['region_score']
        pred_affinity = pred['affinity_score']
        
        # 정답값 추출
        gt_region = kwargs.get('region_maps')
        gt_affinity = kwargs.get('affinity_maps')
        gt_mask = kwargs.get('masks', None)
        
        if gt_mask is not None:
            print(f"gt_mask: {gt_mask.shape}")
        
        # 크기 불일치 확인 및 조정
        if gt_region.shape[2:] != pred_region.shape[2:]:
            gt_region = F.interpolate(gt_region, size=pred_region.shape[2:], mode='bilinear', align_corners=False)
            gt_affinity = F.interpolate(gt_affinity, size=pred_affinity.shape[2:], mode='bilinear', align_corners=False)
            if gt_mask is not None:
                gt_mask = F.interpolate(gt_mask, size=pred_region.shape[2:], mode='nearest')
        
        if gt_mask is None:
            # 마스크가 없으면 모든 픽셀을 유효하다고 가정
            gt_mask = torch.ones_like(gt_region)
        
        # 문자 영역 맵에 대한 마스크 생성 (OHEM)
        region_mask = self.ohem_single(pred_region, gt_region, gt_mask)
        
        # 문자 영역 맵 손실 계산 (이진 교차 엔트로피)
        loss_region = F.binary_cross_entropy(
            pred_region,
            gt_region,
            reduction='none'
        )
        loss_region = (loss_region * region_mask).sum() / (region_mask.sum() + self.eps)
        
        # 연결성 맵에 대한 마스크 생성 (OHEM)
        affinity_mask = self.ohem_single(pred_affinity, gt_affinity, gt_mask)
        
        # 연결성 맵 손실 계산 (이진 교차 엔트로피)
        loss_affinity = F.binary_cross_entropy(
            pred_affinity,
            gt_affinity,
            reduction='none'
        )
        loss_affinity = (loss_affinity * affinity_mask).sum() / (affinity_mask.sum() + self.eps)
        
        # 가중 합계로 총 손실 계산
        loss = self.region_weight * loss_region + self.affinity_weight * loss_affinity
        
        # 각 손실 구성 요소를 딕셔너리에 저장
        loss_dict = OrderedDict(
            loss_region=loss_region,
            loss_affinity=loss_affinity,
            total_loss=loss
        )
        
        return loss, loss_dict
