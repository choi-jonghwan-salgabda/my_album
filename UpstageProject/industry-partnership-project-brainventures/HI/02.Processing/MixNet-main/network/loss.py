# -*- coding: utf-8 -*-
# @Time    : 10/1/21
# @Author  : GXYM
import torch
import torch.nn as nn
from cfglib.config import config as cfg
# from network.Seg_loss import SegmentLoss
from network.Reg_loss import PolyMatchingLoss
import torch.nn.functional as F
from .emb_loss import EmbLoss_v2
from .overlap_loss import overlap_loss
import pytorch_ssim

import cv2

class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.MSE_loss = torch.nn.MSELoss(reduce=False, size_average=False)
        self.BCE_loss = torch.nn.BCELoss(reduce=False, size_average=False)
        self.PolyMatchingLoss = PolyMatchingLoss(cfg.num_points, cfg.device)
        if cfg.mid:
            self.midPolyMatchingLoss = PolyMatchingLoss(cfg.num_points // 2, cfg.device)
        # self.embed_loss = EmbLoss_v2()
        self.ssim = pytorch_ssim.SSIM()
        self.overlap_loss = overlap_loss()

    @staticmethod
    def single_image_loss(pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1)) * 0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        eps = 0.001
        for i in range(batch_size):
            average_number = 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= eps)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= eps)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < eps)]) < 3 * positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < eps)])
                    average_number += len(pre_loss[i][(loss_label[i] < eps)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < eps)], 3 * positive_pixel)[0])
                    average_number += 3 * positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 100)[0])
                average_number += 100
                sum_loss += nega_loss
            # sum_loss += loss/average_number

        return sum_loss/batch_size

    def cls_ohem(self, predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()

        n_pos = pos.float().sum()
        if n_pos.item() > 0:
            loss_pos = self.BCE_loss(predict[pos], target[pos]).sum()
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.)
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = 100
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    @staticmethod
    def loss_calc_flux(pred_flux, gt_flux, weight_matrix, mask, train_mask):
        """
        방향 필드 손실을 계산하는 함수
        
        Args:
            pred_flux: 예측된 방향 필드 텐서, 형태 [B, 2, H, W]
            gt_flux: 실측 방향 필드 텐서, 형태 [B, 2, H, W] 또는 다른 형태
            weight_matrix: 가중치 행렬, 형태 [B, H, W] 또는 [B, 1, H, W]
            mask: 마스크 텐서, 형태 [B, H, W] 또는 [B, 1, H, W]
            train_mask: 훈련 마스크 텐서, 형태 [B, H, W] 또는 [B, 1, H, W]
            
        Returns:
            norm_loss: 크기 손실 텐서
            angle_loss: 각도 손실 텐서
        """
        # 디버깅 출력
        print(f"pred_flux shape: {pred_flux.shape}, gt_flux shape: {gt_flux.shape}")
        print(f"weight_matrix shape: {weight_matrix.shape}, mask shape: {mask.shape}, train_mask shape: {train_mask.shape}")
        
        # gt_flux 차원 조정
        if gt_flux.dim() == 4:
            # gt_flux 형태가 [B, H, W, 2]인 경우 [B, 2, H, W]로 변환
            if gt_flux.shape[-1] == 2 and gt_flux.shape[1] != 2:
                gt_flux = gt_flux.permute(0, 3, 1, 2)
            # gt_flux 형태가 [B, 2, H, W]인 경우 그대로 사용
        elif gt_flux.dim() == 3 and gt_flux.shape[-1] == 2:
            # gt_flux 형태가 [B, H, W, 2]인 경우 [B, 2, H, W]로 변환
            gt_flux = gt_flux.permute(0, 3, 1, 2)
        
        # 차원 크기 일치 확인
        B, C, H, W = pred_flux.shape
        if gt_flux.shape[2:] != pred_flux.shape[2:]:
            # gt_flux 크기가 pred_flux와 다른 경우 크기 조정
            gt_flux = F.interpolate(gt_flux, size=(H, W), mode='bilinear', align_corners=False)
        
        # 정규화된 방향 벡터 계산
        gt_norm = torch.norm(gt_flux, p=2, dim=1, keepdim=True)
        gt_flux = gt_flux / (gt_norm + 1e-3)
        
        # weight_matrix 차원 조정
        if weight_matrix.dim() == 3:  # [B, H, W]
            weight_matrix = weight_matrix.unsqueeze(1)  # [B, 1, H, W]
        
        # weight_matrix 크기 조정
        if weight_matrix.shape[2:] != (H, W):
            weight_matrix = F.interpolate(weight_matrix, size=(H, W), mode='bilinear', align_corners=False)
        
        # train_mask 차원 조정
        if train_mask.dim() == 3:  # [B, H, W]
            train_mask = train_mask.unsqueeze(1)  # [B, 1, H, W]
        
        # train_mask 크기 조정
        if train_mask.shape[2:] != (H, W):
            train_mask = F.interpolate(train_mask.float().unsqueeze(1), size=(H, W), mode='nearest').squeeze(1).bool()
            train_mask = train_mask.unsqueeze(1)
        
        # 제곱 차이 계산
        squared_diff = ((pred_flux - gt_flux) ** 2)  # [B, 2, H, W]
        mean_squared_diff = torch.mean(squared_diff, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 가중치와 마스크 적용
        weighted_loss = mean_squared_diff * weight_matrix * train_mask
        norm_loss = weighted_loss.sum() / (train_mask.sum() + 1e-6)
        
        # 각도 손실 계산
        if mask.dim() == 3:  # [B, H, W]
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        
        # mask 크기 조정
        if mask.shape[2:] != (H, W):
            mask = F.interpolate(mask.float().unsqueeze(1), size=(H, W), mode='nearest').squeeze(1).bool()
            mask = mask.unsqueeze(1)
        
        combined_mask = train_mask * mask
        
        # 예측 방향 벡터 정규화
        pred_norm = torch.norm(pred_flux, p=2, dim=1, keepdim=True)
        pred_flux_normalized = pred_flux / (pred_norm + 1e-3)
        
        # 코사인 유사도 계산
        dot_product = torch.sum(pred_flux_normalized * gt_flux, dim=1)  # [B, H, W]
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle_diff = torch.acos(dot_product) / torch.tensor(3.14159, device=dot_product.device)
        
        # 마스크 적용
        if combined_mask.dim() == 4 and angle_diff.dim() == 3:  # mask [B, 1, H, W], angle_diff [B, H, W]
            combined_mask = combined_mask.squeeze(1)  # [B, H, W]
        
        # 마스크가 적용된 영역의 평균 손실 계산
        if combined_mask.sum() > 0:
            angle_loss = (angle_diff * combined_mask).sum() / (combined_mask.sum() + 1e-6)
        else:
            angle_loss = torch.tensor(0., device=angle_diff.device)
        
        return norm_loss, angle_loss
    
    @staticmethod
    def get_poly_energy(energy_field, img_poly, ind, h, w):
        img_poly = img_poly.clone().float()
        img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
        img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

        batch_size = energy_field.size(0)
        gcn_feature = torch.zeros([img_poly.size(0), energy_field.size(1), img_poly.size(1)]).to(img_poly.device)
        for i in range(batch_size):
            poly = img_poly[ind == i].unsqueeze(0)
            gcn_feature[ind == i] = torch.nn.functional.grid_sample(energy_field[i:i + 1], poly)[0].permute(1, 0, 2)
        return gcn_feature

    def loss_energy_regularization(self, energy_field, img_poly, inds, h, w):
        energys = []
        for i, py in enumerate(img_poly):
            energy = self.get_poly_energy(energy_field.unsqueeze(1), py, inds, h, w)
            energys.append(energy.squeeze(1).sum(-1))

        regular_loss = torch.tensor(0.)
        energy_loss = torch.tensor(0.)
        for i, e in enumerate(energys[1:]):
            regular_loss += torch.clamp(e - energys[i], min=0.0).mean()
            energy_loss += torch.where(e <= 0.01, torch.tensor(0.), e).mean()

        return (energy_loss+regular_loss)/len(energys[1:])

    def dice_loss(self, x, target, mask):
        b = x.shape[0]
        x = torch.sigmoid(x)

        x = x.contiguous().reshape(b, -1)
        target = target.contiguous().reshape(b, -1)
        mask = mask.contiguous().reshape(b, -1)

        x = x * mask
        target = target.float()
        target = target * mask

        a = torch.sum(x * target, 1)
        b = torch.sum(x * x, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)

        loss = 1 - d
        loss = torch.mean(loss)
        return loss


    def forward(self, input_dict, output_dict, eps=None):
        """
        단순화된 손실 함수 - 기본적인 손실만 계산하고 복잡한 부분은 비활성화
        """
        # 기본 입력 획득
        fy_preds = output_dict["fy_preds"]
        train_mask = input_dict['train_mask'].float()
        tr_mask = input_dict['tr_mask'] > 0
        distance_field = input_dict['distance_field']
        conf = tr_mask.float()
        
        # 크기 조정 (필요한 경우)
        if cfg.scale > 1:
            train_mask = F.interpolate(train_mask.float().unsqueeze(1),
                                    scale_factor=1/cfg.scale, mode='bilinear').squeeze().bool()
            tr_mask = F.interpolate(tr_mask.float().unsqueeze(1),
                                    scale_factor=1/cfg.scale, mode='bilinear').squeeze().bool()
            distance_field = F.interpolate(distance_field.unsqueeze(1),
                                        scale_factor=1/cfg.scale, mode='bilinear').squeeze()

        print(f"train_mask 합계: {train_mask.sum().item()}")
        print(f"tr_mask 합계: {tr_mask.sum().item()}")
        print(f"distance_field 합계: {distance_field.sum().item()}")

        # train_mask가 모두 0인 경우 처리
        if train_mask.sum() == 0:
            print("경고: train_mask가 모두 0입니다! 임의의 마스크를 생성합니다.")
            # 임의의 마스크 생성 (일부 픽셀 활성화)
            train_mask = torch.zeros_like(train_mask)
            train_mask[:, train_mask.shape[1]//4:train_mask.shape[1]//2, 
                    train_mask.shape[2]//4:train_mask.shape[2]//2] = 1.0

        # tr_mask가 모두 0인 경우 처리
        if tr_mask.sum() == 0:
            print("경고: tr_mask가 모두 0입니다! 임의의 마스크를 생성합니다.")
            tr_mask = torch.zeros_like(tr_mask)
            tr_mask[:, tr_mask.shape[1]//4:tr_mask.shape[1]//2, 
                tr_mask.shape[2]//4:tr_mask.shape[2]//2] = True

        if distance_field.sum() == 0 and eps <= 5:  # 초기 5 에폭 동안만
            print("경고: distance_field가 모두 0입니다! 가짜 텍스트 영역을 추가합니다.")
            fake_text_region = torch.zeros_like(distance_field)
            # 이미지 중앙에 작은 텍스트 영역 추가
            h, w = fake_text_region.shape[1], fake_text_region.shape[2]
            fake_text_region[:, h//3:2*h//3, w//3:2*w//3] = 1.0
            
            # 가짜 영역을 실제 거리 필드에 추가
            distance_field = distance_field + fake_text_region
            
            # tr_mask와 train_mask도 업데이트
            tr_mask = tr_mask | (fake_text_region > 0)
            train_mask = train_mask | (fake_text_region > 0)
            
            # conf(tr_mask float) 업데이트
            conf = tr_mask.float()

        # 1. 기본 분류 손실 (세그먼테이션)
        cls_loss = self.BCE_loss(fy_preds[:, 0, :, :], conf)
        cls_loss = torch.mul(cls_loss, train_mask).mean()
        
        print(f"cls_loss 계산 전: {self.BCE_loss(fy_preds[:, 0, :, :], conf).mean().item()}")
        print(f"cls_loss 계산 후: {cls_loss.item()}")
        
        # 2. 거리 필드 손실
        dis_loss = self.MSE_loss(fy_preds[:, 1, :, :], distance_field)
        dis_loss = torch.mul(dis_loss, train_mask)
        
        print(f"dis_loss 계산 전: {self.MSE_loss(fy_preds[:, 1, :, :], distance_field).mean().item()}")
        print(f"dis_loss 마스킹 후: {torch.mul(dis_loss, train_mask).mean().item()}")
        
        dis_loss = self.single_image_loss(dis_loss, distance_field)
        
        print(f"dis_loss 최종: {dis_loss.item()}")

        # 3. 나머지 복잡한 손실들 비활성화 (0으로 설정)
        norm_loss = torch.tensor(0.0, device=fy_preds.device)
        angle_loss = torch.tensor(0.0, device=fy_preds.device)
        point_loss = torch.tensor(0.0, device=fy_preds.device)
        midline_loss = torch.tensor(0.0, device=fy_preds.device)
        energy_loss = torch.tensor(0.0, device=fy_preds.device)
        embed_loss = torch.tensor(0.0, device=fy_preds.device)
        
        # 가중치 계수 설정
        alpha = 1.0
        beta = 3.0
        
        # 최종 손실 계산 (단순화)
        loss = alpha * cls_loss + beta * dis_loss
        
        # 손실 딕셔너리 구성
        loss_dict = {
            'total_loss': loss,
            'cls_loss': alpha * cls_loss,
            'distance_loss': beta * dis_loss,
            'dir_loss': norm_loss + angle_loss,
            'norm_loss': norm_loss,
            'angle_loss': angle_loss,
            'point_loss': point_loss,
            'energy_loss': energy_loss,
            'embed_loss': embed_loss,
        }
        
        if cfg.mid:
            loss_dict['midline_loss'] = midline_loss
        
        return loss_dict

class knowledge_loss(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.KLDloss = torch.nn.KLDivLoss(size_average = False)
        self.T = T
    def forward(self, pred, know):
        log_pred = F.log_softmax(pred / self.T, dim = 1)
        sftknow = F.softmax(know / self.T, dim=1)
        kldloss = self.KLDloss(log_pred, sftknow)
        # print(pred.shape)
        kldloss = kldloss * (self.T**2) / (pred.shape[0] * pred.shape[2] * pred.shape[3])
        return kldloss    