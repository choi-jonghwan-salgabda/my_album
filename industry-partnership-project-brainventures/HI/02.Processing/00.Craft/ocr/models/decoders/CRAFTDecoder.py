'''
*****************************************************************************************
* CRAFT: Character Region Awareness for Text Detection
*
* 참고 논문:
* Character Region Awareness for Text Detection
* https://arxiv.org/abs/1904.01941
*
* 원본 및 참고 구현:
* https://github.com/clovaai/CRAFT-pytorch
*****************************************************************************************
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) × 2"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class CRAFTDecoder(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512], out_channels=128):
        super(CRAFTDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1x1 = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1) 
            for ch in in_channels
        ])
        
        self.upconv1 = DoubleConv(out_channels, out_channels)
        self.upconv2 = DoubleConv(out_channels * 2, out_channels)
        self.upconv3 = DoubleConv(out_channels * 2, out_channels)
        self.upconv4 = DoubleConv(out_channels * 2, out_channels)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, features):        
        # 특징 맵과 컨볼루션 레이어 매핑 확인
        if not hasattr(self, '_initialized') or not self._initialized:
            actual_channels = [feat.size(1) for feat in features]
            if actual_channels != [conv.weight.size(1) for conv in self.conv_1x1]:
                print(f"Warning: Reinitializing conv_1x1 for channels {actual_channels}")
                self.conv_1x1 = nn.ModuleList([
                    nn.Conv2d(ch, self.out_channels, kernel_size=1).to(features[0].device) 
                    for ch in actual_channels
                ])
            self._initialized = True
        
        # 1x1 컨볼루션으로 채널 수 조정
        processed_feats = [conv(feat) for feat, conv in zip(features, self.conv_1x1)]
        
        # 역순으로 처리 (깊은 레이어부터)
        processed_feats = processed_feats[::-1]  # [feat4, feat3, feat2, feat1] -> 깊은 레이어가 앞에 오도록
        
        # 디코더 블록 적용
        x = self.upconv1(processed_feats[0])
        
        # 상향 샘플링 및 특징 맵 결합 (스킵 커넥션)
        results = [x]
        
        # 두 번째 단계: feat3와 결합
        x = F.interpolate(x, size=processed_feats[1].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, processed_feats[1]], dim=1)
        x = self.upconv2(x)
        results.append(x)
        
        # 세 번째 단계: feat2와 결합
        x = F.interpolate(x, size=processed_feats[2].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, processed_feats[2]], dim=1)
        x = self.upconv3(x)
        results.append(x)
        
        # 네 번째 단계: feat1와 결합
        x = F.interpolate(x, size=processed_feats[3].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, processed_feats[3]], dim=1)
        x = self.upconv4(x)
        results.append(x)
        
        # 역순으로 반환하여 원래 순서대로
        return results[::-1]