import torch
import torch.nn as nn
import torchvision.models as models


class VGGBackbone(nn.Module):
    def __init__(self, model_name='vgg16_bn', pretrained=True, features_only=True, out_indices=[1, 2, 3, 4], checkpoint_path=None):
        """
        CRAFT 모델을 위한 VGG 기반 백본
        
        Args:
            model_name (str): VGG 모델 변형 ('vgg16' 또는 'vgg16_bn')
            pretrained (bool): 사전학습된 가중치 사용 여부
            features_only (bool): 특징 추출기로만 사용 여부
            out_indices (list): 추출할 특징 맵의 인덱스 (VGG 블록 단위)
        """
        super(VGGBackbone, self).__init__()
        
        # VGG 모델 가져오기
        if model_name == 'vgg16':
            vgg = models.vgg16(pretrained=pretrained)
        elif model_name == 'vgg16_bn':
            vgg = models.vgg16_bn(pretrained=pretrained)
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
            
        # VGG의 features 부분만 가져오기
        features = vgg.features
        
        # 각 특징 블록의 시작 레이어 인덱스 매핑
        # VGG16 구조: 64(2/3) - 128(2/3) - 256(3/4) - 512(3/4) - 512(3/4)
        if 'bn' in model_name:  # BatchNorm 버전인 경우
            self.block_indices = {
                1: (0, 6),    # 두 번째 MaxPool 전까지 (64 채널)
                2: (7, 13),   # 세 번째 MaxPool 전까지 (128 채널)
                3: (14, 23),  # 네 번째 MaxPool 전까지 (256 채널)
                4: (24, 33),  # 다섯 번째 MaxPool 전까지 (512 채널)
                5: (34, 43)   # 마지막까지 (512 채널)
            }
        else:  # 일반 VGG16 (BatchNorm 없음)
            self.block_indices = {
                1: (0, 4),    # 두 번째 MaxPool 전까지
                2: (5, 9),    # 세 번째 MaxPool 전까지
                3: (10, 16),  # 네 번째 MaxPool 전까지
                4: (17, 23),  # 다섯 번째 MaxPool 전까지
                5: (24, 30)   # 마지막까지
            }
        
        # 각 블록 생성
        self.blocks = nn.ModuleDict()
        for idx in out_indices:
            start_idx, end_idx = self.block_indices[idx]
            block = nn.Sequential(*list(features[start_idx:end_idx+1]))
            self.blocks[str(idx)] = block
            
        self.out_indices = out_indices
        self.features_only = features_only
        
    def forward(self, x):
        """
        순전파: 이미지를 입력 받아 선택된 각 단계의 특징 맵 반환
        
        Args:
            x: 입력 이미지 [B, C, H, W]
            
        Returns:
            features: 선택된 단계의 특징 맵 리스트
        """
        outputs = []
        
        # 첫 번째 블록인 경우 원본 입력 사용
        if 1 in self.out_indices:
            feat1 = self.blocks['1'](x)
            outputs.append(feat1)
            prev_feat = feat1
        else:
            prev_feat = self.blocks['1'](x)
            
        # 나머지 블록 처리
        for idx in range(2, max(self.out_indices) + 1):
            if str(idx) in self.blocks:
                curr_feat = self.blocks[str(idx)](prev_feat)
                if idx in self.out_indices:
                    outputs.append(curr_feat)
                prev_feat = curr_feat
                
        return outputs


if __name__ == "__main__":
    # 간단한 테스트 코드
    import torch
    model = VGGBackbone(model_name='vgg16_bn', out_indices=[1, 2, 3, 4])
    x = torch.randn(2, 3, 640, 640)
    features = model(x)
    for i, feat in enumerate(features):
        print(f"Feature {i+1} shape:", feat.shape)