# -*- coding: utf-8 -*-
"""
1. 현재 구상하고 있는것은 내desktop 컴퓨터의 사진들을 이미지만 학습하여 분류를 하고자 합니다. 
2. 내가 보관중인 사진은 이름(label)이 없습니다.
3. 분류후 특정인의 얼굴을 노트북 카메라로 비추고 사진의 이름을 입력하면 
4. 학습한 사진들중 가장 카메라로 보여준 사진과 근접한 분류에 그이름을 정의하고 
5. 그사진이 포함된 다른 사진들을 보여주도록 할것입니다.
6. 새로 입력된 사진이 분류에 속하는 것이 없으면 새로운 분류를 추가할 것입니다. 이때에는 이름이 있을수 있고 없을수도 있습니다.
7. 새로운 분류 또는 이미지가 학습되면 전체의 index를 재 설정 할 것입니다.
8. 향후 이것을 설굴사진외의 자연사진, 간판사진등에 확장할 것입니다. 

먼저 train_myresnet.py를 수정하여 디렉토리 기반 이미지 분류 기능을 안정적으로 만듭니다.
얼굴 검출 및 임베딩 추출 기능을 별도로 구현하고 테스트합니다.
추출된 얼굴 임베딩과 이미지 정보를 저장하고 검색하는 간단한 방법을 구현합니다 (예: 딕셔너리, JSON 파일, SQLite).
유사도 검색 및 이름 태깅 기능을 구현합니다.
새로운 카테고리 처리 및 점진적 학습/인덱싱 방법을 연구하고 적용합니다.


Gemini
네, 이해했습니다! 레이블 없는 이미지들로 비지도 학습 또는 자기지도 학습을 통해 모델이 이미지의 특징을 배우게 한 다음, 사용자님의 얼굴 사진을 입력하면 그 모델이 학습했던 데이터 중에서 사용자님의 얼굴이 포함된 다른 사진들을 찾아 출력하는 시스템을 만들고 싶으신 거군요.


이 목표를 달성하기 위한 주요 단계는 다음과 같습니다:

특징 추출 모델 준비:

옵션 A (사전 학습 모델 활용): ImageNet 등으로 미리 학습된 모델(예: timm의 ResNet, Vision Transformer 등)을 불러옵니다. 이 모델은 이미 일반적인 이미지 특징을 잘 추출할 수 있습니다. 얼굴 인식 성능을 높이려면 얼굴 인식용으로 사전 학습된 모델(예: FaceNet, ArcFace 기반 모델)을 사용하는 것이 더 좋습니다. (face_recognition 라이브러리가 내부적으로 이런 모델을 사용합니다.)
옵션 B (자기지도 학습): 가지고 계신 레이블 없는 대규모 이미지 데이터셋으로 SimCLR, MoCo 같은 자기지도 학습 기법을 사용하여 직접 특징 추출 모델을 학습시킵니다. (이 방법은 상당한 컴퓨팅 자원과 시간이 필요할 수 있습니다.)
데이터셋 특징 추출 및 인덱싱:

준비된 특징 추출 모델을 사용하여 가지고 계신 모든 이미지에 대해 특징 벡터(임베딩)를 추출합니다.
얼굴 인식 강화: 더 정확하게 사용자님의 얼굴을 찾으려면, 각 이미지에서 얼굴 영역을 먼저 검출하고 (예: face_recognition 라이브러리 사용), 검출된 얼굴 영역에 대해서만 특징 벡터(얼굴 임베딩)를 추출하는 것이 좋습니다.
추출된 모든 (얼굴) 임베딩과 해당 이미지가 어떤 파일인지(파일 경로)를 함께 저장하고, 빠른 유사도 검색을 위해 인덱싱합니다. (간단하게는 NumPy 배열과 리스트로 관리하거나, 대규모 데이터셋의 경우 Faiss 같은 라이브러리를 사용합니다.)
쿼리 및 검색:

사용자님이 본인 얼굴 사진(쿼리 이미지)을 입력합니다.
쿼리 이미지에서도 얼굴 영역을 검출하고, 동일한 특징 추출 모델을 사용하여 (얼굴) 임베딩을 추출합니다.
이 쿼리 임베딩과 가장 유사한 임베딩들을 인덱스에서 검색합니다. (예: 코사인 유사도 또는 유클리드 거리 사용)
유사도 검색 시, 단순히 가장 가까운 몇 개를 찾는 것보다 일정 유사도 임계값(threshold) 이상인 임베딩만 선택하는 것이 좋습니다. 이를 통해 다른 사람의 얼굴이나 얼굴이 아닌 이미지가 검색 결과에 포함되는 것을 줄일 수 있습니다.
결과 출력:

검색된 유사 임베딩에 해당하는 이미지 파일 경로들을 가져와 사용자에게 보여줍니다.
구현을 위한 라이브러리 예시:

"""

import os
import sys
import yaml
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ExifTags
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import timm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split # 데이터 분할용

# --- 경로 설정 (개선된 방식) ---
try:
    # 1. 현재 스크립트 파일의 절대 경로 얻기
    current_file_path = Path(__file__).resolve()
    # 2. 스크립트가 있는 디렉토리 (sorc 디렉토리)
    WORK_DIR = current_file_path.parent
    # 3. 프로젝트 루트 디렉토리 (WORK_DIR의 부모 디렉토리)
    PROJ_DIR = WORK_DIR.parent

    # 4. 프로젝트 루트에 있는 .my_config.yaml 로드
    dir_config_path_yaml = PROJ_DIR / ".my_config.yaml" # pathlib의 / 연산자 사용
    if not dir_config_path_yaml.is_file():
        print(f"❌ 프로젝트 경로구성 설정 파일({dir_config_path_yaml})을 찾을 수 없습니다.")
        sys.exit("프로젝트 경로구성 설정 파일 없음")

    with open(dir_config_path_yaml, "r", encoding="utf-8") as file:
        dir_config = yaml.safe_load(file)

    # --- 필요한 경로 변수 설정 (Path 객체 사용 및 get 활용) ---
    # 설정 파일에서 프로젝트 디렉토리 확인 (선택적)
    config_proj_dir = Path(dir_config.get('resnet34_path', {}).get('PROJ_DIR', str(PROJ_DIR)))
    if PROJ_DIR != config_proj_dir:
        print(f"경고: 스크립트 위치 기반 프로젝트 경로({PROJ_DIR})와 설정 파일 경로({config_proj_dir})가 다릅니다.")
        # 필요시 여기서 처리 또는 종료
        # PROJ_DIR = config_proj_dir # 설정 파일 기준으로 맞출 경우

    # 나머지 경로들을 PROJ_DIR 기준으로 설정 (설정 파일 값 우선)
    resnet34_paths = dir_config.get('resnet34_path', {})
    worker_paths = resnet34_paths.get('worker_path', {})
    dataset_paths = resnet34_paths.get('dataset_path', {})
    output_paths = resnet34_paths.get('output_path', {})

    # 기본값 설정 시 PROJ_DIR 기반으로 설정
    WORK_DIR = Path(worker_paths.get('WORK_DIR', PROJ_DIR / 'sorc')) # WORK_DIR 재정의 (설정 우선)
    LOG_DIR = Path(worker_paths.get('LOG_DIR', WORK_DIR / 'logs'))
    MODEL_CFG_DIR = Path(worker_paths.get('CONFIG_DIR', WORK_DIR / 'config')) # 모델 설정 디렉토리 (변수명 변경: MODEL_DIR -> MODEL_CFG_DIR)
    DATA_DIR = Path(dataset_paths.get('DATA_DIR', PROJ_DIR / 'data'))
    IMAGES_DIR = Path(dataset_paths.get('IMAGES_DIR', DATA_DIR / 'images'))
    JSONS_DIR = Path(dataset_paths.get('JSONS_DIR', DATA_DIR / 'jsons')) # JSONS_DIR 추가 (필요시)
    OUTPUT_DIR = Path(output_paths.get('OUTPUT_DIR', PROJ_DIR / 'outputs'))

    # 출력 및 로그 디렉토리 생성 (존재하지 않으면)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 모델 설정 파일 경로
    model_config_path = MODEL_CFG_DIR / ".model.yaml" # MODEL_CFG_DIR 사용

except FileNotFoundError as e:
    print(f"❌ 설정 파일 관련 오류: {e}")
    sys.exit(1)
except KeyError as e:
    print(f"❌ 설정 파일 키 오류: 필요한 키 '{e}'를 찾을 수 없습니다.")
    sys.exit(1)
except Exception as e:
    print(f"❌ 초기 설정 중 오류 발생: {e}")
    import traceback
    traceback.print_exc() # 상세 오류 출력
    sys.exit(1)

# --- 로거 설정 ---
worker_name = Path(__file__).stem # 파일명만 추출 (확장자 제외)
log_file_path = LOG_DIR / f"{worker_name}.log" # 로그 파일 경로

# 기본 로거 설정 (파일 및 콘솔 출력)
logging.basicConfig(level=logging.INFO, # 로그 레벨 INFO로 설정 (DEBUG는 너무 상세할 수 있음)
                    format='%(asctime)s - %(levelname)-6s - %(filename)s:%(lineno)d - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout), # 콘솔 출력 핸들러
                              logging.FileHandler(log_file_path, encoding='utf-8')]) # 파일 출력 핸들러
logger = logging.getLogger(__name__) # 로거 객체 생성

logger.info(f"--- 스크립트 시작: {worker_name} ---")
logger.info(f"프로젝트 루트 (PROJ_DIR): {PROJ_DIR}")
logger.info(f"작업 디렉토리 (WORK_DIR) : {WORK_DIR}")
logger.info(f"데이터 루트 (DATA_DIR)  : {DATA_DIR}")
logger.info(f"이미지 디렉토리 (IMAGES_DIR): {IMAGES_DIR}")
logger.info(f"JSON 디렉토리 (JSONS_DIR) : {JSONS_DIR}") # JSONS_DIR 로깅 추가
logger.info(f"출력 디렉토리 (OUTPUT_DIR): {OUTPUT_DIR}")
logger.info(f"로그 디렉토리 (LOG_DIR)   : {LOG_DIR}")
logger.info(f"모델 설정 디렉토리 (MODEL_CFG_DIR): {MODEL_CFG_DIR}")
logger.info(f"모델 설정 파일: {model_config_path}")

# --- 모델 설정 로드 ---
if not model_config_path.is_file():
    logger.error(f"❌ 모델 설정 파일({model_config_path})을 찾을 수 없습니다.")
    sys.exit("모델 설정 파일 없음")

try:
    with open(model_config_path, "r", encoding="utf-8") as file:
        model_config = yaml.safe_load(file)
        # 설정값 로드 및 기본값 설정 (get, setdefault 사용)
        model_config['LR'] = float(model_config.get('LR', 1e-3))
        model_config.setdefault('EPOCHS', 10)
        model_config.setdefault('BATCH_SIZE', 32)
        model_config.setdefault('NUM_WORKERS', 2)
        model_config.setdefault('img_size', 224) # 이미지 크기 기본값 설정
        model_config.setdefault('model_name', 'resnet34')
        model_config.setdefault('SEED', 42)
        logger.info("모델 설정 로드 완료.")
        logger.debug(f"모델 설정 내용: {model_config}") # DEBUG 레벨로 설정 내용 로깅
except FileNotFoundError as e:
    logger.error(f"❌ 모델 설정 파일 관련 오류: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"❌ 모델 설정 파일 로드 중 오류 발생: {e}", exc_info=True) # 상세 오류 로깅
    sys.exit(1)

# --- 사용되지 않는 함수 제거 ---
# def substitute_vars(data): # 이 함수는 사용되지 않으므로 제거합니다.
#     ...

# --- 환경 초기화 ---
def init_env(seed: int):
    """환경 시드 고정 함수"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    # torch.backends.cudnn.deterministic = True # 주석 처리 (성능 저하 가능성)
    torch.backends.cudnn.benchmark = True # 일반적으로 성능 향상에 도움 (입력 크기 고정 시)
    logger.info(f"환경 시드 고정 완료 (SEED={seed})")

# --- 이미지 회전 처리 ---
# EXIF Orientation 태그 값 찾기
EXIF_ORIENTATION_TAG = None
for k, v in ExifTags.TAGS.items():
    if v == 'Orientation':
        EXIF_ORIENTATION_TAG = k
        break

def rotate_image_based_on_exif(image: Image.Image) -> Image.Image:
    """EXIF 정보에 따라 이미지를 회전시킵니다."""
    if EXIF_ORIENTATION_TAG is None: return image # Orientation 태그 정보 없으면 원본 반환
    try:
        exif = image.getexif() # 이미지에서 EXIF 데이터 추출
        orientation = exif.get(EXIF_ORIENTATION_TAG) # Orientation 값 가져오기
    except Exception: # EXIF 정보가 없거나 손상된 경우
        orientation = None

    # Orientation 값에 따라 이미지 회전 또는 반전 적용
    if orientation == 2: return image.transpose(Image.FLIP_LEFT_RIGHT) # 좌우 반전
    elif orientation == 3: return image.rotate(180) # 180도 회전
    elif orientation == 4: return image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT) # 180도 회전 후 좌우 반전
    elif orientation == 5: return image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT) # 시계방향 90도 회전 후 좌우 반전
    elif orientation == 6: return image.rotate(-90, expand=True) # 시계방향 90도 회전
    elif orientation == 7: return image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT) # 반시계방향 90도 회전 후 좌우 반전
    elif orientation == 8: return image.rotate(90, expand=True) # 반시계방향 90도 회전
    return image # 해당 없으면 원본 반환

# --- 이미지 분류 데이터셋 ---
class ClassificationDataset(Dataset):
    """
    train.csv 파일을 읽어 이미지 분류 학습용 데이터를 제공하는 Dataset 클래스.
    """
    def __init__(self, df: pd.DataFrame, image_root: Path, transform: Optional[A.Compose] = None):
        """
        Args:
            df (pd.DataFrame): 'ID' (이미지 파일명)와 'target' (레이블) 컬럼을 포함하는 데이터프레임.
            image_root (Path): 이미지 파일들이 있는 루트 디렉토리 경로.
            transform (Optional[A.Compose]): Albumentations 변환 파이프라인.
        """
        self.df = df.reset_index(drop=True) # 인덱스 재설정 (iloc 접근 용이)
        self.image_root = image_root
        self.transform = transform
        logger.info(f"Dataset 생성: {len(self.df)}개 샘플, 이미지 루트: {self.image_root}")

    def __len__(self) -> int:
        """데이터셋의 총 샘플 수를 반환합니다."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """데이터셋에서 인덱스(idx)에 해당하는 단일 샘플(이미지 텐서, 레이블)을 가져옵니다."""
        image_id = self.df.loc[idx, 'ID'] # 이미지 파일명 가져오기
        target = int(self.df.loc[idx, 'target']) # 레이블 가져오기 (정수형으로 변환)
        image_path = self.image_root / image_id # 이미지 전체 경로 생성

        try:
            # PIL을 사용하여 이미지 로드 및 RGB 변환
            image = Image.open(image_path).convert('RGB')
            # EXIF 정보 기반 회전 적용
            image = rotate_image_based_on_exif(image)
            # Albumentations 적용을 위해 NumPy 배열로 변환
            image_np = np.array(image)

        except FileNotFoundError:
            logger.warning(f"이미지 파일 없음: {image_path} (인덱스 {idx}). 더미 데이터 반환.")
            # 오류 발생 시 더미 데이터 반환 (모델 입력 크기에 맞춰서)
            dummy_image = torch.zeros((3, model_config['img_size'], model_config['img_size']), dtype=torch.float32)
            return dummy_image, -1 # 오류 식별용 레이블 (-1)
        except Exception as e:
            logger.error(f"이미지 로딩/처리 오류 {image_path}: {e}. 더미 데이터 반환.", exc_info=True)
            dummy_image = torch.zeros((3, model_config['img_size'], model_config['img_size']), dtype=torch.float32)
            return dummy_image, -1

        # Albumentations 변환 적용
        if self.transform:
            try:
                transformed = self.transform(image=image_np) # 변환 적용
                image_tensor = transformed['image'] # 변환된 이미지 텐서 추출
            except Exception as e:
                 logger.error(f"Albumentations 변환 적용 오류 {image_path}: {e}. 더미 데이터 반환.", exc_info=True)
                 image_tensor = torch.zeros((3, model_config['img_size'], model_config['img_size']), dtype=torch.float32)
                 return image_tensor, -1
        else:
            # 변환(transform)이 정의되지 않은 경우, 기본적인 Tensor 변환 수행 (권장하지 않음)
            logger.warning("Albumentations 변환(transform)이 정의되지 않았습니다. 기본 Tensor 변환을 수행합니다.")
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0 # HWC -> CHW, 0-1 정규화

        return image_tensor, target

# --- 데이터 로더 생성 ---
def build_dataloader(df: pd.DataFrame, image_root: Path, model_cfg: Dict, is_train: bool) -> DataLoader:
    """
    주어진 데이터프레임과 설정을 기반으로 PyTorch DataLoader를 생성합니다.

    Args:
        df (pd.DataFrame): 데이터 정보 (ID, target)를 담은 데이터프레임.
        image_root (Path): 이미지 파일 루트 경로.
        model_cfg (Dict): 모델 및 학습 관련 설정값 딕셔너리.
        is_train (bool): 학습용 데이터 로더인지 여부 (True: 학습용, False: 검증/테스트용).

    Returns:
        DataLoader: 생성된 PyTorch DataLoader 객체.
    """
    img_size = model_cfg['img_size']
    batch_size = model_cfg['BATCH_SIZE']
    num_workers = model_cfg['NUM_WORKERS']

    # 데이터 증강(Augmentation) 및 전처리 설정
    if is_train:
        # 학습용 변환: 크기 조정, 좌우 반전, 회전, 밝기/대비 조절, 정규화, 텐서 변환
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
            A.Rotate(limit=15, p=0.3), # -15 ~ +15도 범위 내에서 30% 확률로 회전
            A.RandomBrightnessContrast(p=0.2), # 20% 확률로 밝기/대비 랜덤 조절
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet 표준 정규화
            ToTensorV2(), # NumPy 배열을 PyTorch 텐서로 변환 (CHW 순서)
        ])
        logger.info("학습용 데이터 변환 설정 완료 (증강 포함).")
    else:
        # 검증/테스트용 변환: 크기 조정, 정규화, 텐서 변환 (데이터 증강 없음)
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        logger.info("검증/테스트용 데이터 변환 설정 완료 (증강 없음).")

    # 데이터셋 인스턴스 생성
    dataset = ClassificationDataset(df=df, image_root=image_root, transform=transform)

    # DataLoader 생성
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train, # 학습 시에만 데이터 섞기 (is_train=True)
        num_workers=num_workers, # 데이터 로딩 병렬 처리 워커 수
        pin_memory=True, # GPU 사용 시 데이터 전송 속도 향상을 위해 메모리 고정
        drop_last=False # 마지막 배치가 배치 크기보다 작더라도 사용
    )
    logger.info(f"{'학습' if is_train else '검증'} DataLoader 생성 완료 (Batch size: {batch_size}, Num workers: {num_workers})")
    return loader

# --- 학습 및 검증 루프 ---
def train_one_epoch(loader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module, device: torch.device, epoch_num: int, total_epochs: int) -> Dict[str, float]:
    """1 에폭(epoch) 학습을 수행하고 결과를 반환합니다."""
    model.train() # 모델을 학습 모드로 설정 (Dropout, BatchNorm 등 활성화)
    total_loss = 0.0
    all_preds = [] # 모든 예측값을 저장할 리스트
    all_targets = [] # 모든 실제 레이블을 저장할 리스트
    dummy_data_count = 0 # 더미 데이터 개수 카운트

    # tqdm을 사용하여 진행률 표시 (에폭 정보 포함)
    pbar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{total_epochs} Training")
    for images, targets in pbar:
        # 데이터를 지정된 장치(GPU 또는 CPU)로 이동
        images = images.to(device, non_blocking=True) # non_blocking=True: 비동기 전송 (GPU 사용 시)
        targets = targets.to(device, non_blocking=True)

        # 데이터셋에서 오류로 인해 반환된 더미 데이터(-1 레이블) 건너뛰기
        if -1 in targets:
            num_dummy = torch.sum(targets == -1).item()
            dummy_data_count += num_dummy
            logger.warning(f"더미 데이터 {num_dummy}개 포함된 배치를 건너<0xEB><0><0x8F>니다.")
            continue

        # 옵티마이저 그래디언트 초기화
        optimizer.zero_grad()

        # 모델 예측 수행 (Forward pass)
        preds = model(images)

        # 손실 계산
        loss = loss_fn(preds, targets)

        # 손실 값이 NaN인지 확인 (학습 불안정성 감지)
        if torch.isnan(loss):
            logger.error(f"NaN 손실 감지 (Epoch {epoch_num+1}). 배치를 건너<0xEB><0><0x8F>니다.")
            continue

        # 역전파 수행 (Backward pass)
        loss.backward()
        # 옵티마이저 파라미터 업데이트
        optimizer.step()

        # 배치 손실 누적 (데이터셋 전체 평균 계산 위함)
        total_loss += loss.item() * images.size(0) # 배치 크기 곱해서 더함
        # 예측값(가장 확률 높은 클래스 인덱스)과 실제 레이블 저장 (나중에 정확도, F1 계산용)
        all_preds.extend(preds.argmax(dim=1).detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

        # 진행률 표시줄에 현재 배치 손실 표시 (소수점 4자리까지)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 에폭 평균 손실 계산
    avg_loss = total_loss / (len(loader.dataset) - dummy_data_count) if (len(loader.dataset) - dummy_data_count) > 0 else 0.0
    # 정확도 계산
    accuracy = accuracy_score(all_targets, all_preds)
    # F1 스코어 계산 (Macro average: 각 클래스 F1 점수 평균, zero_division=0: 특정 클래스 예측 없어도 오류 방지)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    # 더미 데이터 처리 로그
    if dummy_data_count > 0:
        logger.warning(f"Epoch {epoch_num+1} Training: 총 {dummy_data_count}개의 더미 데이터 처리됨.")

    # 결과 딕셔너리 반환
    return {"train_loss": avg_loss, "train_acc": accuracy, "train_f1": f1}

def validate_one_epoch(loader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: torch.device, epoch_num: int, total_epochs: int) -> Dict[str, float]:
    """1 에폭(epoch) 검증을 수행하고 결과를 반환합니다."""
    model.eval() # 모델을 평가 모드로 설정 (Dropout, BatchNorm 등 비활성화)
    total_loss = 0.0
    all_preds = []
    all_targets = []
    dummy_data_count = 0

    # tqdm을 사용하여 진행률 표시
    pbar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{total_epochs} Validation")
    # 그래디언트 계산 비활성화 (메모리 절약 및 속도 향상)
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 더미 데이터 건너뛰기
            if -1 in targets:
                num_dummy = torch.sum(targets == -1).item()
                dummy_data_count += num_dummy
                continue

            # 모델 예측
            preds = model(images)
            # 손실 계산
            loss = loss_fn(preds, targets)

            # NaN 손실 확인
            if torch.isnan(loss):
                logger.error(f"NaN 손실 감지 (Epoch {epoch_num+1} Validation). 배치를 건너<0xEB><0><0x8F>니다.")
                continue

            # 손실 누적 및 예측/실제 레이블 저장
            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            # 진행률 표시줄에 손실 표시
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 평균 손실, 정확도, F1 스코어 계산
    avg_loss = total_loss / (len(loader.dataset) - dummy_data_count) if (len(loader.dataset) - dummy_data_count) > 0 else 0.0
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    # 더미 데이터 처리 로그
    if dummy_data_count > 0:
        logger.warning(f"Epoch {epoch_num+1} Validation: 총 {dummy_data_count}개의 더미 데이터 처리됨.")

    return {"val_loss": avg_loss, "val_acc": accuracy, "val_f1": f1}


# --- 메인 학습 함수 ---
def training():
    """메인 학습 프로세스를 실행합니다."""
    # 환경 초기화 (시드 고정)
    init_env(seed=model_config['SEED'])

    # 장치 설정 (GPU 우선 사용)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 디바이스: {device}")

    # --- 데이터 로드 및 분할 ---
    try:
        train_csv_path = DATA_DIR / "train.csv" # DATA_DIR 사용
        if not train_csv_path.is_file():
            logger.error(f"학습 데이터 파일({train_csv_path})을 찾을 수 없습니다.")
            sys.exit(1)

        # train.csv 파일 로드
        full_df = pd.read_csv(train_csv_path)
        logger.info(f"{train_csv_path} 로드 완료. 총 {len(full_df)}개 데이터.")

        # 클래스 개수 확인 (meta.csv 또는 train.csv의 target 값 기준)
        meta_csv_path = DATA_DIR / "meta.csv" # DATA_DIR 사용
        if meta_csv_path.is_file():
            meta_df = pd.read_csv(meta_csv_path)
            num_classes = len(meta_df)
            logger.info(f"meta.csv 기반 클래스 개수: {num_classes}")
        else:
            num_classes = full_df['target'].nunique() # train.csv에서 고유 레이블 개수 계산
            logger.warning(f"meta.csv 파일을 찾을 수 없습니다. train.csv 기반 클래스 개수: {num_classes}")

        # 클래스 개수가 2개 미만이면 오류 처리
        if num_classes < 2:
            logger.error("분류할 클래스가 2개 미만입니다. 데이터 또는 설정을 확인하세요.")
            sys.exit(1)

        # 학습/검증 데이터 분할 (예: 80% 학습, 20% 검증)
        # stratify=full_df['target']: 분할 시 각 클래스의 비율을 원본 데이터와 유사하게 유지
        train_df, val_df = train_test_split(
            full_df,
            test_size=0.2, # 검증 데이터 비율 (20%)
            random_state=model_config['SEED'], # 재현성을 위한 랜덤 시드
            stratify=full_df['target'] # 클래스 비율 유지 옵션
        )
        logger.info(f"데이터 분할 완료: 학습 {len(train_df)}개, 검증 {len(val_df)}개")

        # 이미지 루트 디렉토리 설정 (IMAGES_DIR 사용)
        if not IMAGES_DIR.is_dir():
             logger.error(f"이미지 디렉토리({IMAGES_DIR})가 존재하지 않습니다.")
             # 필요시 DATA_DIR을 대체 경로로 사용하거나 종료
             # image_root_dir = DATA_DIR
             # logger.warning(f"IMAGES_DIR({IMAGES_DIR})가 존재하지 않아 DATA_DIR({DATA_DIR})를 이미지 루트로 사용합니다.")
             sys.exit(1)
        image_root_dir = IMAGES_DIR

        # 데이터 로더 생성
        train_loader = build_dataloader(train_df, image_root_dir, model_config, is_train=True)
        val_loader = build_dataloader(val_df, image_root_dir, model_config, is_train=False)

    except Exception as e:
        logger.error(f"데이터 로딩/분할/로더 생성 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)

    # --- 모델 생성 ---
    try:
        # timm 라이브러리를 사용하여 사전 학습된 모델 로드
        model = timm.create_model(
            model_name=model_config['model_name'], # 설정 파일에서 모델 이름 가져오기
            pretrained=True, # ImageNet 등으로 사전 학습된 가중치 사용
            num_classes=num_classes # 계산된 클래스 개수 적용 (마지막 분류 레이어 수정)
        ).to(device) # 모델을 지정된 장치(GPU/CPU)로 이동
        logger.info(f"모델 '{model_config['model_name']}' 생성 완료 (클래스 수: {num_classes})")
    except Exception as e:
        logger.error(f"모델 생성 실패 '{model_config['model_name']}': {e}", exc_info=True)
        sys.exit(1)

    # --- 손실 함수 및 옵티마이저 ---
    loss_fn = nn.CrossEntropyLoss() # 다중 클래스 분류에 표준적으로 사용되는 CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=model_config['LR']) # Adam 옵티마이저 사용
    # 학습률 스케줄러 (선택 사항): 특정 조건에 따라 학습률 동적 조절
    # 예: scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # 5 에폭마다 학습률 0.1배 감소
    logger.info(f"손실 함수: CrossEntropyLoss, 옵티마이저: Adam (LR={model_config['LR']})")

    # --- 학습 루프 ---
    epochs = model_config['EPOCHS'] # 설정 파일에서 총 에폭 수 가져오기
    logger.info(f"--- 총 {epochs} 에폭 학습 시작 ---")

    best_val_f1 = 0.0 # 최고 검증 F1 점수 추적용 변수 초기화

    for epoch in range(epochs):
        # 1 에폭 학습 수행 및 결과 저장
        train_results = train_one_epoch(train_loader, model, optimizer, loss_fn, device, epoch, epochs)
        # 1 에폭 검증 수행 및 결과 저장
        val_results = validate_one_epoch(val_loader, model, loss_fn, device, epoch, epochs)

        # 에폭 결과 로깅 (학습/검증 손실, 정확도, F1 스코어)
        log_msg = (
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_results['train_loss']:.4f}, Acc: {train_results['train_acc']:.4f}, F1: {train_results['train_f1']:.4f} | "
            f"Val Loss: {val_results['val_loss']:.4f}, Acc: {val_results['val_acc']:.4f}, F1: {val_results['val_f1']:.4f}"
        )
        logger.info(log_msg)

        # 학습률 스케줄러 업데이트 (사용 시)
        # scheduler.step()

        # --- 모델 저장 (체크포인트) ---
        current_val_f1 = val_results['val_f1']
        is_best = current_val_f1 > best_val_f1 # 현재 F1이 최고 F1보다 좋은지 확인

        if is_best:
            best_val_f1 = current_val_f1
            # 저장 파일명 형식: best_model_epoch_{에폭번호}_f1_{F1점수}.pth
            save_path = OUTPUT_DIR / f"best_model_epoch_{epoch+1}_f1_{best_val_f1:.4f}.pth" # OUTPUT_DIR 사용
            try:
                # 체크포인트 저장 (모델 상태, 옵티마이저 상태, 에폭 번호, 최고 점수, 설정 등)
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(), # 모델 파라미터
                    'optimizer_state_dict': optimizer.state_dict(), # 옵티마이저 상태
                    'best_val_f1': best_val_f1, # 최고 검증 F1 점수
                    # 'scheduler_state_dict': scheduler.state_dict(), # 스케줄러 사용 시 상태 저장
                    'model_config': model_config # 학습 시 사용한 설정 저장
                }
                torch.save(checkpoint, save_path)
                logger.info(f"🚀 최고 성능 모델 체크포인트 저장 완료: {save_path} (Val F1: {best_val_f1:.4f})")
            except Exception as e:
                logger.error(f"모델 체크포인트 저장 실패: {e}", exc_info=True)

        # 마지막 에폭 모델 저장 (선택 사항, state_dict만 저장)
        if epoch == epochs - 1:
             last_save_path = OUTPUT_DIR / f"last_model_epoch_{epoch+1}.pth" # OUTPUT_DIR 사용
             try:
                 # 마지막 에폭 모델의 파라미터만 저장
                 torch.save(model.state_dict(), last_save_path)
                 logger.info(f"마지막 에폭 모델 저장 완료: {last_save_path}")
             except Exception as e:
                 logger.error(f"마지막 모델 저장 실패: {e}", exc_info=True)

    logger.info(f"--- 총 {epochs} 에폭 학습 완료 ---")
    logger.info(f"최고 검증 F1 점수: {best_val_f1:.4f}")

# ===================== 스크립트 실행 진입점 =====================
if __name__ == "__main__":
    try:
        training() # 메인 학습 함수 호출
    except Exception as e:
        # 예상치 못한 오류 발생 시 로깅 및 종료
        logger.exception(f"스크립트 실행 중 치명적인 오류 발생: {e}") # 예외 정보 포함 로깅
        sys.exit(1) # 비정상 종료 상태 코드 반환
