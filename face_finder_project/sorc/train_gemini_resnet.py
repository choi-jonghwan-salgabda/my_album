# -*- coding: utf-8 -*-
"""
train_myresnet.py (수정 버전)

[목적]
- .my_config.yaml 설정을 기반으로 이미지 분류 모델(예: ResNet)을 지도 학습 방식으로 학습시킵니다.
- PyTorch와 timm 라이브러리를 사용합니다.
- 학습된 모델은 향후 특징 추출 및 유사성 검색에 활용될 수 있습니다.

[주요 변경 사항]
- 경로 설정, 하이퍼파라미터, 로깅 메시지 등을 .my_config.yaml 파일에서 로드하여 사용합니다.
- pathlib를 사용하여 경로를 객체 지향적으로 처리합니다.
- 설정 파일 로드 및 처리 부분을 강화했습니다.
- 데이터셋 클래스 및 데이터 로더가 설정 파일의 경로를 사용하도록 수정했습니다.
- 학습 루프에서 설정 파일의 하이퍼파라미터를 사용하도록 수정했습니다.
- 모델 저장 시 설정 파일의 출력 경로를 사용합니다.
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
from PIL import Image, ExifTags, UnidentifiedImageError # UnidentifiedImageError 추가
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import timm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split # 데이터 분할용

# --- 경로 설정 및 확장 함수 ---
def expand_path_vars(path_str: Union[str, Path], proj_dir: Path, data_dir: Optional[Path] = None, root_dir: Optional[Path] = None) -> Path:
    """YAML에서 읽은 경로 문자열 내의 변수(${PROJ_DIR}, ${DATA_DIR}, $(ROOT_DIR))와 ~, $~ 를 확장합니다."""
    if isinstance(path_str, Path): # 이미 Path 객체면 절대 경로 보장
        return path_str.resolve()
    if not isinstance(path_str, str):
        raise TypeError(f"Path must be a string or Path object, got: {type(path_str)}")

    expanded_str = path_str

    # $(ROOT_DIR) 치환 (root_dir이 제공된 경우)
    if root_dir and '$(ROOT_DIR)' in expanded_str:
        expanded_str = expanded_str.replace('$(ROOT_DIR)', str(root_dir))

    # ${PROJ_DIR} 치환
    expanded_str = expanded_str.replace('${PROJ_DIR}', str(proj_dir))

    # ${DATA_DIR} 치환 (data_dir이 결정된 후에 사용 가능)
    # YAML의 ${DATAS_DIR}는 ${DATA_DIR}의 오타로 간주
    if data_dir:
        expanded_str = expanded_str.replace('${DATA_DIR}', str(data_dir))
        expanded_str = expanded_str.replace('${DATAS_DIR}', str(data_dir)) # 오타 가능성 고려

    # ~ (홈 디렉토리) 확장 및 비표준 $~/ 처리 시도
    if '$~/' in expanded_str:
        expanded_str = expanded_str.replace('$~/','~/')
    if expanded_str.startswith('~'):
        expanded_str = os.path.expanduser(expanded_str)

    # Path 객체 생성 및 절대 경로화 (resolve)
    try:
        final_path = Path(expanded_str)
        # 상대 경로일 경우 PROJ_DIR 기준으로 절대 경로화 시도
        if not final_path.is_absolute():
             final_path = (proj_dir / final_path).resolve()
        else:
             final_path = final_path.resolve() # 이미 절대경로여도 resolve()는 안전
        return final_path
    except Exception as e:
        print(f"오류: 경로 문자열 '{path_str}' (확장 후: '{expanded_str}') 처리 중 오류: {e}")
        raise

# --- 설정 로드 및 경로 변수 설정 ---
try:
    current_file_path = Path(__file__).resolve()
    WORK_DIR_SCRIPT = current_file_path.parent
    PROJ_DIR_SCRIPT = WORK_DIR_SCRIPT.parent
    dir_config_path_yaml = PROJ_DIR_SCRIPT / ".my_config.yaml"

    if not dir_config_path_yaml.is_file():
        print(f"❌ 프로젝트 경로구성 설정 파일({dir_config_path_yaml})을 찾을 수 없습니다.")
        sys.exit("프로젝트 경로구성 설정 파일 없음")

    with open(dir_config_path_yaml, "r", encoding="utf-8") as file:
        dir_config = yaml.safe_load(file)

    # --- 기본 경로 결정 ---
    # ROOT_DIR (설정 파일 우선, 없으면 기본값 추정)
    raw_root_dir = dir_config.get('ROOT_DIR')
    ROOT_DIR = Path(raw_root_dir).resolve() if raw_root_dir else PROJ_DIR_SCRIPT.parent # 기본값: 프로젝트 상위

    # PROJ_DIR (설정 파일 우선, 없으면 스크립트 위치 기반)
    raw_proj_dir = dir_config.get('resnet34_path', {}).get('PROJ_DIR')
    if raw_proj_dir:
        PROJ_DIR = expand_path_vars(raw_proj_dir, PROJ_DIR_SCRIPT, root_dir=ROOT_DIR) # ROOT_DIR 참조 확장
    else:
        PROJ_DIR = PROJ_DIR_SCRIPT
        print(f"정보: 설정 파일에 PROJ_DIR이 없어 스크립트 위치 기반 경로 사용: {PROJ_DIR}")

    # --- 세부 경로 설정 (설정 파일 값 우선, 없으면 기본값 + 확장) ---
    resnet34_paths = dir_config.get('resnet34_path', {})
    worker_paths = resnet34_paths.get('worker_path', {})
    dataset_paths = resnet34_paths.get('dataset_path', {})
    output_paths = resnet34_paths.get('output_path', {})
    message_str = resnet34_paths.get('MESSAGE', {})
    model_params_cfg = resnet34_paths.get('model_params', {}) # 모델 파라미터 섹션 로드

    # WORK_DIR
    raw_work_dir = worker_paths.get('WORK_DIR', '${PROJ_DIR}/sorc') # 기본값에 변수 사용
    WORK_DIR = expand_path_vars(raw_work_dir, PROJ_DIR, root_dir=ROOT_DIR)

    # LOG_DIR
    raw_log_dir = worker_paths.get('LOG_DIR', '${PROJ_DIR}/logs')
    LOG_DIR = expand_path_vars(raw_log_dir, PROJ_DIR, root_dir=ROOT_DIR)

    # DATA_DIR
    raw_data_dir = dataset_paths.get('DATA_DIR', '${PROJ_DIR}/data')
    DATA_DIR = expand_path_vars(raw_data_dir, PROJ_DIR, root_dir=ROOT_DIR)

    # IMAGES_DIR (DATA_DIR 결정 후 확장)
    raw_images_dir = dataset_paths.get('IMAGES_DIR', '${DATA_DIR}/train') # 기본값 변경: 학습 이미지 경로
    IMAGES_DIR = expand_path_vars(raw_images_dir, PROJ_DIR, data_dir=DATA_DIR, root_dir=ROOT_DIR)

    # TRAIN_CSV 경로
    raw_train_csv = dataset_paths.get('TRAIN_CSV', '${DATA_DIR}/train.csv')
    TRAIN_CSV_PATH = expand_path_vars(raw_train_csv, PROJ_DIR, data_dir=DATA_DIR, root_dir=ROOT_DIR)

    # META_CSV 경로
    raw_meta_csv = dataset_paths.get('META_CSV', '${DATA_DIR}/meta.csv')
    META_CSV_PATH = expand_path_vars(raw_meta_csv, PROJ_DIR, data_dir=DATA_DIR, root_dir=ROOT_DIR)

    # OUTPUT_DIR
    raw_output_dir = output_paths.get('OUTPUT_DIR', '${PROJ_DIR}/outputs')
    OUTPUT_DIR = expand_path_vars(raw_output_dir, PROJ_DIR, root_dir=ROOT_DIR)

    # 디렉토리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- 모델 하이퍼파라미터 로드 ---
    model_config = {
        'LR': float(model_params_cfg.get('LR', 1e-3)),
        'EPOCHS': int(model_params_cfg.get('EPOCHS', 10)),
        'BATCH_SIZE': int(model_params_cfg.get('BATCH_SIZE', 32)),
        'NUM_WORKERS': int(model_params_cfg.get('NUM_WORKERS', 2)),
        'IMG_SIZE': int(model_params_cfg.get('IMG_SIZE', 224)),
        'MODEL_NAME': model_params_cfg.get('MODEL_NAME', 'resnet34'),
        'SEED': int(model_params_cfg.get('SEED', 42))
    }

except FileNotFoundError as e:
    print(f"❌ 설정 파일 관련 오류: {e}")
    sys.exit(1)
except KeyError as e:
    print(f"❌ 설정 파일 키 오류: 필요한 키 '{e}'를 찾을 수 없습니다.")
    sys.exit(1)
except Exception as e:
    print(f"❌ 초기 설정 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 로거 설정 ---
worker_name = Path(__file__).stem
log_file_path = LOG_DIR / f"{worker_name}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)-6s - %(filename)s:%(lineno)d - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(log_file_path, encoding='utf-8')])
logger = logging.getLogger(__name__)

# --- 로깅 메시지 (설정 파일 메시지 사용) ---
logger.info(f"{message_str.get('START', '발자욱 만들기 시작')}:10s: {worker_name}")
logger.info(f"{message_str.get('PROJ_DIR', '프로젝트 위치')}:10s: {PROJ_DIR}")
logger.info(f"{message_str.get('WORK_DIR', '작업 디렉토리')}:10s: {WORK_DIR}")
logger.info(f"{message_str.get('DATA_DIR', '데이터 위치')}:10s: {DATA_DIR}")
logger.info(f"{message_str.get('IMAGES_DIR', '이미지 위치')}:10s: {IMAGES_DIR}")
logger.info(f"{message_str.get('OUTPUT_DIR', '출력 위치')}:10s: {OUTPUT_DIR}")
logger.info(f"{message_str.get('LOG_DIR', '로그 위치')}:10s: {LOG_DIR}")
logger.info(f"학습 데이터 CSV: {TRAIN_CSV_PATH}")
logger.info(f"메타 데이터 CSV: {META_CSV_PATH}")
logger.info(f"모델 설정: {model_config}")


# --- 환경 초기화 ---
def init_env(seed: int):
    """환경 시드 고정 함수"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    torch.backends.cudnn.benchmark = True
    logger.info(f"환경 시드 고정 완료 (SEED={seed})")

# --- 이미지 회전 처리 ---
EXIF_ORIENTATION_TAG = None
for k, v in ExifTags.TAGS.items():
    if v == 'Orientation':
        EXIF_ORIENTATION_TAG = k
        break

def rotate_image_based_on_exif(image: Image.Image) -> Image.Image:
    """EXIF 정보에 따라 이미지를 회전시킵니다."""
    if EXIF_ORIENTATION_TAG is None: return image
    try:
        exif = image.getexif()
        orientation = exif.get(EXIF_ORIENTATION_TAG)
    except Exception:
        orientation = None

    if orientation == 2: return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3: return image.rotate(180)
    elif orientation == 4: return image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 5: return image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 6: return image.rotate(-90, expand=True)
    elif orientation == 7: return image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 8: return image.rotate(90, expand=True)
    return image

# --- 이미지 분류 데이터셋 ---
class ClassificationDataset(Dataset):
    """
    train.csv 파일을 읽어 이미지 분류 학습용 데이터를 제공하는 Dataset 클래스.
    """
    def __init__(self, df: pd.DataFrame, image_root: Path, img_size: int, transform: Optional[A.Compose] = None):
        """
        Args:
            df (pd.DataFrame): 'ID' (이미지 파일명)와 'target' (레이블) 컬럼을 포함하는 데이터프레임.
            image_root (Path): 이미지 파일들이 있는 루트 디렉토리 경로.
            img_size (int): 모델 입력 이미지 크기 (더미 데이터 생성 시 사용).
            transform (Optional[A.Compose]): Albumentations 변환 파이프라인.
        """
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.img_size = img_size # 더미 데이터 생성 위해 저장
        logger.info(f"Dataset 생성: {len(self.df)}개 샘플, 이미지 루트: {self.image_root}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_id = self.df.loc[idx, 'ID']
        target = int(self.df.loc[idx, 'target'])
        image_path = self.image_root / image_id

        try:
            image = Image.open(image_path).convert('RGB')
            image = rotate_image_based_on_exif(image)
            image_np = np.array(image)

        except FileNotFoundError:
            logger.warning(f"이미지 파일 없음: {image_path} (인덱스 {idx}). 더미 데이터 반환.")
            dummy_image = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            return dummy_image, -1
        except UnidentifiedImageError: # PIL이 식별할 수 없는 이미지 처리
             logger.warning(f"이미지 파일 식별 불가: {image_path}. 더미 데이터 반환.")
             dummy_image = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
             return dummy_image, -1
        except Exception as e:
            logger.error(f"이미지 로딩/처리 오류 {image_path}: {e}. 더미 데이터 반환.", exc_info=False) # exc_info=False로 변경 (너무 길어짐 방지)
            dummy_image = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            return dummy_image, -1

        if self.transform:
            try:
                transformed = self.transform(image=image_np)
                image_tensor = transformed['image']
            except Exception as e:
                 logger.error(f"Albumentations 변환 적용 오류 {image_path}: {e}. 더미 데이터 반환.", exc_info=False)
                 image_tensor = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
                 return image_tensor, -1
        else:
            # 기본 변환 (Albumentations 없을 경우)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, target

# --- 데이터 로더 생성 ---
def build_dataloader(df: pd.DataFrame, image_root: Path, model_cfg: Dict, is_train: bool) -> DataLoader:
    """DataLoader 생성 함수"""
    img_size = model_cfg['IMG_SIZE'] # 키 이름 변경 IMG_SIZE
    batch_size = model_cfg['BATCH_SIZE']
    num_workers = model_cfg['NUM_WORKERS']

    if is_train:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        logger.info("학습용 데이터 변환 설정 완료 (증강 포함).")
    else:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        logger.info("검증/테스트용 데이터 변환 설정 완료 (증강 없음).")

    # 데이터셋 인스턴스 생성 시 img_size 전달
    dataset = ClassificationDataset(df=df, image_root=image_root, img_size=img_size, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    logger.info(f"{'학습' if is_train else '검증'} DataLoader 생성 완료 (Batch size: {batch_size}, Num workers: {num_workers})")
    return loader

# --- 학습 및 검증 루프 ---
def train_one_epoch(loader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module, device: torch.device, epoch_num: int, total_epochs: int) -> Dict[str, float]:
    """1 에폭 학습"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    dummy_data_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{total_epochs} Training")
    for images, targets in pbar:
        # 더미 데이터(-1 레이블) 건너뛰기
        valid_indices = targets != -1
        if not valid_indices.all(): # 더미 데이터가 하나라도 있으면
            num_dummy = (~valid_indices).sum().item()
            dummy_data_count += num_dummy
            # 유효한 데이터만 필터링
            images = images[valid_indices]
            targets = targets[valid_indices]
            if images.size(0) == 0: # 배치 전체가 더미 데이터면 건너뛰기
                logger.warning(f"배치 전체가 더미 데이터 ({num_dummy}개). 건너<0xEB><0><0x8F>니다.")
                continue
            logger.warning(f"배치 내 더미 데이터 {num_dummy}개 제외하고 처리.")

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, targets)

        if torch.isnan(loss):
            logger.error(f"NaN 손실 감지 (Epoch {epoch_num+1}). 배치를 건너<0xEB><0><0x8F>니다.")
            continue

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.argmax(dim=1).detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 유효하게 처리된 샘플 수 계산 (전체 - 더미)
    num_valid_samples = len(all_targets) # 필터링 후 실제 처리된 샘플 수
    avg_loss = total_loss / num_valid_samples if num_valid_samples > 0 else 0.0
    accuracy = accuracy_score(all_targets, all_preds) if num_valid_samples > 0 else 0.0
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) if num_valid_samples > 0 else 0.0

    if dummy_data_count > 0:
        logger.warning(f"Epoch {epoch_num+1} Training: 총 {dummy_data_count}개의 더미 데이터 처리됨.")

    return {"train_loss": avg_loss, "train_acc": accuracy, "train_f1": f1}

def validate_one_epoch(loader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: torch.device, epoch_num: int, total_epochs: int) -> Dict[str, float]:
    """1 에폭 검증"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    dummy_data_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{total_epochs} Validation")
    with torch.no_grad():
        for images, targets in pbar:
            # 더미 데이터 처리
            valid_indices = targets != -1
            if not valid_indices.all():
                num_dummy = (~valid_indices).sum().item()
                dummy_data_count += num_dummy
                images = images[valid_indices]
                targets = targets[valid_indices]
                if images.size(0) == 0:
                    continue

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = model(images)
            loss = loss_fn(preds, targets)

            if torch.isnan(loss):
                logger.error(f"NaN 손실 감지 (Epoch {epoch_num+1} Validation). 배치를 건너<0xEB><0><0x8F>니다.")
                continue

            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    num_valid_samples = len(all_targets)
    avg_loss = total_loss / num_valid_samples if num_valid_samples > 0 else 0.0
    accuracy = accuracy_score(all_targets, all_preds) if num_valid_samples > 0 else 0.0
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) if num_valid_samples > 0 else 0.0

    if dummy_data_count > 0:
        logger.warning(f"Epoch {epoch_num+1} Validation: 총 {dummy_data_count}개의 더미 데이터 처리됨.")

    return {"val_loss": avg_loss, "val_acc": accuracy, "val_f1": f1}


# --- 메인 학습 함수 ---
def training():
    """메인 학습 프로세스를 실행합니다."""
    init_env(seed=model_config['SEED'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 디바이스: {device}")

    # --- 데이터 로드 및 분할 ---
    try:
        if not TRAIN_CSV_PATH.is_file():
            logger.error(f"학습 데이터 파일({TRAIN_CSV_PATH})을 찾을 수 없습니다.")
            sys.exit(1)
        full_df = pd.read_csv(TRAIN_CSV_PATH)
        logger.info(f"{TRAIN_CSV_PATH} 로드 완료. 총 {len(full_df)}개 데이터.")

        # 클래스 개수 확인
        if META_CSV_PATH.is_file():
            meta_df = pd.read_csv(META_CSV_PATH)
            num_classes = len(meta_df)
            logger.info(f"{META_CSV_PATH} 기반 클래스 개수: {num_classes}")
        else:
            num_classes = full_df['target'].nunique()
            logger.warning(f"{META_CSV_PATH} 파일을 찾을 수 없습니다. {TRAIN_CSV_PATH} 기반 클래스 개수: {num_classes}")

        if num_classes < 2:
            logger.error("분류할 클래스가 2개 미만입니다. 데이터 또는 설정을 확인하세요.")
            sys.exit(1)

        # 학습/검증 데이터 분할
        train_df, val_df = train_test_split(
            full_df,
            test_size=0.2,
            random_state=model_config['SEED'],
            stratify=full_df['target']
        )
        logger.info(f"데이터 분할 완료: 학습 {len(train_df)}개, 검증 {len(val_df)}개")

        # 이미지 루트 디렉토리 확인
        if not IMAGES_DIR.is_dir():
             logger.error(f"이미지 디렉토리({IMAGES_DIR})가 존재하지 않습니다.")
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
        model = timm.create_model(
            model_name=model_config['MODEL_NAME'],
            pretrained=True,
            num_classes=num_classes
        ).to(device)
        logger.info(f"모델 '{model_config['MODEL_NAME']}' 생성 완료 (클래스 수: {num_classes})")
    except Exception as e:
        logger.error(f"모델 생성 실패 '{model_config['MODEL_NAME']}': {e}", exc_info=True)
        sys.exit(1)

    # --- 손실 함수 및 옵티마이저 ---
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_config['LR'])
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # 필요시 사용
    logger.info(f"손실 함수: CrossEntropyLoss, 옵티마이저: Adam (LR={model_config['LR']})")

    # --- 학습 루프 ---
    epochs = model_config['EPOCHS']
    logger.info(f"--- 총 {epochs} 에폭 학습 시작 ---")
    best_val_f1 = 0.0

    for epoch in range(epochs):
        train_results = train_one_epoch(train_loader, model, optimizer, loss_fn, device, epoch, epochs)
        val_results = validate_one_epoch(val_loader, model, loss_fn, device, epoch, epochs)

        log_msg = (
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_results['train_loss']:.4f}, Acc: {train_results['train_acc']:.4f}, F1: {train_results['train_f1']:.4f} | "
            f"Val Loss: {val_results['val_loss']:.4f}, Acc: {val_results['val_acc']:.4f}, F1: {val_results['val_f1']:.4f}"
        )
        logger.info(log_msg)

        # scheduler.step() # 필요시 사용

        # --- 모델 저장 ---
        current_val_f1 = val_results['val_f1']
        is_best = current_val_f1 > best_val_f1

        # 항상 마지막 에폭 모델 저장
        last_save_path = OUTPUT_DIR / f"last_model_epoch_{epoch+1}.pth"
        try:
            torch.save(model.state_dict(), last_save_path)
            # logger.info(f"마지막 에폭 모델 저장 완료: {last_save_path}") # 너무 자주 로깅될 수 있으므로 주석 처리
        except Exception as e:
            logger.error(f"마지막 모델 저장 실패: {e}", exc_info=False)

        if is_best:
            best_val_f1 = current_val_f1
            save_path = OUTPUT_DIR / f"best_model_f1_{best_val_f1:.4f}.pth" # 파일명 간소화
            try:
                # 최고 성능 모델 저장 (state_dict만 저장하거나 전체 checkpoint 저장)
                # 여기서는 state_dict만 저장
                torch.save(model.state_dict(), save_path)
                logger.info(f"🚀 최고 성능 모델 저장 완료: {save_path} (Val F1: {best_val_f1:.4f})")

                # 이전 최고 성능 모델 삭제 (선택 사항)
                # for f in OUTPUT_DIR.glob("best_model_f1_*.pth"):
                #     if f != save_path:
                #         f.unlink()

            except Exception as e:
                logger.error(f"최고 성능 모델 저장 실패: {e}", exc_info=True)


    logger.info(f"--- 총 {epochs} 에폭 학습 완료 ---")
    logger.info(f"최고 검증 F1 점수: {best_val_f1:.4f}")

# ===================== 스크립트 실행 진입점 =====================
if __name__ == "__main__":
    try:
        training()
    except Exception as e:
        logger.exception(f"스크립트 실행 중 치명적인 오류 발생: {e}")
        sys.exit(1)
    logger.info(f"{message_str.get('ENDED', '발자욱 만들기 마침')}: {worker_name}")
