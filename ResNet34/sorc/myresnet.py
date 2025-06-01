# ⚙️1. 프로그램 환경설정
## 1) 필요한 라이브러리 설치
import os
import sys
import yaml

### 2) 전역변수의 정의
#실행 스크립트가 있는 디렉토리를 PATH환경에 덧붙이기
SORC_DIRS = os.path.dirname(os.path.abspath(__file__))
print(f"SORC_DIRS = {SORC_DIRS}")
sys.path.append(SORC_DIRS)                              

#프로젝트 디렉토리를 PATH환경에 덧붙이가
PRJT_DIRS = os.path.abspath(os.path.join(SORC_DIRS, '..'))
print(f"PRJT_DIRS = {PRJT_DIRS}")
sys.path.append(PRJT_DIRS)    #실행 스크립트가 있는 디렉토리를 PATH환경에 덧붙이가

yaml_path = os.path.join(PRJT_DIRS, ".my_config.yaml")
print(f"도움의 구성 정보(yaml_path) : {yaml_path}")
with open(yaml_path, "r", encoding="utf-8") as file:
    init_config = yaml.safe_load(file)

UTIL_DIRS = os.path.join(PRJT_DIRS, init_config.get("UTIL_DIRS", "./my_utility"))
print(f"도움의 함수 위치(UTIL_DIRS) : {UTIL_DIRS}")
sys.path.append(UTIL_DIRS)    #실행 스크립트가 있는 디렉토리를 PATH환경에 덧붙이가

from my_utility import setup_logger, get_logger, combine_paths

# if os.path.abspath(PRJT_DIRS) != os.path.abspath(init_config.get('PRJT_DIRS', os.path.abspath('.'))):
#     print(f"❌ 입력한 프로젝트 위치 PRJT_DIRS : {os.path.abspath(PRJT_DIRS)}")
#     print(f"❌ 찾아낸 프로젝트 위치 PRJT_DIRS : {os.path.abspath(init_config.get('PRJT_DIRS', os.path.abspath('.')))}")
#     sys.exit(1)

LOGS_DIRS = combine_paths(PRJT_DIRS, init_config.get('LOGS_DIRS', './logs'))
log = setup_logger(log_dir=LOGS_DIRS, console=True)
log.info(f"지금부터는 로그로 기록합니다. 위치는 : {LOGS_DIRS}")

FileName = os.path.basename(__file__)
log.info(f"지금 일하는 파일이름은 : {FileName}")

OUTPUT_DIRS = combine_paths(PRJT_DIRS, init_config.get('OUTPUT_DIRS', './logs'))
log.info(f"프로젝트의 결과물은 디렉토리(OUTPUT_DIRS) : {OUTPUT_DIRS}")
DATA_DIRS = combine_paths(PRJT_DIRS, init_config.get('DATA_DIRS', './data'))
log.info(f"프로젝트의 입력물은 디렉토리(DATA_DIRS)   : {DATA_DIRS}")

#===================== 여기까지가 기본설정임 ============================
import os
import time
import random

import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import Dataset


yaml_path = os.path.join(PRJT_DIRS, ".model_config.yaml")
print(f"모델 구성 정보(yaml_path) : {yaml_path}")
with open(yaml_path, "r", encoding="utf-8") as file:
    model_config = yaml.safe_load(file)
model_config['LR'] = float(model_config['LR'])

# 시드를 고정합니다.
def init_env():
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

       
# 데이터셋 클래스를 정의합니다.
class OCRDataset(Dataset):
    def __init__(self, data_list_dir, transform):
        self.data_list_dir = Path(data_list_dir)  # JSON 파일들이 있는 디렉토리
        self.transform = transform

        self.image_path = Path(f"{DATA_DIRS}/train/image")
        self.json_path = Path(f"{DATA_DIRS}/train/json")

        self.anns = OrderedDict()

        # 1. data_list_dir 안의 파일 이름에서 확장자를 제거
        for file in sorted(self.data_list_dir.glob("*.json")):
            base_name = file.stem  # ex: 간판_가로형간판_000013

            # 2. train/image/에서 이미지 파일 찾기
            image_file = None
            for ext in ['jpg', 'jpeg', 'png']:
                candidate = self.image_path / f"{base_name}.{ext}"  # Path 객체로 변환
                if candidate.exists():
                    image_file = candidate.name
                    break

            if not image_file:
                print(f"[경고] 이미지 파일 없음: {base_name}")
                continue

            # 3. train/json/에서 해당 JSON 파일 열기
            json_file = self.json_path / f"{base_name}.json"  # Path 객체로 변환
            if not json_file.exists():
                print(f"[경고] JSON 파일 없음: {base_name}")
                continue

            with open(json_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            # 4. bbox → polygon 변환
            image_info = annotations['images'][0]
            img_id = image_info['id']

            polygons = []
            for ann in annotations.get('annotations', []):
                if ann['image_id'] == img_id:
                    bbox = ann['bbox']
                    polygon = np.array([[
                        [bbox[0], bbox[1]],
                        [bbox[0]+bbox[2], bbox[1]],
                        [bbox[0]+bbox[2], bbox[1]+bbox[3]],
                        [bbox[0], bbox[1]+bbox[3]]
                    ]], dtype=np.int32)
                    polygons.append(polygon)

            self.anns[image_file] = polygons if polygons else None
            
    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        image_filename = list(self.anns.keys())[idx]
        image = Image.open(self.image_path / image_filename).convert('RGB')

        # EXIF정보를 확인하여 이미지 회전
        exif = image.getexif()
        if exif and EXIF_ORIENTATION in exif:
            image = OCRDataset.rotate_image(image, exif[EXIF_ORIENTATION])
        org_shape = image.size

        item = OrderedDict(image=image, image_filename=image_filename, shape=org_shape)

        # polygon 정보 불러오기
        polygons = self.anns[image_filename] or None

        if self.transform is None:
            raise ValueError("Transform function is a required value.")

        # transform 적용
        transformed = self.transform(image=np.array(image), polygons=polygons)
        item.update(image=transformed['image'],
                    polygons=transformed['polygons'],
                    inverse_matrix=transformed['inverse_matrix'],
                    )

        return item

    @staticmethod
    def rotate_image(image, orientation):
        if orientation == 2:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            return image.rotate(180)
        elif orientation == 4:
            return image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            return image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            return image.rotate(-90, expand=True)
        elif orientation == 7:
            return image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            return image.rotate(90, expand=True)
        return image

def read_file_list(file_path):
    # 로컬 파일 목록 읽기
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def data_load(data_path):
    """
    # augmentation을 위한 transform 코드
    # ImageDataset에서 각 데이터를 변형하는 기준을 정의한다. 
    """
    trn_transform = A.Compose([
        # 이미지 크기 조정
        A.Resize(height=model_config.get('img_size', 64), width=model_config.get('img_size', 64)),
        # images normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # numpy 이미지나 PIL 이미지를 PyTorch 텐서로 변환
        ToTensorV2(),
    ])

    """
    # Dataset 정의
    # python의 DataLoader가 먹이감으로 받는 클래스를 정의하는 정의이다.
    # ImageDataset에서 각 데이터를 처리하도록 하여 그 값을 python이 
    # trn_loader의 리스트값으로 주어진다.
    """
    trn_dataset = OCRDataset(
        f"{DATA_DIRS}/json_file_list/train",  # 경로 수정
        transform=trn_transform
    )

    # DataLoader 정의
    trn_loader = DataLoader(
        trn_dataset,    # clase OCRDataset의 먹이감감을 정의함.
        batch_size=model_config.get('BATCH_SIZE', 10),
        shuffle=True,
        num_workers=model_config.get('NUM_WORKERS', 1),
        pin_memory=True,
        drop_last=False
    )

    return trn_loader  # DataLoader 반환

"""
주어진 데이타목록을 학습한다.(batch size = DataLoader의 먹이감임.)
"""
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader())
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        preds = model(image)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f"Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    ret = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }

    return ret

def training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(
        model_name = model_config.get('model_name', 10),
        pretrained=True,
        num_classes=17
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()


    optimizer = Adam(model.parameters(), lr=model_config.get('LR', 1e-3))

    for epoch in range(model_config.get('EPOCHS', 10)):
        ret = train_one_epoch(data_load(DATA_DIRS), model, optimizer, loss_fn, device=device)
        ret['epoch'] = epoch

        log = ""
        for k, v in ret.items():
            log += f"{k}: {v:.4f}\n"
        print(log)

    preds_list = []


#===================== 여기부터가 기본 main 설정임 =======================
if __name__ == "__main__":

    training()                                               # DBNet 기본 구조를 살펴보면, 입력 이미지의 1/4 크기로 출력을 구함

