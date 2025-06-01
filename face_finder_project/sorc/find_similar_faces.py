# -*- coding: utf-8 -*-
"""
find_similar_faces.py (기존 train_myresnet.py에서 수정됨)

[목적]
- 레이블 없는 대규모 이미지 데이터셋 내에서 특정 인물의 얼굴과 유사한 얼굴이 포함된 다른 이미지들을 효율적으로 검색하는 시스템.
- `face_recognition` 라이브러리를 활용하여 얼굴 검출, 인코딩, 유사도 비교 수행.
- 지도 학습 기반의 이미지 분류 대신, 얼굴 특징 기반의 유사성 검색에 초점.

[구조]
1. 설정 및 초기화: 필요한 라이브러리 임포트, 경로 설정, 로거 설정, 환경 초기화.
2. 핵심 함수 정의:
    - build_face_index: 이미지 디렉토리를 스캔하여 얼굴을 감지하고 인코딩(임베딩)하여 인덱스 파일(.pkl)로 저장.
    - load_face_index: 저장된 인덱스 파일을 로드.
    - find_similar_faces: 쿼리 이미지의 얼굴 인코딩과 로드된 인덱스를 비교하여 유사한 얼굴이 포함된 이미지 경로 목록 반환.
    - display_results: 검색 결과를 콘솔에 출력.
3. 메인 실행 블록 (`if __name__ == "__main__":`):
    - 명령줄 인자(argparse)를 사용하여 실행 모드('build_index' 또는 'search') 및 관련 경로/옵션 입력 받음.
    - 선택된 모드에 따라 핵심 함수들을 호출하여 인덱싱 또는 검색 수행.

[입력]
1. 실행 모드 (명령줄 인자 `mode`):
    - `build_index`: 얼굴 인덱스를 생성하는 모드.
    - `search`: 생성된 인덱스를 사용하여 유사 얼굴을 검색하는 모드.
2. 이미지 디렉토리 (명령줄 인자 `--image_dir`):
    - 인덱싱하거나 검색 대상이 되는 이미지 파일들이 포함된 루트 디렉토리 경로. 하위 디렉토리 포함.
    - 기본값: 설정 파일(`.my_config.yaml`)의 `IMAGES_DIR` 경로 (확장 후, 보통 train 관련).
3. 인덱스 파일 경로 (명령줄 인자 `--index_file`):
    - 생성된 얼굴 인덱스를 저장하거나 로드할 파일 경로. Pickle(.pkl) 형식 사용.
    - 기본값: 설정 파일의 `OUTPUT_DIR` 아래 `face_index.pkl` (확장 후).
4. 쿼리 이미지 경로 (명령줄 인자 `--query_image`):
    - `search` 모드에서 검색 기준으로 사용할 얼굴 사진 파일 경로.
    - 기본값: 설정 파일(`.my_config.yaml`)의 `QUERY_IMAGE_DEFAULT` 경로 (확장 후, 보통 test 관련).
5. 유사도 허용 오차 (명령줄 인자 `--tolerance`, 선택 사항):
    - `search` 모드에서 얼굴 비교 시 사용할 거리 임계값. 낮을수록 더 엄격하게 동일 인물로 판단. (기본값: 0.5)
6. 강제 재인덱싱 옵션 (명령줄 인자 `--force_reindex`, 선택 사항):
    - `build_index` 모드에서 기존 인덱스 파일이 있어도 무시하고 새로 생성.

[출력]
1. 얼굴 인덱스 파일 (`.pkl`):
    - `build_index` 모드 실행 시 생성됨. 얼굴 인코딩 벡터 목록과 해당 이미지 파일 경로 목록을 포함.
2. 콘솔 로그:
    - 스크립트 실행 과정, 인덱싱 진행 상황, 오류 메시지, 검색 결과 등을 출력.
3. 검색 결과 (콘솔):
    - `search` 모드 실행 시, 쿼리 얼굴과 유사한 얼굴이 포함된 것으로 판단되는 이미지 파일들의 경로 목록을 출력.
"""

# --- 표준 라이브러리 임포트 ---
import os
import sys
import yaml
import random
import logging
import pickle # 파이썬 객체 직렬화/역직렬화 (인덱스 저장/로드용)
import argparse # 명령줄 인자 파싱
from pathlib import Path # 객체 지향 파일 시스템 경로
from typing import List, Dict, Any, Optional, Tuple, Union # 타입 힌팅용

# --- 서드파티 라이브러리 임포트 ---
import numpy as np # 수치 연산
from PIL import Image, ExifTags, UnidentifiedImageError # 이미지 처리 (열기, 회전, 오류 처리)

# face_recognition 라이브러리 임포트 (얼굴 검출, 인코딩, 비교)
try:
    import face_recognition
except ImportError:
    print("오류: 'face_recognition' 라이브러리를 찾을 수 없습니다.")
    print("설치 방법: pip install face_recognition")
    sys.exit(1)

# tqdm 라이브러리 임포트 (진행률 표시줄)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None # tqdm 없으면 진행률 표시 안 함

# --- 경로 설정 및 확장 함수 ---
def expand_path_vars(path_str: Union[str, Path], proj_dir: Path, data_dir: Optional[Path] = None) -> Path:
    """
    YAML에서 읽은 경로 문자열 내의 변수(${PROJ_DIR}, ${DATA_DIR})와 홈 디렉토리(~)를 확장하여 절대 경로 Path 객체로 반환합니다.

    입력:
        path_str (Union[str, Path]): 확장할 경로 문자열 또는 Path 객체.
        proj_dir (Path): 프로젝트 루트 디렉토리의 절대 경로. 상대 경로 확장의 기준이 됩니다.
        data_dir (Optional[Path]): 데이터 디렉토리의 절대 경로. ${DATA_DIR} 변수 치환에 사용됩니다.

    반환:
        Path: 확장되고 절대 경로로 변환된 Path 객체.

    오류:
        TypeError: path_str이 문자열이나 Path 객체가 아닐 경우 발생.
        Exception: 경로 처리 중 예외 발생 시 원래 예외를 다시 발생시킴.
    """
    if isinstance(path_str, Path): # 이미 Path 객체면 절대 경로 보장 후 반환
        return path_str.resolve()
    if not isinstance(path_str, str):
        raise TypeError(f"Path must be a string or Path object, got: {type(path_str)}")

    # ${PROJ_DIR} 변수 치환
    expanded_str = path_str.replace('${PROJ_DIR}', str(proj_dir))

    # ${DATA_DIR} 변수 치환 (data_dir이 제공된 경우)
    # YAML의 ${DATAS_DIR}는 ${DATA_DIR}의 오타로 간주하여 함께 처리
    if data_dir:
        expanded_str = expanded_str.replace('${DATA_DIR}', str(data_dir))
        expanded_str = expanded_str.replace('${DATAS_DIR}', str(data_dir)) # 오타 가능성 고려

    # ~ (홈 디렉토리) 확장 및 비표준 $~/ 처리 시도
    if '$~/' in expanded_str:
        # print(f"경고: 비표준 경로 형식 '$~/' 발견됨: '{path_str}'. '~/'로 처리합니다.")
        expanded_str = expanded_str.replace('$~/','~/')
    # 문자열 시작 부분의 '~'만 os.path.expanduser로 확장
    if expanded_str.startswith('~'):
         expanded_str = os.path.expanduser(expanded_str)

    # Path 객체 생성 및 절대 경로화 (resolve)
    try:
        final_path = Path(expanded_str)
        # 경로가 절대 경로가 아니면, proj_dir을 기준으로 절대 경로 생성
        if not final_path.is_absolute():
             final_path = (proj_dir / final_path).resolve()
        else:
             final_path = final_path.resolve() # 이미 절대경로여도 resolve()는 안전
        return final_path
    except Exception as e:
        print(f"오류: 경로 문자열 '{path_str}' (확장 후: '{expanded_str}') 처리 중 오류: {e}")
        raise # 오류를 다시 발생시켜 처리 중단

# --- 설정 파일 로드 및 경로 변수 초기화 ---
try:
    # 현재 스크립트 파일 및 프로젝트 디렉토리 경로 결정
    current_file_path = Path(__file__).resolve() # 이 스크립트 파일의 절대 경로
    WORK_DIR_SCRIPT = current_file_path.parent # 스크립트가 있는 디렉토리 (sorc)
    PROJ_DIR_SCRIPT = WORK_DIR_SCRIPT.parent # 프로젝트 루트 디렉토리 (sorc의 부모)
    dir_config_path_yaml = PROJ_DIR_SCRIPT / ".my_config.yaml" # 설정 파일 경로

    # 설정 파일 존재 여부 확인
    if not dir_config_path_yaml.is_file():
        print(f"❌ 프로젝트 경로구성 설정 파일({dir_config_path_yaml})을 찾을 수 없습니다.")
        sys.exit("프로젝트 경로구성 설정 파일 없음")

    # YAML 설정 파일 로드
    with open(dir_config_path_yaml, "r", encoding="utf-8") as file:
        dir_config = yaml.safe_load(file)

    # 1. PROJ_DIR 결정: 설정 파일 값 우선, 없으면 스크립트 위치 기반
    raw_proj_dir_config = dir_config.get('resnet34_path', {}).get('PROJ_DIR')
    ROOT_DIR_CONFIG = dir_config.get('ROOT_DIR') # ROOT_DIR 값 읽기 (참조용)
    if isinstance(raw_proj_dir_config, str) and raw_proj_dir_config:
        proj_dir_str = raw_proj_dir_config
        # 설정된 PROJ_DIR 값에 $(ROOT_DIR) 변수가 있으면 치환
        if ROOT_DIR_CONFIG and '$(ROOT_DIR)' in proj_dir_str:
             proj_dir_str = proj_dir_str.replace('$(ROOT_DIR)', ROOT_DIR_CONFIG)
        PROJ_DIR = Path(proj_dir_str).resolve() # 최종 PROJ_DIR 결정
        # 스크립트 기반 경로와 다르면 정보 메시지 출력
        if PROJ_DIR != PROJ_DIR_SCRIPT:
             print(f"정보: 설정 파일의 PROJ_DIR({PROJ_DIR})을 사용합니다. (스크립트 위치 기반: {PROJ_DIR_SCRIPT})")
    else:
         PROJ_DIR = PROJ_DIR_SCRIPT # 설정 없으면 스크립트 위치 기반 사용
         print(f"정보: 설정 파일에 PROJ_DIR이 없거나 유효하지 않아 스크립트 위치 기반 경로 사용: {PROJ_DIR}")

    # 2. 세부 경로 설정 (설정 파일 값 우선, 없으면 기본값 + 확장)
    resnet34_paths = dir_config.get('resnet34_path', {})
    worker_paths = resnet34_paths.get('worker_path', {})
    dataset_paths = resnet34_paths.get('dataset_path', {})
    output_paths = resnet34_paths.get('output_path', {})
    message_str = resnet34_paths.get('MESSAGE', {}) # 로깅 메시지 설정

    # WORK_DIR: 작업 디렉토리 (기본값: ${PROJ_DIR}/sorc)
    raw_work_dir = worker_paths.get('WORK_DIR', PROJ_DIR / 'sorc')
    WORK_DIR = expand_path_vars(raw_work_dir, PROJ_DIR)

    # LOG_DIR: 로그 파일 저장 디렉토리 (기본값: ${WORK_DIR}/logs)
    raw_log_dir = worker_paths.get('LOG_DIR', WORK_DIR / 'logs')
    LOG_DIR = expand_path_vars(raw_log_dir, PROJ_DIR) # PROJ_DIR 기준으로 확장

    # DATA_DIR: 데이터 루트 디렉토리 (기본값: ${PROJ_DIR}/data)
    raw_data_dir = dataset_paths.get('DATA_DIR', PROJ_DIR / 'data')
    DATA_DIR = expand_path_vars(raw_data_dir, PROJ_DIR) # PROJ_DIR 기준으로 확장

    # IMAGES_DIR: 인덱싱/검색 대상 이미지 디렉토리 (기본값: ${DATA_DIR}/train)
    # 기본적으로 train 관련 경로를 가리키도록 설정
    raw_images_dir = dataset_paths.get('IMAGES_DIR', '${DATA_DIR}/train') # 기본값을 train으로 명시
    IMAGES_DIR = expand_path_vars(raw_images_dir, PROJ_DIR, data_dir=DATA_DIR) # DATA_DIR 결정 후 확장

    # QUERY_IMAGE_DEFAULT_PATH: 기본 쿼리 이미지 경로 (test 관련)
    # .my_config.yaml의 dataset_path 아래 QUERY_IMAGE_DEFAULT 키를 찾음
    # 없으면 DATA_DIR 아래 test/default_query.jpg 를 기본값으로 사용
    raw_query_image_default = dataset_paths.get('QUERY_IMAGE_DEFAULT', '${DATA_DIR}/test/default_query.jpg')
    QUERY_IMAGE_DEFAULT_PATH = expand_path_vars(raw_query_image_default, PROJ_DIR, data_dir=DATA_DIR)

    # OUTPUT_DIR: 결과물(인덱스 파일 등) 저장 디렉토리 (기본값: ${PROJ_DIR}/outputs)
    raw_output_dir = output_paths.get('OUTPUT_DIR', PROJ_DIR / 'outputs')
    OUTPUT_DIR = expand_path_vars(raw_output_dir, PROJ_DIR)

    # 3. 필요한 디렉토리 생성 (존재하지 않을 경우)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

except Exception as e:
    # 초기 설정 중 오류 발생 시 처리
    print(f"❌ 초기 설정 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 로거 설정 ---
worker_name = Path(__file__).stem # 현재 스크립트 파일 이름 (확장자 제외)
log_file_path = LOG_DIR / f"{worker_name}.log" # 로그 파일 전체 경로
logging.basicConfig(level=logging.INFO, # 기본 로그 레벨 설정 (INFO 이상만 기록)
                    format='%(asctime)s - %(levelname)-6s - %(filename)s:%(lineno)d - %(message)s', # 로그 형식 지정
                    handlers=[logging.StreamHandler(sys.stdout), # 콘솔(stdout) 출력 핸들러
                              logging.FileHandler(log_file_path, encoding='utf-8')]) # 파일 출력 핸들러 (UTF-8 인코딩)
logger = logging.getLogger(__name__) # 로거 객체 생성

# --- 설정된 경로 및 정보 로깅 ---
# 설정 파일의 MESSAGE 섹션 값 또는 기본 문자열 사용, 30칸 왼쪽 정렬
logger.info(f"{message_str.get('START', '스크립트 시작'):35s}: {worker_name}")
logger.info(f"{message_str.get('PROJ_DIR', '프로젝트 위치'):35s}: {PROJ_DIR}")
logger.info(f"{message_str.get('WORK_DIR', '작업 디렉토리'):35s}: {WORK_DIR}")
logger.info(f"{message_str.get('DATA_DIR', '데이터 위치'):35s}: {DATA_DIR}")
logger.info(f"{message_str.get('IMAGES_DIR', '이미지 위치(Train)'):35s}: {IMAGES_DIR}") # 이름 명확화 (기본값이 train 관련)
logger.info(f"{message_str.get('QUERY_IMAGE_DEFAULT', '기본 쿼리 이미지(Test)'):35s}: {QUERY_IMAGE_DEFAULT_PATH}") # 추가된 기본 쿼리 경로 로깅
logger.info(f"{message_str.get('OUTPUT_DIR', '출력 위치'):35s}: {OUTPUT_DIR}")
logger.info(f"{message_str.get('LOG_DIR', '로그 위치'):35s}: {LOG_DIR}")

# --- 환경 초기화 함수 ---
def init_env(seed: int = 42):
    """
    스크립트 실행 환경의 재현성을 위해 랜덤 시드를 고정합니다.

    입력:
        seed (int): 고정할 시드 값 (기본값: 42).
    """
    os.environ['PYTHONHASHSEED'] = str(seed) # 파이썬 해시 시드 고정
    random.seed(seed) # 파이썬 내장 random 모듈 시드 고정
    np.random.seed(seed) # NumPy 랜덤 시드 고정
    # PyTorch 사용 시 추가:
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logger.info(f"환경 시드 고정 완료 (SEED={seed})")

# --- 이미지 회전 처리 함수 ---
# EXIF 태그 중 Orientation에 해당하는 숫자 코드 찾기
EXIF_ORIENTATION_TAG = None
for k, v in ExifTags.TAGS.items():
    if v == 'Orientation':
        EXIF_ORIENTATION_TAG = k
        break

def rotate_image_based_on_exif(image: Image.Image) -> Image.Image:
    """
    이미지의 EXIF 메타데이터에 저장된 Orientation 정보에 따라 이미지를 올바르게 회전시킵니다.

    입력:
        image (PIL.Image.Image): 회전시킬 PIL 이미지 객체.

    반환:
        PIL.Image.Image: EXIF Orientation에 맞게 회전된 PIL 이미지 객체. Orientation 정보가 없거나 처리 중 오류 발생 시 원본 이미지 반환.
    """
    if EXIF_ORIENTATION_TAG is None: return image # Orientation 태그 코드를 못 찾았으면 원본 반환
    try:
        exif = image.getexif() # 이미지에서 EXIF 데이터 추출
        orientation = exif.get(EXIF_ORIENTATION_TAG) # Orientation 값 가져오기
    except Exception:
        # EXIF 데이터 읽기 실패 시 (예: EXIF 정보 없음, 손상된 데이터)
        orientation = None

    # Orientation 값에 따른 회전/반전 처리
    if orientation == 2: return image.transpose(Image.FLIP_LEFT_RIGHT) # 좌우 반전
    elif orientation == 3: return image.rotate(180) # 180도 회전
    elif orientation == 4: return image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT) # 180도 회전 후 좌우 반전
    elif orientation == 5: return image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT) # 시계방향 90도 회전 후 좌우 반전
    elif orientation == 6: return image.rotate(-90, expand=True) # 시계방향 90도 회전
    elif orientation == 7: return image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT) # 반시계방향 90도 회전 후 좌우 반전
    elif orientation == 8: return image.rotate(90, expand=True) # 반시계방향 90도 회전
    return image # 해당하는 Orientation 값이 없으면 원본 반환

# --- 얼굴 인덱싱 함수 ---
def build_face_index(image_directory: Path, index_save_path: Path, skip_existing: bool = True):
    """
    지정된 디렉토리(하위 디렉토리 포함)의 모든 이미지에서 얼굴을 찾아 인코딩(임베딩)하고,
    결과(인코딩 리스트, 해당 이미지 경로 리스트)를 Pickle 파일로 저장합니다.

    입력:
        image_directory (Path): 얼굴을 찾을 이미지들이 포함된 루트 디렉토리 경로.
        index_save_path (Path): 생성된 얼굴 인덱스를 저장할 파일 경로 (.pkl).
        skip_existing (bool): True일 경우, 인덱스 파일이 이미 존재하면 인덱싱을 건너<0xEB><0><0x8F><0xBC>니다 (기본값: True).
                               False일 경우, 기존 파일을 덮어쓰고 새로 인덱싱합니다.

    반환:
        None. 인덱스 파일 생성 또는 건너뛰기 메시지 출력.
    """
    # 기존 인덱스 파일 존재 시 처리
    if skip_existing and index_save_path.exists():
        logger.warning(f"인덱스 파일이 이미 존재하여 빌드를 건너<0xEB><0><0x8F><0xBC>니다: {index_save_path}")
        logger.info("새로 빌드하려면 기존 인덱스 파일을 삭제하거나 --force-reindex 옵션을 사용하세요.")
        return

    # 인덱스 데이터 저장용 리스트 초기화
    known_face_encodings = [] # 얼굴 인코딩(128차원 벡터) 저장 리스트
    known_face_image_paths = [] # 해당 얼굴이 발견된 이미지 파일 경로 저장 리스트

    # 이미지 파일 탐색 (주요 이미지 확장자, 대소문자 구분 없이)
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.heic', '*.heif', '*.webp']
    image_files = []
    for pattern in image_patterns:
        # rglob: 지정된 패턴과 일치하는 파일을 하위 디렉토리까지 재귀적으로 찾음
        image_files.extend(image_directory.rglob(pattern))
        image_files.extend(image_directory.rglob(pattern.upper())) # 대문자 확장자도 고려

    # 찾은 파일 목록에서 중복 제거 및 정렬
    image_files = sorted(list(set(image_files)))

    logger.info(f"'{image_directory}' 에서 얼굴 인덱싱 시작 ({len(image_files)}개 이미지 스캔)...")

    # 처리 통계 변수 초기화
    processed_count = 0 # 처리 시도한 이미지 수
    skipped_no_face = 0 # 얼굴이 없거나 인코딩 실패하여 건너뛴 이미지 수
    error_count = 0 # 처리 중 오류 발생한 이미지 수

    # tqdm 진행률 표시줄 설정 (tqdm 라이브러리가 있을 경우)
    pbar = tqdm(image_files, desc="얼굴 인덱싱 중", unit="img") if tqdm else image_files

    # 각 이미지 파일 처리 루프
    for image_path in pbar:
        try:
            # face_recognition 라이브러리를 사용하여 이미지 로드
            # 내부적으로 Pillow 사용하며, numpy 배열로 반환
            image = face_recognition.load_image_file(str(image_path))

            # (선택 사항) EXIF 회전 처리: face_recognition이 EXIF를 처리 못할 경우
            # Pillow로 먼저 열고 회전시킨 후 numpy 배열로 변환하여 전달
            # try:
            #     pil_image = Image.open(image_path)
            #     rotated_image = rotate_image_based_on_exif(pil_image)
            #     image = np.array(rotated_image)
            # except UnidentifiedImageError: ...
            # except Exception as pil_e: ...

            # 이미지에서 얼굴 위치 찾기 (model="hog" 또는 "cnn")
            # "hog": 빠르지만 덜 정확, CPU 기반
            # "cnn": 느리지만 더 정확, GPU 가속 가능 (dlib GPU 빌드 필요)
            face_locations = face_recognition.face_locations(image, model="hog")

            # 얼굴이 발견된 경우
            if face_locations:
                # 발견된 얼굴 위치에 대해 얼굴 인코딩(128차원 벡터) 추출
                face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

                # 인코딩 성공 시 리스트에 추가
                if face_encodings:
                    for encoding in face_encodings:
                        known_face_encodings.append(encoding)
                        known_face_image_paths.append(str(image_path)) # 각 인코딩에 대해 이미지 경로 저장
                else:
                    # 얼굴 위치는 찾았으나 인코딩 실패 (드문 경우)
                    logger.warning(f"얼굴 위치는 찾았으나 인코딩 실패: {image_path}")
                    skipped_no_face += 1 # 실패도 건너뛴 것으로 간주
            else:
                # 얼굴이 발견되지 않은 경우
                skipped_no_face += 1

            processed_count += 1 # 처리 시도 카운트 증가

            # tqdm 진행률 표시줄 업데이트 (tqdm 사용 시)
            if tqdm and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    "처리": processed_count,
                    "얼굴없음": skipped_no_face,
                    "오류": error_count
                }, refresh=True) # refresh=True: 즉시 업데이트

        except UnidentifiedImageError:
            # PIL이 이미지를 식별할 수 없는 경우 (손상, 지원 불가 형식 등)
            logger.warning(f"이미지 파일 열기 실패 (손상 또는 지원 불가 형식?): {image_path}")
            error_count += 1
        except Exception as e:
            # 기타 예외 처리 (메모리 부족 등)
            logger.error(f"이미지 처리 오류 {image_path}: {e}", exc_info=False) # 기본적으로 상세 traceback 제외
            error_count += 1
        finally:
             # 메모리 누수 방지를 위해 루프 마지막에 변수 삭제 시도 (선택적)
             if 'image' in locals(): del image
             if 'face_locations' in locals(): del face_locations
             if 'face_encodings' in locals(): del face_encodings

    # tqdm 진행률 표시줄 닫기 (tqdm 사용 시)
    if isinstance(pbar, tqdm): pbar.close()

    # 인덱싱 결과 요약 로깅
    logger.info(f"얼굴 인덱싱 완료: 총 {processed_count}개 이미지 처리 시도.")
    logger.info(f"  - 얼굴 인코딩된 총 개수: {len(known_face_encodings)} (이미지 내 모든 얼굴 포함)")
    logger.info(f"  - 얼굴이 없거나 인코딩 실패한 이미지 수: {skipped_no_face}")
    logger.info(f"  - 처리 중 오류 발생 이미지 수: {error_count}")

    # 인코딩된 얼굴 데이터가 있을 경우 파일로 저장
    if known_face_encodings:
        # 저장할 데이터 구조 (딕셔너리)
        index_data = {
            "encodings": known_face_encodings,
            "paths": known_face_image_paths
        }
        try:
            # 저장 경로의 상위 디렉토리 생성 (없으면)
            index_save_path.parent.mkdir(parents=True, exist_ok=True)
            # Pickle을 사용하여 바이너리 쓰기 모드("wb")로 파일 저장
            with open(index_save_path, "wb") as f:
                pickle.dump(index_data, f)
            logger.info(f"얼굴 인덱스 저장 완료: {index_save_path}")
        except Exception as e:
            logger.error(f"인덱스 파일 저장 실패 {index_save_path}: {e}", exc_info=True) # 저장 실패 시 상세 오류 로깅
    else:
        logger.warning("인덱싱할 얼굴 데이터를 찾지 못했습니다. 인덱스 파일을 저장하지 않습니다.")

# --- 얼굴 인덱스 로드 함수 ---
def load_face_index(index_load_path: Path) -> Optional[Dict[str, Any]]:
    """
    저장된 얼굴 인덱스 Pickle 파일을 로드하여 딕셔너리 형태로 반환합니다.

    입력:
        index_load_path (Path): 로드할 얼굴 인덱스 파일 경로 (.pkl).

    반환:
        Optional[Dict[str, Any]]: 로드 성공 시 얼굴 인덱스 데이터 딕셔너리 ({"encodings": [...], "paths": [...]}).
                                  파일이 없거나 로드/검증 실패 시 None 반환.
    """
    # 인덱스 파일 존재 여부 확인
    if not index_load_path.is_file():
        logger.error(f"인덱스 파일을 찾을 수 없습니다: {index_load_path}")
        logger.error("먼저 'build_index' 모드로 인덱스를 생성해야 합니다.")
        return None
    try:
        # Pickle 파일을 바이너리 읽기 모드("rb")로 열어 데이터 로드
        with open(index_load_path, "rb") as f:
            index_data = pickle.load(f)

        # 로드된 데이터 형식 및 내용 검증
        if isinstance(index_data, dict) and "encodings" in index_data and "paths" in index_data:
            num_encodings = len(index_data['encodings'])
            num_paths = len(index_data['paths'])
            # 인코딩 개수와 경로 개수가 일치하는지 확인 (데이터 무결성)
            if num_encodings == num_paths:
                logger.info(f"얼굴 인덱스 로드 완료: {index_load_path} ({num_encodings}개 인코딩)")
                return index_data
            else:
                logger.error(f"인덱스 파일 데이터 무결성 오류: 인코딩({num_encodings})과 경로({num_paths}) 개수 불일치.")
                return None
        else:
            logger.error(f"인덱스 파일 형식이 잘못되었습니다 (딕셔너리 형태 및 키 부재): {index_load_path}")
            return None
    except pickle.UnpicklingError as e:
         # Pickle 역직렬화 오류 처리
         logger.error(f"인덱스 파일 로드 실패 (Pickle 오류) {index_load_path}: {e}", exc_info=True)
         return None
    except Exception as e:
        # 기타 로드 중 예외 처리
        logger.error(f"인덱스 파일 로드 중 예외 발생 {index_load_path}: {e}", exc_info=True)
        return None

# --- 유사 얼굴 검색 함수 ---
def find_similar_faces(query_image_path: Path, index_data: Dict[str, Any], tolerance: float = 0.5) -> List[str]:
    """
    쿼리 이미지에서 얼굴을 찾아 인코딩하고, 로드된 인덱스 데이터와 비교하여
    유사도(거리)가 tolerance 값 이하인 얼굴이 포함된 이미지 파일 경로 목록을 반환합니다.

    입력:
        query_image_path (Path): 검색 기준으로 사용할 얼굴이 포함된 이미지 파일 경로.
        index_data (Dict[str, Any]): `load_face_index` 함수로 로드된 얼굴 인덱스 데이터 딕셔너리.
        tolerance (float): 얼굴 비교 시 사용할 거리 임계값 (기본값: 0.5).
                           값이 낮을수록 더 엄격하게 동일 인물로 판단합니다.
                           (face_recognition 라이브러리의 기본값은 0.6)

    반환:
        List[str]: 쿼리 얼굴과 유사한 얼굴이 포함된 것으로 판단되는 고유한 이미지 파일 경로 목록 (정렬됨).
                   유사 얼굴을 찾지 못하거나 오류 발생 시 빈 리스트 반환.
    """
    # 인덱스 데이터 유효성 검사
    if not index_data:
        logger.error("인덱스 데이터가 로드되지 않았거나 유효하지 않습니다.")
        return []

    known_face_encodings = index_data.get("encodings") # 인덱스 내 모든 얼굴 인코딩 리스트
    known_face_image_paths = index_data.get("paths") # 각 인코딩에 해당하는 이미지 경로 리스트

    # 인덱스 내 인코딩 데이터 유효성 검사
    if not known_face_encodings or not isinstance(known_face_encodings, list) or len(known_face_encodings) == 0:
        logger.warning("인덱스에 유효한 얼굴 인코딩 데이터가 없습니다.")
        return []

    try:
        # 1. 쿼리 이미지 로드 및 얼굴 인코딩 (첫 번째 얼굴만 사용)
        query_image = face_recognition.load_image_file(str(query_image_path))
        query_face_locations = face_recognition.face_locations(query_image, model="hog") # 얼굴 위치 찾기

        # 쿼리 이미지에서 얼굴을 찾지 못한 경우
        if not query_face_locations:
            logger.warning(f"쿼리 이미지에서 얼굴을 찾을 수 없습니다: {query_image_path}")
            return []

        # 쿼리 이미지에서 얼굴 인코딩 추출 (여러 얼굴이 있어도 첫 번째 얼굴만 사용)
        query_face_encodings = face_recognition.face_encodings(query_image, known_face_locations=query_face_locations)
        if not query_face_encodings:
             logger.warning(f"쿼리 이미지에서 얼굴은 찾았으나 인코딩에 실패했습니다: {query_image_path}")
             return []

        query_face_encoding = query_face_encodings[0] # 첫 번째 얼굴의 인코딩 사용
        logger.info(f"쿼리 이미지에서 얼굴 인코딩 추출 완료 (첫 번째 얼굴 사용): {query_image_path}")

        # 2. 유사도 비교
        logger.info(f"인덱스 내 {len(known_face_encodings)}개 얼굴과 비교 시작 (tolerance={tolerance})...")
        try:
            # face_recognition.compare_faces 함수 사용
            # 입력: 인덱스 인코딩 리스트, 쿼리 인코딩, 허용 오차
            # 반환: 각 인덱스 인코딩이 쿼리 인코딩과 유사한지 여부 (True/False) 리스트
            matches = face_recognition.compare_faces(known_face_encodings, query_face_encoding, tolerance=tolerance)
        except Exception as comp_e:
             # 비교 중 오류 발생 시 (예: 데이터 타입 불일치)
             logger.error(f"얼굴 비교 중 오류 발생: {comp_e}", exc_info=True)
             # known_face_encodings가 numpy 배열 리스트가 아닐 경우 변환 시도
             # known_face_encodings = [np.array(enc) for enc in known_face_encodings]
             return []

        # 3. 결과 집계 (유사한 얼굴이 포함된 고유한 이미지 경로)
        matched_image_paths = set() # 중복 제거를 위해 set 사용
        match_count = 0 # 유사한 얼굴 인스턴스 개수 카운트
        for i, match in enumerate(matches):
            if match: # 유사하다고 판단된 경우
                match_count += 1
                # 해당 인덱스의 이미지 경로를 set에 추가
                if i < len(known_face_image_paths): # 인덱스 범위 확인
                    matched_image_paths.add(known_face_image_paths[i])
                else:
                    logger.warning(f"매칭 인덱스 {i}가 경로 리스트 범위를 벗어납니다.")

        logger.info(f"비교 완료. 총 {match_count}개의 유사한 얼굴 인스턴스 발견.")
        if matched_image_paths:
            logger.info(f"결과: {len(matched_image_paths)}개의 고유 이미지에서 유사 얼굴 발견.")
            # 결과를 정렬된 리스트로 변환하여 반환
            return sorted(list(matched_image_paths))
        else:
            logger.info("결과: 유사한 얼굴을 포함하는 이미지를 찾지 못했습니다.")
            return []

    except FileNotFoundError:
        logger.error(f"쿼리 이미지 파일을 찾을 수 없습니다: {query_image_path}")
        return []
    except UnidentifiedImageError:
        logger.warning(f"쿼리 이미지 파일 열기 실패 (손상?): {query_image_path}")
        return []
    except Exception as e:
        logger.error(f"쿼리 이미지 처리 또는 비교 중 오류 발생 {query_image_path}: {e}", exc_info=True)
        return []
    finally:
        # 메모리 관리 (선택적)
        if 'query_image' in locals(): del query_image
        if 'query_face_locations' in locals(): del query_face_locations
        if 'query_face_encodings' in locals(): del query_face_encodings
        if 'query_face_encoding' in locals(): del query_face_encoding
        if 'matches' in locals(): del matches

# --- 결과 표시 함수 ---
def display_results(image_paths: List[str]):
    """
    검색된 유사 이미지 경로 목록을 콘솔에 보기 좋게 출력합니다.

    입력:
        image_paths (List[str]): `find_similar_faces` 함수에서 반환된 이미지 경로 리스트.

    반환:
        None. 콘솔에 결과 출력.
    """
    if not image_paths:
        print("\n>> 검색 결과: 유사한 얼굴을 포함한 이미지를 찾지 못했습니다.")
        return
    print(f"\n>> 검색 결과: {len(image_paths)}개의 이미지에서 유사한 얼굴 발견")
    print("-" * 50) # 구분선
    for i, path in enumerate(image_paths):
        print(f"  {i+1}: {path}") # 번호 매겨서 경로 출력
    print("-" * 50) # 구분선


# ===================== 스크립트 실행 진입점 =====================
# ===================== 스크립트 실행 진입점 =====================
if __name__ == "__main__":
    init_env() # 환경 시드 고정

    # --- 명령줄 인자 파서 설정 ---
    parser = argparse.ArgumentParser(
        description="이미지 데이터셋에서 얼굴을 인덱싱하고, 쿼리 얼굴과 유사한 이미지를 검색합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 인자 도움말에 기본값 표시
    )
    # 필수 인자: 실행 모드 ('build_index' 또는 'search')
    parser.add_argument("mode", choices=['build_index', 'search'],
                        help="실행할 모드 선택: 'build_index' (얼굴 인덱스 생성), 'search' (유사 얼굴 검색)")

    # 선택적 인자들 (기본값 설정)
    # --image_dir: 인덱싱/검색 대상 이미지 디렉토리 (기본값: 설정 파일의 IMAGES_DIR)
    parser.add_argument("--image_dir", type=str, default=str(IMAGES_DIR),
                        help="얼굴을 검색하거나 인덱싱할 이미지가 있는 루트 디렉토리 경로 (기본값: 설정 파일의 학습 이미지 경로)")
    # --index_file: 인덱스 파일 경로 (기본값: 설정 파일의 OUTPUT_DIR 아래 face_index.pkl)
    parser.add_argument("--index_file", type=str, default=str(OUTPUT_DIR / "face_index.pkl"),
                        help="얼굴 인덱스를 저장하거나 로드할 파일 경로")
    # --query_image: 검색 모드에서 사용할 쿼리 이미지 경로 (기본값: 설정 파일의 QUERY_IMAGE_DEFAULT_PATH)
    parser.add_argument("--query_image", type=str, default=str(QUERY_IMAGE_DEFAULT_PATH),
                        help="'search' 모드에서 사용할 쿼리 얼굴 이미지 파일 경로 (기본값: 설정 파일의 테스트 이미지 경로)")
    # --tolerance: 얼굴 비교 허용 오차 (기본값: 0.5)
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="'search' 모드에서 사용할 얼굴 비교 허용 오차 (낮을수록 엄격, face_recognition 기본값은 0.6)")
    # --force_reindex: 기존 인덱스 강제 재생성 여부 (기본값: False)
    parser.add_argument("--force_reindex", action="store_true",
                        help="'build_index' 모드에서 기존 인덱스 파일이 있어도 강제로 다시 생성합니다.")
    # --debug: 디버그 로그 활성화 여부 (기본값: False)
    parser.add_argument("--debug", action="store_true",
                        help="디버그 레벨 로그를 활성화합니다.")

    # 명령줄 인자 파싱
    args = parser.parse_args()

    # --- 로그 레벨 설정 (명령줄 --debug 인자 기반) ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level) # 로거의 레벨 설정
    # 모든 핸들러(화면, 파일)의 레벨도 동일하게 설정
    for handler in logger.handlers:
        handler.setLevel(log_level)
    if args.debug:
        logger.info("디버그 로깅 활성화됨.")
        # 디버그 모드 시 파싱된 인자 값 로깅
        logger.debug("--- ArgumentParser 결과 ---")
        for arg, value in vars(args).items():
             logger.debug(f"  {arg}: {value}")
        logger.debug("--------------------------")

    # --- 경로 인자 처리 및 유효성 검사 ---
    # argparse에서 받은 문자열 경로를 Path 객체로 변환하고 resolve()로 절대 경로화
    try:
        image_dir_path = Path(args.image_dir).resolve()
        index_file_path = Path(args.index_file).resolve()
        # query_image는 None일 수 있으므로 체크 후 Path 객체 생성
        query_image_path = Path(args.query_image).resolve() if args.query_image else None
    except Exception as path_e:
         logger.error(f"명령줄 인자로 받은 경로 처리 중 오류: {path_e}", exc_info=True)
         sys.exit(1)

    # --- 선택된 모드에 따라 기능 실행 ---
    if args.mode == 'build_index':
        logger.info("--- 얼굴 인덱스 생성 모드 ---")
        # 이미지 디렉토리 존재 및 접근 가능 여부 확인
        if not image_dir_path.is_dir():
            logger.error(f"이미지 디렉토리를 찾거나 접근할 수 없습니다: {image_dir_path}")
            sys.exit(1)
        # 얼굴 인덱스 빌드 함수 호출
        # skip_existing=not args.force_reindex: --force_reindex가 있으면 skip_existing=False
        build_face_index(image_dir_path, index_file_path, skip_existing=not args.force_reindex)

    elif args.mode == 'search':
        logger.info("--- 유사 얼굴 검색 모드 ---")
        # 쿼리 이미지 경로 필수 확인
        if not query_image_path or not str(query_image_path).strip():
            logger.error("'search' 모드에서는 --query_image 인자가 반드시 필요합니다.")
            parser.print_help() # 도움말 출력
            sys.exit(1)

        # 쿼리 이미지 파일 존재 및 접근 가능 여부 확인
        if not query_image_path.is_file():
            logger.error(f"쿼리 이미지 파일을 찾거나 접근할 수 없습니다: {query_image_path}")
            sys.exit(1)

        # 얼굴 인덱스 로드 (파일 존재 여부는 내부에서 체크)
        face_index_data = load_face_index(index_file_path)

        # 인덱스 로드 성공 시 유사 얼굴 검색 수행
        if face_index_data:
            found_image_paths = find_similar_faces(query_image_path, face_index_data, tolerance=args.tolerance)
            display_results(found_image_paths) # 결과 표시 함수 호출
        else:
            # 인덱스 로드 실패 시 오류 메시지 출력 및 종료
            logger.error("인덱스 로드 실패 또는 데이터 없음으로 검색을 진행할 수 없습니다.")
            sys.exit(1)

    else:
        # argparse의 choices 설정으로 인해 이 부분은 이론적으로 실행되지 않음
        logger.error(f"알 수 없는 모드입니다: {args.mode}")
        sys.exit(1)

    # 스크립트 정상 종료 로깅
    logger.info(f"--- 스크립트 종료: {worker_name} ---")
