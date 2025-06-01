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
    - 기본값: 설정 파일(`.my_config.yaml`)의 `IMAGES_DIR` 경로 (확장 후).
3. 인덱스 파일 경로 (명령줄 인자 `--index_file`):
    - 생성된 얼굴 인덱스를 저장하거나 로드할 파일 경로. Pickle(.pkl) 형식 사용.
    - 기본값: 설정 파일의 `OUTPUT_DIR` 아래 `face_index.pkl` (확장 후).
4. 쿼리 이미지 경로 (명령줄 인자 `--query_image`):
    - `search` 모드에서 검색 기준으로 사용할 얼굴 사진 파일 경로 (필수).
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

import os
import sys
import yaml
import random
import logging
import pickle # 인덱스 저장/로드용
import argparse # 명령줄 인자 처리용
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
# pandas, torch 등 제거
from PIL import Image, ExifTags, UnidentifiedImageError # UnidentifiedImageError 추가

# face_recognition 라이브러리 임포트
try:
    import face_recognition
except ImportError:
    print("오류: 'face_recognition' 라이브러리를 찾을 수 없습니다.")
    print("설치 방법: pip install face_recognition")
    sys.exit(1)

# tqdm 임포트 (진행률 표시용)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None # tqdm 없으면 None으로 설정

# --- 경로 설정 및 확장 함수 ---
def expand_path_vars(path_str: Union[str, Path], proj_dir: Path, data_dir: Optional[Path] = None) -> Path:
    """YAML에서 읽은 경로 문자열 내의 변수(${PROJ_DIR}, ${DATA_DIR})와 ~ 를 확장합니다."""
    if isinstance(path_str, Path): # 이미 Path 객체면 그대로 반환
        return path_str.resolve() # 절대 경로 보장
    if not isinstance(path_str, str):
        raise TypeError(f"Path must be a string or Path object, got: {type(path_str)}")

    # ${PROJ_DIR} 치환
    expanded_str = path_str.replace('${PROJ_DIR}', str(proj_dir))

    # ${DATA_DIR} 치환 (data_dir이 결정된 후에 사용 가능)
    # YAML의 ${DATAS_DIR}는 ${DATA_DIR}의 오타로 간주
    if data_dir:
        expanded_str = expanded_str.replace('${DATA_DIR}', str(data_dir))
        expanded_str = expanded_str.replace('${DATAS_DIR}', str(data_dir)) # 오타 가능성 고려

    # ~ (홈 디렉토리) 확장 및 비표준 $~/ 처리 시도
    if '$~/' in expanded_str:
        # print(f"경고: 비표준 경로 형식 '$~/' 발견됨: '{path_str}'. '~/'로 처리합니다.")
        expanded_str = expanded_str.replace('$~/','~/')
    if '~' in expanded_str: # 반드시 처음에 있지 않아도 확장 시도
        # os.path.expanduser는 문자열 시작 부분의 ~만 처리하므로 주의
        # 여기서는 문자열 내 모든 ~를 홈 디렉토리 경로로 바꾼다고 가정 (덜 일반적)
        # 표준적인 사용은 경로 시작의 '~/' 또는 '~' 입니다.
        # 여기서는 간단히 문자열 시작의 '~'만 처리하도록 수정합니다.
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
        raise # 오류를 다시 발생시켜 처리 중단

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

    # 1. PROJ_DIR 결정 (설정 파일 우선, 없으면 스크립트 위치 기반)
    raw_proj_dir_config = dir_config.get('resnet34_path', {}).get('PROJ_DIR')
    # $(ROOT_DIR) 같은 복잡한 참조는 직접 처리하지 않음. 단순 문자열 경로로 가정.
    # ROOT_DIR 값을 읽어서 치환하는 로직 추가 가능
    ROOT_DIR_CONFIG = dir_config.get('ROOT_DIR')
    if isinstance(raw_proj_dir_config, str) and raw_proj_dir_config:
        proj_dir_str = raw_proj_dir_config
        if ROOT_DIR_CONFIG and '$(ROOT_DIR)' in proj_dir_str:
             proj_dir_str = proj_dir_str.replace('$(ROOT_DIR)', ROOT_DIR_CONFIG)
        PROJ_DIR = Path(proj_dir_str).resolve()
        if PROJ_DIR != PROJ_DIR_SCRIPT:
             print(f"정보: 설정 파일의 PROJ_DIR({PROJ_DIR})을 사용합니다. (스크립트 위치 기반: {PROJ_DIR_SCRIPT})")
    else:
         PROJ_DIR = PROJ_DIR_SCRIPT # 기본값: 스크립트 위치 기반
         print(f"정보: 설정 파일에 PROJ_DIR이 없거나 유효하지 않아 스크립트 위치 기반 경로 사용: {PROJ_DIR}")

    # 경로 설정 가져오기
    resnet34_paths = dir_config.get('resnet34_path', {})
    worker_paths = resnet34_paths.get('worker_path', {})
    dataset_paths = resnet34_paths.get('dataset_path', {})
    output_paths = resnet34_paths.get('output_path', {})
    message_str = resnet34_paths.get('MESSAGE', {})

    # 경로 확장 적용
    # WORK_DIR: 설정값 우선, 없으면 기본값 사용 후 확장
    raw_work_dir = worker_paths.get('WORK_DIR', PROJ_DIR / 'sorc')
    WORK_DIR = expand_path_vars(raw_work_dir, PROJ_DIR)

    # LOG_DIR: 설정값 우선, 없으면 기본값 사용 후 확장
    raw_log_dir = worker_paths.get('LOG_DIR', WORK_DIR / 'logs') # 기본값을 WORK_DIR 기반으로 변경 가능
    LOG_DIR = expand_path_vars(raw_log_dir, PROJ_DIR) # PROJ_DIR 기준으로 확장

    # DATA_DIR: 설정값 우선, 없으면 기본값 사용 후 확장
    raw_data_dir = dataset_paths.get('DATA_DIR', PROJ_DIR / 'data')
    DATA_DIR = expand_path_vars(raw_data_dir, PROJ_DIR) # PROJ_DIR 기준으로 확장

    # IMAGES_DIR: 설정값 우선, 없으면 *결정된* DATA_DIR 기반 기본값 사용 후 확장
    raw_images_dir = dataset_paths.get('IMAGES_DIR')
    if raw_images_dir:
         # 설정 파일에 값이 있으면 해당 값을 PROJ_DIR 및 DATA_DIR 기준으로 확장 시도
         IMAGES_DIR = expand_path_vars(raw_images_dir, PROJ_DIR, DATA_DIR)
    else:
         # 설정 파일에 값이 없으면, 결정된 DATA_DIR 아래 images를 기본값으로 사용
         IMAGES_DIR = DATA_DIR / 'images'

    # OUTPUT_DIR: 설정값 우선, 없으면 기본값 사용 후 확장
    raw_output_dir = output_paths.get('OUTPUT_DIR', PROJ_DIR / 'outputs')
    OUTPUT_DIR = expand_path_vars(raw_output_dir, PROJ_DIR)

    # 디렉토리 생성 (확장된 경로 사용)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

except Exception as e:
    print(f"❌ 초기 설정 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    # 로거 설정 전이므로 print 사용
    # 로거가 설정되었다면 logger.exception 사용 가능
    sys.exit(1)

# --- 로거 설정 ---
worker_name = Path(__file__).stem
log_file_path = LOG_DIR / f"{worker_name}.log" # 결정된 LOG_DIR 사용
logging.basicConfig(level=logging.INFO, # 기본 레벨 INFO
                    format='%(asctime)s - %(levelname)-6s - %(filename)s:%(lineno)d - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(log_file_path, encoding='utf-8')])
logger = logging.getLogger(__name__)

# --- 로깅 메시지 (개선된 경로 변수 사용) ---
logger.info(f"{message_str.get('START', '스크립트 시작'):30s}: {worker_name}")
logger.info(f"{message_str.get('PROJ_DIR', '프로젝트 위치'):30s}: {PROJ_DIR}")
logger.info(f"{message_str.get('WORK_DIR', '작업 디렉토리'):30s}: {WORK_DIR}")
logger.info(f"{message_str.get('DATA_DIR', '데이터 위치'):30s}: {DATA_DIR}")
logger.info(f"{message_str.get('IMAGES_DIR', '이미지 위치'):30s}: {IMAGES_DIR}")
logger.info(f"{message_str.get('OUTPUT_DIR', '출력 위치'):30s}: {OUTPUT_DIR}")
logger.info(f"{message_str.get('LOG_DIR', '로그 위치'):30s}: {LOG_DIR}")

# --- 환경 초기화 ---
def init_env(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"환경 시드 고정 완료 (SEED={seed})")

# --- 이미지 회전 처리 (기존 코드 유지) ---
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

# --- 얼굴 인덱싱 및 검색 함수 (기존 코드 유지, 단 로깅 개선) ---

def build_face_index(image_directory: Path, index_save_path: Path, skip_existing: bool = True):
    """
    지정된 디렉토리(하위 포함)의 이미지에서 얼굴을 찾아 인코딩하고 인덱스 파일로 저장합니다.
    얼굴이 없는 이미지는 자동으로 건너<0xEB><0><0x8F><0xBC>니다.
    """
    if skip_existing and index_save_path.exists():
        logger.warning(f"인덱스 파일이 이미 존재하여 빌드를 건너<0xEB><0><0x8F><0xBC>니다: {index_save_path}")
        logger.info("새로 빌드하려면 기존 인덱스 파일을 삭제하거나 --force-reindex 옵션을 사용하세요.")
        return

    known_face_encodings = []
    known_face_image_paths = []
    # 이미지 파일 탐색 (주요 이미지 확장자) - 대소문자 구분 없이 rglob 사용 가능 (Python 3.5+)
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.heic', '*.heif', '*.webp']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(image_directory.rglob(pattern))
        image_files.extend(image_directory.rglob(pattern.upper())) # 대문자 확장자도 고려

    # 중복 제거 (경로 객체는 hashable)
    image_files = sorted(list(set(image_files)))

    logger.info(f"'{image_directory}' 에서 얼굴 인덱싱 시작 ({len(image_files)}개 이미지 스캔)...")

    processed_count = 0
    skipped_no_face = 0
    error_count = 0

    # tqdm 사용 가능 여부 확인 및 pbar 설정
    pbar = tqdm(image_files, desc="얼굴 인덱싱 중", unit="img") if tqdm else image_files

    for image_path in pbar:
        try:
            # Pillow로 먼저 열어서 회전 처리 후 face_recognition에 numpy 배열 전달 고려 가능
            # 여기서는 face_recognition.load_image_file 유지 (내부적으로 Pillow 사용)
            # 단, EXIF 회전은 face_recognition에서 자동으로 처리하지 않을 수 있음
            # 필요시 Pillow로 열고 rotate_image_based_on_exif 적용 후 numpy 변환
            image = face_recognition.load_image_file(str(image_path))

            # Pillow로 열어 회전 처리 (선택 사항, face_recognition이 EXIF 처리 못할 경우)
            # try:
            #     pil_image = Image.open(image_path)
            #     rotated_image = rotate_image_based_on_exif(pil_image)
            #     image = np.array(rotated_image) # face_recognition은 numpy 배열 필요
            # except UnidentifiedImageError:
            #      logger.warning(f"이미지 파일 열기 실패 (Pillow): {image_path}")
            #      error_count += 1
            #      continue # 다음 파일로
            # except Exception as pil_e:
            #      logger.error(f"Pillow 이미지 처리 오류 {image_path}: {pil_e}", exc_info=False)
            #      error_count += 1
            #      continue

            face_locations = face_recognition.face_locations(image, model="hog") # 또는 "cnn" (더 정확하지만 느림, dlib GPU 빌드 필요)

            if face_locations:
                # 얼굴 발견 시에만 인코딩 수행
                face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
                if face_encodings: # 인코딩 성공 여부 확인
                    for encoding in face_encodings:
                        known_face_encodings.append(encoding)
                        known_face_image_paths.append(str(image_path))
                    # logger.debug(f"얼굴 발견 및 인코딩 ({len(face_encodings)}개): {image_path}") # DEBUG 레벨
                else:
                    # 얼굴 위치는 찾았으나 인코딩 실패 (매우 드묾)
                    logger.warning(f"얼굴 위치는 찾았으나 인코딩 실패: {image_path}")
                    skipped_no_face += 1 # 실패도 건너뛴 것으로 간주
            else:
                skipped_no_face += 1
                # logger.debug(f"얼굴 없음 (건너뛰기): {image_path}") # DEBUG 레벨

            processed_count += 1
            if tqdm and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    "처리": processed_count,
                    "얼굴없음": skipped_no_face,
                    "오류": error_count
                }, refresh=True)

        except UnidentifiedImageError:
            logger.warning(f"이미지 파일 열기 실패 (손상 또는 지원 불가 형식?): {image_path}")
            error_count += 1
        except Exception as e:
            # 상세 오류 로깅 (exc_info=True)은 디버그 시 유용
            logger.error(f"이미지 처리 오류 {image_path}: {e}", exc_info=False) # 기본은 False 유지
            error_count += 1
        finally:
             # 메모리 관리 (선택적): 큰 이미지 반복 처리 시 명시적 해제 고려
             del image # face_recognition.load_image_file 결과 해제 시도
             if 'face_locations' in locals(): del face_locations
             if 'face_encodings' in locals(): del face_encodings

    if isinstance(pbar, tqdm): pbar.close() # tqdm 사용 시 루프 종료 후 닫기

    logger.info(f"얼굴 인덱싱 완료: 총 {processed_count}개 이미지 처리 시도.")
    logger.info(f"  - 얼굴 인코딩된 총 개수: {len(known_face_encodings)} (고유 얼굴 아님, 이미지 내 얼굴 수 합계)")
    logger.info(f"  - 얼굴이 없거나 인코딩 실패한 이미지 수: {skipped_no_face}")
    logger.info(f"  - 처리 중 오류 발생 이미지 수: {error_count}")

    if known_face_encodings:
        index_data = {
            "encodings": known_face_encodings,
            "paths": known_face_image_paths
        }
        try:
            # 인덱스 저장 전 상위 디렉토리 존재 확인 및 생성
            index_save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(index_save_path, "wb") as f:
                pickle.dump(index_data, f)
            logger.info(f"얼굴 인덱스 저장 완료: {index_save_path}")
        except Exception as e:
            logger.error(f"인덱스 파일 저장 실패 {index_save_path}: {e}", exc_info=True)
    else:
        logger.warning("인덱싱할 얼굴 데이터를 찾지 못했습니다. 인덱스 파일을 저장하지 않습니다.")


def load_face_index(index_load_path: Path) -> Optional[Dict[str, Any]]:
    """ 저장된 얼굴 인덱스 파일을 로드합니다. """
    if not index_load_path.is_file():
        logger.error(f"인덱스 파일을 찾을 수 없습니다: {index_load_path}")
        logger.error("먼저 'build_index' 모드로 인덱스를 생성해야 합니다.")
        return None
    try:
        with open(index_load_path, "rb") as f:
            index_data = pickle.load(f)
        if isinstance(index_data, dict) and "encodings" in index_data and "paths" in index_data:
            num_encodings = len(index_data['encodings'])
            num_paths = len(index_data['paths'])
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
         logger.error(f"인덱스 파일 로드 실패 (Pickle 오류) {index_load_path}: {e}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"인덱스 파일 로드 중 예외 발생 {index_load_path}: {e}", exc_info=True)
        return None


def find_similar_faces(query_image_path: Path, index_data: Dict[str, Any], tolerance: float = 0.5) -> List[str]:
    """
    쿼리 이미지의 얼굴과 인덱스 내 유사한 얼굴을 찾아 해당 이미지 경로 리스트를 반환합니다.
    """
    if not index_data:
        logger.error("인덱스 데이터가 로드되지 않았거나 유효하지 않습니다.")
        return []

    known_face_encodings = index_data.get("encodings")
    known_face_image_paths = index_data.get("paths")

    if not known_face_encodings or not isinstance(known_face_encodings, list) or len(known_face_encodings) == 0:
        logger.warning("인덱스에 유효한 얼굴 인코딩 데이터가 없습니다.")
        return []

    try:
        # 쿼리 이미지 로드 및 얼굴 인코딩 (첫 번째 얼굴만 사용)
        query_image = face_recognition.load_image_file(str(query_image_path))
        query_face_locations = face_recognition.face_locations(query_image, model="hog") # 또는 "cnn"
        if not query_face_locations:
            logger.warning(f"쿼리 이미지에서 얼굴을 찾을 수 없습니다: {query_image_path}")
            return []

        # 여러 얼굴이 찾아져도 첫 번째 얼굴의 인코딩만 사용
        query_face_encodings = face_recognition.face_encodings(query_image, known_face_locations=query_face_locations)
        if not query_face_encodings:
             logger.warning(f"쿼리 이미지에서 얼굴은 찾았으나 인코딩에 실패했습니다: {query_image_path}")
             return []

        query_face_encoding = query_face_encodings[0]
        logger.info(f"쿼리 이미지에서 얼굴 인코딩 추출 완료 (첫 번째 얼굴 사용): {query_image_path}")

        # 유사도 비교
        logger.info(f"인덱스 내 {len(known_face_encodings)}개 얼굴과 비교 시작 (tolerance={tolerance})...")
        # known_face_encodings가 numpy 배열 리스트가 아닐 경우 변환 필요할 수 있음
        # face_recognition 함수는 일반적으로 numpy 배열 리스트를 기대함
        try:
            matches = face_recognition.compare_faces(known_face_encodings, query_face_encoding, tolerance=tolerance)
        except Exception as comp_e:
             logger.error(f"얼굴 비교 중 오류 발생: {comp_e}", exc_info=True)
             # 데이터 타입 문제일 수 있음 (예: list of lists 대신 list of numpy arrays 필요)
             # known_face_encodings = [np.array(enc) for enc in known_face_encodings] # 필요시 시도
             return []


        # 결과 집계 (고유한 이미지 경로만)
        matched_image_paths = set()
        match_count = 0
        for i, match in enumerate(matches):
            if match:
                match_count += 1
                # 경로 인덱스가 유효한지 확인
                if i < len(known_face_image_paths):
                    matched_image_paths.add(known_face_image_paths[i])
                else:
                    logger.warning(f"매칭 인덱스 {i}가 경로 리스트 범위를 벗어납니다.")

        logger.info(f"비교 완료. 총 {match_count}개의 유사한 얼굴 인스턴스 발견.")
        if matched_image_paths:
            logger.info(f"결과: {len(matched_image_paths)}개의 고유 이미지에서 유사 얼굴 발견.")
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
        # 메모리 관리
        del query_image
        if 'query_face_locations' in locals(): del query_face_locations
        if 'query_face_encodings' in locals(): del query_face_encodings
        if 'query_face_encoding' in locals(): del query_face_encoding
        if 'matches' in locals(): del matches


# --- 결과 표시 함수 (기존 코드 유지) ---
def display_results(image_paths: List[str]):
    """ 검색된 이미지 경로 목록을 출력합니다. """
    if not image_paths:
        print("\n>> 검색 결과: 유사한 얼굴을 포함한 이미지를 찾지 못했습니다.")
        return
    print(f"\n>> 검색 결과: {len(image_paths)}개의 이미지에서 유사한 얼굴 발견")
    print("-" * 50)
    for i, path in enumerate(image_paths):
        # 경로가 너무 길 경우 잘라서 표시 고려
        print(f"  {i+1}: {path}")
    print("-" * 50)


# ===================== 스크립트 실행 진입점 =====================
if __name__ == "__main__":
    init_env() # 시드 고정

    # --- 명령줄 인자 파서 설정 ---
    parser = argparse.ArgumentParser(
        description="이미지 데이터셋에서 얼굴을 인덱싱하고, 쿼리 얼굴과 유사한 이미지를 검색합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 기본값 표시 방식 개선
    )
    parser.add_argument("mode", choices=['build_index', 'search'],
                        help="실행할 모드 선택: 'build_index' (얼굴 인덱스 생성), 'search' (유사 얼굴 검색)")
    # 기본값을 설정 파일에서 읽어온 경로로 지정
    parser.add_argument("--image_dir", type=str, default=str(IMAGES_DIR),
                        help="얼굴을 검색하거나 인덱싱할 이미지가 있는 루트 디렉토리 경로")
    parser.add_argument("--index_file", type=str, default=str(OUTPUT_DIR / "face_index.pkl"),
                        help="얼굴 인덱스를 저장하거나 로드할 파일 경로")
    parser.add_argument("--query_image", type=str, default=None, # 기본값 None으로 변경
                        help="'search' 모드에서 사용할 쿼리 얼굴 이미지 파일 경로 (필수)")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="'search' 모드에서 사용할 얼굴 비교 허용 오차 (낮을수록 엄격, face_recognition 기본값은 0.6)")
    parser.add_argument("--force_reindex", action="store_true",
                        help="'build_index' 모드에서 기존 인덱스 파일이 있어도 강제로 다시 생성합니다.")
    parser.add_argument("--debug", action="store_true", # 디버그 옵션 추가
                        help="디버그 레벨 로그를 활성화합니다.")

    args = parser.parse_args()

    # --- 로그 레벨 설정 (명령줄 인자 기반) ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level)
    # 모든 핸들러(화면, 파일)의 레벨도 동일하게 설정
    for handler in logger.handlers:
        handler.setLevel(log_level)
    if args.debug:
        logger.info("디버그 로깅 활성화됨.")
        logger.debug("--- ArgumentParser 결과 ---")
        for arg, value in vars(args).items():
             logger.debug(f"  {arg}: {value}")
        logger.debug("--------------------------")


    # --- 경로 객체 변환 및 유효성 검사 ---
    # argparse에서 받은 문자열 경로를 Path 객체로 변환하고 resolve()로 절대 경로화
    try:
        image_dir_path = Path(args.image_dir).resolve()
        index_file_path = Path(args.index_file).resolve()
        query_image_path = Path(args.query_image).resolve() if args.query_image else None
    except Exception as path_e:
         logger.error(f"명령줄 인자로 받은 경로 처리 중 오류: {path_e}", exc_info=True)
         sys.exit(1)


    # --- 모드별 실행 ---
    if args.mode == 'build_index':
        logger.info("--- 얼굴 인덱스 생성 모드 ---")
        if not image_dir_path.is_dir():
            logger.error(f"이미지 디렉토리를 찾거나 접근할 수 없습니다: {image_dir_path}")
            sys.exit(1)
        # 인덱스 파일 저장 경로의 부모 디렉토리 생성은 build_face_index 함수 내부에서 처리
        build_face_index(image_dir_path, index_file_path, skip_existing=not args.force_reindex)

    elif args.mode == 'search':
        logger.info("--- 유사 얼굴 검색 모드 ---")
        if not query_image_path: # query_image 인자 필수 체크
            logger.error("'search' 모드에서는 --query_image 인자가 반드시 필요합니다.")
            parser.print_help() # 도움말 출력
            sys.exit(1)

        if not query_image_path.is_file():
            logger.error(f"쿼리 이미지 파일을 찾거나 접근할 수 없습니다: {query_image_path}")
            sys.exit(1)

        # 인덱스 파일 존재 여부는 load_face_index 내부에서 체크
        face_index_data = load_face_index(index_file_path)

        if face_index_data:
            found_image_paths = find_similar_faces(query_image_path, face_index_data, tolerance=args.tolerance)
            display_results(found_image_paths) # 결과 표시 함수 호출
        else:
            logger.error("인덱스 로드 실패 또는 데이터 없음으로 검색을 진행할 수 없습니다.")
            sys.exit(1) # 인덱스 로드 실패 시 종료

    else:
        # argparse의 choices 설정으로 인해 이 부분은 실행되지 않아야 함
        logger.error(f"알 수 없는 모드입니다: {args.mode}")
        sys.exit(1)

    logger.info(f"--- 스크립트 종료: {worker_name} ---")
