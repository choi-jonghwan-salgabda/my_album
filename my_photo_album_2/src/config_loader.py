# src/config_loader.py
import yaml
from pathlib import Path
import logging # 로깅 추가
import numpy as np # Numpy 추가
import face_recognition # face_recognition 추가
from PIL import Image # PIL 추가
import cv2 # OpenCV 추가

logger = logging.getLogger(__name__) # 로거 가져오기

def load_config(config_path):
    """설정 파일을 로드합니다."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical(f"❌ 설정 파일({config_path})을 찾을 수 없습니다.")
        raise # 오류를 다시 발생시켜 호출자가 처리하도록 함
    except Exception as e:
        logger.critical(f"❌ 설정 파일 로딩 실패 ({config_path}): {e}")
        raise

# --- 공용 얼굴 인코딩 함수 ---
def get_face_encodings(image_source, model="hog"):
    """
    이미지 소스(경로, Numpy 배열, PIL 이미지)에서 얼굴 인코딩 목록을 추출합니다.

    Args:
        image_source: 이미지 파일 경로(str 또는 Path), Numpy 배열, 또는 PIL 이미지 객체.
        model (str): 얼굴 검출 모델 ('hog' 또는 'cnn'). 기본값 'hog'.

    Returns:
        list: 찾은 얼굴 인코딩(Numpy 배열)의 리스트. 얼굴을 찾지 못하면 빈 리스트 반환.
    """
    try:
        if isinstance(image_source, (str, Path)):
            # 경로인 경우 이미지 로드
            image_np = face_recognition.load_image_file(str(image_source))
        elif isinstance(image_source, Image.Image):
            # PIL 이미지인 경우 Numpy 배열로 변환 (RGB 확인)
            if image_source.mode != 'RGB':
                image_source = image_source.convert('RGB')
            image_np = np.array(image_source)
        elif isinstance(image_source, np.ndarray):
            # Numpy 배열인 경우 (채널 순서가 RGB라고 가정)
            # OpenCV로 로드된 BGR 배열은 미리 RGB로 변환해야 함
            if image_source.ndim == 3 and image_source.shape[2] == 3:
                 # 간단한 체크 (BGR 가능성 있음 - 필요시 명시적 변환 추가)
                 pass
            else:
                 # 유효하지 않은 배열 형태
                 logger.warning(f"⚠️ 유효하지 않은 Numpy 배열 형태: {image_source.shape}")
                 return []
            image_np = image_source
        else:
            logger.error(f"❌ 지원하지 않는 이미지 소스 타입: {type(image_source)}")
            return []

        # 얼굴 위치 찾기
        face_locations = face_recognition.face_locations(image_np, model=model)
        if not face_locations:
            # logger.debug(f"이미지에서 얼굴을 찾지 못했습니다 (모델: {model}).") # 디버그 레벨로 변경 가능
            return []

        # 얼굴 인코딩 추출
        face_encodings = face_recognition.face_encodings(image_np, known_face_locations=face_locations)
        return face_encodings

    except FileNotFoundError:
        logger.error(f"❌ 이미지 파일을 찾을 수 없습니다: {image_source}")
        return []
    except Exception as e:
        logger.error(f"❌ 얼굴 인코딩 중 오류 발생 (모델: {model}): {e}", exc_info=True)
        return []

# --- 기존 load_config 함수는 그대로 둡니다 ---
# def load_config(config_path):
#     ... (기존 코드) ...
# src/config_loader.py

def load_config_org(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
