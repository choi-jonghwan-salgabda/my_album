# import sys # sys 모듈 임포트 불필요
# import os # os 모듈 임포트 불필요 (Pathlib 사용 시)
# sys.path.append(...) 코드 두 줄 제거 또는 주석 처리

import os
from pathlib import Path
from PIL import Image
import logging

# 로깅 설정 개선 (포맷 추가)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-6s - %(message)s')
logger = logging.getLogger(__name__) # 로거 이름 지정

import yaml
#from pathlib import Path
#import logging # 로깅 추가
import numpy as np # Numpy 추가
import face_recognition # face_recognition 추가
#from PIL import Image # PIL 추가
import cv2 # OpenCV 추가

logger = logging.getLogger(__name__) # 로거 가져오기

# 같은 디렉토리에 있으므로 상대 경로 임포트 사용
#from .config_loader import load_config
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

#import cv2
#import numpy as np
import mediapipe as mp
#from PIL import Image
from typing import Union, List, Dict
import hashlib
import json
#from pathlib import Path

mp_face_detection = mp.solutions.face_detection

# 같은 디렉토리에 있으므로 상대 경로 임포트 사용
#from .face_detector import detect_faces_with_hash, save_face_json
def compute_sha256(image: Union[np.ndarray, Image.Image]) -> str:
    """이미지의 SHA-256 해시를 반환합니다."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    _, buffer = cv2.imencode('.png', image)
    return hashlib.sha256(buffer.tobytes()).hexdigest()

def detect_faces_with_hash(
    image: Union[np.ndarray, Image.Image],
    image_path: Union[str, Path] = None,
    min_confidence: float = 0.5,
    target_size: tuple = (224, 224)
) -> Dict:
    """
    얼굴 검출 및 해시 포함 결과 생성

    Returns:
        Dict: {
            image_hash: str,
            image_path: str,
            faces: List[Dict]
        }
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    height, width, _ = image.shape
    image_hash = compute_sha256(image)

    faces = []

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_confidence) as detector:
        results = detector.process(image)

        if results.detections:
            for i, det in enumerate(results.detections):
                box = det.location_data.relative_bounding_box
                x = int(box.xmin * width)
                y = int(box.ymin * height)
                w = int(box.width * width)
                h = int(box.height * height)

                face_info = {
                    "face_id": i,
                    "box": {"x": x, "y": y, "width": w, "height": h},
                    "score": float(det.score[0])
                }
                faces.append(face_info)

    return {
        "image_hash": image_hash,
        "image_path": str(image_path) if image_path else None,
        "faces": faces
    }

def save_face_json(json_data: Dict, output_path: Union[str, Path]):
    """검출된 얼굴 정보를 JSON으로 저장합니다."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)



def create_jsons(config_path: str): # 기본값 제거, 명시적 전달 권장
    """지정된 이미지 디렉토리에서 얼굴을 감지하고 JSON 파일을 생성합니다."""
    try:
        config = load_config(config_path)

        image_dir = Path(config["raw_image_dir"])
        json_dir = Path(config["json_output_dir"])
        min_conf = config.get("min_detection_confidence", 0.5)

        # JSON 저장 디렉토리 생성 (없으면)
        json_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"JSON 저장 디렉토리 확인/생성: {json_dir.resolve()}")

        if not image_dir.is_dir(): # is_dir() 사용 권장
            logger.error(f"❌ 이미지 디렉토리 없음 또는 디렉토리가 아님: {image_dir}")
            return

        # 지원하는 이미지 확장자 목록
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        image_files = []
        logger.info(f"이미지 검색 시작 (하위 디렉토리 포함): {image_dir.resolve()}")
        for ext in image_extensions:
            # rglob()을 사용하여 하위 디렉토리까지 재귀적으로 검색
            image_files.extend(list(image_dir.rglob(ext)))

        if not image_files:
            logger.warning(f"⚠️ {image_dir} 및 하위 디렉토리에서 이미지 파일을 찾을 수 없습니다.")
            return

        logger.info(f"📂 총 {len(image_files)}개의 이미지 발견 (하위 디렉토리 포함)")

        processed_count = 0
        error_count = 0
        for img_path in image_files:
            try:
                # with 문으로 파일 핸들 자동 관리
                with Image.open(img_path) as img:
                    # 얼굴 감지 함수 호출
                    result = detect_faces_with_hash(
                        image=img,
                        image_path=img_path,
                        min_confidence=min_conf
                    )

                # JSON 파일 경로 생성
                json_out_path = json_dir / f"{img_path.stem}.json"
                # JSON 저장 함수 호출
                save_face_json(result, json_out_path)

                logger.info(f"✅ {img_path.name} → {json_out_path.name}")
                processed_count += 1

            except Exception as e:
                # 오류 발생 시 상세 정보 로깅 (exc_info=True)
                logger.error(f"❌ {img_path.name} 처리 실패: {e}", exc_info=True)
                error_count += 1

        logger.info(f"🎉 JSON 파일 생성 완료 (성공: {processed_count}, 실패: {error_count})")

    except KeyError as e:
        logger.critical(f"❌ 설정 파일에 필요한 키가 없습니다: {e} (경로: {config_path})")
    except FileNotFoundError:
        logger.critical(f"❌ 설정 파일({config_path})을 찾을 수 없습니다.")
    except Exception as e:
        logger.critical(f"❌ 스크립트 실행 중 예상치 못한 오류 발생: {e}", exc_info=True)


if __name__ == "__main__":
    # 스크립트가 src 안에 있으므로, 설정 파일 경로는 프로젝트 루트 기준으로 계산
    project_root = os.getcwd()
    default_config_path = os.path.join(project_root, "config", ".my_config.yaml")

    # 설정 파일 경로를 명시적으로 전달
    create_jsons(config_path=str(default_config_path))

