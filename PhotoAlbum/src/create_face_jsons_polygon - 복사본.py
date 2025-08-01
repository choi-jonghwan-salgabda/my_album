"""
✅ 기능 요약
raw_image_dir 내의 모든 이미지 파일 반복

MediaPipe Face Mesh로 얼굴 윤곽 검출

이미지의 SHA-256 해시 생성

JSON으로 저장 (save_face_json_with_polygon)

설정은 .my_config.yaml을 사용
"""

import os
import hashlib
from pathlib import Path
from PIL import Image

import logging
from datetime import datetime
from pathlib import Path

# 로그 디렉토리 설정
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

# 날짜 기반 로그 파일 이름
log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

# 로깅 설정: 콘솔 + 파일
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__) # 로거 가져오기

# 이(아래) 코드는 MediaPipe의 Face Mesh를 활용하여 다각형으로 얼굴을 자르고 PNG로 저장합니다.
# from src.face_cropper_polygon import detect_faces_with_polygon
# from pathlib import Path
# from PIL import Image
from typing import List
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def get_face_landmarks(image: np.ndarray) -> List[np.ndarray]:
    """MediaPipe Face Mesh를 사용하여 얼굴 랜드마크 좌표를 반환"""
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return []
        return [
            np.array([(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                      for landmark in face_landmarks.landmark], dtype=np.int32)
            for face_landmarks in results.multi_face_landmarks
        ]

def crop_polygon_face(image: np.ndarray, landmarks: np.ndarray, output_size=(224, 224)) -> np.ndarray:
    """랜드마크를 기반으로 다각형 마스크를 생성하고 얼굴 영역만 자름"""
    contour_idx = list(range(0, 17)) + list(range(68, 83))  # 턱선 + 눈썹 아래쪽
    polygon = landmarks[contour_idx]

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    face_only = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(polygon)
    cropped = face_only[y:y+h, x:x+w]

    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
    return resized

def crop_face_from_file(img_path: Path, output_dir: Path) -> bool:
    image = np.array(Image.open(img_path).convert("RGB"))
    landmarks_list = get_face_landmarks(image)
    if not landmarks_list:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, landmarks in enumerate(landmarks_list):
        cropped = crop_polygon_face(image, landmarks)
        out_path = output_dir / f"{img_path.stem}_face{i}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    return True


"""
detect_faces_with_polygon(image: Image.Image, min_confidence: float = 0.5) -> List[Dict]:
MediaPipe Face Mesh를 이용해 이미지에서 얼굴의 윤곽선 다각형 추출

from typing import List, Dict
from PIL import Image
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh


score, polygon 좌표, bounding box 등의 정보를 포함하는 JSON-like 결과 반환
return
[
  {
    "score": 1.0,
    "polygon": [
      {"x": 120.5, "y": 210.3},
      {"x": 122.2, "y": 208.9},
      ...
    ]
  }
]
✅ 연계 활용
create_face_jsons_polygon.py에서 사용해 JSON 생성

face_cropper_polygon.py와 함께 써서 얼굴 이미지를 자르면서 JSON도 생성

face_search_web.py에서 업로드된 이미지의 윤곽 추출에도 사용 가능
"""

def detect_faces_with_polygon(image: Image.Image, min_confidence: float = 0.5) -> List[dict]:
    """
    이미지에서 얼굴을 다각형 윤곽 기반으로 검출
    반환: 각 얼굴에 대해 polygon 좌표 및 score 포함된 dict 리스트
    """
    image_np = np.array(image.convert("RGB"))
    height, width = image_np.shape[:2]

    faces = []

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, refine_landmarks=True,
                                min_detection_confidence=min_confidence) as face_mesh:
        results = face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            return []

        for face_landmarks in results.multi_face_landmarks:
            # 턱선 (0~16), 눈썹 하단 (68~83) 중심으로 polygon 구성
            indices = list(range(0, 17)) + list(range(68, 83))
            polygon = [
                {
                    "x": round(face_landmarks.landmark[i].x * width, 2),
                    "y": round(face_landmarks.landmark[i].y * height, 2)
                } for i in indices
            ]

            faces.append({
                "score": 1.0,  # MediaPipe Face Mesh는 별도 score 없음
                "polygon": polygon
            })

    return faces

"""
def save_face_json_with_polygon(image_path: Path, image_hash: str, faces: List[Dict], output_path: Path) -> None:

✅ 기능 요약
MediaPipe Face Mesh를 이용해 이미지에서 얼굴의 윤곽선 다각형 추출

score, polygon 좌표, bounding box 등의 정보를 포함하는 JSON-like 결과 반환

매개변수	설명
image_path	원본 이미지 경로 (Path)
image_hash	SHA-256 해시 (중복 제거 및 식별자 용도)
faces	detect_faces_with_polygon() 함수의 결과
output_path	저장할 JSON 파일 경로

"""

import json
# from pathlib import Path
# from typing import List, Dict

def save_face_json_with_polygon(image_path: Path, image_hash: str, faces: List[dict], output_path: Path) -> None:
    """
    얼굴 윤곽 정보와 이미지 해시를 포함한 JSON 파일 저장
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_data = {
        "image_path": str(image_path),
        "image_hash": image_hash,
        "num_faces": len(faces),
        "faces": faces
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

#from config_loader import load_config
#from .config_loader import load_config
#from pathlib import Path
#import logging # 로깅 추가
#import numpy as np # Numpy 추가
#from PIL import Image # PIL 추가
#import cv2 # OpenCV 추가
import yaml
#import face_recognition # face_recognition 추가


# 같은 디렉토리에 있으므로 상대 경로 임포트 사용

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

"""
✅ src/create_face_jsons_polygon.py

"""

def compute_image_hash(image: Image.Image) -> str:
    """SHA-256 해시 계산"""
    return hashlib.sha256(image.tobytes()).hexdigest()

def process_images(config_path: str):
    config = load_config(config_path)

    image_dir = Path(config["raw_image_dir"]).expanduser()
    output_dir = Path(config["json_output_dir"]).expanduser()
    min_conf = config.get("min_detection_confidence", 0.5)

    if not image_dir.exists():
        logging.error(f"❌ 이미지 폴더가 존재하지 않습니다: {image_dir}")
        return

    image_files = [p for p in image_dir.glob("**/*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    logging.info(f"🔍 {len(image_files)}장의 이미지를 처리합니다...")

    no_face_list = []

    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.warning(f"⚠️ 이미지를 열 수 없음: {img_path} ({e})")
            continue

        img_hash = compute_image_hash(img)
        faces = detect_faces_with_polygon(img, min_conf)

        if not faces:
            logging.info(f"❌ 얼굴 없음: {img_path.name}")
            no_face_list.append(str(img_path.resolve()))
            continue  # JSON 저장하지 않음

        json_path = output_dir / f"{img_path.stem}.json"
        save_face_json_with_polygon(img_path, img_hash, faces, json_path)
        logging.info(f"✅ {img_path.name} → 얼굴 {len(faces)}개 저장 완료")

    # ⏳ 얼굴 없는 파일 목록 저장
    if no_face_list:
        no_face_file = output_dir / "no_faces_found.txt"
        with open(no_face_file, "w") as f:
            f.write("\n".join(no_face_list))
        logging.info(f"📄 얼굴 없는 이미지 목록 저장됨: {no_face_file}")


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/.my_config.yaml"
    process_images(config_file)


"""
✅ 실행 방법


poetry run python src/create_face_jsons_polygon.py
 or 
poetry run python src/create_face_jsons_polygon.py config/.my_config.yaml

"""