# save_face_json_with_polygon.py

import os
import sys
import yaml
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union, Any # <--- Add Union and Any here
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

# === 설정 ===
mp_face_mesh = mp.solutions.face_mesh

# === 로깅 설정 ===
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-8s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === 기능 함수 ===
class ProjectConfig:
    """
    YAML 설정 파일을 로드하고,
    ${root_dir}, ${dataset_dir}, ${output_dir} 같은 플레이스홀더를 실제 경로로 치환한 뒤
    필요한 구성 단위(dataset, output, source, models 등)를 반환하는 클래스
    """

    def __init__(self, config_path: str):
        """
        [생성자]
        - 입력: config_path (str) -> YAML 설정 파일 경로
        - 출력: 없음 (클래스 내부에 config 저장)
        - 기능: 설정 파일 로드 및 경로 플레이스홀더 치환
        """
        self.config_path = Path(config_path)
        self.config = self._load_and_resolve_config()

    def _load_yaml(self) -> dict:
        """
        [비공개 메서드] YAML 파일을 읽어서 Python 딕셔너리로 변환
        - 입력: 없음 (self.config_path 사용)
        - 출력: config (dict)
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_placeholders(self, config: dict, context: dict) -> dict:
        """
        [비공개 메서드] 설정 딕셔너리 내 플레이스홀더(${var})를 실제 값으로 치환
        - 입력:
          - config: 원본 설정 딕셔너리
          - context: 치환할 키-값 매핑(dict) (ex: {"root_dir": "/home/user/project"})
        - 출력: 치환이 완료된 설정 딕셔너리
        """
        pattern = re.compile(r"\$\{([^}]+)\}")  # ${} 안의 변수를 찾는 정규식 패턴

        def resolve_value(value):
            if isinstance(value, str):
                matches = pattern.findall(value)
                for match in matches:
                    if match in context:
                        value = value.replace(f"${{{match}}}", str(context[match]))
            return value

        def recursive_resolve(obj):
            if isinstance(obj, dict):
                return {k: recursive_resolve(resolve_value(v)) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_resolve(v) for v in obj]
            else:
                return resolve_value(obj)

        return recursive_resolve(config)

    def _load_and_resolve_config(self) -> dict:
        """
        [비공개 메서드] 설정 파일 로드 후, 플레이스홀더 치환까지 완료
        - 입력: 없음
        - 출력: 치환이 완료된 전체 설정 딕셔너리
        """
        raw_config = self._load_yaml()
        context = {
            "root_dir": raw_config["project"]["root_dir"],
            "dataset_dir": raw_config["project"]["dataset"]["dataset_dir"],
            "output_dir": raw_config["project"]["output"]["output_dir"],
            "src_dir": raw_config["project"]["source"]["src_dir"]
        }
        return self._resolve_placeholders(raw_config, context)

    # ======= 외부에 제공하는 메서드 =======

    def get_project_config(self) -> dict:
        """
        [공개 메서드] project 전체 정보 반환
        - 입력: 없음
        - 출력: project 섹션 (dict)
        """
        return self.config.get("project", {})

    def get_dataset_config(self) -> dict:
        """
        [공개 메서드] dataset 구성 정보 반환
        - 입력: 없음
        - 출력: dataset 섹션 (dict)
        """
        return self.config["project"].get("dataset", {})

    def get_output_config(self) -> dict:
        """
        [공개 메서드] output 구성 정보 반환
        - 입력: 없음
        - 출력: output 섹션 (dict)
        """
        return self.config["project"].get("output", {})

    def get_source_config(self) -> dict:
        """
        [공개 메서드] source 구성 정보 반환
        - 입력: 없음
        - 출력: source 섹션 (dict)
        """
        return self.config["project"].get("source", {})

    def get_models_config(self) -> dict:
        """
        [공개 메서드] models 설정 정보 반환
        - 입력: 없음
        - 출력: models 섹션 (dict)
        """
        return self.config.get("models", {})

# === 사용 예시 ===
# if __name__ == "__main__":
#     config = ProjectConfig("config/my_photo_album_3.yaml")
    
#     print("📂 dataset 정보:", config.get_dataset_config())
#     print("📂 output 정보:", config.get_output_config())
#     print("📂 source 정보:", config.get_source_config())
#     print("🧠 models 정보:", config.get_models_config())
    
# def load_config(config_path: str) -> dict:
#     """YAML 설정 파일 로드"""
#     try:
#         with open(config_path, "r", encoding="utf-8") as f:
#             config = yaml.safe_load(f)
#         return config
#     except Exception as e:
#         logger.critical(f"설정 파일 로딩 실패: {e}")
#         raise

def resolve_path_placeholders(
    config_data: Union[Dict[str, Any], list],
    placeholder: str,
    base_path: Path
) -> Union[Dict[str, Any], list]:
    """
    딕셔너리나 리스트 내의 문자열 값에서 플레이스홀더를 주어진 경로로 재귀적으로 치환합니다.
    원본 데이터 구조를 직접 수정합니다(in-place).

    Args:
        config_data: 처리할 딕셔너리 또는 리스트.
        placeholder: 치환할 플레이스홀더 문자열 (예: "${base_dir}").
        base_path: 플레이스홀더 대신 사용할 Path 객체.

    Returns:
        수정된 config_data (in-place 수정됨).
    """
    if isinstance(config_data, dict):
        for key, value in config_data.items():
            if isinstance(value, str):
                original_value = value
                # Path 객체를 문자열로 변환하여 replace 함수에 사용
                updated_value = value.replace(placeholder, str(base_path))
                if updated_value != original_value:
                    logger.debug(f"  Replacing placeholder in '{key}': '{original_value}' -> '{updated_value}'")
                    config_data[key] = updated_value
            elif isinstance(value, (dict, list)):
                # 하위 딕셔너리나 리스트에 대해 재귀 호출
                resolve_path_placeholders(value, placeholder, base_path)
    elif isinstance(config_data, list):
        for i, item in enumerate(config_data):
            if isinstance(item, str):
                original_item = item
                updated_item = item.replace(placeholder, str(base_path))
                if updated_item != original_item:
                    logger.debug(f"  Replacing placeholder in list index {i}: '{original_item}' -> '{updated_item}'")
                    config_data[i] = updated_item
            elif isinstance(item, (dict, list)):
                # 리스트 내의 딕셔너리나 리스트에 대해 재귀 호출
                resolve_path_placeholders(item, placeholder, base_path)
    return config_data # 수정된 데이터 구조 반환 (실제로는 in-place 수정)


# 0. detect_faces_with_polygon Face Detector 사용

def detect_faces_with_polygon(image: Image.Image, min_confidence: float = 0.5) -> List[Dict]:
    """이미지에서 얼굴 다각형 검출"""
    image_np = np.array(image.convert("RGB"))
    height, width = image_np.shape[:2]
    faces = []

    with ㅊ(static_image_mode=True, max_num_faces=10, refine_landmarks=True,
                                min_detection_confidence=min_confidence) as face_mesh:
        results = face_mesh.process(image_np)
        if not results.multi_face_landmarks:
            return []

        for face_landmarks in results.multi_face_landmarks:
            indices = list(range(0, 17)) + list(range(68, 83))
            polygon = [{"x": round(face_landmarks.landmark[i].x * width, 2),
                        "y": round(face_landmarks.landmark[i].y * height, 2)} for i in indices]
            faces.append({"score": 1.0, "polygon": polygon})

    return faces


# 1. MediaPipe Face Detection 사용

# MediaPipe Face Detection 초기화 (루프 밖에서 한 번만 실행 권장)
mp_face_detection = mp.solutions.face_detection
# face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) # model_selection=0은 단거리, 1은 장거리

def detect_faces_mp_detection(image: Image.Image, min_confidence: float = 0.5) -> List[Dict]:
    """MediaPipe Face Detection을 이용한 얼굴 검출"""
    # FaceDetection 객체는 with 문을 사용하지 않으므로, 미리 생성된 객체를 사용합니다.
    # 여기서는 예시를 위해 함수 내에서 생성 (비효율적)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_confidence) as face_detector:
        image_np = np.array(image.convert("RGB"))
        height, width = image_np.shape[:2]
        results = face_detector.process(image_np)
        faces = []

        if results.detections:
            for detection in results.detections:
                score = detection.score[0] # 검출 신뢰도
                box = detection.location_data.relative_bounding_box
                # 상대 좌표를 절대 좌표로 변환
                x = int(box.xmin * width)
                y = int(box.ymin * height)
                w = int(box.width * width)
                h = int(box.height * height)

                # 경계 상자를 다각형으로 표현 (사각형)
                polygon = [{"x": x, "y": y}, {"x": x + w, "y": y},
                           {"x": x + w, "y": y + h}, {"x": x, "y": y + h}]

                faces.append({
                    "score": round(float(score), 4),
                    "polygon": polygon,
                    "bounding_box": {"x": x, "y": y, "width": w, "height": h} # 경계 상자 정보도 추가
                })
        return faces

# 2. OpenCV Haar Cascades 사용

# Haar Cascade 모델 로드 (루프 밖에서 한 번만 실행 권장)
# cascade_path = Path("my_photo_album_3/config/haarcascade_frontalface_default.xml") # 실제 경로 확인
# if cascade_path.exists():
#     face_cascade = cv2.CascadeClassifier(str(cascade_path))
# else:
#     # 오류 처리
#     face_cascade = None

def detect_faces_haar(image: Image.Image, face_cascade: cv2.CascadeClassifier, scaleFactor: float = 1.1, minNeighbors: int = 5, minSize: tuple = (30, 30)) -> List[Dict]:
    """OpenCV Haar Cascade를 이용한 얼굴 검출"""
    if face_cascade is None:
        logger.error("Haar Cascade 분류기가 로드되지 않았습니다.")
        return []

    image_np = np.array(image.convert("L")) # 그레이스케일로 변환
    faces_rects = face_cascade.detectMultiScale(image_np,
                                                scaleFactor=scaleFactor,
                                                minNeighbors=minNeighbors,
                                                minSize=minSize)
    faces = []
    for (x, y, w, h) in faces_rects:
        # Haar는 신뢰도 점수를 제공하지 않음
        # 경계 상자를 다각형으로 표현 (사각형)
        polygon = [{"x": x, "y": y}, {"x": x + w, "y": y},
                   {"x": x + w, "y": y + h}, {"x": x, "y": y + h}]
        faces.append({
            "score": 1.0, # 임의 점수
            "polygon": polygon,
            "bounding_box": {"x": x, "y": y, "width": w, "height": h}
        })
    return faces

# 사용 예시:
# config = load_config(...)
# cascade_path = Path(config["project"]["config"]["haar_cascade_path"]) # 설정에서 경로 가져오기
# if cascade_path.exists():
#    face_cascade_classifier = cv2.CascadeClassifier(str(cascade_path))
#    # ... 이미지 루프 내 ...
#    faces = detect_faces_haar(img, face_cascade_classifier)

# 3. OpenCV DNN Face Detector 사용

# DNN 모델 로드 (루프 밖에서 한 번만 실행 권장)
# prototxt_path = Path("path/to/deploy.prototxt")
# model_path = Path("path/to/res10_300x300_ssd_iter_140000.caffemodel")
# if prototxt_path.exists() and model_path.exists():
#     net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
# else:
#     # 오류 처리
#     net = None

def detect_faces_dnn(image: Image.Image, net: cv2.dnn_Net, confidence_threshold: float = 0.5) -> List[Dict]:
    """OpenCV DNN을 이용한 얼굴 검출"""
    if net is None:
         logger.error("OpenCV DNN 네트워크가 로드되지 않았습니다.")
         return []

    image_np = np.array(image.convert("RGB"))
    (h, w) = image_np.shape[:2]
    # 이미지 전처리 (모델에 따라 다를 수 있음)
    blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    faces = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            width_box = endX - startX
            height_box = endY - startY

            # 경계 상자를 다각형으로 표현 (사각형)
            polygon = [{"x": startX, "y": startY}, {"x": endX, "y": startY},
                       {"x": endX, "y": endY}, {"x": startX, "y": endY}]

            faces.append({
                "score": round(float(confidence), 4),
                "polygon": polygon,
                "bounding_box": {"x": startX, "y": startY, "width": width_box, "height": height_box}
            })
    return faces

# 사용 예시:
# config = load_config(...)
# prototxt = Path(config["models"]["dnn_prototxt_path"]) # 설정에서 경로 가져오기
# model = Path(config["models"]["dnn_model_path"])
# if prototxt.exists() and model.exists():
#     dnn_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model))
#     # ... 이미지 루프 내 ...
#     faces = detect_faces_dnn(img, dnn_net, min_conf)

# 4. Dlib HOG 사용

# Dlib HOG 검출기 로드 (루프 밖에서 한 번만 실행 권장)
# hog_face_detector = dlib.get_frontal_face_detector()

def detect_faces_dlib_hog(image: Image.Image, detector: dlib.fhog_object_detector) -> List[Dict]:
    """Dlib HOG를 이용한 얼굴 검출"""
    if detector is None:
        logger.error("Dlib HOG 검출기가 로드되지 않았습니다.")
        return []

    image_np = np.array(image.convert("RGB")) # Dlib은 RGB 이미지를 사용
    dets = detector(image_np, 1) # 두 번째 인자는 업샘플링 횟수
    faces = []

    for d in dets:
        x = d.left()
        y = d.top()
        w = d.width()
        h = d.height()

        # 경계 상자를 다각형으로 표현 (사각형)
        polygon = [{"x": x, "y": y}, {"x": x + w, "y": y},
                   {"x": x + w, "y": y + h}, {"x": x, "y": y + h}]

        faces.append({
            "score": 1.0, # HOG는 신뢰도 점수를 제공하지 않음
            "polygon": polygon,
            "bounding_box": {"x": x, "y": y, "width": w, "height": h}
        })
    return faces


# 사용 예시:
# hog_detector = dlib.get_frontal_face_detector()
# # ... 이미지 루프 내 ...
# faces = detect_faces_dlib_hog(img, hog_detector)


# 5. Dlib CNN 사용

# Dlib CNN 모델 로드 (루프 밖에서 한 번만 실행 권장)
# cnn_model_path = Path("path/to/mmod_human_face_detector.dat") # 실제 경로 확인
# if cnn_model_path.exists():
#     cnn_face_detector = dlib.cnn_face_detection_model_v1(str(cnn_model_path))
# else:
#     # 오류 처리
#     cnn_face_detector = None

def detect_faces_dlib_cnn(image: Image.Image, detector: dlib.cnn_face_detection_model_v1, confidence_threshold: float = 0.5) -> List[Dict]:
    """Dlib CNN을 이용한 얼굴 검출"""
    if detector is None:
        logger.error("Dlib CNN 검출기가 로드되지 않았습니다.")
        return []

    image_np = np.array(image.convert("RGB"))
    dets = detector(image_np, 1) # 두 번째 인자는 업샘플링 횟수
    faces = []

    for d in dets:
        confidence = d.confidence
        if confidence >= confidence_threshold:
            rect = d.rect
            x = rect.left()
            y = rect.top()
            w = rect.width()
            h = rect.height()

            # 경계 상자를 다각형으로 표현 (사각형)
            polygon = [{"x": x, "y": y}, {"x": x + w, "y": y},
                       {"x": x + w, "y": y + h}, {"x": x, "y": y + h}]

            faces.append({
                "score": round(float(confidence), 4),
                "polygon": polygon,
                "bounding_box": {"x": x, "y": y, "width": w, "height": h}
            })
    return faces

# 사용 예시:
# config = load_config(...)
# cnn_path = Path(config["models"]["dlib_cnn_model_path"]) # 설정에서 경로 가져오기
# if cnn_path.exists():
#     cnn_detector = dlib.cnn_face_detection_model_v1(str(cnn_path))
#     # ... 이미지 루프 내 ...
#     faces = detect_faces_dlib_cnn(img, cnn_detector, min_conf)


# 6. MTCNN 사용

# MTCNN 검출기 로드 (루프 밖에서 한 번만 실행 권장)
# detector = MTCNN()

def detect_faces_mtcnn(image: Image.Image, detector: MTCNN, confidence_threshold: float = 0.9) -> List[Dict]:
    """MTCNN을 이용한 얼굴 검출"""
    if detector is None:
        logger.error("MTCNN 검출기가 로드되지 않았습니다.")
        return []

    image_np = np.array(image.convert("RGB"))
    results = detector.detect_faces(image_np)
    faces = []

    for result in results:
        confidence = result['confidence']
        if confidence >= confidence_threshold:
            x, y, w, h = result['box']

            # 경계 상자를 다각형으로 표현 (사각형)
            polygon = [{"x": x, "y": y}, {"x": x + w, "y": y},
                       {"x": x + w, "y": y + h}, {"x": x, "y": y + h}]

            faces.append({
                "score": round(float(confidence), 4),
                "polygon": polygon,
                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                "keypoints": result['keypoints'] # MTCNN은 주요 특징점도 제공
            })
    return faces

# 사용 예시:
# mtcnn_detector = MTCNN()
# # ... 이미지 루프 내 ...
# faces = detect_faces_mtcnn(img, mtcnn_detector, min_conf)


def compute_image_hash(image) -> str:
    """
    SHA-256 해시를 계산하는 함수
    입력 (in):
        - image: PIL.Image.Image 객체 또는 numpy.ndarray 객체
    출력 (out):
        - str: SHA-256 해시 문자열
    기능:
        - 이미지가 PIL 객체가 아니면 자동 변환 시도
        - PIL Image 객체에서 tobytes()로 해시 계산
    """
    if isinstance(image, np.ndarray):
        # NumPy 배열이면 PIL 이미지로 변환
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        # PIL.Image.Image 타입이 아니면 오류 발생
        raise TypeError(f"지원하지 않는 이미지 타입입니다: {type(image)}")

    return hashlib.sha256(image.tobytes()).hexdigest()
def save_face_json_with_polygon(image_path: Path, image_hash: str, faces: List[Dict], output_path: Path) -> None:
    """얼굴 검출 결과를 JSON 파일로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_data = {
        "image_name": image_path.name,
        "image_path": str(image_path.resolve()),
        "image_hash": image_hash,
        "faces": faces
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

def process_images(config: ProjectConfig):
    """설정에 따라 이미지 처리 전체 흐름"""
    models_info = config.get_project_config()
    min_conf = float(models_info.get("min_detection_confidence", 0.5))

    detected_list = []
    undetected_list = []
    fail_list = []

    if not raw_image_dir.exists():
        logger.error(f"❌ 이미지 폴더 없음: {raw_image_dir}")
        return

    # 지원하는 이미지 확장자 목록 (설정 파일에서 가져오도록 수정 가능)
    supported_extensions = {ext.lower() for ext in config.get("models", {}).get("supported_image_extensions", [".jpg", ".jpeg", ".png"])}
    image_files = [p for p in raw_image_dir.glob("**/*") if p.is_file() and p.suffix.lower() in supported_extensions]
    logger.info(f"🔍 총 {len(image_files)}장의 이미지를 처리합니다.")

    try:
        # with 문을 사용하여 여러 파일을 동시에 안전하게 엽니다.
        with open(detected_list_path, 'a', encoding='utf-8') as detected_file, \
             open(undetect_list_path, 'a', encoding='utf-8') as undetected_file, \
             open(failed_list_path, 'a', encoding='utf-8') as fail_file:

            # 처리 결과 카운터
            detected_count = 0
            undetected_count = 0
            fail_count = 0

            for img_path in image_files:
                try:
                    # with 문을 사용하여 이미지 파일 핸들 자동 관리
                    # PIL.Image.open(), Pillow (PIL.Image)라이브러리
                    with Image.open(img_path) as img:       #RGB (Red-Green-Blue)
                        img_rgb = img.convert("RGB") # RGB로 변환
                        img_hash = compute_image_hash(img_rgb)
                        faces = detect_faces_with_polygon(img_rgb, min_conf)

                    if faces:
                        # JSON 파일 경로 생성 (원본과 동일한 하위 폴더 구조 유지)
                        relative_path = img_path.relative_to(raw_image_dir)
                        json_path = raw_jsons_dir / relative_path.with_suffix(".json")
                        json_path.parent.mkdir(parents=True, exist_ok=True) # JSON 저장 폴더 생성

                        save_face_json_with_polygon(img_path, img_hash, faces, json_path)
                        # --- 파일에 직접 쓰기 ---
                        detected_file.write(f"{img_path.resolve()}\n")
                        detected_count += 1
                        logger.info(f"✅ 얼굴 검출 성공: {img_path.name}")
                    else:
                        # --- 파일에 직접 쓰기 ---
                        undetected_file.write(f"{img_path.resolve()}\n")
                        undetected_count += 1
                        logger.info(f"⚠️ 얼굴 검출 실패: {img_path.name}")

                except FileNotFoundError:
                    logger.warning(f"⚠️ 처리 중 이미지 파일 없음 (이동/삭제되었을 수 있음): {img_path}")
                    # --- 파일에 직접 쓰기 ---
                    fail_file.write(f"{img_path.resolve()} (File not found during processing)\n")
                    fail_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ 이미지 처리 실패: {img_path} ({e})")
                    # --- 파일에 직접 쓰기 ---
                    fail_file.write(f"{img_path.resolve()} (Error: {e})\n")
                    fail_count += 1
                    # 오류 발생 시 다음 이미지로 넘어감 (continue는 필요 없음, 루프가 계속됨)

            # --- 최종 결과 요약 로그 ---
            logger.info(f"--- 처리 완료 ---")
            logger.info(f"✅ 얼굴 검출 성공: {detected_count} 건")
            logger.info(f"⚠️ 얼굴 검출 실패: {undetected_count} 건")
            logger.info(f"❌ 처리 오류 발생: {fail_count} 건")
            logger.info(f"📄 결과 목록 파일 위치:")
            logger.info(f"   - 성공: {detected_list_path}")
            logger.info(f"   - 실패(미검출): {undetect_list_path}")
            logger.info(f"   - 오류: {failed_list_path}")

    except IOError as e:
        logger.critical(f"결과 목록 파일 열기/쓰기 오류: {e}")
    except Exception as e:
        logger.critical(f"이미지 처리 중 예상치 못한 오류 발생: {e}")



# === 메인 실행 ===

if __name__ == "__main__":
    # 0. 기즘 내가 일하는 곳은"
    direction_dir = os.getcwd()
    print(f"지금 쥔계서 계신곳(direction_dir) : {direction_dir}")
    
    worker_path_obj = Path(__file__).resolve()
    project_root_path = worker_path_obj.parent.parent
    print(f"지금 일꾼이 일하는곳(worker_dir_name)  : {project_root_path}")

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = f"{project_root_path}/config/{project_root_path.name}.yaml"
        print(f"설정 파일 경로: {config_path}")

    try:
#     config = ProjectConfig("config/my_photo_album_3.yaml")
        config = ProjectConfig(config_path)
        logger.info(f"Loaded config from: {config_path}")
    except Exception as e:
        logger.critical(f"Failed to load config: {config_path} - {e}")
        return # 설정 로드 실패 시 종료
    try:
        process_images(config)
    except FileNotFoundError:
        # load_config에서 발생한 FileNotFoundError 처리
        logging.critical("스크립트 실행 중단: 설정 파일을 찾을 수 없습니다.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"스크립트 실행 중 예상치 못한 오류 발생: {e}", exc_info=True)
        sys.exit(1)
