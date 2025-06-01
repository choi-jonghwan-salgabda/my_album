# ======== 🧱 표준 라이브러리 ========
import os
import sys
import gc
import re
import math
import shutil
import pickle
import tempfile
from pathlib import Path
from datetime import datetime

# ======== 🧪 과학 및 수치 계산 ========
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# ======== 🧠 머신러닝/딥러닝 ========
# from sklearn.manifold import TSNE  # 사용 시 주석 해제
import mediapipe as mp

# ======== 📋 설정, 직렬화, 로깅 ========
import yaml
import logging
import hashlib

# ======== 🧾 타입 힌팅 ========
from typing import List, Dict, Union, Any
# mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection # <--- 이 부분이 face_indexer_landmark.py에 누락됨

# # 일단 화면(console)용 스트림 핸들러만 설정
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(logging.Formatter(
#     "%(asctime)s - %(levelname)-8s - %(message)s"
# ))
# logger.addHandler(stream_handler)

import math # math 모듈 import 필요

digit_width = 0 # 전역 변수 초기화

def print_log(func_name: str, mess: str) -> None:
    """
    함수 이름(문자열)의 길이를 나타내는 데 필요한 자릿수를 계산합니다.
    새로운 길이가 현재 저장된 최대 자릿수보다 크면 전역 변수 digit_width를 업데이트합니다.
    업데이트된 전역 digit_width 값을 반환합니다.
    """
    global digit_width # 함수 내에서 전역 변수 'digit_width'를 수정할 것임을 명시
    name_len = len(func_name) # 입력된 함수 이름의 길이를 구합니다.
    # <--- 해당 print 라인이 삭제됨
    # 이름 길이가 0이면 math.log10(0)에서 오류가 발생하므로 처리합니다.
    if name_len > 0:
        # ... (나머지 코드) ...
        required_digits = math.floor(math.log10(name_len)) + 1
        if digit_width < required_digits:
            digit_width = required_digits

    print(f"[{func_name:{digit_width}s}] {mess}")



# def add_file_logger(log_dir: str):
#     try:
#         # 파일 핸들러 생성 및 추가
#         file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
#         file_handler.setFormatter(logging.Formatter(
#             "%(asctime)s - %(levelname)-8s - %(message)s"
#         ))
#         logger.addHandler(file_handler)
#         logger.info(f"파일 로거 추가 성공: {log_file_path}")
#     except Exception as e:
#         logger.error(f"파일 로거 추가 실패 ({log_file_path}): {e}")

# === 기능 함수 ===
class processing_config:
    """
    YAML 설정 파일을 로드하고, 플레이스홀더를 치환하며,
    경로 요청 시 해당 디렉토리가 존재하도록 보장하는 클래스.
    """
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_and_resolve_config()

    def _load_yaml(self) -> dict:
        # ... (기존과 동일) ...
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # _resolve_placeholders 메서드는 이전 답변의 코드를 사용하면 됩니다.
    # (플레이스홀더 치환 후 Path 객체 변환)
    def _resolve_placeholders(self, config: dict, context: dict) -> dict:
        # ... (이전 답변의 _resolve_placeholders 코드) ...
        pattern = re.compile(r"\$\{([^}]+)\}")

        def resolve_value(key, value):
            """값 치환 및 Path 객체 변환"""
            resolved_value = value
            original_value_for_debug = value # 로깅을 위해 원본 저장

            if isinstance(value, str):
                # 플레이스홀더 치환 로직 (context는 이미 완전히 해석된 값들을 가짐)
                resolved_value = self._resolve_single_value(value, context) # 수정된 도우미 함수 사용
                # 치환이 발생했는지 로깅
                if resolved_value != original_value_for_debug:
                     print(f"[resolve_value] 키 '{key:20s}': 값이 바뀜 '{original_value_for_debug:35s}' -> '{resolved_value}'")

            # 키 이름 규칙에 따라 Path 객체로 변환
            if isinstance(resolved_value, str) and key is not None:
                if key.endswith("_dir") or key.endswith("_path"):
                    try:
                        path_obj = Path(resolved_value).expanduser() # 이제 resolved_value는 완전한 경로 문자열
                        return path_obj
                    except Exception as e:
                        logger.error(f"경로 문자열을 Path 객체로 변환 중 오류 ('{key}': '{resolved_value}'): {e}")
                        return resolved_value
            return resolved_value

        def recursive_resolve(obj):
            if isinstance(obj, dict):
                return {k: recursive_resolve(resolve_value(k, v)) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_resolve(v) for v in obj]
            else:
                return resolve_value(None, obj)

        return recursive_resolve(config)

    def _resolve_single_value(self, value: str, context: dict) -> str:
        """단일 문자열 값 내의 플레이스홀더를 치환합니다."""
        pattern = re.compile(r"\$\{([^}]+)\}")
        matches = pattern.findall(value)
        resolved_value = value
        # 여러 플레이스홀더가 있을 수 있으므로 반복 치환
        # (주의: 순환 참조가 있으면 무한 루프 가능성 있음)
        for _ in range(5): # 최대 5번 반복하여 중첩된 플레이스홀더 처리 시도
            made_change = False
            temp_value = resolved_value
            matches = pattern.findall(temp_value)
            if not matches:
                break
            for match in matches:
                if match in context:
                    replacement = str(context[match])
                    if f"${{{match}}}" in temp_value: # 실제 치환이 일어나는지 확인
                        temp_value = temp_value.replace(f"${{{match}}}", replacement)
                        made_change = True
            resolved_value = temp_value
            if not made_change: # 더 이상 치환이 일어나지 않으면 종료
                break
        return resolved_value

    def _load_and_resolve_config(self) -> dict:
        """
        [비공개 메서드] 설정 파일 로드 후, 플레이스홀더 치환까지 완료
        """
        raw_config = self._load_yaml()
        context = {}

        try:
            # 1. root_dir 먼저 결정 (절대 경로로)
            raw_root_dir = raw_config.get("project", {}).get("root_dir")
            if raw_root_dir:
                # expanduser()와 resolve()를 사용하여 절대 경로 Path 객체 생성
                resolved_root_dir_path = Path(raw_root_dir).expanduser().resolve()
                context["root_dir"] = str(resolved_root_dir_path) # context에는 문자열로 저장
            else:
                # root_dir이 없으면 다른 경로 해석 불가, 오류 발생 또는 기본값 설정 필요
                raise ValueError("설정 파일에 project.root_dir이 정의되지 않았습니다.")

            # 2. root_dir을 기반으로 다른 기본 경로들 결정 (절대 경로 문자열로)
            base_context_for_paths = {"root_dir": context["root_dir"]} # root_dir만 있는 context

            raw_dataset_dir = raw_config.get("project", {}).get("dataset", {}).get("dataset_dir")
            if raw_dataset_dir:
                resolved_dataset_dir_str = self._resolve_single_value(raw_dataset_dir, base_context_for_paths)
                context["dataset_dir"] = str(Path(resolved_dataset_dir_str).expanduser().resolve())

            raw_output_dir = raw_config.get("project", {}).get("output", {}).get("output_dir")
            if raw_output_dir:
                resolved_output_dir_str = self._resolve_single_value(raw_output_dir, base_context_for_paths)
                context["output_dir"] = str(Path(resolved_output_dir_str).expanduser().resolve())

            raw_src_dir = raw_config.get("project", {}).get("source", {}).get("src_dir")
            if raw_src_dir:
                resolved_src_dir_str = self._resolve_single_value(raw_src_dir, base_context_for_paths)
                context["src_dir"] = str(Path(resolved_src_dir_str).expanduser().resolve())

            # context에 None 값 제거 (이미 절대 경로이므로 None이 없을 것으로 예상)
            context = {k: v for k, v in context.items() if v is not None}

        except KeyError as e:
            raise ValueError(f"YAML에서 context 생성을 위한 키 누락: {e}") from e
        except Exception as e:
            raise

        # 3. 최종적으로 완성된 context를 사용하여 전체 설정 재귀적 치환 및 Path 객체 변환
        return self._resolve_placeholders(raw_config, context)

    # ======= 외부에 제공하는 메서드 =======

    def get_path(self, key: str, default: Any = None, ensure_exists: bool = True) -> Union[Path, Any]:
        """
        설정에서 경로 키에 해당하는 Path 객체를 직접 반환합니다.
        ensure_exists=True일 경우, _dir 키는 해당 디렉토리를, _path 키는 부모 디렉토리를 생성 시도합니다.
        검색 순서: dataset -> output -> source -> models -> project (최상위)
        """
        value = None
        found_in_section = False

        search_sections = [
            self.get_dataset_config(),
            self.get_output_config(),
            self.get_source_config(),
            self.get_models_config(),
            self.get_project_config()
        ]

        for section in search_sections:
            if isinstance(section, dict) and key in section:
                value = section[key]
                found_in_section = True
                break # 첫 번째 섹션에서 찾으면 중단

        # 최상위 project 키 바로 아래에도 있는지 확인 (섹션에서 못 찾았을 경우)
        if not found_in_section:
            project_config = self.get_project_config()
            if key in project_config:
                 value = project_config[key]

        # 값을 찾았는지 확인 및 Path 객체 처리
        if value is not None:
            path_obj = None
            if isinstance(value, Path):
                path_obj = value
            # _resolve_placeholders에서 변환 실패했을 경우 대비 문자열 체크 추가
            elif isinstance(value, str) and (key.endswith("_dir") or key.endswith("_path")):
                 try:
                     path_obj = Path(value).expanduser()
                 except Exception as e:
                     logger.error(f"get_path에서 경로 변환 중 오류 ('{key}'): {e}")
                     return default

            if path_obj is not None:
                if ensure_exists: # 디렉토리 존재 확인 및 생성 로직
                    try:
                        target_dir_to_create = None
                        if key.endswith("_dir"):
                            target_dir_to_create = path_obj
                        elif key.endswith("_path"):
                            target_dir_to_create = path_obj.parent

                        if target_dir_to_create and not target_dir_to_create.exists():
                            target_dir_to_create.mkdir(parents=True, exist_ok=True)
                        # else: # 이미 존재하는 경우
                        #     print(f"  [get_path] 키 '{key}' 관련 디렉토리 이미 존재: {target_dir_to_create}") # 디버깅용

                    except OSError as e:
                        logger.warning(f"경로 자동 생성 실패 (권한 확인 필요): {target_dir_to_create} - {e}")
                    except Exception as e:
                        logger.error(f"경로 확인/생성 중 오류 발생 ('{key}'): {e}")
                        # 생성 실패 시에도 일단 경로 객체는 반환하거나, default 반환 결정 필요
                        # return default # 오류 시 기본값 반환하도록 변경 가능

                return path_obj # 최종 Path 객체 반환
            else:
                # 경로 키가 아니거나 Path 변환 실패 시 원본 값 반환
                return value

        # 모든 섹션에서 찾지 못하면 기본값 반환
        return default

    # get_project_config, get_dataset_config 등 다른 getter는 그대로 유지
    def get_project_config(self) -> dict:
        return self.config.get("project", {})
    def get_dataset_config(self) -> dict:
        return self.config["project"].get("dataset", {})
    def get_output_config(self) -> dict:
        return self.config["project"].get("output", {})
    def get_source_config(self) -> dict:
        return self.config["project"].get("source", {})
    def get_models_config(self) -> dict:
        return self.config.get("models", {})


# # === 사용 예시 ===
# if __name__ == "__main__":
#     # 설정 파일 경로를 지정해서 클래스 생성
#     config = ProjectConfig("config/my_photo_album_3.yaml")
    
#     # 각 구성 정보를 가져와 출력
#     print("📂 dataset 정보:", config.get_dataset_config())
#     print("📂 output 정보:", config.get_output_config())
#     print("📂 source 정보:", config.get_source_config())
#     print("🧠 models 정보:", config.get_models_config())
import hashlib
from pathlib import Path
from typing import Union, BinaryIO

# SHA-256 해시 계산 함수 (마루님께서 제공하신 함수 기반)
# 이 함수는 PIL Image 또는 NumPy 배열을 받아 이미지 데이터의 해시를 계산합니다.
def compute_sha256(image: Union[np.ndarray, Image.Image]) -> str:
    """
    SHA-256 해시를 계산하는 함수
    입력 (in):
        - image: PIL.Image.Image 객체 또는 numpy.ndarray 객체
    출력 (out):
        - str: SHA-256 해시 문자열
    기능:
        - 이미지가 PIL 객체이면 numpy 배열로 자동 변환
        - numpy.ndarray 객체의 tobytes()로 해시 계산
        - 지원하지 않는 타입이면 TypeError 발생
    """
    func_name = "compute_sha256"
    print_log(func_name, "시작")

    img_array = None
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
        print_log(func_name, f"PIL Image를 NumPy 배열로 변환 완료.")
    elif isinstance(image, np.ndarray):
        img_array = image
        print_log(func_name, f"입력 이미지는 이미 NumPy 배열입니다.")
    else:
        # numpy.ndarray 타입도 PIL Image 타입도 아니면 오류 발생
        print_log(func_name, f"지원하지 않는 이미지 타입입니다: {type(image)}")
        raise TypeError(f"지원하지 않는 이미지 타입입니다: {type(image)}")

    # NumPy 배열의 바이트 데이터를 SHA-256으로 해시 계산
    # image_array가 None이 아니라고 가정하고 진행 (위에서 타입 체크 했으므로)
    try:
        img_bytes = img_array.tobytes()
        print_log(func_name, "NumPy 배열 tobytes() 변환 완료.")
        image_hash_value = hashlib.sha256(img_bytes).hexdigest()
        print_log(func_name, f"SHA-256 해시 계산 완료.")
        return image_hash_value
    except Exception as e:
        print_log(func_name, f"해시 계산 중 오류 발생: {e}")
        # 해시 계산 실패 시 None 또는 빈 문자열 반환 고려
        return None


def detect_faces_with_hash(
    image: Union[np.ndarray, Image.Image],
    image_path: Union[str, Path] = None,
    min_detection_confidence: float = 0.5,
    target_size = [224, 224]
    ) -> Dict:
    """
    이미지에서 얼굴을 검출하고 SHA-256 해시를 포함하여 결과를 반환합니다.

    Args:
        image: 얼굴을 검출할 이미지 (NumPy 배열 또는 PIL Image).
        image_path: 이미지 파일 경로 (결과에 포함될 정보로만 사용).
        min_detection_confidence: 얼굴 검출 최소 신뢰도 (0.0 ~ 1.0).

    Returns:
        Dict: {
            "image_hash": str 또는 None, # 계산된 SHA-256 해시
            "image_path": str 또는 None, # 제공된 이미지 경로
            "faces": List[Dict] # 검출된 얼굴 목록
        }
        각 얼굴 Dict: {
            "face_id": int, # 이미지 내 순번
            "box": {"x": int, "y": int, "width": int, "height": int}, # 픽셀 좌표
            "score": float # 검출 신뢰도
        }
        얼굴이 검출되지 않거나 오류 발생 시 faces는 빈 리스트가 됩니다.
    """
    func_name = "detect_faces_with_hash"
    print_log(func_name, f"함수 시작, 받은 image 타입: {type(image)}")

    # PIL Image를 NumPy 배열로 변환 (MediaPipe 입력 형식에 맞춤)
    image_rgb = None
    if isinstance(image, Image.Image):
        # PIL 이미지는 convert("RGB") 후 numpy 배열로 변환
        image_rgb = np.array(image.convert("RGB"))
        print_log(func_name, f"PIL Image -> NumPy 배열 (RGB) 변환 완료.")
    elif isinstance(image, np.ndarray):
        # 이미 NumPy 배열이면 그대로 사용 (RGB 형식인지 확인 필요할 수 있음)
        if image.ndim == 3 and image.shape[2] == 3: # 3차원, 3채널 확인 (간단)
             image_rgb = image
             print_log(func_name, f"NumPy 배열 입력 확인.")
        else:
             print_log(func_name, f"NumPy 배열 형태가 예상과 다름: {image.shape}")
             # 필요에 따라 여기서 오류 처리 또는 변환 로직 추가
             return {
                "image_hash": None,
                "image_path": str(image_path) if image_path else None,
                "faces": []
             }
    else:
        # 지원하지 않는 이미지 타입일 경우
        print_log(func_name, f"지원하지 않는 이미지 타입입니다: {type(image)}")
        return {
            "image_hash": None,
            "image_path": str(image_path) if image_path else None,
            "faces": []
        }

    # 변환된 이미지(NumPy 배열)의 shape 정보
    height, width, _ = image_rgb.shape
    print_log(func_name, f"처리할 이미지 shape: {image_rgb.shape}")

    # --- 이미지 해시 계산 ---
    # compute_sha256 함수를 호출하여 해시 값을 얻습니다.
    # compute_sha256 함수가 이미지 데이터를 받으므로 image_rgb를 전달합니다.
    # 파일 내용 자체의 해시가 필요하다면 image_path를 사용하여 파일을 읽고 해시를 계산해야 합니다.
    image_hash_value = compute_sha256(image_rgb)
    print_log(func_name, f"계산된 이미지 해시: {image_hash_value}")

    # --- MediaPipe 얼굴 검출 ---
    faces = []
    try:
        # MediaPipe FaceDetection 객체를 'with' 구문으로 안전하게 사용
        # model_selection=1: 넓은 범위의 얼굴 감지 모델 (성능 및 정확도 고려)
        with mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
        ) as detector:
            # MediaPipe는 입력 이미지를 RGB 채널로 기대합니다.
            results = detector.process(image_rgb)
            print_log(func_name, f"MediaPipe process 호출 완료. 결과: {results}")

            if results.detections:
                print_log(func_name, f"총 {len(results.detections)}개의 얼굴 검출됨.")
                # 검출된 각 얼굴 정보 처리
                for i, det in enumerate(results.detections):
                    # 바운딩 박스 정보 추출 (상대 좌표)
                    box = det.location_data.relative_bounding_box
                    # 상대 좌표를 픽셀 좌표로 변환
                    x = int(box.xmin * width)
                    y = int(box.ymin * height)
                    w = int(box.width * width)
                    h = int(box.height * height)

                    # 검출 영역이 이미지 경계를 벗어나지 않도록 보정 (안정성 증가)
                    x = max(0, x)
                    y = max(0, y)
                    # w, h는 시작점에서 이미지 끝까지의 길이와 비교하여 조정
                    w = min(width - x, w)
                    h = min(height - y, h)

                    face_info = {
                        "face_id": i, # 루프 변수 i 사용
                        "box": {"x": x, "y": y, "width": w, "height": h},
                        "score": float(det.score[0]) # score는 보통 리스트의 첫 번째 요소입니다.
                    }
                    faces.append(face_info)
                    print_log(func_name, f"얼굴 [{i}] 정보: {face_info}")
            else:
                 print_log(func_name, "이미지에서 얼굴이 검출되지 않았습니다.")

    except Exception as e:
        # MediaPipe 처리 또는 다른 과정에서 오류 발생 시
        print_log(func_name, f"얼굴 검출 처리 중 오류 발생: {e}")
        # 오류 발생 시에도 빈 faces 리스트를 포함한 결과 반환
        return {
            "image_hash": image_hash_value, # 오류 발생 시에도 계산된 해시 반환 시도
            "image_path": str(image_path) if image_path else None,
            "faces": [] # 오류 발생 시 얼굴 목록은 비어 있음
        }

    # 최종 결과 반환
    return {
        "image_hash": image_hash_value,
        "image_path": str(image_path) if image_path else None,
        "faces": faces
    }

# --- 사용 예시 ---
# 이 부분은 실제 사용하실 때 파일 경로에 맞게 수정하셔야 합니다.
# try:
#     # 예시 1: 파일 경로를 사용하여 이미지 로드 및 함수 호출
#     image_file_path = "path/to/your/image.jpg" # <-- 실제 파일 경로로 변경
#     with Image.open(image_file_path) as img:
#          # PIL Image 객체와 파일 경로를 함수에 전달
#          detection_result = detect_faces_with_hash(image=img, image_path=image_file_path, min_detection_confidence=0.7)
#          print("\n--- 검출 결과 (파일) ---")
#          import json
#          print(json.dumps(detection_result, indent=4))
#
#     # 예시 2: NumPy 배열 이미지를 직접 생성하여 함수 호출
#     # (실제 이미지처럼 동작하지는 않지만 함수 테스트용)
#     # image_np_dummy = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8) # 더미 이미지 (높이, 너비, 채널)
#     # detection_result_np = detect_faces_with_hash(image=image_np_dummy, min_detection_confidence=0.6)
#     # print("\n--- 검출 결과 (NumPy 배열) ---")
#     # print(json.dumps(detection_result_np, indent=4))
#
# except FileNotFoundError:
#     print(f"오류: 지정된 이미지 파일을 찾을 수 없습니다.")
# except ImportError:
#      print("필요한 라이브러리(mediapipe, pillow, numpy, hashlib)가 설치되지 않았습니다.")
#      print("pip install mediapipe Pillow numpy")
# except Exception as e:
#     print(f"함수 실행 중 예상치 못한 오류 발생: {e}")

def save_face_json(json_data: Dict, json_path: Union[str, Path]):
    """검출된 얼굴 정보를 JSON으로 저장합니다."""
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

# save_index, plot_distribution, save_face 함수는 변경 없음
# ... (save_index, plot_distribution, save_face 함수 정의) ...
def save_index(index_file, encodings, paths):
    try:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=index_file.parent, suffix=".tmp") as temp_f:
            pickle.dump({"encodings": encodings, "paths": paths}, temp_f)
            temp_path = Path(temp_f.name)
        shutil.move(str(temp_path), index_file)
        return True
    except Exception as e:
        logging.error(f"❌ Failed to save index: {e}")
        return False

def plot_distribution(encodings, output_path):
    if len(encodings) < 2:
        logging.info("📉 시각화 생략 (encoding 2개 미만)")
        return
    try:
        reduced = TSNE(n_components=2, random_state=42).fit_transform(np.array(encodings))
        plt.figure(figsize=(16, 10))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7)
        plt.title("Face Index Distribution (t-SNE)")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"📊 분포도 저장 완료: {output_path}")
    except Exception as e:
        logging.error(f"❌ t-SNE 시각화 실패: {e}")

def save_face(image, bbox, save_path):
    h, w, _ = image.shape
    x, y, width, height = bbox
    left = max(int(x), 0)
    top = max(int(y), 0)
    right = min(int(x + width), w)
    bottom = min(int(y + height), h)
    if top >= bottom or left >= right:
        return False
    face = image[top:bottom, left:right]
    # 저장 시 BGR로 변환
    cv2.imwrite(str(save_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    return True

def save_face_json_with_polygon(
        image_path: Path, 
        image_hash: str, 
        faces: List[Dict], 
        json_path: Path
    ) -> None:
    """얼굴 검출 결과를 JSON 파일로 저장"""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_data = {
        "image_name": image_path.name,
        "image_path": str(image_path.resolve()),
        "image_hash": image_hash,
        "faces": faces
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Union, Dict, List
import os # 폰트 파일 경로 처리를 위해 필요할 수 있습니다.

# 이전에 사용하신 로그 출력 함수
def print_log(func, msg):
    print(f"[{func}] {msg}")

def draw_detections_on_image(
    image: Union[np.ndarray, Image.Image],
    detection_result: Dict,
    box_color: str = "red", # 바운딩 박스 색상
    box_thickness: int = 2, # 바운딩 박스 두께
    text_color: str = "white", # 텍스트 색상
    font_size: int = 15 # 텍스트 폰트 크기
) -> Image.Image:
    """
    원본 이미지 위에 검출된 얼굴 바운딩 박스 및 정보를 그려주는 함수.

    Args:
        image: 원본 이미지 (NumPy 배열 또는 PIL Image).
        detection_result: detect_faces_with_hash 함수의 반환 결과 딕셔너리.
        box_color: 바운딩 박스 색상 (PIL에서 지원하는 색상 이름 또는 튜플).
        box_thickness: 바운딩 박스 선의 두께.
        text_color: 텍스트 색상.
        font_size: 텍스트 폰트 크기.

    Returns:
        PIL.Image.Image: 바운딩 박스가 그려진 새로운 이미지 객체.
                         검출된 얼굴이 없거나 오류 발생 시 원본 이미지의 복사본을 반환합니다.
    """
    func_name = "draw_detections_on_image"
    print_log(func_name, f"함수 시작, 받은 image 타입: {type(image)}")

    # PIL Image 객체로 변환하여 그리기 준비
    if isinstance(image, np.ndarray):
        # NumPy 배열이면 PIL Image로 변환. MediaPipe는 RGB를 사용하므로 RGB로 변환.
        # 그릴 수 있도록 'RGB' 모드로 변환합니다.
        image_pil = Image.fromarray(image).convert("RGB")
        print_log(func_name, "NumPy 배열을 PIL Image로 변환 완료.")
    elif isinstance(image, Image.Image):
        # 이미 PIL Image이면 복사하여 사용 (원본 이미지 변경 방지)
        image_pil = image.copy().convert("RGB")
        print_log(func_name, "PIL Image 복사 완료.")
    else:
        print_log(func_name, f"지원하지 않는 이미지 타입입니다: {type(image)}")
        # 지원하지 않는 타입이면 None 반환하거나 오류 처리
        return None # 또는 image.copy() 등으로 원본 복사본 반환

    # 그림을 그릴 ImageDraw 객체 생성
    draw = ImageDraw.Draw(image_pil)

    # 폰트 로드 (시스템 폰트나 특정 경로의 폰트 사용)
    # 예시: 기본 폰트 또는 noto-sans-kr 같은 폰트 사용 시 경로 설정 필요
    try:
        # 시스템 폰트나 특정 경로의 폰트를 지정할 수 있습니다.
        # Windows: "arial.ttf"
        # macOS: "/Library/Fonts/Arial.ttf" 또는 다른 폰트 경로
        # Linux: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 또는 다른 폰트 경로
        # 실제 시스템에 맞는 폰트 경로를 찾아 사용하시거나,
        # 프로젝트 폴더에 폰트 파일을 두고 사용하세요.
        # 여기서는 ImageFont.load_default()를 사용합니다 (영문만 가능).
        # 한글 폰트 사용 시에는 ImageFont.truetype("폰트파일경로", font_size) 사용
        font = ImageFont.load_default() # 영문 기본 폰트
        # 예시 (한글 지원 폰트):
        # font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" # Linux 예시
        # if os.path.exists(font_path):
        #     font = ImageFont.truetype(font_path, font_size)
        # else:
        #     print_log(func_name, f"경고: 지정된 폰트 파일 '{font_path}'을 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        #     font = ImageFont.load_default()


    except Exception as e:
        print_log(func_name, f"폰트 로딩 중 오류 발생: {e}. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default() # 오류 발생 시 기본 폰트 사용

    # 검출 결과 딕셔너리가 유효하고 'faces' 키가 있는지 확인
    if detection_result and "faces" in detection_result and detection_result["faces"]:
        print_log(func_name, f"총 {len(detection_result['faces'])}개의 얼굴에 박스를 그립니다.")
        # 검출된 각 얼굴 정보에 대해 반복
        for face_info in detection_result["faces"]:
            try:
                # 바운딩 박스 정보 가져오기
                box = face_info["box"]
                x = box["x"]
                y = box["y"]
                w = box["width"]
                h = box["height"]

                # 바운딩 박스 그리기 (PIL은 (x1, y1, x2, y2) 형식을 사용)
                # (x, y)는 좌상단, (x+w, y+h)는 우하단 좌표
                draw.rectangle([(x, y), (x + w, y + h)], outline=box_color, width=box_thickness)

                # 텍스트 정보 (얼굴 ID 및 신뢰도) 생성
                # score는 소수점 첫째 자리까지만 표시
                text = f"ID: {face_info.get('face_id', 'N/A')} Score: {face_info.get('score', 0.0):.1f}"

                # 텍스트 위치 계산 (바운딩 박스 상단에 표시)
                text_x = x
                text_y = y - font_size # 폰트 크기만큼 위로 올림

                # 이미지를 벗어나지 않도록 텍스트 위치 조정
                if text_y < 0:
                    text_y = y + h + 2 # 바운딩 박스 아래에 표시

                # 텍스트 그리기
                draw.text((text_x, text_y), text, fill=text_color, font=font)

                print_log(func_name, f"얼굴 ID {face_info.get('face_id', 'N/A')}에 박스 및 텍스트 그림.")

            except KeyError as ke:
                print_log(func_name, f"오류: face_info 딕셔너리에 필수 키가 없습니다: {ke}. 해당 얼굴 정보 건너뜀.")
                print_log(func_name, f"문제의 face_info: {face_info}")
            except Exception as e:
                print_log(func_name, f"얼굴 정보 그리던 중 오류 발생: {e}. 해당 얼굴 정보 건너뜀.")

    else:
        print_log(func_name, "검출된 얼굴 정보가 없거나 유효하지 않습니다. 박스를 그리지 않습니다.")

    print_log(func_name, "함수 종료.")
    return image_pil

# --- 사용 예시 ---
# 이 부분은 실제 사용하실 때 detect_faces_with_hash 함수 호출 결과와 함께 사용하세요.

# 예시를 위해 더미 이미지와 더미 검출 결과 생성
# (실제로는 detect_faces_with_hash 함수를 호출하여 결과를 받아야 합니다.)

# # 1. 더미 이미지 생성 (NumPy 배열)
# dummy_image_np = np.zeros((600, 800, 3), dtype=np.uint8) # 검은색 이미지
# # 또는 PIL Image로 생성
# # dummy_image_pil = Image.new('RGB', (800, 600), color = 'white') # 흰색 이미지

# # 2. 더미 검출 결과 생성 (실제 detect_faces_with_hash 결과와 유사한 구조)
# # (실제로는 detect_faces_with_hash("path/to/your/image.jpg") 등으로 얻어야 합니다)
# dummy_detection_result = {
#     "image_hash": "a1b2c3d4e5f67890...",
#     "image_path": "path/to/dummy_image.jpg",
#     "faces": [
#         {"face_id": 0, "box": {"x": 100, "y": 100, "width": 150, "height": 200}, "score": 0.98},
#         {"face_id": 1, "box": {"x": 400, "y": 150, "width": 100, "height": 150}, "score": 0.95},
#         {"face_id": 2, "box": {"x": 600, "y": 50, "width": 80, "height": 120}, "score": 0.85}
#     ]
# }

# # 3. 함수 호출하여 이미지에 박스 그리기
# # numpy 배열 입력 시
# # image_with_boxes_np = draw_detections_on_image(image=dummy_image_np, detection_result=dummy_detection_result)
#
# # pil image 입력 시
# # image_with_boxes_pil = draw_detections_on_image(image=dummy_image_pil, detection_result=dummy_detection_result)
#
# # 4. 결과 이미지 저장 또는 표시
# # image_with_boxes_np.save("dummy_image_with_boxes_np.png")
# # image_with_boxes_pil.save("dummy_image_with_boxes_pil.png")
#
# # 주피터 노트북 등에서는 다음과 같이 바로 표시 가능
# # image_with_boxes_pil # 마지막 줄에 변수 이름을 쓰면 표시됨


def detect_faces_landmark(config: processing_config):
    func_name = "detect_faces_landmark"
    print_log(func_name, "시작")

    try:
        # 설정에서 Path 객체 바로 가져오기 (자동 생성됨)
        # 필수 경로들을 가져오는 헬퍼 함수 정의
        def get_required_path(config_obj: processing_config, key: str) -> Path:
            """설정에서 필수 경로를 가져오고 Path 객체인지 확인, 아니면 오류 발생"""
            path_obj = config_obj.get_path(key) # get_path는 디렉토리 생성도 시도함
            if not isinstance(path_obj, Path):
                # get_path가 Path 객체를 반환하지 못한 경우 (설정에 없거나 유효하지 않음)
                error_msg = f"필수 경로 '{key}'가 설정 파일에 없거나 유효하지 않습니다. 확인된 값: {path_obj} (타입: {type(path_obj)})"
                logger.critical(error_msg)
                raise ValueError(error_msg) # 오류 발생시켜 프로그램 중단 유도
            print_log(func_name, f"경로 확인됨 - {key:20s}: {path_obj}")
            return path_obj

        # 헬퍼 함수를 사용하여 필수 경로 가져오기
        raw_image_dir =     get_required_path(config, "raw_image_dir")
        raw_jsons_dir =     get_required_path(config, "raw_jsons_dir")
        detected_list_path =get_required_path(config, "detected_list_path")
        undetect_list_path =get_required_path(config, "undetect_list_path")
        failed_list_path =  get_required_path(config, "failed_list_path")

        # --- 설정 파일에서 tolerance 신뢰도 값 읽기 ---
        models_config =     config.get_models_config()
        min_detection_confidence = float(models_config.get("min_detection_confidence", 0.6)) # models 섹션에서 가져오기
        target_size_tuple = tuple(models_config.get("target_size", [224, 224])) # 기본값 [224, 224]
        print_log(func_name, f"사용할 정밀도(min_detection_confidence): {min_detection_confidence}, target_size: {target_size_tuple}") # 로깅 추가 (선택 사항)

        ext_list     = [".jpg", ".jpeg", ".png"]
        ext_list = models_config.get("supported_image_extensions", ext_list)
    except (KeyError, TypeError, AttributeError) as e:
        logger.critical(f"모델 변수값 가저오기 오류 발생: {e}")
        return

    supported_extensions = {ext.lower() for ext in ext_list}
    print_log(func_name, f"📂 이미지 supported_extensions: {supported_extensions}")
    print_log(func_name, f"📂 이미지 raw_image_dir: {raw_image_dir}")
    images = [p for p in raw_image_dir.glob("**/*") if ( p.is_file() and p.suffix.lower()) in supported_extensions]
    image_count = len(images)
    if image_count == 0:
        logging.warning(f"⚠️ {raw_image_dir} 에서 이미지를 찾을 수 없습니다.")
        return
    width = math.floor(math.log10(image_count)) + 1
    print_log(func_name, f"📂 이미지 {image_count}장 탐색됨")

    processed_files_count = 0 # 처리된 얼굴 수 카운트
    detected_face_count = 0
    image_read_faild_count = 0

    for idx, img_path in enumerate(images, 1):
        try:
            img_gbr = cv2.imread(str(img_path)) # OpenCV (cv2) 라이브러리
            # numpy.ndarray (NumPy 배열), BGR (Blue-Green-Red, 8비트 정수형 (uint8)
            if img_gbr is None:
                image_read_faild_count += 1
                print_log(func_name, f"[{image_read_faild_count:0{width}d}/{image_count}] ⚠️ 이미지 로딩 실패: {img_path.name}")
                continue
            print_log(func_name, f"[{idx:0{width}d}/{image_count}] 번째 파일 읽음: {img_path.name}")
            # Mediapipe는 RGB 이미지를 사용, numpy.ndarray
            # detect_faces_with_hash 호출 시 설정에서 읽은 target_size_tuple 사용
            print_log(func_name, f"[{idx:0{width}d}/{image_count}] 번째 파일 해시 만들러가기: {img_path.name}")
            image_hash, image_path, faces = detect_faces_with_hash(
                image=img_gbr,
                image_path=img_path,
                min_detection_confidence=min_detection_confidence,
                target_size = target_size_tuple
            )
            if faces:
                processed_files_count += 1
                detected_face_count += len(faces)
                print_log(func_name, f"[{idx:0{width}d}/{image_count}] 번째 json파일 만들러가기: {img_path.name}")
                
                #JSON 경로 생성 시 Path 객체 연산 사용
                jsons_path = Path(raw_jsons_dir)/f"{img_path.stem}.json" # 문자열 변환 불필요
                save_face_json_with_polygon(img_path, img_hash, bbox_list, jsons_path) # jsons_path는 이미 Path 객체
            else:
                image_read_faild_count += 1
        except Exception as e:
            # 오류 발생 시에도 메모리 정리 시도
            # print_log(func_name, f"⚠️ 처리 중 오류 ({img_path.name}): {e}", exc_info=True) # 상세 오류 로깅
            gc.collect()

    print_log(func_name, f"✅총 처리수[얼굴:{detected_face_count:{width}d}개]/파일:{processed_files_count:{width}d}개/총파일:{image_count}개]")
    print_log(func_name, "🎉 인덱싱 완료.")

if __name__ == "__main__":
    func_name = "main"
    print_log(func_name, "시작")

    # 0. 기즘 내가 일하는 곳은"
    direction_dir = os.getcwd()
    print_log(func_name, f"지금 쥔계서 계신곳(direction_dir)      : {direction_dir}")
    worker_path_obj = Path(__file__).resolve()
    project_root_path = worker_path_obj.parent.parent
    print_log(func_name, f"지금 일꾼이 일하는곳(worker_dir_name)  : {project_root_path}")

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = f"{project_root_path}/config/{project_root_path.name}.yaml"
        print_log(func_name, f"만들어진 파일 경로(config_path)        : {config_path}")

    try:
        config = processing_config(config_path)

        # get_path를 사용하여 로그 디렉토리 Path 객체 가져오기
        # ensure_exists=True (기본값)이므로, get_path 내부에서 디렉토리 생성 시도
        log_dir_path_obj = config.get_path("worker_logs_dir")

        # Path 객체로 제대로 가져왔는지 확인
        if not isinstance(log_dir_path_obj, Path):
            print_log(func_name, f"worker_logs_dir'를 Path 객체로 가져오지 못했습니다.")
            print_log(func_name, f"log_dir_path_obj의 타입: {type(log_dir_path_obj)}")
            sys.exit(1)

        # 최종 로그 파일 경로 생성
        log_file_path = log_dir_path_obj / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        print_log(func_name, f"로그 파일 경로: {log_file_path}")

        # 수정된 add_file_logger 호출 (최종 파일 경로 전달)
#        add_file_logger(log_file_path)

        print_log(func_name, f"발자국 그리기 시작 : {log_file_path}")

    except Exception as e:
        logger.error(func_name, f"설정 로드 또는 로깅 설정 중 오류 발생: {config_path} - {e}", exc_info=True)
        sys.exit(1)
    
