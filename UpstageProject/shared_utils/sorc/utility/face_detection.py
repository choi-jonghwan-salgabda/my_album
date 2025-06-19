# ======== 🧱 표준 라이브러리 ========
import os
import sys
#import gc
#import re
import math # 로그 자릿수 계산을 위해 math 모듈 추가
#import shutil
#import pickle
#import tempfile
import json # JSON 모듈 import 추가
from pathlib import Path

# ======== 🧪 과학 및 수치 계산 ========
import numpy as np
import cv2
from PIL import Image

import dlib
import hashlib

# ======== 🧠 머신러닝/딥러닝 ========
# from sklearn.manifold import TSNE  # 사용 시 주석 해제
import mediapipe as mp

# ======== 🧾 타입 힌팅 ========
from typing import List, Dict, Union, Any, BinaryIO

from .config_manager import print_log, ProjectConfig

def initialize_face_detector():
    """
    얼굴 검출기를 초기화합니다.
    Returns:
        detector: Dlib의 HOG 기반 얼굴 검출기
    """
    func_name = "initialize_face_detector"
    print_log(func_name, f"시작")

    detector = dlib.get_frontal_face_detector()
    return detector


def return_np_array(image: Union[np.ndarray, Image.Image]):
    # PIL Image를 NumPy 배열로 변환 (MediaPipe 입력 형식에 맞춤)
    func_name = "return_np_array"
    print_log(func_name, f"시작")

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

def save_face_json_with_polygon(
        image_path: Path, 
        image_hash: str, 
        faces: List[Dict], 
        json_path: Path
    ) -> None:
    """얼굴 검출 결과를 JSON 파일로 저장"""
    func_name = "save_face_json_with_polygon"
    print_log(func_name, "시작")

    try:
        # 1. 부모 디렉토리 생성 시도
        target_dir = json_path.parent
        print_log(func_name, f"JSON 저장 디렉토리 확인/생성 시도: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        print_log(func_name, f"디렉토리 준비 완료:  {target_dir}")

        # 2. JSON 데이터 준비
        json_data = {
            "image_name": image_path.name,
            "image_path": str(image_path.resolve()),
            "image_hash": image_hash,
            "faces": faces
        }

        # 3. 파일 쓰기 시도
        print_log(func_name, f"JSON 파일 쓰기 시도: {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print_log(func_name, f"JSON 파일 저장 완료: {json_path}")

    except OSError as e:
        print_log(func_name, f"파일/디렉토리 작업 오류 발생 ({json_path}): {e}")
        # 필요하다면 여기서 더 구체적인 오류 처리 또는 로깅 추가
    except TypeError as e:
        print_log(func_name, f"JSON 직렬화 오류 발생 ({json_path}): {e}")
    except Exception as e:
        print_log(func_name, f"JSON 저장 중 예상치 못한 오류 발생 ({json_path}): {e}")
        # 오류 발생 시에도 계속 진행할지, 아니면 여기서 프로그램을 중단할지 결정 필요
        # raise # 오류를 다시 발생시켜 상위 호출자에게 알리려면 주석 해제

def detect_faces_dlib(detector, image):
    """
    주어진 이미지에서 얼굴을 검출합니다.

    Args:
        image: 얼굴을 검출할 이미지 (NumPy 배열).

    Returns:
        List[Dict]: 검출된 얼굴의 위치 정보 (x, y, width, height).
    """
    # Dlib의 얼굴 검출기 초기화
    func_name = "detect_faces_dlib"
    print_log(func_name, f"시작")

    # 얼굴 검출
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환
    faces = detector(gray_image)

    face_locations = []
    for i, face in enumerate(faces):
        # 얼굴의 위치 정보 저장
        # 올바른 딕셔너리 구조로 수정하고 null 대신 None 사용
        face_info = {
            "face_id": i,
            "box": {
                "x": face.left(),
                "y": face.top(),
                "width": face.right() - face.left(),
                "height": face.bottom() - face.top()
            },
            "name": None  # 파이썬에서는 None을 사용합니다.
        }
        face_locations.append(face_info)

    return face_locations

mp_face_detection = mp.solutions.face_detection # <--- 이 부분이 face_indexer_landmark.py에 누락됨
def detect_faces_FaceDetection(detector, image: Union[np.ndarray, Image.Image], mosel_config):
    """
    주어진 이미지에서 얼굴을 검출합니다.

    Args:
        image: 얼굴을 검출할 이미지 (NumPy 배열).

    Returns:
        List[Dict]: 검출된 얼굴의 위치 정보 (x, y, width, height).
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

    # --- MediaPipe 얼굴 검출 ---
    faces = []
    try:
        # MediaPipe FaceDetection 객체를 'with' 구문으로 안전하게 사용
        # model_selection=1: 넓은 범위의 얼굴 감지 모델 (성능 및 정확도 고려)
        min_detection_confidence = float(models_config.get("min_detection_confidence", 0.6)) # models 섹션에서 가져오기
        target_size_tuple = tuple(models_config.get("target_size", [224, 224])) # 기본값 [224, 224]
        model_selection = int(models_config.get("model_selection", 1))
        print_log(func_name, f"사용할 정밀도(min_detection_confidence): {min_detection_confidence}") # 로깅 추가 (선택 사항)
        print_log(func_name, f"사용할 사진크기(target_size): {target_size_tuple}") # 로깅 추가 (선택 사항)
        print_log(func_name, f"사용할 사진거리(model_selection): {model_selection}") # 로깅 추가 (선택 사항)
        with mp_face_detection.FaceDetection(
            model_selection=model_selection,
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

def detect_objects_opencv_dnn(
    net: cv2.dnn.Net, 
    image: Union[np.ndarray], 
    conf_threshold: float = 0.5, 
    nms_threshold: float = 0.4
    ) -> list:
    """
    OpenCV DNN 모델을 사용하여 이미지에서 객체를 검출합니다.

    Args:
        net: 미리 로드된 OpenCV DNN 네트워크 객체.
        image: 객체 검출을 수행할 입력 이미지 (numpy 배열).
               PIL Image 객체를 지원하려면 코드 수정이 필요합니다.
        conf_threshold: 검출 결과의 신뢰도 임계값. 이 값보다 낮은 결과는 무시됩니다.
        nms_threshold: Non-Maximum Suppression (NMS) 임계값. 중복된 바운딩 박스 제거에 사용됩니다.

    Returns:
        검출된 객체 목록. 각 객체는 바운딩 박스 좌표, 신뢰도, 클래스 ID를 포함합니다.
        예: [[left, top, width, height, confidence, class_id], ...]
    """
    # 이미지의 높이와 너비를 가져옵니다.
    height, width = image.shape[:2]

    # 이미지에서 Blob을 생성합니다.
    # 모델에 따라 scale, size, mean, swapRB 등을 조정해야 할 수 있습니다.
    # 예시: blobFromImage(image, scale factor, size, mean, swapRB=True, crop=False)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    # 네트워크의 입력으로 Blob을 설정합니다.
    net.setInput(blob)

    # 네트워크의 출력 레이어 이름을 가져옵니다. (모델에 따라 다를 수 있습니다)
    # 일반적으로 객체 검출 모델은 하나의 출력 레이어를 가집니다.
    output_layers = net.getUnconnectedOutLayersNames()

    # 순방향 추론(forward pass)을 수행하여 결과를 얻습니다.
    outs = net.forward(output_layers)

    # 검출된 객체 정보를 저장할 리스트를 초기화합니다.
    detections = []
    class_ids = []
    confidences = []
    boxes = []

    # 검출 결과를 분석합니다.
    # 결과 형태는 모델(YOLO, SSD 등)에 따라 다릅니다. 여기서는 YOLO 스타일 출력을 가정합니다.
    # YOLO 출력은 보통 (num_detections, 5 + num_classes) 형태입니다.
    for out in outs:
        for detection in out:
            # 클래스별 신뢰도와 객체 신뢰도를 분리합니다.
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 설정된 신뢰도 임계값보다 높은 검출 결과만 고려합니다.
            if confidence > conf_threshold:
                # 바운딩 박스 좌표를 계산합니다. (센터 x, 센터 y, 너비, 높이 형태)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 좌상단 좌표로 변환합니다.
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression (NMS)를 적용하여 중복된 박스를 제거합니다.
    # cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    # score_threshold는 사실상 conf_threshold와 같은 역할을 하지만, NMSBoxes 함수는 별도로 받습니다.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    final_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            final_detections.append({
                "box": [box[0], box[1], box[2], box[3]], # x, y, w, h
                "confidence": confidences[i],
                "class_id": class_ids[i]
            })

    return final_detections

# --- 함수 사용 예시 ---
if __name__ == '__main__':
    # 모델 파일 경로 (예: YOLOv3)
    # 실제 모델 파일은 별도로 다운로드해야 합니다.
    # cfg 파일: 모델의 구조 정보
    # weights 파일: 훈련된 가중치 정보
    # names 파일: 클래스 이름 목록 (선택 사항)
    model_cfg_path = "path/to/yolov3.cfg"
    model_weights_path = "path/to/yolov3.weights"
    class_names_path = "path/to/coco.names" # 예시: COCO 데이터셋 클래스 이름

    # 1. DNN 네트워크 로드
    try:
        # cv2.dnn.readNet(weights, config) 함수를 사용하여 네트워크를 로드합니다. [[3]](https://junstar92.tistory.com/411)
        net = cv2.dnn.readNet(model_weights_path, model_cfg_path)

        # 사용할 백엔드와 타겟을 설정할 수 있습니다. (예: CPU, GPU)
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # 또는 cv2.dnn.DNN_TARGET_CUDA 등

        print("DNN 네트워크 로드 성공")

    except Exception as e:
        print(f"DNN 네트워크 로드 실패: {e}")
        print("모델 설정(.cfg) 파일과 가중치(.weights) 파일 경로를 확인해주세요.")
        net = None # 로드 실패 시 net 변수를 None으로 설정

    if net:
        # 2. 검출할 이미지 로드
        image_path = "path/to/your/image.jpg" # 실제 이미지 파일 경로로 변경
        image = cv2.imread(image_path)

        if image is not None:
            # 3. 객체 검출 함수 호출
            # 신뢰도 임계값과 NMS 임계값은 필요에 따라 조정하세요.
            detections = detect_objects_dnn(net, image, conf_threshold=0.5, nms_threshold=0.4)

            # 4. 결과 출력 또는 시각화
            print(f"검출된 객체 수: {len(detections)}")

            # 클래스 이름 로드 (가능하다면)
            class_names = []
            try:
                with open(class_names_path, "r") as f:
                    class_names = [line.strip() for line in f.readlines()]
            except FileNotFoundError:
                print(f"'{class_names_path}' 파일을 찾을 수 없습니다. 클래스 이름은 출력되지 않습니다.")
                class_names = [str(i) for i in range(1000)] # 임의의 클래스 ID 사용

            output_image = image.copy()
            for det in detections:
                box = det["box"]
                confidence = det["confidence"]
                class_id = det["class_id"]

                x, y, w, h = box
                label = f"{class_names[class_id]}: {confidence:.2f}"

                # 바운딩 박스 그리기
                color = (0, 255, 0) # 초록색
                cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)

                # 클래스 이름 및 신뢰도 텍스트 추가
                cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                print(f"  - 객체: {class_names[class_id]}, 신뢰도: {confidence:.2f}, 박스: ({x}, {y}, {w}, {h})")

            # 결과를 화면에 표시 (OpenCV GUI 사용)
            # cv2.imshow("Object Detection Result", output_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 또는 파일로 저장
            cv2.imwrite("detection_result.jpg", output_image)
            print("검출 결과 이미지가 'detection_result.jpg'로 저장되었습니다.")


        else:
            print(f"이미지 로드 실패: '{image_path}' 파일을 찾거나 읽을 수 없습니다.")

    else:
        print("네트워크가 로드되지 않아 객체 검출을 수행할 수 없습니다.")


# === 사용 예시 ===
if __name__ == "__main__":
    func_name = "main"
    print_log(func_name, "시작")

    # 0. 지금 내가 일하는 곳은 - 정체성 정의 "
    direction_dir = os.getcwd()
    print_log(func_name, f"지금 쥔께서 계신곳- O/S(direction_dir)    : {direction_dir}")
    worker_dir = Path(__file__).resolve().parent
    print_log(func_name, f"지금 일꾼이 일하는곳 (worker_dir)         : {worker_dir}")
    project_root_dir = worker_dir.parent
    print_log(func_name, f"지금 쥔께서 계신곳-계산(project_root_dir) : {project_root_dir}")

    # 1.일하며 걸어간 발자국을 적는다.
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = f"{project_root_dir}/config/{project_root_dir.name}.yaml"
    print_log(func_name, f"구성파일 경로-입력/계산(config_path)      : {config_path}")

    # 할 일에 대한 환경구성정보 읽기
    try:
        config = ProjectConfig(config_path)

        # 계산된 것과 구성파일 정보 검증
        dir_obj = config.get_project_config()
        # Path 객체로 제대로 가져왔는지 확인
        if project_root_dir != dir_obj.get('root_dir', ''):
            print_log(func_name, f"프로젝트 루트(읽어옮): {dir_obj.get('root_dir', '')}")
            print_log(func_name, f"프로젝트 루트(계산됨): {project_root_dir}")
            print_log(func_name, f"다시 확인해 주세요")
            sys.exit(1)

        dir_obj = config.get_utility_config()
        if worker_dir != dir_obj.get('utility_dir', ''):
            print_log(func_name, f"일꾼이 있는곳(읽어옮): {dir_obj.get('utility_dir', '')}")
            print_log(func_name, f"일꾼이 있는곳(계산됨): {worker_dir}")
            print_log(func_name, f"다시 확인해 주세요")
            sys.exit(1)

        worker_logs_dir = f"{project_root_dir}/outputs/worker_logs"
        dir_obj = config.get_outputs_config()
        if worker_logs_dir != str(dir_obj.get('worker_logs_dir', '')):
            print_log(func_name, f"발자국 그리는곳(읽어옮): {str(dir_obj.get('worker_logs_dir', ''))}")
            print_log(func_name, f"발자국 그리는곳(계산됨): {worker_logs_dir}")
            print_log(func_name, f"다시 확인해 주세요")
            sys.exit(1)

        # 여기까즈는 프로젝트의환경에 관한 정의였습니다.
        # 이제부터는  프로젝트의 실행에 직접적인 정보의설정입니다.
        dir_obj = config.get_dataset_config()
        raw_image_dir = Path(dir_obj.get('raw_image_dir', '')).expanduser().resolve()
        if not (isinstance(raw_image_dir, Path) and raw_image_dir.is_dir()):
            # 이 블록은 raw_image_path_obj가 유효한 디렉토리가 아닐 때 실행됩니다.
            print_log(func_name, f"'{raw_image_dir}'는 유효한 디렉토리가 아닙니다.")
            sys.exit(1)

        raw_jsons_dir = Path(dir_obj.get('raw_jsons_dir', '')).expanduser().resolve()
        # raw_jsons_dir가 Path 객체인지 확인하고, 디렉토리인지 확인합니다.
        if not isinstance(raw_jsons_dir, Path):
            # Path 객체가 아닌 경우 (설정 오류 등)
            print_log(func_name, f"'{raw_jsons_dir}'는 유효한 Path 객체가 아닙니다.")
            sys.exit(1)
        if not raw_jsons_dir.is_dir():
            print_log(func_name, f"'{raw_jsons_dir}' 디렉토리가 존재하지 않아 새로 생성합니다.")
            try:
                raw_jsons_dir.mkdir(parents=True, exist_ok=True) # 디렉토리 생성 (부모 디렉토리 포함, 이미 있어도 오류 없음)
            except OSError as e:
                print_log(func_name, f"'{raw_jsons_dir}' 디렉토리 생성 실패: {e}")
                sys.exit(1) # 생성 실패 시 종료

        # --- 설정 파일에서 tolerance 신뢰도 값 읽기 ---
        models_config =     config.get_models_config()
        # --- 설정 파일에서 tolerance 신뢰도 값 읽기 ---
        models_config =     config.get_models_config()
        min_detection_confidence = float(models_config.get("min_detection_confidence", 0.6)) # models 섹션에서 가져오기
        target_size_tuple = tuple(models_config.get("target_size", [224, 224])) # 기본값 [224, 224]
        print_log(func_name, f"사용할 정밀도(min_detection_confidence): {min_detection_confidence}, target_size: {target_size_tuple}") # 로깅 추가 (선택 사항)

        # 읽을 화일의 종류를 정함.
        ext_list     = [".jpg", ".jpeg", ".png"]
        ext_list = models_config.get("supported_image_extensions", ext_list)
        supported_extensions = {ext.lower() for ext in ext_list}
        print_log(func_name, f"📂 이미지 supported_extensions: {supported_extensions}")
        print_log(func_name, f"📂 이미지 raw_image_dir: {raw_image_dir}")
        print_log(func_name, f"📂 이미지 raw_jsons_dir: {raw_jsons_dir}")

        all_items = list(raw_image_dir.glob("**/*"))
        # print_log(func_name, f"👀 glob 결과 ({len(all_items)}개): {[str(p) for p in all_items[:20]]}") # 처음 20개 항목만 출력 (너무 많을 경우 대비)
        
        images = [p for p in raw_image_dir.glob("**/*") if ( p.is_file() and p.suffix.lower()) in supported_extensions]
        image_count = len(images)
        if image_count == 0:
            print_log(func_name, f"⚠️ {raw_image_dir} 에서 이미지를 찾을 수 없습니다.")
            sys.exit(1) # 생성 실패 시 종료
        width = math.floor(math.log10(image_count)) + 1
        print_log(func_name, f"📂 이미지 {image_count}장 탐색됨")

        processed_files_count = 0 # 처리된 얼굴 수 카운트
        detected_face_count = 0
        image_read_faild_count = 0
        
        face_detector = initialize_face_detector()
        print_log(func_name, f"initialize_face_detector() 완료")

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

                # 얼굴 검출
                # --- 이미지 해시 계산 ---
                # compute_sha256 함수를 호출하여 해시 값을 얻습니다.
                # compute_sha256 함수가 이미지 데이터를 받으므로 image_rgb를 전달합니다.
                # 파일 내용 자체의 해시가 필요하다면 image_path를 사용하여 파일을 읽고 해시를 계산해야 합니다.
                image_hash_value = compute_sha256(img_gbr)
                print_log(func_name, f"계산된 이미지 해시: {image_hash_value}")
                detected_faces = detect_faces_FaceDetection(face_detector, img_gbr, models_config)
                # detected_faces = detect_faces_dlib(face_detector, img_gbr)
                print_log(func_name, f"검출된 얼굴 수: {len(detected_faces)}")

                # 검출된 얼굴 정보 출력
                if detected_faces:
                    processed_files_count += 1
                    detected_face_count += len(detected_faces)
                    print_log(func_name, f"[{idx:0{width}d}/{image_count}] 번째 json파일 만들러가기: {img_path.name}")
                    
                    #JSON 경로 생성 시 Path 객체 연산 사용
                    jsons_path = Path(raw_jsons_dir)/f"{img_path.stem}.json" # 문자열 변환 불필요
                    print_log(func_name, f"img_path :{str(img_path)}")
                    print_log(func_name, f"json_path:{str(jsons_path)}")
                    print_log(func_name, f"image_hash_value :{str(image_hash_value)}")
                    print_log(func_name, f"detected_faces:{detected_faces}")
                    save_face_json_with_polygon(img_path, image_hash_value, detected_faces, jsons_path) # jsons_path는 이미 Path 객체
                else:
                    image_read_faild_count += 1
            except Exception as e:
                # 오류 발생 시에도 메모리 정리 시도
                # print_log(func_name, f"⚠️ 처리 중 오류 ({img_path.name}): {e}", exc_info=True) # 상세 오류 로깅
                gc.collect()

        print_log(func_name, f"✅읽은 사진 총수:        {image_count}개]")
        print_log(func_name, f"✅읽기를 실패한 사진 총수:{image_read_faild_count:{width}d}개]/{image_count}개]")
        print_log(func_name, f"✅읽기를 성골한 사진 총수:{processed_files_count:{width}d}개]/{image_count}개]")
        print_log(func_name, f"✅찾아낸 얼궇 사진 총수  :{detected_face_count:{width}d}개]/{image_count}개]")
        print_log(func_name, "🎉 인덱싱 완료.")

    except (KeyError, TypeError, AttributeError) as e:
        print_log(func_name, f"모델 변수값 가저오기 오류 발생: {e}")
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

