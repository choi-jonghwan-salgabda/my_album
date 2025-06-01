"""
이 스크립트의 주요 목적은 이미지에서 이미 탐지된 객체(예: 사람)들의 영역 내에서 얼굴을 추가로 탐지하고, 그 결과를 JSON 파일과 이미지 파일로 저장하는 것입니다. object_detector.py와 같은 이전 단계에서 생성된 JSON 파일을 입력으로 사용합니다.

주요 기능은 다음과 같이 나눌 수 있습니다.

초기 설정 및 로깅:

필요한 라이브러리(OpenCV, PyTorch, YOLO, PIL 등)와 자체 유틸리티(configger, SimpleLogger, object_utils)를 임포트합니다.
initialize_application_logging(): SimpleLogger를 사용하여 애플리케이션 전체의 로깅 설정을 초기화합니다. 로그 파일 경로, 로그 레벨, 포맷 등을 설정하여 실행 과정을 기록합니다.
핵심 헬퍼 함수 - detect_faces_in_crop_yolo_internal():

입력:
image_crop: 잘라낸 객체 이미지 (OpenCV cv2.Mat 형식).
obj_bbox_origin_xyxy: 잘라낸 객체의 원본 이미지 기준 좌표 [x1, y1, x2, y2].
yolo_face_model: 미리 로드된 YOLO 얼굴 탐지 모델 객체.
confidence_threshold: 얼굴 탐지에 사용할 최소 신뢰도 값.
동작:
입력된 image_crop (객체 이미지)에 대해 yolo_face_model을 실행하여 얼굴을 탐지합니다.
탐지된 각 얼굴의 경계 상자(bounding box)는 image_crop 내의 로컬 좌표로 얻어집니다.
이 로컬 좌표를 obj_bbox_origin_xyxy (객체의 원본 이미지 내 위치)를 이용하여 원본 이미지 전체에서의 절대 좌표로 변환합니다.
출력: 원본 이미지 좌표 기준으로 변환된 얼굴들의 경계 상자(bbox_xyxy), 신뢰도(confidence), 레이블(label, 여기서는 "face")을 담은 딕셔너리 리스트를 반환합니다.
메인 처리 함수 - run_face_detection_from_json():

설정 값 로드:
configger 객체(cfg)를 통해 YAML 설정 파일에서 필요한 경로들(입력 JSON 디렉토리, 주석 처리된 이미지 저장 디렉토리, 업데이트된 JSON 저장 디렉토리, 잘라낸 얼굴 이미지 저장 디렉토리)과 모델 관련 파라미터(모델 이름, 가중치 경로, 신뢰도, CPU 사용 여부 등)를 가져옵니다.
YOLO 얼굴 탐지 모델 로드:
설정에 따라 CPU 또는 GPU (CUDA 가능 시)를 사용 장치로 선택합니다.
설정된 모델 가중치 경로(model_weights_path) 또는 모델 이름(model_name)을 사용하여 Ultralytics YOLO 모델을 로드합니다.
로드된 모델을 선택된 장치로 이동시킵니다.
선택적으로 모델 워밍업(작은 더미 이미지로 첫 추론 실행)을 수행하여 실제 이미지 처리 시 첫 추론 속도를 개선합니다.
입력 JSON 파일 탐색:
설정된 input_jsons_dir에서 .json 확장자를 가진 파일들을 찾아 처리할 목록을 만듭니다. 이 JSON 파일들은 object_detector.py에서 생성된, 이미지 내 객체 탐지 정보를 담고 있는 파일들입니다.
통계 변수 초기화:
처리된 JSON 파일 수, 얼굴이 탐지된 이미지 수, 총 처리 객체 수, 얼굴이 탐지된 객체 수, 총 탐지 얼굴 수 등을 기록하기 위한 변수들을 초기화합니다.
메인 루프 (JSON 파일 순회):
탐색된 각 JSON 파일을 순회하며 다음 작업을 수행합니다.
JSON 데이터 로드: load_json 유틸리티 함수를 사용하여 현재 JSON 파일의 내용을 읽어옵니다.
이미지 정보 추출: JSON 데이터에서 원본 이미지 경로(image_path)와 object_detector.py가 탐지한 객체 목록(detected_objects_list, JSON 내 "faces" 키 아래 저장됨)을 가져옵니다.
원본 이미지 로드: cv2.imread를 사용하여 원본 이미지를 로드합니다. 주석 처리를 위해 이미지 복사본(annotated_image_copy)도 만듭니다.
내부 루프 (이미지 내 객체 순회):
현재 이미지에서 탐지된 각 객체(obj_data)를 순회합니다.
객체의 경계 상자(obj_bbox_xyxy)를 가져옵니다.
좌표 유효성을 검사하고, 원본 이미지에서 해당 객체 영역을 잘라냅니다 (object_crop_image).
detect_faces_in_crop_yolo_internal() 함수를 호출하여 이 object_crop_image 내에서 얼굴을 탐지합니다.
얼굴이 탐지되면:
탐지된 얼굴 정보(faces_in_object)를 현재 객체 데이터(obj_data)에 "detected_faces"라는 새 키로 추가합니다.
통계 변수들을 업데이트합니다.
annotated_image_copy에 탐지된 얼굴의 경계 상자를 녹색으로 그립니다.
save_cropped_face_image() 유틸리티 함수를 호출하여 잘라낸 각 얼굴 이미지를 output_cropped_face_dir에 저장합니다. 파일명은 원본이미지명_obj객체인덱스_face얼굴인덱스.png 형식입니다.
(얼굴 정보가 추가되었을 수 있는) obj_data를 updated_objects_data_list에 추가합니다.
JSON 데이터 업데이트: 현재 이미지에 대한 image_data_from_json의 "faces" 키 값을 updated_objects_data_list로 교체합니다.
주석 처리된 이미지 저장: annotated_image_copy (객체 및 얼굴 경계 상자가 그려진 이미지)를 output_annotated_image_dir에 저장합니다.
업데이트된 JSON 저장: save_object_json_with_polygon() 유틸리티 함수를 사용하여 얼굴 탐지 결과가 추가된 image_data_from_json을 output_updated_json_dir에 새로운 JSON 파일로 저장합니다.
최종 통계 로깅:
모든 JSON 파일 처리가 끝나면, 수집된 통계 정보를 로그로 출력합니다.
스크립트 실행 블록 (if __name__ == "__main__":):

스크립트가 직접 실행될 때의 진입점입니다.
initialize_application_logging()을 호출하여 로깅을 설정합니다.
configger를 사용하여 설정 파일(my_yolo_tiny.yaml)을 로드합니다.
run_face_detection_from_json() 함수에 configger 객체를 전달하여 메인 프로세스를 실행합니다.
실행 중 발생할 수 있는 예외를 처리하고, 최종적으로 로거를 종료합니다.
요약하자면, face_detector_with_json.py는 객체 탐지 결과를 입력받아, 각 객체 영역 내에서 더 상세한 얼굴 탐지를 수행하고, 이 추가 정보를 기존 데이터에 통합하여 새로운 JSON과 이미지로 출력하는 2단계 탐지 파이프라인의 일부라고 볼 수 있습니다. 설정 파일을 통해 유연하게 경로와 모델 파라미터를 관리하며, 로깅을 통해 처리 과정을 상세히 기록합니다.

"""
import json
import os
import logging # SimpleLogger를 통해 이미 사용 가능하지만, 직접 사용이 필요할 경우를 위해
import sys
from pathlib import Path
from typing import List, Dict, Any
import cv2
import torch
import numpy as np # 모델 워밍업용 더미 이미지 생성에 사용
from ultralytics import YOLO
from PIL import Image # 얼굴 이미지 저장을 위해 PIL.Image 사용
from datetime import datetime

# my_utils 유틸리티 임포트
try:
    from my_utils.config_utils.SimpleLogger import logger # 공유 로거
    from my_utils.config_utils.configger import configger # 설정 관리자
    # object_utils에 일반적인 load_json, save_json 함수가 있다고 가정합니다.
    # 만약 save_object_json_with_polygon 같은 특정 함수를 사용해야 한다면, 그에 맞게 수정 필요.
    from my_utils.photo_utils.object_utils import compute_sha256, load_json, save_object_json_with_polygon, save_cropped_face_image # configger 임포트
except ImportError as e:
    # SimpleLogger 임포트 실패 시 기본 로거 사용 (SimpleLogger가 자체 설정을 처리해야 함)
    # 이 기본 설정은 SimpleLogger가 루트 로거를 설정할 경우 충돌할 수 있습니다.
    # 가급적 SimpleLogger의 설정에 의존하는 것이 좋습니다.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(f"my_utils 임포트 오류: {e}. 기본 로거를 사용합니다.")
    # configger나 json 유틸리티는 중요하므로, 임포트 실패 시 심각한 오류로 간주할 수 있습니다.
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. 경로 및 의존성을 확인해주세요: {e}")
    sys.exit(1)


def initialize_application_logging(min_level="INFO"):
    """
    공유 SimpleLogger를 사용하여 애플리케이션 전체 로깅을 초기화합니다.
    이 함수는 `logger` (SimpleLogger에서 임포트된)가 이미 임포트되었다고 가정합니다.
    """
    app_root = Path(__file__).resolve().parent # 현재 스크립트가 있는 디렉토리 (src)
    log_directory = app_root / "logs" # 로그 디렉토리 (src/logs)
    # 로그 파일 이름: 스크립트이름_날짜.log
    log_file_name = f"{Path(__file__).stem}_{datetime.now().strftime('%Y%m%d')}.log"
    log_file_path = log_directory / log_file_name

    # 로그 디렉토리 생성 (이미 존재하면 무시)
    log_directory.mkdir(parents=True, exist_ok=True)

    logger.setup(
        logger_path=str(log_file_path),
        min_level=min_level, # 기본 로그 레벨, 설정 파일에서 재정의 가능
        include_function_name=True,
        pretty_print=True, # 선호에 따라 True 또는 False
        async_file_writing=False # 대용량 로깅 시 필요에 따라 True로 설정
    )
    logger.info(f"애플리케이션 로거 초기화 완료. 로그 파일: {log_file_path}")
    logger.show_config() # 현재 로거 설정 표시


def detect_faces_in_crop_yolo_internal(
    image_crop: cv2.Mat,
    obj_bbox_origin_xyxy: List[int], # 크롭된 객체의 원본 이미지 내 [x1, y1, x2, y2] 좌표
    yolo_face_model: YOLO, # 로드된 YOLO 얼굴 모델 객체
    confidence_threshold: float
) -> List[Dict]:
    """
    크롭된 이미지 영역에서 YOLO 모델을 사용하여 얼굴을 탐지합니다.
    탐지된 얼굴의 경계 상자는 원본 이미지 좌표 기준으로 반환됩니다.
    yolo_face_model은 이미 적절한 장치(CPU/GPU)에 로드되어 있어야 합니다.
    """
    if image_crop is None or image_crop.size == 0:
        logger.warning("얼굴 탐지를 위한 입력 이미지 크롭이 비어있습니다.")
        return []

    # 모델은 이미 올바른 장치에 있다고 가정합니다.
    # verbose=False로 설정하여 YOLO의 콘솔 출력을 줄입니다.
    results = yolo_face_model(image_crop, conf=confidence_threshold, verbose=False)
    
    faces_in_original_coords = []
    # results는 리스트 형태일 수 있으며, 각 요소는 단일 이미지(크롭)에 대한 결과입니다.
    for r in results: 
        # r.boxes는 해당 이미지(크롭)에서 탐지된 모든 경계 상자 정보를 담고 있습니다.
        for box in r.boxes: 
            score = float(box.conf[0]) # 신뢰도 점수

            # 텐서에서 경계 상자 좌표 [x1, y1, x2, y2]를 가져옵니다.
            face_crop_bbox_tensor = box.xyxy[0].tolist()
            face_crop_bbox = [int(coord) for coord in face_crop_bbox_tensor] # 정수형으로 변환

            # 크롭 이미지 내 로컬 bbox를 원본 이미지 좌표로 변환합니다.
            obj_x1_orig, obj_y1_orig, _, _ = obj_bbox_origin_xyxy
            face_orig_bbox = [
                obj_x1_orig + face_crop_bbox[0], # 원본 x1 = 객체 원본 x1 + 얼굴 크롭 x1
                obj_y1_orig + face_crop_bbox[1], # 원본 y1 = 객체 원본 y1 + 얼굴 크롭 y1
                obj_x1_orig + face_crop_bbox[2], # 원본 x2 = 객체 원본 x1 + 얼굴 크롭 x2
                obj_y1_orig + face_crop_bbox[3], # 원본 y2 = 객체 원본 y1 + 얼굴 크롭 y2
            ]
            faces_in_original_coords.append({
                "bbox_xyxy": face_orig_bbox, # 원본 이미지 기준 bbox
                "confidence": score,
                "label": "face" # 명확성을 위해 레이블 추가
            })
    return faces_in_original_coords


def run_face_detection_from_json(cfg: configger):
    logger.info("얼굴 탐지 프로세스 (JSON 객체 데이터 기반) 시작.")

    # 1. 설정 값 가져오기
    try:
        # object_detector.py의 JSON 출력물이 있는 디렉토리
        # object_detector.py는 JSON을 `project.paths.datasets.raw_jsons_dir`에 저장합니다.
        input_jsons_dir = cfg.get_path('project.paths.datasets.raw_jsons_dir', ensure_exists=True)
        logger.info(f"입력 JSON 디렉토리: {input_jsons_dir}")

        # 잘라낸 얼굴 이미지를 저장할 디렉토리 (이전에는 output_annotated_image_dir로 사용되었으나, 명확히 분리)
        # 탐지된 얼굴이 주석 처리된 이미지를 저장할 출력 디렉토리
        output_annotated_image_dir = cfg.get_path('project.paths.outputs.detected_face_images_dir', ensure_exists=True)
        logger.info(f"주석 처리된 이미지 저장 경로: {output_annotated_image_dir}")

        # 얼굴 탐지 결과가 포함된 업데이트된 JSON 파일을 저장할 출력 디렉토리
        output_updated_json_dir = cfg.get_path('project.paths.outputs.detected_face_json_dir', ensure_exists=True)
        logger.info(f"업데이트된 JSON 파일 저장 경로: {output_updated_json_dir}")

       # 잘라낸 '순수' 얼굴 이미지를 저장할 디렉토리
        output_cropped_face_dir = cfg.get_path('project.paths.outputs.detected_face_images_dir', ensure_exists=True) # YAML에 정의된 경로 사용
        logger.info(f"잘라낸 얼굴 이미지 저장 디렉토리: {output_cropped_face_dir}")

        # 설정 파일에서 YOLO 얼굴 탐지 모델 설정 가져오기
        # 예: 'models.object_yolo_tiny_model.face_detection_model' 구조 가정
        model_config_prefix = 'models.object_yolo_tiny_model.face_detection_model'

        model_name = cfg.get_value(f'{model_config_prefix}.model_name', ensure_exists=True) # 경로가 우선, 이름은 차선
        logger.info(f"model_name: {model_name}")
        
        model_weights_path_str = cfg.get_path(f'{model_config_prefix}.model_weights_path', ensure_exists=True)
        logger.info(f"model_weights_path_str: {model_weights_path_str}")

        model_weights_path = Path(model_weights_path_str) if model_weights_path_str else None
        logger.info(f"model_weights_path: {model_weights_path}")
        
        confidence_threshold = float(cfg.get_value(f'{model_config_prefix}.confidence_threshold', ensure_exists=True))
        use_cpu = cfg.get_value(f'{model_config_prefix}.use_cpu', ensure_exists=True)
        logger.info(f"model_weights_path: {model_weights_path}, 모델 이름: {model_name}, 신뢰도: {confidence_threshold}, CPU 사용: {use_cpu}")
        
        supported_json_extensions = ['.json'] # 처리할 JSON 파일 확장자
        logger.info(f"supported_json_extensions: {model_weights_path}, 모델 이름: {model_name}, 신뢰도: {confidence_threshold}, CPU 사용: {use_cpu}")

    except Exception as e:
        logger.error(f"설정 값 가져오기 오류: {e}")
        sys.exit(1)

    # 2. YOLO 얼굴 탐지 모델 로드
    try:
        if use_cpu:
            selected_device = 'cpu'
            logger.info("얼굴 탐지 모델에 CPU가 명시적으로 선택되었습니다.")
        elif torch.cuda.is_available():
            selected_device = 'cuda'
            logger.info("CUDA GPU가 감지되어 얼굴 탐지 모델에 사용됩니다.")
        else:
            selected_device = 'cpu'
            logger.info("CUDA GPU가 없거나 CPU가 선호되어, 얼굴 탐지 모델에 CPU를 사용합니다.")

        #  object_detector.py의 것을 참조하여 고처봄 - 시작

        logger.info(f"얼굴 탐지 모델 로딩: {model_weights_path}, 장치: {selected_device}")

        # YOLO() 생성자에 모델 파일 경로와 device 인자를 전달하여 로드합니다.
        # 이전 코드에서는 device 인자를 제거했지만, 명시적으로 지정하는 것이 더 확실합니다.
        if model_weights_path and model_weights_path.exists():
            # 수정: YOLO() 생성자에서 device 인자를 제거
            yolo_face_model = YOLO(str(model_weights_path))
            yolo_face_model.to(selected_device) # 모델을 선택된 장치로 이동 (redundant if device is set in YOLO(), but safe)

        elif model_name:
            logger.info(f"모델 이름으로 로딩 (자동 다운로드): {model_name}")
            # 수정: YOLO() 생성자에서 device 인자를 제거
            yolo_face_model = YOLO(model_name)
        else:
            logger.error("설정에 유효한 'model_name' 또는 'model_weights_path'가 지정되지 않았습니다.")
            sys.exit(1)

        logger.info("YOLO 모델 로딩 성공.")

        # 모델 워밍업 (선택 사항이지만, 첫 추론 속도 개선에 도움)
        try:
            logger.info("얼굴 탐지 모델 워밍업 수행 중...")
            dummy_img_np = np.zeros((64, 64, 3), dtype=np.uint8) # 작은 더미 이미지 생성
            yolo_face_model(dummy_img_np, verbose=False) # verbose=False로 출력 억제
            logger.info("얼굴 탐지 모델 워밍업 완료.")
        except Exception as e_warmup:
            logger.warning(f"얼굴 탐지 모델 워밍업 실패: {e_warmup}")

    except Exception as e:
        logger.error(f"YOLO 얼굴 탐지 모델 로드 오류: {e}")
        sys.exit(1)

    # 3. 처리할 JSON 파일 목록 가져오기
    json_files_to_process = []
    try:
        logger.info(f"입력 디렉토리 '{input_jsons_dir}'에서 JSON 파일 스캔 중...")
        for file_path in input_jsons_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_json_extensions:
                json_files_to_process.append(file_path)
        json_files_to_process.sort() # 일관된 순서로 처리
        logger.info(f"처리할 JSON 파일 {len(json_files_to_process)}개 발견.")
        if not json_files_to_process:
            logger.info("입력 디렉토리에 처리할 JSON 파일이 없습니다. 종료합니다.")
            return # 파일 없으면 종료

    except Exception as e:
        logger.error(f"JSON 파일 스캔 오류: {e}")
        sys.exit(1)

    # 4. 통계 변수 초기화
    total_json_files = len(json_files_to_process)
    processed_json_count = 0
    images_had_faces_count = 0      # 객체 내에서 하나 이상의 얼굴이 발견된 이미지 수
    total_objects_processed_count = 0 # 처리된 총 객체 수 (모든 JSON 파일 합산)
    objects_with_face_detections_count = 0 # 얼굴이 탐지된 객체 수
    total_faces_detected_count = 0  # 탐지된 총 얼굴 수

    # 5. 각 JSON 파일 처리
    for idx, json_file_path in enumerate(json_files_to_process):
        logger.info(f"JSON 파일 처리 중 [{idx+1}/{total_json_files}]: {json_file_path.name}")
        processed_json_count += 1
        
        try:
            # 현재 이미지에 대한 JSON 데이터 로드
            # `load_json`은 `my_utils.photo_utils.object_utils`에서 가져옴
            image_data_from_json = load_json(str(json_file_path))
            if not image_data_from_json or not isinstance(image_data_from_json, dict):
                logger.warning(f"JSON 파일이 비어있거나 예상된 dict 형식이 아님: {json_file_path}. 건너뜁니다.")
                continue

            # 이미지 경로 및 탐지된 객체 목록 추출 (object_detector.py의 출력 형식 기준)
            # object_detector.py는 객체를 "faces" 키 아래에 저장함 (실제로는 일반 객체)
            # object_detector.py의 save_object_json_with_polygon 함수는 객체를 "detected_obj" 키 아래에 저장합니다.
            image_path_str = image_data_from_json.get("image_path")
            # `object_detector.py`의 출력 JSON에서 객체 목록의 키는 'faces' 였습니다.
            # 이 detected_objects_list는 원본 JSON에서 가져온 객체(예: 사람) 목록입니다.
            # detected_objects_list = image_data_from_json.get("faces", []) 
            detected_objects_list = image_data_from_json.get("detected_obj", []) 

            if not image_path_str:
                logger.warning(f"JSON 파일에 'image_path'가 없음: {json_file_path}. 건너뜁니다.")
                continue
            
            image_full_path = Path(image_path_str)
            if not image_full_path.is_file():
                logger.error(f"이미지 파일 없음: {image_full_path} (JSON: {json_file_path} 참조). 건너뜁니다.")
                continue

            # 원본 이미지 로드
            original_image = cv2.imread(str(image_full_path))
            if original_image is None:
                logger.error(f"이미지 읽기 실패: {image_full_path}. 건너뜁니다.")
                continue
            
            annotated_image_copy = original_image.copy() # 얼굴 경계 상자 그리기를 위한 복사본
            current_image_has_any_face = False # 현재 이미지에서 얼굴이 하나라도 탐지되었는지 여부
            
            # updated_objects_data_list는 최종적으로 저장될 객체 목록입니다.
            # 각 객체는 원본 정보를 포함하며, 얼굴이 탐지된 경우 'detected_faces' 정보가 추가됩니다.
            updated_objects_data_list = [] # 얼굴 탐지 결과를 포함하여 업데이트될 객체 목록

            for obj_idx, obj_data in enumerate(detected_objects_list): # 객체 인덱스 추가
                # obj_data는 원본 객체 하나의 정보(딕셔너리)를 담고 있습니다.
                # 예: {"box_xyxy": [...], "class_name": "person", ...}
                total_objects_processed_count += 1
                # `object_detector.py`는 객체 bbox를 "box_xyxy"로 저장함
                obj_bbox_xyxy = obj_data.get("box_xyxy")

                if not obj_bbox_xyxy or len(obj_bbox_xyxy) != 4:
                    logger.warning(f"JSON 파일 {json_file_path.name}의 객체에 유효하지 않거나 누락된 'box_xyxy'가 있습니다. 객체 데이터: {obj_data}")
                    # 원본 객체 데이터를 그대로 updated_objects_data_list에 추가합니다.
                    updated_objects_data_list.append(obj_data) # 원본 객체 데이터 유지
                    continue
                
                try:
                    # bbox 좌표를 정수형으로 변환
                    x1, y1, x2, y2 = map(int, obj_bbox_xyxy)
                except (ValueError, TypeError) as e_map:
                    logger.warning(f"JSON 파일 {json_file_path.name}의 객체 bbox 좌표 {obj_bbox_xyxy}를 파싱할 수 없습니다: {e_map}")
                    updated_objects_data_list.append(obj_data) # 원본 객체 데이터 유지
                    continue

                # 좌표가 이미지 경계 내에 있고 유효한지 확인
                img_h, img_w = original_image.shape[:2]
                # 크롭 좌표를 이미지 경계 내로 제한
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(img_w, x2), min(img_h, y2)

                if x1_c >= x2_c or y1_c >= y2_c: # 크롭 영역이 유효하지 않은 경우
                    logger.warning(f"객체 bbox {obj_bbox_xyxy}가 이미지 {image_full_path.name}에 대해 유효하지 않은 크롭 [{x1_c},{y1_c},{x2_c},{y2_c}]을 생성합니다. 이 객체는 건너뜁니다.")
                    updated_objects_data_list.append(obj_data) # 원본 객체 데이터 유지
                    continue
                
                # 원본 이미지에서 객체 영역 크롭
                object_crop_image = original_image[y1_c:y2_c, x1_c:x2_c]

                # 이 객체 크롭 내에서 얼굴 탐지
                faces_in_object = detect_faces_in_crop_yolo_internal(
                    image_crop=object_crop_image,
                    obj_bbox_origin_xyxy=[x1_c, y1_c, x2_c, y2_c], # 크롭 영역의 원본 이미지 내 시작 좌표 (x1, y1)
                    yolo_face_model=yolo_face_model,
                    confidence_threshold=confidence_threshold
                )

                # <<<--- 요청하신 핵심 부분 시작 --- >>>
                # 만약 현재 객체(obj_data) 내에서 얼굴(faces_in_object)이 탐지되었다면,
                # 기존 객체 정보(obj_data)에 'detected_faces'라는 키로 탐지된 얼굴 정보를 추가합니다.
                # 만약 현재 객체(obj_data) 내에서 얼굴(faces_in_object)이 탐지되었다면,                
                # 기존 객체 정보(obj_data)에 'detected_face_crop'이라는 키로 탐지된 얼굴 정보를 추가합니다.
                if faces_in_object:
                    # obj_data["detected_faces"] = faces_in_object # 탐지된 얼굴 정보를 새 키로 추가
                    # 이제 obj_data는 기존 객체 정보와 그 아래 탐지된 얼굴 정보를 모두 포함합니다.
                    obj_data["detected_face_crop"] = faces_in_object # 탐지된 얼굴 정보를 'detected_face_crop' 키로 추가

                    # 이제 obj_data는 기존 객체 정보와 그 아래 탐지된 얼굴 정보를 모두 포함합니다.
                    current_image_has_any_face = True
                    objects_with_face_detections_count += 1
                    total_faces_detected_count += len(faces_in_object)

                    for face_idx, face in enumerate(faces_in_object): 
                        fb = face["bbox_xyxy"] 
                        cv2.rectangle(annotated_image_copy, (fb[0], fb[1]), (fb[2], fb[3]), (0, 255, 0), 2) 
                        if fb[2] > fb[0] and fb[3] > fb[1]: 
                            cropped_face_np = original_image[fb[1]:fb[3], fb[0]:fb[2]]
                            if cropped_face_np.size > 0:
                                try:
                                    cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face_np, cv2.COLOR_BGR2RGB))
                                    save_cropped_face_image(
                                        cropped_face_pil=cropped_face_pil,
                                        output_dir=output_cropped_face_dir,
                                        original_image_stem=image_full_path.stem,
                                        obj_idx=obj_idx, 
                                        face_idx=face_idx 
                                    )
                                except Exception as e_crop_save:
                                    logger.error(f"얼굴 크롭 또는 저장 중 오류 발생 (이미지: {image_full_path.name}, obj: {obj_idx}, face: {face_idx}): {e_crop_save}")
                            else:
                                logger.warning(f"얼굴 좌표로 자른 이미지가 비어있습니다. 저장 건너<0xEB><0><0x8F><0xBB>니다. (이미지: {image_full_path.name}, obj: {obj_idx}, face: {face_idx})")
                        else:
                            logger.warning(f"얼굴 바운딩 박스 좌표가 유효하지 않아 이미지를 자를 수 없습니다. (이미지: {image_full_path.name}, obj: {obj_idx}, face: {face_idx})")
                
                # 이렇게 업데이트된 (또는 얼굴이 없어 원본 그대로인) obj_data를
                # updated_objects_data_list에 추가합니다.
                updated_objects_data_list.append(obj_data)
                # # <<<--- 요청하신 핵심 부분 끝 --- >>>


                # if faces_in_object:
                #     obj_data["detected_faces"] = faces_in_object # 탐지된 얼굴 정보를 새 키로 추가
                #     current_image_has_any_face = True
                #     objects_with_face_detections_count += 1
                #     total_faces_detected_count += len(faces_in_object)

                #     # 탐지된 얼굴들의 경계 상자를 annotated_image_copy에 그립니다.
                #     for face_idx, face in enumerate(faces_in_object): # 얼굴 인덱스 추가
                #         fb = face["bbox_xyxy"] # 원본 이미지 기준 bbox
                #         cv2.rectangle(annotated_image_copy, (fb[0], fb[1]), (fb[2], fb[3]), (0, 255, 0), 2) # 얼굴은 녹색으로 표시
                #         # 선택 사항: 신뢰도 점수 텍스트 추가
                #         # score_text = f"{face['confidence']:.2f}"
                #         # cv2.putText(annotated_image_copy, score_text, (fb[0], fb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                #         # 잘라낸 얼굴 이미지 저장 로직 추가
                #         # 원본 이미지(original_image)에서 얼굴 영역 자르기
                #         if fb[2] > fb[0] and fb[3] > fb[1]: # x_max > x_min and y_max > y_min
                #             # fb 좌표는 이미 원본 이미지 기준이므로, original_image에서 직접 크롭
                #             cropped_face_np = original_image[fb[1]:fb[3], fb[0]:fb[2]]
                #             if cropped_face_np.size > 0:
                #                 try:
                #                     # OpenCV BGR 이미지를 PIL RGB로 변환
                #                     cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face_np, cv2.COLOR_BGR2RGB))
                                    
                #                     # 잘라낸 얼굴 이미지 저장 함수 호출
                #                     save_cropped_face_image(
                #                         cropped_face_pil=cropped_face_pil,
                #                         output_dir=output_cropped_face_dir,
                #                         original_image_stem=image_full_path.stem,
                #                         obj_idx=obj_idx, # 현재 객체의 인덱스
                #                         face_idx=face_idx # 현재 객체 내 얼굴의 인덱스
                #                     )
                #                 except Exception as e_crop_save:
                #                     logger.error(f"얼굴 크롭 또는 저장 중 오류 발생 (이미지: {image_full_path.name}, obj: {obj_idx}, face: {face_idx}): {e_crop_save}")
                #             else:
                #                 logger.warning(f"얼굴 좌표로 자른 이미지가 비어있습니다. 저장 건너<0xEB><0><0x8F><0xBB>니다. (이미지: {image_full_path.name}, obj: {obj_idx}, face: {face_idx})")
                #         else:
                #             logger.warning(f"얼굴 바운딩 박스 좌표가 유효하지 않아 이미지를 자를 수 없습니다. (이미지: {image_full_path.name}, obj: {obj_idx}, face: {face_idx})")
                
                # updated_objects_data_list.append(obj_data) # (업데이트되었을 수 있는) 객체 데이터를 목록에 추가
                
            # 현재 이미지에 대한 주 JSON 데이터 구조 업데이트
            # 원본 'faces' 키의 값을 업데이트된 객체 목록으로 교체
            # image_data_from_json["faces"] = updated_objects_data_list 
            image_data_from_json["detected_obj"] = updated_objects_data_list 

            if current_image_has_any_face:
                images_had_faces_count += 1
            
            # 주석 처리된 이미지 저장
            output_image_name = image_full_path.name # 원본 이미지 파일 이름 사용
            save_annotated_image_path = output_annotated_image_dir / output_image_name
            cv2.imwrite(str(save_annotated_image_path), annotated_image_copy)
            logger.debug(f"주석 처리된 이미지 저장 완료: {save_annotated_image_path}")

            # 업데이트된 JSON 데이터 저장
            output_json_name = json_file_path.name # 원본 JSON 파일 이름 사용
            save_updated_json_path = output_updated_json_dir / output_json_name

            current_image_hash = image_data_from_json.get("image_hash")
            # 여기서 current_objects_list는 image_data_from_json["faces"]와 동일하며,
            # 이미 위에서 updated_objects_data_list로 업데이트된 상태입니다.
            # 업데이트된 객체 목록을 가져옵니다 (키 이름 일치).
            current_objects_list = image_data_from_json.get("detected_obj", [])
#             current_objects_list = image_data_from_json.get("faces", [])

            if current_image_hash is None:
                logger.warning(f"JSON 데이터 '{json_file_path.name}'에 'image_hash'가 없습니다. 해시 정보 없이 저장됩니다.")
                current_image_hash = "N/A" # 또는 다른 기본값 설정

            save_object_json_with_polygon(
                image_path=image_full_path,       # 원본 이미지의 Path 객체
                image_hash=current_image_hash,    # 이미지 해시 문자열
#                 faces=current_objects_list,       # 업데이트된 객체(얼굴 포함 가능) 목록
                ditec_obj=current_objects_list,   # 업데이트된 객체 목록 (함수 파라미터 이름 ditec_obj 사용)
                json_path=save_updated_json_path  # 저장될 JSON 파일의 Path 객체
            )
            # save_object_json_with_polygon 함수 내부에 성공 로깅(INFO 레벨)이 이미 있으므로,
            # 여기서는 추가적인 logger.debug 메시지를 생략하거나 필요에 따라 수정합니다.
            logger.debug(f"업데이트된 JSON 데이터 저장 요청 완료: {save_updated_json_path}")

        except Exception as e_file_proc:
            logger.error(f"JSON 파일 {json_file_path.name} 처리 중 오류 발생: {e_file_proc}")
            # 선택 사항: 오류 발생 파일 수 카운트
            continue # 다음 JSON 파일로 이동

    # 6. 최종 통계 로깅
    logger.info("--- JSON 기반 얼굴 탐지 - 처리 통계 ---")
    logger.info(f"스캔한 총 JSON 파일 수: {total_json_files}")
    logger.info(f"처리된 JSON 파일 수: {processed_json_count}")
    logger.info(f"객체 내에서 하나 이상의 얼굴이 탐지된 이미지 수: {images_had_faces_count}")
    logger.info(f"모든 JSON에서 처리된 총 객체 수: {total_objects_processed_count}")
    logger.info(f"얼굴이 탐지된 객체 수: {objects_with_face_detections_count}")
    logger.info(f"모든 객체에서 탐지된 총 개별 얼굴 수: {total_faces_detected_count}")
    logger.info("------------------------------------------------------")


if __name__ == "__main__":
    # 1. 애플리케이션 로깅 초기화 (SimpleLogger 사용)
    # 이 함수는 시작 시 한 번만 호출되어야 합니다.
    initialize_application_logging()
    
    logger.info("얼굴 탐지기 (JSON 객체 데이터 기반) 애플리케이션 시작.")

    # 2. configger를 사용하여 설정 로드
    # 이 스크립트 파일을 기준으로 프로젝트 루트 결정.
    # 예: 스크립트가 'my_yolo_tiny/src/'에 있고 설정 파일이 'my_yolo_tiny/config/'에 있다고 가정
    try:
        project_root_abs = Path(__file__).resolve().parent.parent 
        config_file_relative = "config/my_yolo_tiny.yaml"
        abs_config_path = project_root_abs / config_file_relative

        if not abs_config_path.is_file():
            logger.error(f"메인 설정 파일 없음: {abs_config_path}")
            logger.error("configger를 초기화할 수 없습니다. 애플리케이션을 종료합니다.")
            sys.exit(1)

        logger.info(f"configger 초기화: root='{project_root_abs}', config 파일='{config_file_relative}'")
        cfg_object = configger(root_dir=str(project_root_abs), config_path=config_file_relative)
        
        # 예시: 중요한 설정 값이 로드되었는지 확인
        # test_project_name = cfg_object.get_value("project.name", default="프로젝트 이름 설정 안됨")
        # logger.info(f"프로젝트 '{test_project_name}'에 대한 설정 로드 성공.")

        # 3. 메인 얼굴 탐지 프로세스 실행
        run_face_detection_from_json(cfg_object)

    except FileNotFoundError as e_fnf: # 설정 파일 관련 오류
        logger.error(f"설정 파일 오류: {e_fnf}")
    except Exception as e_main: # 그 외 메인 애플리케이션 흐름에서 발생한 예외
        logger.error(f"메인 애플리케이션 흐름에서 처리되지 않은 오류 발생: {e_main}")
    finally:
        logger.info("얼굴 탐지기 (JSON 객체 데이터 기반) 애플리케이션 종료.")
        # SimpleLogger의 종료 처리 (예: 비동기 핸들러 사용 시)
        if hasattr(logger, 'shutdown') and callable(logger.shutdown):
             logger.shutdown()
        # 성공/실패 플래그에 따라 종료 코드 관리 가능
        # sys.exit(0) 
