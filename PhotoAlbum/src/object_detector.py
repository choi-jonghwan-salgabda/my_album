# my_yolo_tiny/sorc/object_detector.py
"""
이미지 내 객체 탐지를 위한 고성능 YOLO 기반 스크립트입니다.

이 스크립트는 지정된 디렉토리의 모든 이미지 파일을 스캔하여, YOLO(You Only Look Once)
객체 탐지 모델을 사용해 이미지 내의 객체(예: 사람, 휴대폰 등)를 식별합니다.
탐지된 객체의 정보(클래스, 경계 상자, 신뢰도 점수 등)는 각 이미지에 해당하는
별도의 JSON 파일로 저장됩니다.

주요 기능:
1.  객체 탐지:
    - Ultralytics YOLO 모델을 활용하여 이미지에서 다양한 객체를 탐지합니다.
    - 탐지할 클래스, 신뢰도 임계값 등은 설정 파일을 통해 유연하게 제어할 수 있습니다.

2.  메타데이터 생성:
    - 각 이미지에 대해 탐지된 객체 정보와 이미지의 고유 해시(SHA256) 값을
      포함하는 JSON 파일을 생성합니다.
    - 객체가 탐지되지 않은 이미지에 대해서도 빈 객체 목록을 가진 JSON 파일이 생성되어
      모든 이미지의 처리 상태를 일관되게 관리할 수 있습니다.

3.  고성능 병렬 처리 (`--parallel`):
    - CPU 집약적인 객체 탐지 작업을 여러 프로세스에 분산하여 동시에 처리합니다.
    - 이를 통해 수천, 수만 장의 이미지도 CPU 코어를 최대한 활용하여 빠르고
      효율적으로 처리할 수 있습니다.
    - 작업 파이프라인을 '파일 스캔 -> 병렬 탐지 -> 순차 저장'의 3단계로 분리하여
      I/O 병목 현상을 최소화하고 성능을 극대화합니다.

4.  안전한 로깅 및 오류 관리:
    - 병렬 처리 중 발생하는 모든 로그는 중앙 리스너로 전송되어 순차적으로 안전하게
      기록되므로, 로그 파일 충돌이 발생하지 않습니다.
    - 객체가 탐지되지 않은 파일은 별도의 디렉토리로 복사하고 목록을 기록하여
      후속 분석을 용이하게 합니다.

사용법 예시:
# 여러 CPU 코어를 사용하여 병렬로 객체 탐지 실행
python object_detector.py --parallel

# 단일 코어로 순차적으로 실행
python object_detector.py
"""
import os
import sys
import copy
import shutil
import functools
import threading
import multiprocessing
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from tqdm import tqdm
from ultralytics import YOLO

# shared_utils 패키지에서 configger 클래스 가져오기
# shared_utils 프로젝트의 src/utility/configger.py에 configger 클래스가 있다고 가정
# object_detector.py 파일 내 임포트 구문
# 사용자 정의 유틸리티 모듈 임포트
try:
    from my_utils.config_utils.arg_utils import get_argument, visual_length
    from my_utils.config_utils.SimpleLogger import logger
    from my_utils.config_utils.configger import configger
    from my_utils.config_utils.file_utils import safe_move, safe_copy
    from my_utils.config_utils.display_utils import calc_digit_number, with_progress_bar
    from my_utils.config_utils.JsonManager import JsonConfigHandler
    from my_utils.object_utils.photo_utils import rotate_image_if_needed, calculate_sha256
except ImportError as e:
    print(f"모듈 임포트 중 오류 발생: {e}")
    print(f"자세한 오류 정보:")
    import traceback
    traceback.print_exc() # 전체 트레이스백 출력 (개발 단계에서 유용)
    sys.exit(1)
DEFAULT_STATUS_TEMPLATE  = {
    "total_input_found":       {"value": 0,  "msg": "총 입력 파일 수 (지원 확장자 기준)"},
    "req_process_count":       {"value": 0,  "msg": "총 처리 요청 파일 수"},
    "error_image_rotation":    {"value": 0,  "msg": "이미지 회전중 오류 발생 파일 수"},
    "error_input_file_read":   {"value": 0,  "msg": "입력 파일 읽기 오류 수"},
    "undetection_object_file": {"value": 0,  "msg": "객체가 검출되지 않은 파일 수"},
    "detection_object_file":   {"value": 0,  "msg": "객체가 검출된 파일 수"},
    "error_copied_input_file": {"value": 0, "msg": "미탐지 파일 복사/기록 중 오류 수"},
    "error_output_input_file": {"value": 0, "msg": "출력물 저장처리 오류"},
    "total_object_count":      {"value": 0,  "msg": "검출된 총 객체 수"},
    "total_output_files":      {"value": 0,  "msg": "총 출력 파일수"},
    "error_input_file_process":{"value": 0,  "msg": "입력파일 처리 중 오류 수"},
    "total_duration":          {"value": "00:00:00.00", "msg": "총 처리 소요 시간"}
} 

# --- 병렬 처리 지원을 위한 전역 변수 및 함수 ---
worker_model: Optional[YOLO] = None
worker_json_handler: Optional[JsonConfigHandler] = None

def log_listener_process(queue: multiprocessing.Queue):
    """
    [병렬 처리] 멀티프로세싱 로그 리스너.

    Args:
        queue (multiprocessing.Queue): 워커 프로세스들이 로그 메시지를 보내는 공유 큐.
    """
    while True:
        try:
            record = queue.get()
            if record is None:  # 종료 신호
                break
            level, message = record
            logger.log(level=level, message=message)
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception:
            import traceback
            print(f"로그 리스너 오류:\n{traceback.format_exc()}", file=sys.stderr)

def init_worker(queue: multiprocessing.Queue, cfg: configger):
    """
    [병렬 처리] 각 워커 프로세스를 초기화합니다.
    `ProcessPoolExecutor`가 새로운 워커 프로세스를 생성할 때마다 호출됩니다.
    각 워커는 자신만의 모델과 JSON 핸들러 인스턴스를 전역 변수로 가집니다.
    """
    global worker_model, worker_json_handler
    logger.setup(mp_log_queue=queue)
    
    worker_model = model_load(cfg)
    
    json_cfg = cfg.get_config("json_keys")
    if json_cfg:
        worker_json_handler = JsonConfigHandler(json_cfg)

def model_load(cfg: configger) -> YOLO:
    """
    설정 파일에서 YOLO 모델 관련 설정을 읽어와 모델을 로드하고,
    필요한 설정 값들을 모델 객체의 속성으로 추가하여 반환합니다.

    Args:
        cfg (configger): 설정 객체.

    Returns:
        YOLO: 설정 값들이 속성으로 추가된 YOLO 모델 객체.
    """
    model_key_str = "models.object_yolo_tiny_model.object_detection_model"
    logger.debug(f"model_load-model_key_str: {model_key_str}")
    try:
        cur_cfg = cfg.get_config(model_key_str)
    except Exception as e:
        logger.error(f"설정 경로 '{model_key_str}' 하위 값 읽어오는 중 오류 발생: {e}")
        sys.exit(1)

    try:
        logger.debug(f"model_load-읽어온 모델 설정, cur_cfg:{cur_cfg}")
        model_name = cur_cfg.get("model_name")
        model_weights_path_str = cur_cfg.get("model_weights_path")
        model_weights_path = Path(model_weights_path_str) if model_weights_path_str else None
        logger.debug(f"model_load-model_weights_path: {str(model_weights_path)}")

        confidence_threshold = float(cur_cfg.get("confidence_threshold", 0.25))
        iou_threshold = float(cur_cfg.get("iou_threshold", 0.45))
        classes_to_detect = cur_cfg.get("classes_to_detect", [0, 15, 16, 65, 67, 68, 69])
        use_cpu = cur_cfg.get("use_cpu", False)
        imgsz = int(cur_cfg.get("imgsz", 640))

    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"모델 설정 로딩 중 오류 발생 ('{model_key_str}'): {e}")
        sys.exit(1)

    # 장치 결정 로직
    if use_cpu:
        selected_device = 'cpu'
        logger.debug("설정에서 CPU 사용이 명시되어 CPU를 사용합니다.")
    else:
        if torch.cuda.is_available():
            selected_device = 'cuda'
            logger.debug(f"CUDA 지원 GPU가 감지되었습니다. GPU ({torch.cuda.get_device_name(0)})를 사용합니다.")
            logger.debug(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            # 환경 변수는 없을 수도 있으므로 get을 사용하고 기본값 지정
            logger.debug(f"os.environ['CUDA_VISIBLE_DEVICES']: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
        else:
            selected_device = 'cpu'
            logger.warning("CUDA 지원 GPU를 찾을 수 없습니다.")
            logger.warning(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
            logger.warning(f"GPU 사용이 불가능하므로 CPU를 대신 사용합니다.")

    # YOLO 모델 로딩
    model = None
    model_source = None # 로딩 소스 (경로 또는 이름) 추적
    if model_weights_path and model_weights_path.exists():
        try:
            # model_weights_path (Path 객체) 사용. device 인자는 predict 시점에 주는 것이 일반적입니다.
            model_source = str(model_weights_path)
            model = YOLO(model_source)
            logger.debug(f"로컬 모델 파일 '{model_source}' 로딩 성공.")
        except Exception as e:
             logger.error(f"로컬 모델 파일 로딩 중 오류 발생: {e}")
             sys.exit(1)
    elif model_name:
        try:
            # model_name (문자열) 사용. device 인자는 predict 시점에 주는 것이 일반적입니다.
            model_source = model_name
            model = YOLO(model_source)
            logger.debug(f"허브 모델 '{model_source}' 로딩 성공.")
        except Exception as e:
            logger.error(f"허브 모델 로딩 중 오류 발생: {e}")
            sys.exit(1)
    else:
        logger.error("설정에 유효한 'model_name' 또는 'model_weights_path'가 지정되지 않았습니다.")
        sys.exit(1)

    # 로드된 모델 객체에 설정 값들을 속성으로 추가합니다.
    if model: # 모델이 성공적으로 로드되었는지 확인
        # 모델 객체에 필요한 속성들을 동적으로 추가
        setattr(model, 'confidence_threshold', confidence_threshold)
        setattr(model, 'iou_threshold', iou_threshold)
        setattr(model, 'classes_to_detect', classes_to_detect)
        setattr(model, 'selected_device', selected_device)
        setattr(model, 'imgsz', imgsz)
        setattr(model, 'model_source', model_source) # 로딩 소스도 추가해두면 유용할 수 있습니다.

        logger.debug("YOLO 모델 로딩 완료 및 설정 값 속성 추가 완료.")
    else:
        # 이 경우는 위 sys.exit(1) 때문에 실제로 도달하지 않겠지만, 안전을 위해 추가
        logger.error("YOLO 모델 로딩 실패.")
        sys.exit(1)

    # 수정된 모델 객체만 반환합니다.
    return model

def detect_object_worker(input_path: Path) -> Dict[str, Any]: # 순차 처리 시에도 재사용
    """
    [병렬 워커] 단일 이미지 파일에 대한 객체 탐지를 수행하고 결과를 반환합니다.

    이 함수는 병렬 처리를 위해 각 워커 프로세스에서 독립적으로 실행되는 핵심 작업 단위입니다.
    파일 경로를 입력받아 이미지를 읽고, 전처리(회전 보정) 후 YOLO 모델로 객체를 탐지합니다.
    파일 I/O를 직접 수행하는 대신, 탐지 결과와 관련 메타데이터(해시, 해상도 등)와
    처리 통계를 담은 딕셔너리를 반환하여 메인 프로세스에서 후처리하도록 설계되었습니다.

    Args:
        input_path (Path): 처리할 이미지 파일의 경로.

    Returns:
        Dict[str, Any]: 처리 결과를 담은 딕셔너리. 포함하는 키는 다음과 같습니다:
            - "status" (Dict): 이 파일 처리에 대한 통계 정보.
            - "error" (str | None): 오류 발생 시 오류 유형 문자열, 성공 시 None.
            - "input_path" (Path): 원본 이미지 경로.
            - "is_detected" (bool): 객체 탐지 여부.
            - "json_payload" (Dict): JSON 파일로 저장될 데이터.
    """
    status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)

    if not worker_model or not worker_json_handler:
        logger.error("워커가 제대로 초기화되지 않았습니다. worker_model 또는 worker_json_handler가 None입니다.")
        status["error_input_file_process"]["value"] += 1
        return {"status": status, "error": "worker_not_initialized", "input_path": input_path, "is_detected": False, "json_payload": None}

    if not rotate_image_if_needed(str(input_path)):
        logger.warning(f"이미지 파일 회전 중 오류 발생: '{input_path}'. 건너뜁니다.")
        status["error_image_rotation"]["value"] += 1
        return {"status": status, "error": "rotation_error", "input_path": input_path, "is_detected": False, "json_payload": None}

    try:
        # Use Pillow to open the image, which is more robust to profile errors like iCCP.
        with Image.open(input_path) as img:
            # Ensure image is in a format cv2 can use (e.g., RGB)
            img_rgb = img.convert('RGB')
            # Convert PIL Image to NumPy array for OpenCV (in BGR format)
            frame = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, FileNotFoundError) as e:
        logger.warning(f"이미지 파일 '{input_path}'을 읽을 수 없습니다. 건너뜁니다. 오류: {e}")
        status["error_input_file_read"]["value"] += 1
        return {"status": status, "error": "read_error", "input_path": input_path, "is_detected": False, "json_payload": None}
    except Exception as e:
        logger.warning(f"이미지 파일 '{input_path}' 처리 중 예기치 않은 오류 발생. 건너뜁니다. 오류: {e}")
        status["error_input_file_read"]["value"] += 1
        return {"status": status, "error": "read_error", "input_path": input_path, "is_detected": False, "json_payload": None}
    height, width, channels = frame.shape
    results = worker_model(
        frame,
        conf=worker_model.confidence_threshold,
        iou=worker_model.iou_threshold,
        classes=worker_model.classes_to_detect,
        imgsz=worker_model.imgsz,
        device=worker_model.selected_device,
        verbose=False
    )

    processed_ditec_obj_data = []
    index_per_class = {}
    amount_object = len(results[0].boxes)
    is_detected = False

    if results[0].boxes is not None and amount_object > 0:
        is_detected = True
        status["detection_object_file"]["value"] += 1
        status["total_object_count"]["value"] += amount_object

        for box in results[0].boxes:
            xyxy_coords = box.xyxy.tolist()[0]
            xywh_coords = box.xywh.tolist()[0]
            confidence = float(box.conf)
            class_id = int(box.cls)
            class_name = results[0].names[class_id]

            if class_name not in index_per_class:
                index_per_class[class_name] = {"count": 0}
            index_per_class[class_name]["count"] += 1

            ditec_obj_info = {
                worker_json_handler.object_box_xyxy_key: xyxy_coords,
                worker_json_handler.object_box_xywh_key: xywh_coords,
                worker_json_handler.object_confidence_key: confidence,
                worker_json_handler.object_class_id_key: class_id,
                worker_json_handler.object_class_name_key: class_name,
                worker_json_handler.object_label_key: worker_json_handler.object_label_mask,
                worker_json_handler.object_index_key: index_per_class[class_name]["count"] # 현재 객체의 순번
            }
            processed_ditec_obj_data.append(ditec_obj_info)
    else:
        status["undetection_object_file"]["value"] += 1
        logger.debug(f"'{input_path.name}'에서 객체를 검출하지 못했습니다.")

    # calculate_sha256 함수는 파일 핸들을 인자로 받습니다.
    # 'numpy.ndarray' 객체인 'frame'을 직접 전달하면 'seek' 속성 오류가 발생합니다.
    # 따라서, (회전 처리가 적용되었을 수 있는) 파일을 직접 열어 파일 핸들을 전달합니다.
    try:
        with open(input_path, 'rb') as f:
            image_hash_value = calculate_sha256(f)
    except Exception as e:
        logger.error(f"'{input_path.name}'의 해시 계산 중 오류 발생: {e}")
        # 해시 계산에 실패하더라도 나머지 처리는 계속될 수 있도록 None으로 설정합니다.
        image_hash_value = None

    json_payload = { # JSON 페이로드도 객체 검출 여부와 상관없이 구성
        "image_path": str(input_path),
        "image_hash": image_hash_value,
        "width": width,
        "height": height,
        "channels": channels,
        "detected_objects": processed_ditec_obj_data, # 객체가 없으면 빈 리스트
    }
    return {
        "status": status,
        "error": None,
        "input_path": input_path,
        "is_detected": is_detected,
        "json_payload": json_payload
    }


## --- 메인 실행 함수 ---
def run_main(
    cfg: configger, 
    input_dir: Optional[Path] = None,
    parallel: bool = False,
    max_workers: Optional[int] = None
):
    main_start_time = time.perf_counter()
    status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)

    # 1. 설정에서 경로 및 환경 변수 로드
    try:
        # --- 데이터셋 관련 경로 ---
        dataset_cfg = cfg.get_config("project.paths.datasets")
        if dataset_cfg is None:
            logger.error("'project.paths.datasets' 설정 그룹을 찾을 수 없습니다.")
            sys.exit(1)

        # 입력 디렉토리 결정 (인자 > 설정 파일)
        if input_dir is None:
            logger.debug("입력 디렉토리가 지정되지 않았습니다. 설정 파일에서 경로를 가져옵니다.")
            input_dir_str = dataset_cfg.get('raw_image_dir')
            if not input_dir_str:
                logger.error("설정 파일에 'project.paths.datasets.raw_image_dir' 키가 없습니다.")
                sys.exit(1)
            input_dir = Path(input_dir_str).expanduser()
            logger.debug(f"설정 파일의 경로를 입력 디렉토리로 사용합니다: {input_dir}")

        if not input_dir.exists():
            logger.error(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
            sys.exit(1)

        output_dir_str = dataset_cfg.get('raw_jsons_dir')
        if not output_dir_str:
            logger.error("'project.paths.datasets.raw_jsons_dir' 설정이 없습니다.")
            sys.exit(1)
        output_dir = Path(output_dir_str).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True) 
        logger.debug(f"output_dir: {output_dir}")

        # --- 객체 탐지 관련 경로 ---
        objects_cfg = cfg.get_config("project.paths.objects")
        if objects_cfg is None:
            logger.error("'project.paths.objects' 설정 그룹을 찾을 수 없습니다.")
            sys.exit(1)

        # object_detector.py는 미탐지 '이미지'를 처리하므로, 'undetect_dir'를 사용.
        # 설정 파일의 'undetect_dir'를 이미지 저장용으로 사용하도록 통일.
        undetect_objects_dir_str = objects_cfg.get('undetect_dir')
        if not undetect_objects_dir_str:
            logger.error("'project.paths.objects.undetect_dir' 설정이 없습니다.")
            sys.exit(1)
        undetect_objects_dir = Path(undetect_objects_dir_str).expanduser()
        
        # undetect_dir 처리: 존재하면 삭제 후 재생성
        if undetect_objects_dir.exists():
            logger.debug(f"기존 미검출 이미지 저장 디렉토리 '{undetect_objects_dir}'이(가) 존재하여 삭제 후 재생성합니다.")
            try:
                shutil.rmtree(undetect_objects_dir) # 디렉토리와 내용물 모두 삭제
            except OSError as e:
                logger.error(f"기존 미검출 이미지 저장 디렉토리 '{undetect_objects_dir}' 삭제 중 오류 발생: {e}")
        
        undetect_objects_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"미검출 관련 디렉토리 준비 완료: {undetect_objects_dir}")
        
        undetect_list_path_str = objects_cfg.get('undetect_list_path')
        if not undetect_list_path_str:
            logger.error("'project.paths.objects.undetect_list_path' 설정이 없습니다.")
            sys.exit(1)
        undetect_list_path = Path(undetect_list_path_str).expanduser()
        # 미검출 목록 파일 초기화 (존재하면 삭제)
        if undetect_list_path.exists():
            try:
                undetect_list_path.unlink()
                logger.debug(f"기존 미검출 목록 파일 '{undetect_list_path}'을(를) 삭제했습니다.")
            except OSError as e:
                logger.error(f"기존 미검출 목록 파일 '{undetect_list_path}' 삭제 중 오류 발생: {e}")
        logger.debug(f"undetect_list_path: {undetect_list_path}")

        json_cfg = cfg.get_config("json_keys")
        if json_cfg is None:
            logger.error("설정 파일에 'json_keys' 정보가 누락되었습니다.")
            sys.exit(1)

        # JsonConfigHandler 인스턴스 생성
        try:
            logger.debug(f"JsonConfigHandler 초기화 시작")
            json_handler = JsonConfigHandler(json_cfg)
            logger.debug(f"JsonConfigHandler가 성공적으로 초기화 끝.")
        except Exception as e_json_handler:
            logger.error(f"JsonConfigHandler 초기화 중 오류 발생: {e_json_handler}", exc_info=True)
            sys.exit(1)

    except Exception as e:
        logger.error(f"설정 값 가져오기 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)
    
    # 2. 처리할 이미지 파일 목록 스캔
    try:
        processing_cfg = cfg.get_config("processing") or {}
        supported_extensions = processing_cfg.get("supported_image_extensions", [])
        if not supported_extensions:
            logger.warning("지원 확장자 목록이 비어있어 기본 이미지 확장자를 사용합니다.")
            supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    except Exception as e:
        logger.error(f"설정에서 확장자 정보 로드 중 오류: {e}")
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

    allowed_ext_lower = {ext.lower() for ext in supported_extensions}
    logger.console(f"'{input_dir}'에서 처리할 이미지 파일을 스캔하고 필터링합니다...")
    image_files = [p for p in input_dir.glob('**/*') if p.is_file() and p.suffix.lower() in allowed_ext_lower]

    status["total_input_found"]["value"] = len(image_files)
    logger.debug(f'지원되는 이미지 개수 "total_input_found": {status["total_input_found"]["value"]}')

    if not image_files:
        logger.warning(f"'{input_dir}' 디렉토리에서 처리할 이미지 파일을 찾을 수 없습니다.")
        logger.console("✅ 최종 통계:")
        logger.console(f'   - 탐색된 IMAGE 파일 총 개수: {status["total_input_found"]["value"]}')
        logger.console(f"{Path(__file__).name} 정상 종료 (처리할 파일 없음).")
        sys.exit(0)

    try:
        digit_width = calc_digit_number(status["total_input_found"]["value"])
    except Exception as e:
        digit_width = 0
        logger.error(f"calc_digit_number 값 가져오기 중 오류 발생: {e}")

    logger.console(f'✅ 처리할 Image 파일 {status["total_input_found"]["value"]}개 발견. 숫자 폭(digit_width) : {digit_width}')

    # 3. 병렬/순차 처리 실행 (tqdm 호환 로깅 활성화)
    results = []
    manager = None
    log_queue = None
    listener_thread = None
    initializer = None

    if hasattr(logger, 'set_tqdm_aware'):
        logger.set_tqdm_aware(True)

    try:
        if parallel:
            manager = multiprocessing.Manager()
            log_queue = manager.Queue()
            listener_thread = threading.Thread(target=log_listener_process, args=(log_queue,))
            listener_thread.start()
            # 워커 초기화 함수 설정
            initializer = functools.partial(init_worker, log_queue, cfg)

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=initializer) as executor:
                future_to_path = {executor.submit(detect_object_worker, path): path for path in image_files}
                for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(image_files), desc="객체 탐지 중 (병렬)", file=sys.stdout):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        path = future_to_path[future]
                        logger.error(f"'{path.name}' 처리 중 워커에서 예외 발생: {exc}")
                        status["error_input_file_process"]["value"] += 1
        else:
            # 순차 처리
            global worker_model, worker_json_handler
            worker_model = model_load(cfg)
            worker_json_handler = json_handler
            for img_path in tqdm(image_files, desc="객체 탐지 중 (순차)", unit="파일", file=sys.stdout):
                try:
                    results.append(detect_object_worker(img_path))
                except Exception as e:
                    logger.error(f"이미지 파일 '{img_path}' 처리 중 오류 발생: {e}")
                    status["error_input_file_process"]["value"] += 1
                    continue

        # 4. 결과 후처리 (파일 쓰기, 통계 집계)
        logger.debug("탐지 결과 저장 및 후처리 시작...")
        for res in tqdm(results, desc="결과 저장 및 처리 중", unit="파일", file=sys.stdout):
            status['req_process_count']['value'] += 1
            # 워커에서 반환된 개별 통계 집계
            for key, data in res["status"].items():
                if key in status:
                    status[key]["value"] += data["value"]

            if res["error"]:
                # 워커에서 이미 로그를 남겼으므로 여기서는 집계만 함
                continue

            # JSON 파일 저장
            payload = res["json_payload"]
            output_filename = res["input_path"].with_suffix('.json').name
            json_output_path = output_dir / output_filename
            try:
                json_handler.write_json(
                    image_path=res["input_path"],
                    image_hash=payload["image_hash"],
                    width=payload["width"],
                    height=payload["height"],
                    channels=payload["channels"],
                    detected_objects=payload["detected_objects"],
                    json_path=json_output_path
                )
                logger.debug(f"JSON 데이터가 '{json_output_path}'에 성공적으로 저장되었습니다.")
                status["total_output_files"]["value"] += 1
            except Exception as e:
                status["error_output_input_file"]["value"] += 1
                logger.error(f"'{json_output_path}' JSON 파일 저장 중 오류: {e}")

            # 미탐지 파일 처리
            if not res["is_detected"]:
                try:
                    with open(undetect_list_path, 'a', encoding='utf-8') as f:
                        f.write(f"{str(res['input_path'])}\n")
                    logger.debug(f"처리 실패 이미지 '{res['input_path']}'의 경로를 '{undetect_list_path}'에 기록했습니다.")
                    dest_path = undetect_objects_dir / res['input_path'].name # 미검출 이미지 복사
                    shutil.copy2(str(res['input_path']), str(dest_path)) # 미검출 이미지 복사
                    logger.debug(f"미검출 이미지 '{res['input_path'].name}'을(를) '{dest_path}'로 복사했습니다.")
                except Exception as e:
                    logger.error(f"미탐지 파일 '{res['input_path'].name}' 복사/기록 중 오류: {e}")
                    status["error_copied_input_file"]["value"] += 1
    finally:
        # tqdm 사용이 끝났으므로 로거를 원래 상태로 복원합니다.
        if hasattr(logger, 'set_tqdm_aware'):
            logger.set_tqdm_aware(False)

        if parallel and listener_thread:
            logger.debug("로그 리스너 스레드 종료 신호 전송...")
            if log_queue:
                log_queue.put(None)
            listener_thread.join(timeout=5)
            if listener_thread.is_alive():
                logger.warning("로그 리스너 스레드가 시간 내에 종료되지 않았습니다.")
            else:
                logger.debug("로그 리스너 스레드가 성공적으로 종료되었습니다.")

    # 5. 최종 통계 출력
    total_duration_seconds = time.perf_counter() - main_start_time
    hours, rem = divmod(total_duration_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    status["total_duration"]["value"] = f"{int(hours):02}:{int(minutes):02}:{seconds:05.2f}"

    max_msg_byte_length = 0
    # Calculate max message length
    for key in status.keys():
        if key in DEFAULT_STATUS_TEMPLATE:
            msg_string = DEFAULT_STATUS_TEMPLATE[key]["msg"]
            current_byte_length = visual_length(msg_string, 2)
            if current_byte_length > max_msg_byte_length:
                max_msg_byte_length = current_byte_length

    # Calculate max value length for alignment
    max_value_len = 0
    for data in status.values():
        value_len = len(str(data['value']))
        if value_len > max_value_len:
            max_value_len = value_len

    fill_char = '.'
    logger.console("--- 이미지 파일 처리 통계 ---")
    for key, data in status.items():
        msg = DEFAULT_STATUS_TEMPLATE.get(key, {}).get("msg", key)
        value = data["value"]
        # Use max_value_len for alignment to handle both strings and numbers
        logger.console(f'{msg:{fill_char}<{max_msg_byte_length}}: {str(value):>{max_value_len}}')
    logger.console("------------------------")


# 스크립트가 직접 실행될 때 함수 호출
if __name__ == "__main__":
    script_name = Path(__file__).stem

    # 1. 초기 콘솔 로깅 설정: 파일 경로는 아직 미정
    #    configger가 로드되기 전까지의 모든 로그는 콘솔에만 출력됩니다.
    logger.setup(
        logger_path=None,
        console_min_level="INFO", # 시작 메시지를 볼 수 있도록 INFO 레벨로 설정
        include_function_name=True,
        pretty_print=True
    )
    logger.console(f"애플리케이션 ({script_name}) 시작. 초기 콘솔 로깅 활성화.")

    supported_args_for_script = [
        # 이 스크립트가 지원하는 인자 목록을 명시적으로 정의합니다.
        'source_dir',
        'log_mode',
        'parallel',
        'max_workers'
    ]
    parsed_args = get_argument(
        # object_detector.py는 소스 디렉토리만 필수로 요구합니다.
        # -dst 인자는 사용되지 않으므로 필수 목록에서 제거합니다.
        required_args=['-src'],
        supported_args=supported_args_for_script
    )

    try:
        # root_dir은 configger 초기화에 필요하므로 먼저 설정합니다.
        if parsed_args.root_dir is None:
            parsed_args.root_dir = Path.cwd()
        else:
            parsed_args.root_dir = Path(parsed_args.root_dir)

        logger.debug(f"Configger 초기화 시도: root_dir='{parsed_args.root_dir}', config_path='{parsed_args.config_path}'")
        cfg_object = configger(root_dir=parsed_args.root_dir, config_path=parsed_args.config_path)
        logger.debug(f"Configger 초기화 완료.")

        # 2. 최종 로거 설정: 명령줄 인자 또는 설정 파일의 값을 종합하여 결정합니다.
        try:
            # --- 설정 값 로드 (YAML) ---
            log_cfg = cfg_object.get_config("project.logging") or {}
            cfg_log_dir = log_cfg.get("log_dir")
            cfg_log_console_level = log_cfg.get("console_min_level", "INFO")
            cfg_log_file_level = log_cfg.get("file_min_level", "DEBUG")
            cfg_log_mode = log_cfg.get("log_mode", "sync")

            # --- 최종 설정 값 결정 (우선순위: 명령줄 > YAML > 기본값) ---
            final_log_dir_str = parsed_args.log_dir or cfg_log_dir
            final_log_file_level = (parsed_args.log_level or cfg_log_file_level).upper()
            final_log_console_level = cfg_log_console_level.upper() # 콘솔 레벨은 YAML에서만 제어
            final_log_mode = parsed_args.log_mode or cfg_log_mode

            # --- 파일 로거 설정 ---
            if final_log_dir_str:
                log_dir_path = Path(final_log_dir_str).expanduser().resolve()
                log_dir_path.mkdir(parents=True, exist_ok=True)

                date_str = datetime.now().strftime("%y%m%d%hh%mm")
                log_file_name = f"{script_name}_{date_str}.log"
                full_log_path = log_dir_path / log_file_name

                # 최종 로거 설정 적용
                logger.setup(
                    logger_path=str(full_log_path),
                    file_min_level=final_log_file_level,
                    console_min_level=final_log_console_level,
                    async_file_writing=(final_log_mode == 'async'),
                    include_function_name=True,
                    pretty_print=True
                )
                logger.debug(f"파일 로깅을 시작합니다: {full_log_path}")
                logger.debug(f"최종 로그 설정: 파일 레벨='{final_log_file_level}', 콘솔 레벨='{final_log_console_level}', 모드='{final_log_mode}'")
            else:
                logger.warning("로그 디렉토리가 지정되지 않아 파일 로깅이 비활성화됩니다.")

        except Exception as e:
            logger.error(f"파일 로거 설정 중 오류 발생: {e}. 파일 로깅이 비활성화될 수 있습니다.")

        # 3. 입력 디렉토리 결정 (명령줄 인자 우선)
        input_dir_to_use = None
        if parsed_args.source_dir:
            input_dir_to_use = Path(parsed_args.source_dir).expanduser().resolve()
            logger.debug(f"명령줄 인자(-src)를 입력 디렉토리로 사용합니다: {input_dir_to_use}")
        else:
            # 명령줄 인자가 없으면 run_main 함수가 설정 파일에서 경로를 가져옵니다.
            logger.debug("명령줄 인자(-src)가 없습니다. 설정 파일의 입력 디렉토리를 사용합니다.")

        # 4. 메인 로직 실행
        run_main(
            cfg=cfg_object,
            input_dir=input_dir_to_use,
            parallel=parsed_args.parallel,
            max_workers=parsed_args.max_workers
        )
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}", exc_info=True)
    finally:
        logger.console(f"{script_name} 애플리케이션 종료")
        if hasattr(logger, "shutdown"):
            logger.shutdown()
        sys.exit(0)
