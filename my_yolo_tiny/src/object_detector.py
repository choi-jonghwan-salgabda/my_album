# my_yolo_tiny/sorc/object_detector.py
import inspect 
import types # SimpleNamespace 사용을 위해 types 모듈 임포트
import sys     # 표준 출력 스트림 사용을 위해 필요
from pathlib import Path # 경로 관리를 위해 필요
import torch # torch 모듈을 임포트합니다.
from datetime import datetime
from typing import List, Dict, Any, Tuple
import copy # deepcopy 사용을 위해 copy 모듈 임포트
import shutil # 파일 복사를 위해 shutil 모듈 임포트

import cv2 # OpenCV 라이브러리
from ultralytics import YOLO # ultralytics의 YOLO 모델

# shared_utils 패키지에서 configger 클래스 가져오기
# shared_utils 프로젝트의 src/utility/configger.py에 configger 클래스가 있다고 가정
# object_detector.py 파일 내 임포트 구문
# 사용자 정의 유틸리티 모듈 임포트
try:
    from my_utils.config_utils.SimpleLogger import logger, calc_digit_number, get_argument, visual_length
    from my_utils.config_utils.configger import configger
    from my_utils.object_utils.photo_utils import rotate_image_if_needed, compute_sha256, JsonConfigHandler, _get_string_key_from_config # JsonConfigHandler 임포트
except ImportError as e:
    # 실제 발생한 예외 e를 출력하여 원인 파악
    print(f"모듈 임포트 중 오류 발생: {e}")
    print(f"자세한 오류 정보:")
    import traceback
    traceback.print_exc() # 전체 트레이스백 출력 (개발 단계에서 유용)
    sys.exit(1)
DEFAULT_STATUS_TEMPLATE  = {
    "total_input_found":         {"value": 0,  "msg": "총 입력 파일 수 (지원 확장자 기준)"},
    "req_process_count":         {"value": 0,  "msg": "총 처리 요청 파일 수"},
    "error_extension":   {"value": 0,  "msg": "지원되지 않는 확장자로 건너뛴 파일 수"},
    "error_input_file_process":        {"value": 0,  "msg": "입력파일 읽기 오류 수"},
    "error_image_rotation":          {"value": 0,  "msg": "이미지 회전중 오류 발생 파일 수"},
    "error_input_file_read":        {"value": 0,  "msg": "입력 파일 읽기 오류 수"},
    "undetection_object_file":   {"value": 0,  "msg": "객체가 검출되지 않은 파일 수"},
    "detection_object_file":     {"value": 0,  "msg": "객체가 검출된 파일 수"},
    "error_copied_input_file": {"value": 0, "msg": "객체가없는 파일 복사 중 오류 발생 수"},
    "error_output_input_file": {"value": 0, "msg": "출력물 저장처리 오류"},
    "total_object_count":      {"value": 0,  "msg": "검출된 총 객체 수"},
    "total_output_files":        {"value": 0,  "msg": "총 출력 파일수"}
} 

def model_load(cfg: configger) -> YOLO:
    """
    설정 파일에서 YOLO 모델 관련 설정을 읽어와 모델을 로드하고,
    필요한 설정 값들을 모델 객체의 속성으로 추가하여 반환합니다.

    Args:
        cfg (configger): 설정 객체.
        base_key (str): 설정에서 모델 설정이 있는 경로 문자열 (예: "models.object_yolo_tiny_model.general_detection_model").

    Returns:
        YOLO: 설정 값들이 속성으로 추가된 YOLO 모델 객체.
    models_keys_to_get   = [
        "model_name",
        "model_weights_path",
        "confidence_threshold",
        "iou_threshold",
        "classes_to_detect",
        "use_cpu",
        "imgsz"
    ]
    """
    model_key_str = "models.object_yolo_tiny_model.object_detection_model" # 일반 객체 검출 모델

    logger.debug(f"model_load-model_key_str: {model_key_str}")
    try:
        # # get_config_value 함수를 호출하여 설정 값 딕셔너리를 가져옵니다.
        # # 함수 시그니처에 맞게 base_key 변수를 사용합니다.
        # model_config = get_config_value(cfg, model_key_str, models_keys_to_get) # config_keys_to_get은 문자열 리스트여야 함
        cur_cfg = cfg.get_config(model_key_str) # 이 부분은 삭제될 것 같습니다.
    except Exception as e:
        logger.error(f"설정 경로 '{model_key_str}' 하위 값 읽어오는 중 오류 발생: {e}") # model_key_str 사용
        sys.exit(1)


        # # 반환된 딕셔너리에서 각 변수에 올바른 값을 할당
    try:
        logger.debug(f"model_load-읽어온 모델 설정, cur_cfg:{cur_cfg}")

        model_name = cur_cfg.get("model_name", None)
        logger.debug(f"model_load- model_name: {model_name}")
    except Exception as e:
        logger.error(f"설정 경로 'model_name' 하위 값 읽어오는 중 오류 발생: {e}") # model_key_str 사용
        sys.exit(1)

    try:
        model_weights_path = Path(cur_cfg.get("model_weights_path", None)) # model_config 사용
        # Common image saving logic
        # 디렉토리가 있는지 확인하고 없다면 생성
        # model_weights_path.mkdir(parents=True, exist_ok=True) 
        logger.debug(f"model_load-model_weights_path: {str(model_weights_path)}")
    except Exception as e:
        logger.error(f"설정 경로 'model_weights_path' 하위 값 읽어오는 중 오류 발생: {e}") # model_key_str 사용
        sys.exit(1)

    try:
        confidence_threshold = float(cur_cfg.get("confidence_threshold", 0.25)) # 기본값 추가
        logger.debug(f"model_load-confidence_threshold: {confidence_threshold}")
    except Exception as e:
        logger.error(f"설정 경로 'confidence_threshold' 하위 값 읽어오는 중 오류 발생: {e}") # model_key_str 사용
        sys.exit(1)

    try:
        iou_threshold = float(cur_cfg.get("iou_threshold", 0.45)) # 기본값 추가
        logger.debug(f"model_load-iou_threshold: {iou_threshold}")
    except Exception as e:
        logger.error(f"설정 경로 'iou_threshold' 하위 값 읽어오는 중 오류 발생: {e}") # model_key_str 사용
        sys.exit(1)

    try:
        classes_to_detect = cur_cfg.get("classes_to_detect", [0, 15, 16, 65, 67, 68, 69])
        logger.debug(f"model_load-classes_to_detect: {classes_to_detect}")
    except Exception as e:
        logger.error(f"설정 경로 'classes_to_detect' 하위 값 읽어오는 중 오류 발생: {e}") # model_key_str 사용
        sys.exit(1)

    try:
        use_cpu = cur_cfg.get("use_cpu", False) # 기본값 추가
        logger.debug(f"model_load-use_cpu: {use_cpu}")
    except Exception as e:
        logger.error(f"설정 경로 'use_cpu' 하위 값 읽어오는 중 오류 발생: {e}") # model_key_str 사용
        sys.exit(1)

    try:
        imgsz = int(cur_cfg.get("imgsz", 640)) # 기본값 추가
        logger.debug(f"model_load-imgsz: {imgsz}")
    except Exception as e:
        logger.error(f"설정 경로 'imgsz' 하위 값 읽어오는 중 오류 발생: {e}") # model_key_str 사용
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

        logger.info("YOLO 모델 로딩 완료 및 설정 값 속성 추가 완료.")
    else:
        # 이 경우는 위 sys.exit(1) 때문에 실제로 도달하지 않겠지만, 안전을 위해 추가
        logger.error("YOLO 모델 로딩 실패.")
        sys.exit(1)

    # 수정된 모델 객체만 반환합니다.
    return model

def detect_object(
    base_config: Dict[str, Any],
    input_path: Path, 
    output_dir: Path, 
    undetect_objects_dir: Path, 
    json_handler: JsonConfigHandler, # JsonConfigHandler 인스턴스 추가
    undetect_list_path: Path, 
    model:YOLO
    )->dict:
    # 이미지 파일 읽기 (OpenCV 사용)
    # 통계 정보를 담을 딕셔너리 초기화
    status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)

    # 1. 이미지 파일 읽기 전에 이미지 회전
    if rotate_image_if_needed(str(input_path)) == False: # input_path는 Path 객체이므로 문자열로 변환하여 전달
        logger.warning(f"이미지 파일 회전('rotate_image_if_needed')중 오률발견. 건너뜁니다.")
        status["error_image_rotation"]["value"] += 1
        return status # 파일 읽기 실패 시 다음 파일로 이동

    # 2... 이미지 파일 읽기 부분 ...
    frame = cv2.imread(str(input_path))

    if frame is None:
        logger.warning(f"이미지 파일 '{input_path}'을 읽을 수 없습니다. 건너뜁니다.")
        status["error_input_file_read"]["value"] += 1
        return status # 파일 읽기 실패 시 다음 파일로 이동
    else:
        logger.info(f"이미지 파일: {input_path}를 읽었습니다..")
        height, width, channels = frame.shape
        
    # 3. 이미지에서 객체검출 모델 적용
    results = model(
        frame,                       # ① 입력 데이터
        conf=model.confidence_threshold,   # ② 신뢰도 임계값 (Confidence Threshold)
        iou=model.iou_threshold,           # ③ IoU 임계값 (IoU Threshold) for NMS
        classes=model.classes_to_detect,   # ④ 검출할 클래스 목록 (Classes to Detect)
        imgsz=model.imgsz,                 # ⑤ 입력 이미지 크기 (Image Size)
        device=model.selected_device,      # ⑥ 실행 장치 (Device)
        verbose=False                # ⑦ 상세 출력 여부 (Verbosity)
    )

    # 4. 검출된 객체 수 확인
    # 4.1 검출된 정보를 확인할 설정값을 각저옴. 
    try:
        object_name_key       = base_config.get("object_info_key", {}).get("key", "detected_obj") # base_config는 json_keys 전체여야 함
        object_label_mask     = base_config.get("object_info_key", {}).get("label_mask", "detected_obj") # base_config는 json_keys 전체여야 함
        object_box_xyxy_key   = base_config.get("object_info_key", {}).get("object_box_xyxy_key", "box_xyxy")
        object_box_xywh_key   = base_config.get("object_info_key", {}).get("object_box_xywh_key", "box_xywh")
        object_confidence_key = base_config.get("object_info_key", {}).get("object_confidence_key", "confidence")
        object_class_id_key   = base_config.get("object_info_key", {}).get("object_class_id_key", "class_id")
        object_class_name_key = base_config.get("object_info_key", {}).get("object_class_name_key", "class_name")
        object_label_key      = base_config.get("object_info_key", {}).get("object_label_key", "label")
        object_index_key      = base_config.get("object_info_key", {}).get("object_index_key", "index")
    except Exception as e:
        logger.error(f"detect_object: base_config에서 설정 키 로딩 실패 ('{input_path.name}'): {e}")
        return status

    # 4.2 검출된 정보를 확인. 
    processed_ditec_obj_data = []
    index_per_class = {}
    amount_object = len(results[0].boxes)
    if results[0].boxes is not None and amount_object > 0:
        # if results[0].boxes is not None: # boxes가 있는지 확인
        # If files_with_objects_detected > 0, results[0].boxes is guaranteed to be non-empty.
        # So, `if results[0].boxes is not None:` is redundant here.
        for box in results[0].boxes:
            # box 객체에서 필요한 정보 추출
            xyxy_coords = box.xyxy.tolist()[0]
            xywh_coords = box.xywh.tolist()[0]
            confidence = float(box.conf)
            class_id = int(box.cls)
            class_name = results[0].names[class_id]

            # 클래스별 카운트 업데이트 (객체 순번 부여)
            if class_name not in index_per_class:
                index_per_class[class_name] = {"count": 0}

            index_per_class[class_name]["count"] += 1

            # 필요한 정보들을 딕셔너리로 구성
            ditec_obj_info = {
                object_box_xyxy_key: xyxy_coords,
                object_box_xywh_key: xywh_coords,
                object_confidence_key: confidence,
                object_class_id_key: class_id,
                object_class_name_key: class_name,
                object_label_key: object_label_mask, # <- 순번 값을 직접 할당
                object_index_key: index_per_class[class_name]["count"] # 현재 객체의 순번
                # 필요에 따라 다른 정보(예: 마스크, 키포인트 등) 추가
            }
            processed_ditec_obj_data.append(ditec_obj_info)

        status["total_object_count"]["value"] += amount_object
        status["detection_object_file"]["value"] += 1
        logger.info(f'"{input_path.name}"에서 [{status["total_object_count"]["value"]}]개의 객체를 검출 했습니다.')

    else:
        status["undetection_object_file"]["value"] += 1
        logger.info(f"'{input_path.name}'에서 객체를 검출하지 못했습니다.")
        try:
            with open(undetect_list_path, 'a', encoding='utf-8') as f_failed:
                f_failed.write(f"{str(input_path)}\n")
            logger.info(f"처리 실패 이미지 '{input_path}'의 경로를 '{undetect_list_path}'에 기록했습니다.")
        except Exception as file_e:
            logger.error(f"'{undetect_list_path}'에 실패 이미지 경로를 기록하는 중 오류 발생: {file_e}")
            status["error_output_input_file"]["value"] += 1

        # 미검출 이미지 복사
        try:
            destination_image_path = undetect_objects_dir / input_path.name
            shutil.copy2(str(input_path), str(destination_image_path))
            logger.info(f"미검출 이미지 '{input_path.name}'을(를) '{destination_image_path}'로 복사했습니다.")
        except Exception as copy_e:
            status["error_copied_input_file"]["value"] += 1
            logger.error(f"미검출 이미지 '{input_path.name}' 복사 중 오류 발생: {copy_e}")

    # JSON 파일 저장 (객체 검출 여부와 상관없이)
    # cv2.imwrite 함수로 결과 이미지 저장
    # 해시 계산
    image_hash_value = compute_sha256(frame)

    file_name = input_path.stem
    output_file_name = f"{file_name}.json"
    # output_dir is already a Path object from function arguments
    # The save_object_json_with_polygon function should handle directory creation if needed.
    json_output_path = output_dir / output_file_name

    try:
        json_handler.write_json( # JsonConfigHandler의 write_json 메소드 사용
            image_path  = input_path,
            image_hash  = image_hash_value,
            width = width,
            height = height,
            channels = channels,
            detected_objects = processed_ditec_obj_data,
            json_path = json_output_path
        )
        status["total_output_files"]["value"] += 1
        if status["detection_object_file"]["value"] > 0:
            logger.info(f'"{input_path.name}"에서 [{status["detection_object_file"]["value"]}]개의 객체를 검출하여 "{json_output_path}"에 저장했습니다.')
        else:
            logger.info(f"'{input_path.name}'에서 객체를 검출하지 못했으나, 빈 객체 목록으로 '{json_output_path}'에 저장했습니다.")
    except Exception as e_save:
        logger.error(f"'{json_output_path}' JSON 파일 저장 중 오류 발생: {e_save}")

    return status
## --- 메인 실행 함수 ---
def run_main(cfg: configger):
    # 0. 메인 통계 정보를 담을 딕셔너리 초기화
    status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)

    # 1. YOLO 모델 로딩 (기존 코드와 동일)
    try:
        model = model_load(cfg)
    except Exception as e:
        logger.error(f"설정model_load 중 오류 발생: {e}")
        sys.exit(1)

    # 2. 이미지 파일 목록 가져오기 및 순회 처리
    # IMAGE 파일 목록 가져오기
    """
    파일 개수가 일반적인 수준을 넘어 매우 많다고 판단되신다면, 메모리 사용량을 최소화하면서 
    .json 파일 경로를 처리하는 다른 방법을 고려해볼 수 있습니다. 
    리스트에 모든 경로를 한 번에 저장하는 대신, 파일 경로를 '생성기(generator)' 형태로 다루는 방식입니다.
    가장 좋은 방법은 glob의 결과 자체를 바로 사용하는 것입니다. 
    glob 메소드는 기본적으로 이터레이터(iterator)를 반환합니다. 
    이 이터레이터는 모든 경로를 한 번에 메모리에 로드하는 것이 아니라, 
    필요할 때마다 하나씩 경로를 생성하여 반환합니다. 이렇게 하면 메모리 사용량이 매우 효율적으로 관리됩니다.
    원래 코드에서 리스트 컴프리헨션 [...]을 사용했기 때문에 결과가 리스트로 즉시 메모리에 저장된 것입니다. 
    리스트 컴프리헨션을 사용하지 않고 glob 결과 이터레이터를 직접 순회하면 메모리 부담을 줄일 수 있습니다.
        dataset_keys_to_get = [
        "datasets_dir",
        "raw_image_dir",
        "raw_jsons_dir",
        "undetect_objects_dir",
        "detected_face_images_dir",
        "detected_face_json_dir",
        "undetect_list_path"   #:       ${project.paths.datasets.datasets_dir}/undetect_image.lst
        ]
    """
    # 3. 성정정보(yaml)에서 필요한 정보를 가저옵니다.
    # 3-1. Dataset값 가져와서 환경만들기
    try:
        dataset_key_str = "project.paths.datasets"
        cur_cfg = cfg.get_config(dataset_key_str)
        if cur_cfg is None:
            logger.error(f"'{dataset_key_str}' 설정 그룹을 찾을 수 없습니다.")
            sys.exit(1)

        input_dir_str = cur_cfg.get('raw_image_dir', None)
        input_dir = Path(input_dir_str).expanduser()
        if not input_dir.exists():
            logger.error(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
            sys.exit(1)

        output_dir_str = cur_cfg.get('raw_jsons_dir', None)
        output_dir = Path(output_dir_str).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True) 
        logger.debug(f"output_dir: {output_dir}")

        undetect_dir_str = cur_cfg.get('undetect_dir', None)
        undetect_dir = Path(undetect_dir_str).expanduser()
        # undetect_dir 처리: 존재하면 삭제 후 재생성
        if undetect_dir.exists():
            logger.info(f"기존 미검출 이미지 저장 디렉토리 '{undetect_dir}'이(가) 존재하여 삭제 후 재생성합니다.")
            try:
                shutil.rmtree(undetect_dir) # 디렉토리와 내용물 모두 삭제
            except OSError as e:
                logger.error(f"기존 미검출 이미지 저장 디렉토리 '{undetect_dir}' 삭제 중 오류 발생: {e}")
                # 심각한 오류로 간주하고 종료할 수 있습니다.
                # sys.exit(1)
        
        undetect_dir.mkdir(parents=True, exist_ok=True) # 항상 새로 생성 (또는 exist_ok=True로 인해 오류 없이 넘어감)
        logger.debug(f"미검출 관련 디렉토리 준비 완료: {undetect_dir}")

        undetect_objects_dir_str = cur_cfg.get('undetect_objects_dir', None)
        undetect_objects_dir = Path(undetect_objects_dir_str).expanduser()
        undetect_objects_dir.mkdir(parents=True, exist_ok=True)
        
        undetect_list_path_str = cur_cfg.get('undetect_list_path', None)
        undetect_list_path = Path(undetect_list_path_str).expanduser()
        # 미검출 목록 파일 초기화 (존재하면 삭제)
        if undetect_list_path.exists():
            try:
                undetect_list_path.unlink()
                logger.info(f"기존 미검출 목록 파일 '{undetect_list_path}'을(를) 삭제했습니다.")
            except OSError as e:
                logger.error(f"기존 미검출 목록 파일 '{undetect_list_path}' 삭제 중 오류 발생: {e}")
                # 필요에 따라 여기서 프로그램을 중단할지 결정할 수 있습니다.
                # sys.exit(1)
        logger.debug(f"undetect_list_path: {undetect_list_path}")

        json_cfg = cfg.get_config("json_keys")
        if json_cfg is None: # 수정된 부분
            logger.error(f"설정 파일에 'json_keys' 정보가 누락되었습니다.") # 오류 메시지도 내용에 맞게 수정하는 것이 좋습니다
            sys.exit(1)

        # JsonConfigHandler 인스턴스 생성
        try:
            json_handler = JsonConfigHandler(json_cfg)
        except Exception as e_json_handler:
            logger.error(f"JsonConfigHandler 초기화 중 오류 발생: {e_json_handler}", exc_info=True)
            sys.exit(1)

    except Exception as e:
        logger.error(f"설정{dataset_key_str} 값 가져오기 중 오류 발생: {e}")
        sys.exit(1)
    
    # 3-2. 확장자값 가져와서 환경만들기
    # 모든 파일을 대상으로 glob 실행 후, 확장자 필터링으로 이미지 파일 개수 계산
    # 이 방식은 모든 파일/디렉토리를 순회하므로, 매우 큰 디렉토리에서는 시간이 걸릴 수 있음
    # 더 효율적인 방법은 glob 패턴 자체에 확장자를 포함하는 것이나, supported_extensions가 동적이므로 이 방식 사용
    try:
        """
        processing_keys_to_get    = [
            "supported_image_extensions",
            "frame_skip",
            "output_format"
        ]
        """
        processing_key_str = "processing"
        cur_cfg = cfg.get_config(processing_key_str)

        supported_extensions = cur_cfg.get("supported_image_extensions", None)
        logger.debug(f"지원되는 이미지 확장자 목록: {supported_extensions}")
    except Exception as e:
        logger.error(f"설정{processing_key_str} 값 가져오기 중 오류 발생: {e}")
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

    # 4. 모든 파일에서 객체 정보 누적 (메모리 효율적인 파일 순회)
    input_file_iterator_for_counting = Path(input_dir).glob("**/*")
    if supported_extensions:
        allowed_ext_lower = [ext.lower() for ext in supported_extensions]
        status["total_input_found"]["value"] += sum(1 for p in input_file_iterator_for_counting if p.is_file() and p.suffix.lower() in allowed_ext_lower)
    else: # 지원 확장자 목록이 없으면 모든 파일을 카운트 (주의)
        status["total_input_found"]["value"] = sum(1 for p in input_file_iterator_for_counting if p.is_file())
    logger.debug(f'지원되는 이미지 개수"total_input_found": {status["total_input_found"]["value"]}')

    if status["total_input_found"]["value"] == 0:
        logger.warning(f"'{input_dir}' 디렉토리에서 검출할 image 파일을 찾을 수 없습니다.")
        logger.info("✅ 최종 통계:")
        logger.info(f'   - 탐색된 IMAGE 파일 총 개수: {status["total_input_found"]["value"]}')
        logger.info(f"{Path(__file__).name} 정상 종료 (처리할 파일 없음).")
        sys.exit(0)

    try:
        digit_width = calc_digit_number(status["total_input_found"]["value"])
    except Exception as e:
        digit_width = 0
        logger.error(f"설정{processing_key_str} 값 가져오기 중 오류 발생: {e}")

    logger.info(f'✅ 검출할 Image 파일 {status["total_input_found"]["value"]}개 발견. 숫자 수(digit_width) : {digit_width}')

    # 5. glob 결과를 다시 생성 (실제 처리용)
    # 개수를 세느라 첫 번째 이터레이터가 소모되었으므로, 실제 처리를 위해서는 새로 만들어야 합니다.
    input_file_iterator_for_processing = input_dir.glob("**/*")

    logger.info("IMAGE 파일 내 객체 정보 수집 시작...")

    # 6. 전체 파일을 하나씩 처리합니다.
    # enumerate를 사용하여 진행 상황을 표시하기 위해 리스트로 변환하는 대신,
    # total_input_found 활용하여 수동으로 카운트하며 진행 상황을 표시하는 것이 좋습니다.
    input_files = []
    for input_path in input_file_iterator_for_processing: # 이터레이터를 순회
        if input_path.is_file(): # 파일인지 다시 확인 (glob 결과는 대부분 파일이지만 안전을 위해)
            status["req_process_count"]["value"] += 1
            logger.debug(f'[{(status["req_process_count"]["value"]):{digit_width}}/{status["total_input_found"]["value"]}] 이미지 파일 처리 시작: {input_path.name}')

            # 6.1 확장자 검사 (설정된 확장자 목록이 있는 경우)
            if supported_extensions and isinstance(supported_extensions, list): # supported_extensions가 유효한 리스트인지 확인
                if input_path.suffix.lower() not in allowed_ext_lower:
                    logger.debug(f"지원되지 않는 확장자 '{input_path.suffix}'. 파일 건너뜀: {input_path.name}")
                    status["error_extension"]["value"] += 1
                    continue # 다음 파일로

            # 6.2. 이미지 파일 순회 및 객체 검출 루프 (기존 카메라 루프 대체)
            try:
                """
                def detect_object(
                    base_config: Dict[str, Any],
                    input_path: Path, 
                    output_dir: Path, 
                    undetect_objects_dir = undetect_objects_dir, 
                    undetect_list_path = undetect_list_path, 
                    model:YOLO
                    )->dict:
                """
                ret = detect_object( # base_config 인자 제거
                    base_config = json_cfg,
                    input_path = input_path, 
                    output_dir = output_dir, 
                    json_handler = json_handler, # 생성된 json_handler 인스턴스 전달
                    undetect_objects_dir = undetect_objects_dir, 
                    undetect_list_path = undetect_list_path, 
                    model = model
                    )
            except Exception as e:
                logger.error(f"이미지 파일 '{input_path}' 처리 중 오류 발생: {e}")
                status["error_input_file_process"]["value"] += 1 # 처리 오류 파일 수 증가
                # 오류 발생 시 다음 파일로 이동하거나 종료할 수 있습니다.
                continue # 다음 파일로 이동

            # 6.3 반환값 처리
            # ret (detect_face의 결과 status)에서 main_status로 값 누적
            if ret: # ret이 None이 아닌 경우 (정상 반환된 경우)
                for stat_key in DEFAULT_STATUS_TEMPLATE.keys(): # DEFAULT_STATUS_TEMPLATE의 모든 키에 대해
                    if stat_key in ret and stat_key in status:
                        # ret와 main_status 모두 "value" 키를 가지고 숫자 값을 가진다고 가정
                        if "value" in ret[stat_key] and isinstance(ret[stat_key]["value"], (int, float)):
                            status[stat_key]["value"] += ret[stat_key]["value"]
            logger.info(f"'{(status['req_process_count']['value']):{digit_width}}/{status['total_input_found']['value']}' JSON 파일 처리 완료: {input_path.name}")

    # 9. 모든 이미지 처리 완료 또는 중단 후 자원 해제
    # 9-1. 통계 결과 출력 ---
    # 가장 긴 메시지의 바이트 길이를 저장할 변수 초기화 (UTF-8 기준)
    max_msg_byte_length = 0 

    # DEFAULT_STATUS_TEMPLATE에 있는 메시지들 (status에 해당하는 키만)의 최대 바이트 길이를 계산
    for key in status.keys(): # Iterate over keys present in the status
        if key in DEFAULT_STATUS_TEMPLATE: # Check if the key has a defined message in the template
            msg_string = DEFAULT_STATUS_TEMPLATE[key]["msg"]
            
            # 메시지 문자열을 UTF-8로 인코딩한 후 바이트 길이를 계산합니다.
            current_byte_length = visual_length(msg_string, 2) 
            
            # 현재 메시지의 바이트 길이가 최대 바이트 길이보다 크면 업데이트
            if current_byte_length > max_msg_byte_length:
                max_msg_byte_length = current_byte_length

    # max_msg_byte_length 변수에 가장 긴 메시지의 UTF-8 바이트 길이가 저장됩니다.
    # print(f"가장 긴 메시지의 UTF-8 바이트 길이: {max_msg_byte_length}") # 확인을 위한 예시 출력

    fill_char = '.' # 원하는 채움 문자를 여기에 지정합니다. 예를 들어 '.' 또는 '-' 등
    # --- 통계 결과 출력 ---
    logger.warning("--- 이미지 파일 처리 통계 ---")
    for key, data in status.items():
        # DEFAULT_STATUS_TEMPLATE에 해당 키가 있을 경우에만 메시지를 가져오고, 없으면 키 이름을 사용
        msg = DEFAULT_STATUS_TEMPLATE.get(key, {}).get("msg", key)
        value = data["value"]
        logger.warning(f'{msg:{fill_char}<{max_msg_byte_length}}: {value:{digit_width}}')
    logger.warning("------------------------")
    # --- 통계 결과 출력 끝 ---


# 스크립트가 직접 실행될 때 함수 호출
if __name__ == "__main__":
    """
    설정 파일을 로드하고 YOLO 객체 검출을 수행하는 함수에게 일을 시킴
    """
    # 0. 애플리케이션 아귀먼트 있으면 갖오기
    logger.info(f"애플리케이션 시작")
    parsed_args = get_argument()

    script_name = Path(__file__).stem # Define script_name early for logging
    try:
        # 1. 애플리케이션 로거 설정
        date_str = datetime.now().strftime("%y%m%d")
        log_file_name = f"{script_name}_{date_str}.log"
        full_log_path = Path(parsed_args.log_dir) / log_file_name
        logger.setup(
            logger_path=str(full_log_path),
            min_level=parsed_args.log_level,
            include_function_name=True,
            pretty_print=True
        )
        logger.info(f"애플리케이션({script_name}) 시작")
        logger.info(f"명령줄 인자로 결정된 경로: {vars(parsed_args)}")
    except Exception as e:
        print(f"치명적 오류: 로거 설정 중 오류 발생 - {e}", file=sys.stderr)
        sys.exit(1)

    # 2. configger 인스턴스 생성
    # configger는 이제 위에서 설정된 공유 logger를 사용합니다.
    # root_dir과 config_path는 실제 프로젝트에 맞게 설정해야 합니다.
    # 설정 파일이 실제로 존재하는지 확인

    logger.info(f"Configger 초기화 시도: root_dir='{parsed_args.root_dir}', config_path='{parsed_args.config_path}'")

    try:
        cfg_object = configger(root_dir=parsed_args.root_dir, config_path=parsed_args.config_path)
        logger.info(f"Configger 초기화 끝")

        run_main(cfg_object)
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}", exc_info=True)
    finally:
        # 애플리케이션 종료 시 로거 정리 (특히 비동기 사용 시 중요)
        logger.info("my_yolo_tiny 애플리케이션 종료")
        logger.shutdown()
        exit(0)
