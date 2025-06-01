# my_yolo_tiny/sorc/object_detector.py
import inspect 
import types # SimpleNamespace 사용을 위해 types 모듈 임포트
import sys     # 표준 출력 스트림 사용을 위해 필요
from pathlib import Path # 경로 관리를 위해 필요
import torch # torch 모듈을 임포트합니다.
from datetime import datetime

import cv2 # OpenCV 라이브러리
from ultralytics import YOLO # ultralytics의 YOLO 모델

# shared_utils 패키지에서 configger 클래스 가져오기
# shared_utils 프로젝트의 src/utility/configger.py에 configger 클래스가 있다고 가정
# object_detector.py 파일 내 임포트 구문
# 사용자 정의 유틸리티 모듈 임포트
try:
    from my_utils.config_utils.SimpleLogger import logger, calc_digit_number
    from my_utils.config_utils.configger import configger, get_log_cfg_argument
    from my_utils.photo_utils.object_utils import rotate_image_if_needed, compute_sha256, load_json, save_object_json_with_polygon, save_cropped_face_image
except ImportError as e:
    # 실제 발생한 예외 e를 출력하여 원인 파악
    print(f"모듈 임포트 중 오류 발생: {e}")
    print(f"자세한 오류 정보:")
    import traceback
    traceback.print_exc() # 전체 트레이스백 출력 (개발 단계에서 유용)
    sys.exit(1)

# def get_config_paths(cfg: configger, base_key: str, config_keys_to_get: list) -> dict:
#     """
#     설정 파일에서 지정된 기본 경로 하위의 여러 경로 설정 값을 읽어와
#     속성으로 접근 가능한 SimpleNamespace 객체로 반환합니다.

#     Args:
#         cfg (configger): 설정 객체 (get_path 메서드를 가짐).
#         base_key (str): 설정에서 경로 값들의 상위 기본 경로 문자열 (예: "project.paths.datasets").
#         config_keys_to_get (list): 기본 경로 하위에서 가져올 경로 설정 키 이름 목록 (예: ["raw_image_dir", "raw_jsons_dir"]).

#     Returns:
#         types.SimpleNamespace: 경로 설정 값들을 속성으로 가지는 객체.
#                                  설정을 찾지 못한 키는 None 값을 가집니다.
#     """
#     # 딕셔너리 대신 SimpleNamespace 객체를 생성합니다.
#     paths_namespace = types.SimpleNamespace()

#     try:
#         for key_name in config_keys_to_get:
#             full_key = f"{base_key}.{key_name}"
#             try:
#                 # cfg.get_path 메서드를 호출하여 경로 값을 가져옵니다.
#                 # raw_image_dir는 존재해야 하는 설정이라면 ensure_exists=True를 유지합니다.
#                 # 다른 키들은 필요에 따라 ensure_exists 값을 조정할 수 있습니다.
#                 # 예시에서는 모든 키에 대해 ensure_exists=True를 적용합니다.
#                 # 특정 키에 대해 다르게 처리하려면 조건을 추가해야 합니다.
#                 path_value = cfg.get_path(full_key, ensure_exists=True)

#                 # SimpleNamespace 객체에 속성으로 추가합니다.
#                 # setattr 함수를 사용하여 동적으로 속성을 설정합니다.
#                 setattr(paths_namespace, key_name, path_value)

#                 if path_value:
#                     logger.debug(f"경로 확인: '{full_key}' -> '{path_value}'")
#                 else:
#                     logger.warning(f"'{full_key}' 키에 대한 경로 값을 가져오지 못했습니다 (존재하지 않거나 유효하지 않음).")

#             except Exception as e:
#                  # 개별 경로 가져오기 중 발생한 오류를 로깅하고 계속 진행하거나 (경고),
#                  # 치명적이라고 판단하면 sys.exit(1) 할 수 있습니다.
#                  # 여기서는 일단 로깅하고 해당 키는 None으로 설정되도록 합니다.
#                  logger.error(f"'{full_key}' 경로 값 가져오기 중 오류 발생: {e}") # 전체 traceback 대신 간단히 로깅
#                  setattr(paths_namespace, key_name, None) # 오류 발생 시 해당 속성 값은 None으로 설정

#     except Exception as e:
#         # 기본 경로 자체에 문제가 있는 경우 (예: base_key 형식 오류 등)
#         logger.error(f"'{base_key}' 하위의 설정 경로 값 가져오기 중 치명적인 오류 발생: {e}")
#         sys.exit(1) # 기본 경로 설정 오류는 치명적이라고 가정

#     # SimpleNamespace 객체를 반환합니다.
#     return paths_namespace

# def get_config_value(cfg: configger, base_key: str, config_keys_to_get: list) -> dict:
#     """
#     설정 객체에서 지정된 키에 해당하는 값을 가져오는 함수 (예시)
#     실제 구현에 따라 다를 수 있습니다.
#     여기서는 default_keys 목록에 있는 하위 값들을 딕셔너리로 반환한다고 가정합니다.
#     """
#     config_values = {}
#     try:
#         for key_name in config_keys_to_get:
#             full_key = f"{base_key}.{key_name}"
#             try:
#                 # cfg.get_value는 키가 없거나 오류 발생 시 기본값을 반환하거나 예외를 발생시킬 수 있습니다.
#                 value = cfg.get_value(full_key) 
#                 config_values[key_name] = value
#                 if value is not None:
#                     logger.debug(f"값 확인: '{full_key}' -> '{value}'")
#                 else:
#                     logger.warning(f"'{full_key}' 키에 대한 값을 가져오지 못했거나 설정되지 않았습니다. 반환 값: {value}")
#             except Exception as e_get_val:
#                 logger.error(f"'{full_key}' 값 가져오기 중 오류 발생: {e_get_val}")
#                 config_values[key_name] = None # 오류 발생 시 해당 키 값은 None으로 설정
#     except Exception as e_base:
#         logger.error(f"'{base_key}' 하위의 설정 값 가져오기 중 치명적인 오류 발생: {e_base}")
#         return {} # 오류 발생 시 빈 딕셔너리 반환 또는 sys.exit(1)
#     return config_values

# model_load 함수의 시그니처 변경: cfg와 key_str만 받고, 반환 타입은 YOLO 객체
# default_keys는 get_config_value 내부에서 사용되거나,
# 이 함수에서는 object_detection_model_key_str에 해당하는 고정된 키 목록을 사용합니다.
# 따라서 함수 인자에서 default_keys는 제거합니다.
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

    # # Path 객체 사용을 위해 문자열을 Path 객체로 변환
    # model_weights_path = None # 초기화
    # if model_weights_path_str:
    #      # 이전에 얻은 model_weights_path_str 변수를 사용합니다.
    #      model_weights_path = Path(model_weights_path_str).expanduser() # 사용자 홈 디렉토리 확장
    #      logger.debug(f"모델 가중치 Path 객체: {model_weights_path}")
    # else:
    #      logger.debug("설정에 model_weights_path가 지정되지 않았습니다. model_name을 사용합니다.")

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

def detect_object(input_path: Path, output_dir: Path, detected_objects_dir: Path, undetect_list_path: Path, model:YOLO)->dict:
    # 이미지 파일 읽기 (OpenCV 사용)
  # 통계 정보를 담을 딕셔너리 초기화
    status = {
        "files_read_success": 0, # 파일 읽기 성공 수
        "files_read_error": 0, # 파일 읽기 오류로 건너뛴 파일 수
        "files_with_objects_detected": 0, # 객체가 1개 이상 검출된 파일 수
        "files_with_no_objects_detected": 0, # 객체가 검출되지 않은 파일 수
        "files_detected_image_save_success": 0, # 검출한 객채를 저장중 오류 없이 처리 완료된 파일 수
        "files_detected_image_save_error": 0, # 검출한 객채를 저장중 중 예외 발생으로 건너뛴 파일 수
        "files_processed_successfully": 0 # 오류 없이 처리 완료 파일 수 증가
    }

    # 1. 이미지 파일 읽기 전에 이미지 회전
    rotate_image_if_needed(str(input_path)) # input_path는 Path 객체이므로 문자열로 변환하여 전달

    # ... 이미지 파일 읽기 부분 ...
    frame = cv2.imread(str(input_path))

    if frame is None:
        logger.warning(f"이미지 파일 '{input_path}'을 읽을 수 없습니다. 건너뜁니다.")
        status["files_read_error"] += 1 # 읽기 오류 파일 수 증가        
        return  status # 파일 읽기 실패 시 다음 파일로 이동

    status["files_read_success"] += 1 # 파일 읽기 성공 수 증가

    results = model(
        frame,                       # ① 입력 데이터
        conf=model.confidence_threshold,   # ② 신뢰도 임계값 (Confidence Threshold)
        iou=model.iou_threshold,           # ③ IoU 임계값 (IoU Threshold) for NMS
        classes=model.classes_to_detect,   # ④ 검출할 클래스 목록 (Classes to Detect)
        imgsz=model.imgsz,                 # ⑤ 입력 이미지 크기 (Image Size)
        device=model.selected_device,      # ⑥ 실행 장치 (Device)
        verbose=False                # ⑦ 상세 출력 여부 (Verbosity)
    )

    # 검출된 객체 수 확인
    processed_ditec_obj_data = []
    index_per_class = {}
    if results[0].boxes is not None and len(results[0].boxes) > 0:
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
            object_index_in_class = index_per_class[class_name]["count"] # 현재 객체의 순번

            # 필요한 정보들을 딕셔너리로 구성
            ditec_obj_info = {
                "box_xyxy": xyxy_coords,
                "box_xywh": xywh_coords,
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
                "index_in_class": object_index_in_class # <- 순번 값을 직접 할당
                # 필요에 따라 다른 정보(예: 마스크, 키포인트 등) 추가
            }
            processed_ditec_obj_data.append(ditec_obj_info)

        num_detected_objects = len(results[0].boxes)
        status["files_with_objects_detected"] = num_detected_objects
        logger.info(f"'{input_path.name}'에서 [{num_detected_objects}]개의 객체를 검출 했습니다.")

    else:
        status["files_with_no_objects_detected"] = 1
        logger.info(f"'{input_path.name}'에서 객체를 검출하지 못했습니다.")
        try:
            with open(undetect_list_path, 'a', encoding='utf-8') as f_failed:
                f_failed.write(f"{str(input_path)}\n")
            logger.info(f"처리 실패 이미지 '{input_path}'의 경로를 '{undetect_list_path}'에 기록했습니다.")
        except Exception as file_e:
            logger.error(f"'{undetect_list_path}'에 실패 이미지 경로를 기록하는 중 오류 발생: {file_e}")

    # JSON 파일 저장 (객체 검출 여부와 상관없이)
    # cv2.imwrite 함수로 결과 이미지 저장
    # 해시 계산
    image_hash_value = compute_sha256(frame)

    file_name = input_path.stem
    output_file_name = f"{file_name}.json"
    # output_dir is already a Path object from function arguments
    # The save_object_json_with_polygon function should handle directory creation if needed.
    output_path = output_dir / output_file_name

    save_object_json_with_polygon(
        image_path  = input_path, 
        image_hash  = image_hash_value,
        ditec_obj   = processed_ditec_obj_data, # Parameter name in function is ditec_obj
        json_path   = output_path
    )
    if status["files_with_objects_detected"] > 0:
        logger.info(f"'{input_path.name}'에서 [{status['files_with_objects_detected']}]개의 객체를 검출하여 '{output_path}'에 저장했습니다.")
    else:
        logger.info(f"'{input_path.name}'에서 객체를 검출하지 못했으나, '{output_path}'에 저장했습니다.")
    status["files_processed_successfully"] += 1

    return status
## --- 메인 실행 함수 ---
def run_object_detection(cfg: configger):
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
        "detected_objects_dir",
        "undetect_objects_dir",
        "detected_face_images_dir",
        "detected_face_json_dir",
        "undetect_list_path"   #:       ${project.paths.datasets.datasets_dir}/undetect_image.lst
        ]

    """

    # 3. IMAGE 파일 목록 가져오기 및 총 개수 세기 (메모리 효율적)
    try:
        dataset_key_str = "project.paths.datasets"
        cur_cfg = cfg.get_config(dataset_key_str)

        input_dir = Path(cur_cfg.get('raw_image_dir', None)).expanduser()
        logger.debug(f"run_object_detection-input_dir: {input_dir}")

        output_dir = Path(cur_cfg.get('raw_jsons_dir', None)).expanduser()
        logger.debug(f"run_object_detection-output_dir: {output_dir}")

        detected_objects_dir = Path(cur_cfg.get('detected_objects_dir', None)).expanduser()
        detected_objects_dir.mkdir(parents=True, exist_ok=True) 
        logger.debug(f"run_object_detection-detected_objects_dir: {detected_objects_dir}")

        undetect_list_path = Path(cur_cfg.get('undetect_list_path', None)).expanduser()
        # 파일 이름 제외한 디렉토리 경로 추출, 디렉토리 생성 (상위 디렉토리가 없으면 함께 생성, 이미 존재하면 에러 X)
        undetect_dir = undetect_list_path.parent
        undetect_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"run_object_detection-undetect_dir: {undetect_dir}")
    except Exception as e:
        logger.error(f"run_object_detection-설정{dataset_key_str} 값 가져오기 중 오류 발생: {e}")
        sys.exit(1)
    
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
        logger.debug(f"run_object_detection-지원되는 이미지 확장자 목록: {supported_extensions}")
    except Exception as e:
        logger.error(f"설정{pricessing_key_str} 값 가져오기 중 오류 발생: {e}")
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

    # --- 통계 변수 초기화 ---
    files_read_success = 0                  # 파일 읽기 성공 수
    files_read_error = 0                    # 파일 읽기 오류로 건너뛴 파일 수
    files_skipped_read_error = 0            # 파일 읽기 오류로 건너뛴 파일 수
    files_processed_successfully = 0        # 오류 없이 처리 완료된 파일 수
    files_with_objects_detected = 0         # 객체가 1개 이상 검출된 파일 수
    files_with_no_objects_detected = 0      # 객체가 검출되지 않은 파일 수
    files_detected_image_save_success = 0   # 검출한 객채를 저장중 오류 없이 처리 완료된 파일 수
    files_detected_image_save_error = 0     # 검출한 객채를 저장중 중 예외 발생으로 건너뛴 파일 수
    files_processed_with_error = 0          # 처리 중 예외 발생으로 건너뛴 파일 수
    total_files_read = 0                    # 읽기를 시도한 총 파일 수
    total_object_processed = 0              # 성공적으로 처리된 총 객체 개수
    total_objects_detected = 0              # 검출된 총 오브젝트 수
    total_input_found = 0
    # --- 통계 변수 초기화 끝 ---

    input_file_iterator_for_counting = Path(input_dir).glob("**/*")
    if supported_extensions:
        allowed_ext_lower = [ext.lower() for ext in supported_extensions]
        total_input_found = sum(1 for p in input_file_iterator_for_counting if p.is_file() and p.suffix.lower() in allowed_ext_lower)
    else: # 지원 확장자 목록이 없으면 모든 파일을 카운트 (주의)
        total_input_found = sum(1 for p in input_file_iterator_for_counting if p.is_file())
    logger.debug(f"run_object_detection-지원되는 이미지 개수(total_input_found): {total_input_found}")

    if total_input_found == 0:
        logger.warning(f"'{input_dir}' 디렉토리에서 검출할 image 파일을 찾을 수 없습니다.")
        logger.info("✅ 최종 통계:")
        logger.info(f"   - 탐색된 IMAGE 파일 총 개수: {total_input_found}")
        logger.info(f"{Path(__file__).name} 정상 종료 (처리할 파일 없음).")
        sys.exit(0)

    try:
        digit_width = calc_digit_number(total_input_found)
    except Exception as e:
        digit_width = 0
        logger.error(f"설정{pricessing_key_str} 값 가져오기 중 오류 발생: {e}")

    logger.info(f"✅ 검출할 Image 파일 {total_input_found}개 발견. 숫자 수(digit_width) : {digit_width}")

    # 2. 모든 파일에서 객체 정보 누적 (메모리 효율적인 파일 순회)
    # glob 결과를 다시 생성 (실제 처리용)
    # 개수를 세느라 첫 번째 이터레이터가 소모되었으므로, 실제 처리를 위해서는 새로 만들어야 합니다.
    input_file_iterator_for_processing = input_dir.glob("**/*")

    logger.info("IMAGE 파일 내 객체 정보 수집 시작...")
    # enumerate를 사용하여 진행 상황을 표시하기 위해 리스트로 변환하는 대신,
    # total_input_found 활용하여 수동으로 카운트하며 진행 상황을 표시하는 것이 좋습니다.
    input_files = []
    for input_path in input_file_iterator_for_processing: # 이터레이터를 순회
        if input_path.is_file(): # 파일인지 다시 확인 (glob 결과는 대부분 파일이지만 안전을 위해)
            total_files_read += 1 # 처리 시작 파일 카운트
            logger.debug(f"[{(total_files_read):{digit_width}}/{total_input_found}] 이미지 파일 처리 시작: {input_path.name}")

            # 확장자 검사 (설정된 확장자 목록이 있는 경우)
            if supported_extensions and isinstance(supported_extensions, list): # supported_extensions가 유효한 리스트인지 확인
                if input_path.suffix.lower() not in allowed_ext_lower:
                    logger.debug(f"지원되지 않는 확장자 '{input_path.suffix}'. 파일 건너뜀: {input_path.name}")
                    files_skipped_read_error += 1 # 통계 추가 가능
                    return  status # 다음 파일로

                # 3. 이미지 파일 순회 및 객체 검출 루프 (기존 카메라 루프 대체)
                try:
                    ret = detect_object(input_path, output_dir, detected_objects_dir, undetect_list_path, model)
                    files_read_success += ret["files_read_success"] # 파일 읽기 성공 수
                    files_read_error += ret["files_read_error"] # 파일 읽기 성공 수
                    files_processed_successfully += ret["files_processed_successfully"] # 오류 없이 처리 완료된 파일 수
                    files_with_objects_detected += ret["files_with_objects_detected"] # 객체가 1개 이상 검출된 파일 수
                    files_with_no_objects_detected += ret["files_with_no_objects_detected"] # 객체가 검출되지 않은 파일 수
                    files_detected_image_save_success += ret["files_detected_image_save_success"] # 검출한 객채를 저장중 오류 없이 처리 완료된 파일 수
                    files_detected_image_save_error += ret["files_detected_image_save_error"] # 검출한 객채를 저장중 중 예외 발생으로 건너뛴 파일 수
                except Exception as e:
                    logger.error(f"이미지 파일 '{input_path}' 처리 중 오류 발생: {e}")
                    files_processed_with_error += 1 # 처리 오류 파일 수 증가
                    # 오류 발생 시 다음 파일로 이동하거나 종료할 수 있습니다.
                    return  status # 다음 파일로 이동
            logger.info(f"[{files_read_success}/{total_input_found}] 이미지 파일 처리 완료: {input_path.name}")

    if files_read_success == 0 and total_input_found > 0 : # 탐색은 되었으나, 읽기 성공한 파일이 없는 경우
        logger.warning(f"탐색된 {total_input_found}개의 이미지 파일 중 성공적으로 읽고 처리한 파일이 없습니다.")
    elif total_files_read == 0 and total_input_found == 0: # 이 경우는 이미 위에서 처리됨
        logger.warning("'{input_dir}' 디렉토리에 처리할 이미지 파일이 없습니다.")

    # 9. 모든 이미지 처리 완료 또는 중단 후 자원 해제
    # --- 통계 결과 출력 ---
    logger.info("--- 이미지 파일 처리 통계 ---")
    logger.info(f"총 읽기 시도 파일 수: {total_files_read}")
    logger.info(f"읽기 성공 파일 수: {files_read_success}")
    logger.info(f"읽기 오류 건너뛴 파일 수: {files_skipped_read_error}")
    logger.info(f"오류 없이 처리 완료 파일 수: {files_processed_successfully}")
    logger.info(f"  > 객체 1개 이상 검출된 파일 수: {files_with_objects_detected}")
    logger.info(f"  > 객체 검출되지 않은 파일 수: {files_with_no_objects_detected}")
    logger.info(f"  > 검출한 객채를 저장중 오류 없이 처리 완료된 파일 수: {files_detected_image_save_success}")
    logger.info(f"  > 검출한 객채를 저장중 중 예외 발생으로 건너뛴 파일 수: {files_detected_image_save_error}")
    logger.info(f"처리 중 오류 발생 파일 수: {files_processed_with_error}")
    logger.info(f"총 검출된 오브젝트 수: {total_objects_detected}")
    logger.info("------------------------")
    # --- 통계 결과 출력 끝 ---


# 스크립트가 직접 실행될 때 run_object_detection 함수 호출
if __name__ == "__main__":
    """
    설정 파일을 로드하고 YOLO 객체 검출을 수행하는 함수에게 일을 시킴
    """
    # 0. 애플리케이션 아귀먼트 있으면 갖오기
    logger.info(f"애플리케이션 시작")
    parsed_args = get_log_cfg_argument()

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

        run_object_detection(cfg_object)
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}", exc_info=True)
    finally:
        # 애플리케이션 종료 시 로거 정리 (특히 비동기 사용 시 중요)
        logger.info("my_yolo_tiny 애플리케이션 종료")
        logger.shutdown()
        exit(0)
