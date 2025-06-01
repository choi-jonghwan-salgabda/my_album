"""

"""
import json
import os
import logging # SimpleLogger를 통해 이미 사용 가능하지만, 직접 사용이 필요할 경우를 위해
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import torch
import numpy as np # 모델 워밍업용 더미 이미지 생성에 사용
from ultralytics import YOLO
from PIL import Image # 얼굴 이미지 저장을 위해 PIL.Image 사용
from datetime import datetime
from PIL import Image, ExifTags
import copy # deepcopy 사용을 위해 copy 모듈 임포트
import shutil # 파일 복사를 위해 shutil 모듈 임포트

# shared_utils 패키지에서 configger 클래스 가져오기
# shared_utils 프로젝트의 src/utility/configger.py에 configger 클래스가 있다고 가정
# object_detector.py 파일 내 임포트 구문
# 사용자 정의 유틸리티 모듈 임포트
try:
    from my_utils.config_utils.SimpleLogger import logger, calc_digit_number, get_argument, visual_length
    from my_utils.config_utils.configger import configger
    from my_utils.photo_utils.object_utils import rotate_image_if_needed, compute_sha256, load_json, save_object_json_with_polygon, save_cropped_face_image, read_json_with_config_keys, write_json_from_config
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
    "error_image_rotation":          {"value": 0,  "msg": "이미지 회전중 오류 발생 파일 수"},
    "error_input_file_read":        {"value": 0,  "msg": "입력 파일 읽기 오류 수"},
    "error_target_file_get":        {"value": 0,  "msg": "처리대상(image or json) 파일 읽기 오류 수"},
    "error_input_file_process":        {"value": 0,  "msg": "입렫파일 읽기 오류 수"},
    "total_object_count":      {"value": 0,  "msg": "검출된 총 객체 수"},
    "detection_object_file":     {"value": 0,  "msg": "객체가 검출된 파일 수"},
    "get_object_crop":           {"value": 0,  "msg": "객체가 검출된 객체 수"},
    "error_object_crop":           {"value": 0,  "msg": "객체가 검출된 객체 수"},
    "error_object_bbox":           {"value": 0,  "msg": "객체가 검출된 객체 수"},
    "error_object_bbox_cnt":           {"value": 0,  "msg": "객체가 검출된 객체 수"},
    "error_object_bbox_posit":           {"value": 0,  "msg": "객체가 검출된 객체 수"},
    "undetection_object":   {"value": 0,  "msg": "객체가 검출되지 않은 파일 수"},
    "error_copied_input_file": {"value": 0, "msg": "오류발생 잉ㅂ력팡ㄹ 보관시 실패 수"},
    "detect_faces_in_object":    {"value": 0,  "msg": "객체에서 얼굴검출을 성공한 수"},
    "error_faces_in_object":    {"value": 0,  "msg": "객체에서 얼굴검출을 성공한 수"},
    "unmatched_object_number":   {"value": 0,  "msg": "검출 대상 object수와 검출한 object의 수가 다른 파일수"},
    "total_output_files":        {"value": 0,  "msg": "총 출력 파일수"},
    "total_input_found":         {"value": 0,  "msg": "총 입력 파일 수 (지원 확장자 기준)"},
    "total_output_files":        {"value": 0,  "msg": "총 출력 파일수"},
    "read_input_files_success":          {"value": 0,  "msg": "읽은 입력 파일 수 (detect_object 기준)"},
    "read_input_files_error":          {"value": 0,  "msg": "읽은 입력 파일 수 (detect_object 기준)"},
    "files_json_load":           {"value": 0,  "msg": "JSON 정보 읽은 파일 수"},
    "files_json_update":         {"value": 0,  "msg": "JSON 파일 덧씌우기 성공 파일 수"},
    "error_json_update":         {"value": 0,  "msg": "JSON 파일 덧씌우기 성공 파일 수"},
    "get_image_path_in_json":    {"value": 0,  "msg": "IMAGE 파일 경로 가져온 파일 수"},
    "detection_object_file":     {"value": 0,  "msg": "객체가 검출된 파일 수"},
    "undetected_image_copied_success": {"value": 0, "msg": "미검출 이미지 복사 성공 수"},
    "undetected_image_copied_error": {"value": 0, "msg": "미검출 이미지 복사 실패 수"},
    "undetection_object_file":   {"value": 0,  "msg": "객체가 검출되지 않은 파일 수"},
    "num_detected_objects":      {"value": 0,  "msg": "검출된 총 객체 수"},
    "files_object_crop":         {"value": 0,  "msg": "객체가 있는 파일 수"},
    "error_faild_file_backup":        {"value": 0,  "msg": "읽을때 오류가 난 입력 파일을 보관하는데 오류발생 수"},
    "files_skipped_extension":   {"value": 0,  "msg": "지원되지 않는 확장자로 건너뛴 파일 수"},
    "files_processed_for_log":   {"value": 0,  "msg": "로그용으로 처리 시도한 파일 수"}, # Not for final stats display usually
    "files_processed_main_error":{"value": 0,  "msg": "메인 루프에서 처리 중 오류 발생 파일 수"}
} 


def add_path_to_list_file(file_path: Path, list_file_path: Path) -> bool:
    """
    로드에 실패한 JSON 파일 경로를 지정된 목록 파일에 추가합니다.

    Args:
        file_path (Path): 관리할 파일의 경로.
        list_file_path (Path): 경로를 추가할 목록 파일의 경로.

    Returns:
        bool: 성공적으로 추가하면 True, 그렇지 않으면 False.
    """
    if not list_file_path:
        logger.warning("경로를 기록할 목록 파일 경로가 제공되지 않았습니다. 기록을 건너<0xEB><0><0x8F><0xBB>니다.")
        return False
    try:
        # 목록 파일의 상위 디렉토리 생성 (존재하지 않는 경우)
        list_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(list_file_path, 'a', encoding='utf-8') as f_list:
            # add_failed_json_to_list 함수 수정:
            # 파일 내용을 읽어 각 줄과 비교하여 중복 확인
            # file_path를 문자열로 변환하여 비교해야 합니다.
            file_path_str = str(file_path)
            already_exists = False
            with open(list_file_path, 'r', encoding='utf-8') as f_read:
                if any(file_path_str == line.strip() for line in f_read):
                    already_exists = True
            if not already_exists:
                f_list.write(str(file_path) + "\n")
        logger.info(f"'{file_path}' 경로를 '{list_file_path}'에 추가했습니다.")
        return True
    except Exception as e:
        logger.error(f"'{list_file_path}'에 경로를 기록하는 중 오류 발생: {e}")
        return False
    return True


def model_load(cfg: configger) -> Tuple[YOLO, Dict[str, Any]]:
    """
    설정 파일에서 YOLO 모델 설정을 읽어 모델을 로드하고,
    관련 설정 값들도 함께 반환합니다.

    Returns:
        Tuple[YOLO, Dict]: YOLO 모델 객체와 설정 정보 딕셔너리
    """
    model_config_keys = 'models.object_yolo_tiny_model.face_detection_model'
    logger.info(f"model_key_str: {model_config_keys}")

    # 설정값 로딩
    try:
        model_name = cfg.get_value(f'{model_config_keys}.model_name', ensure_exists=True)
        logger.debug(f"model_name: {model_name}")
    except Exception as e:
        logger.error(f"model_name 설정 값 가져오기 오류: {e}")
        sys.exit(1)

    try:
        model_weights_path = Path(cfg.get_path(f'{model_config_keys}.model_weights_path', ensure_exists=True))
        logger.debug(f"model_weights_path: {model_weights_path}")
    except Exception as e:
        logger.error(f"model_weights_path 설정 값 가져오기 오류: {e}")
        sys.exit(1)

    try:
        confidence_threshold = float(cfg.get_value(f'{model_config_keys}.confidence_threshold', ensure_exists=True))
        logger.debug(f"confidence_threshold: {confidence_threshold}")
    except Exception as e:
        logger.error(f"confidence_threshold 설정 값 가져오기 오류: {e}")
        sys.exit(1)

    try:
        use_cpu = cfg.get_value(f'{model_config_keys}.use_cpu', ensure_exists=True)
        logger.debug(f"use_cpu: {use_cpu}")
    except Exception as e:
        logger.error(f"use_cpu 설정 값 가져오기 오류: {e}")
        sys.exit(1)

    # 디바이스 선택
    if use_cpu:
        selected_device = 'cpu'
        logger.debug("CPU 사용 설정")
    elif torch.cuda.is_available():
        selected_device = 'cuda'
        logger.debug("CUDA 사용")
    else:
        selected_device = 'cpu'
        logger.debug("CUDA 미사용, CPU로 대체")

    logger.info(f"YOLO 모델 로딩 시작 - 경로: {model_weights_path}, 장치: {selected_device}")

    # 모델 로딩
    try:
        if model_weights_path and model_weights_path.exists():
            model_source = str(model_weights_path)
            yolo_face_model = YOLO(model_source)
            yolo_face_model.to(selected_device)
            logger.debug(f"로컬 모델 파일 로딩 성공: {model_source}")
        elif model_name:
            model_source = model_name
            yolo_face_model = YOLO(model_source)
            logger.debug(f"허브 모델 로딩 성공: {model_source}")
        else:
            logger.error("모델 이름 또는 경로가 지정되지 않았습니다.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"YOLO 모델 로딩 오류: {e}")
        sys.exit(1)

    # 모델 워밍업
    try:
        logger.debug("YOLO 워밍업 중...")
        dummy_img_np = np.zeros((64, 64, 3), dtype=np.uint8)
        yolo_face_model(dummy_img_np, verbose=False)
        logger.debug("YOLO 워밍업 완료.")
    except Exception as e_warmup:
        logger.warning(f"YOLO 워밍업 실패: {e_warmup}")

    logger.info("YOLO 모델 최종 로딩 성공")

    # 설정 정보 함께 반환
    return yolo_face_model, {
        "confidence_threshold": confidence_threshold,
        "use_cpu": use_cpu,
        "model_source": model_source,
        "device": selected_device
    }

# 신뢰도가 높은순으로 반환
def detect_faces_in_crop_yolo_internal(
    output_face_info_config: List[int],
    image_crop: cv2.Mat,
    obj_bbox_origin_xyxy: List[int],
    yolo_face_model: YOLO,
    confidence_threshold:  float
) -> List[Dict]:
    """
    YOLO 얼굴 탐지 결과를 신뢰도 순으로 정렬하여 반환하고, 얼굴을 잘라 저장합니다.
    크롭된 이미지 영역에서 YOLO 모델을 사용하여 얼굴을 탐지합니다.
    탐지된 얼굴의 경계 상자는 원본 이미지 좌표 기준으로 반환됩니다.
    yolo_face_model은 이미 적절한 장치(CPU/GPU)에 로드되어 있어야 합니다.
    """
    if image_crop is None or image_crop.size == 0:
        logger.warning("얼굴 탐지를 위한 입력 이미지 크롭이 비어있습니다.")
        return []

    # 0.1. JSON정보에 넣어야 하는 KEY를 설정정보에서 가저오기 "detected_face"
    try:
        face_name_key       = output_face_info_config.get("face_name_key", "detected_face")
        face_box_xyxy_key   = output_face_info_config.get("face_box_xyxy_key", "box_xyxy")
        face_confidence_key = output_face_info_config.get("face_confidence_key", "confidence")  
        face_label_key      = output_face_info_config.get("face_label_key", "label")
        face_class_id_key   = output_face_info_config.get("face_class_id_key", "class_in")  
        face_class_name_key = output_face_info_config.get("face_class_name_key", "calss_name")  
        """
        face_embedding_key  = output_face_info_config.get("face_embedding_key", "embedding")
        face_id_key         = output_face_info_config.get("face_id_key", "face_id")
        face_box_key        = output_face_info_config.get("face_box_key", "box")
        face_score_key      = output_face_info_config.get("face_score_key", "score")
        """
    except Exception as e:
        logger.error(f"detect_object: base_config에서 설정 키 로딩 실패 ({face_name_key}): {e}")
        return []


    # 모델은 이미 올바른 장치에 있다고 가정합니다.
    # verbose=False로 설정하여 YOLO의 콘솔 출력을 줄입니다.
    results = yolo_face_model(image_crop, conf=confidence_threshold, verbose=False)

    faces_in_original_coords = []
    for r in results:
        for box in r.boxes:
            score = float(box.conf[0]) # 신뢰도 점수
            class_id = int(box.cls[0].item()) # 객체 분류 ID
            class_name = r.names[class_id] # 객체 분류 이름

            face_crop_bbox_tensor = box.xyxy[0].tolist()
            face_crop_bbox = [int(coord) for coord in face_crop_bbox_tensor] # 정수형으로 변환

            logger.debug(f"face_crop_bbox: {face_crop_bbox}")

            obj_x1_orig, obj_y1_orig, _, _ = obj_bbox_origin_xyxy
            face_orig_bbox = [
                obj_x1_orig + face_crop_bbox[0],
                obj_y1_orig + face_crop_bbox[1],
                obj_x1_orig + face_crop_bbox[2],
                obj_y1_orig + face_crop_bbox[3],
            ]

            # TODO: 이 부분에 얼굴 크롭 이미지에서 임베딩을 추출하는 코드 추가 필요
            # actual_embedding = extract_embedding(image_crop, face_crop_bbox) # 예시 함수 호출

            faces_in_original_coords.append({
                face_box_xyxy_key: face_orig_bbox,
                face_confidence_key: score,
                face_class_id_key: class_id, # class_id 할당
                face_class_name_key: class_name, # class_name 할당
                face_label_key: "face", # class_name 할당
                # face_embedding_key: actual_embedding, # 실제 임베딩 벡터 할당
                # 'box' 또는 'score' 키가 필요한 경우 추가
                # "box": face_orig_bbox, # 예시: 'box' 키가 필요하다면 이렇게
                # "score": score # 예시: 'score' 키가 신뢰도라면 이렇게
            })
            logger.debug(f"face_orig_bbox: {face_orig_bbox}")

    return faces_in_original_coords

def detect_face(
    base_config: Dict[str, Any],
    input_path: Path,
    undetect_objects_dir: Path, 
    undetect_list_path : Path, 
    yolo_face_model: YOLO, # YOLO 모델 객체를 파라미터로 추가
    model_info: Dict
    ):
    # 이미지 파일 읽기 (OpenCV 사용)
    logger.info(f"이미지 파일 읽기-detect_face- (OpenCV 사용) 시작")

    # 0. 일힐 준비
    # 0.0. 통계 정보를 담을 딕셔너리 초기화
    status = {k: {"value": v["value"], "msg": v["msg"]} for k, v in DEFAULT_STATUS_TEMPLATE.items()}
    
    # 0.1. JSON정보에서 찾아야 하는 KEY를 설정정보에서 가저오기 "detected_obj"
    try:
        # 1. config에서 키 불러오기
        user_profile_cfg = base_config.get("user_profile", {})
        image_info_cfg = base_config.get("image_info_key", {})
        object_info_cfg = base_config.get("object_info_key", {})

        # 사용자 정보
        user_profile_key        = user_profile_cfg.get("key", "user_profile")
        user_name_key           = user_profile_cfg.get("username", {}).get("key", "user_name")
        user_name_val           = user_profile_cfg.get("username", {}).get("name", "unknown_user")
        email_key               = user_profile_cfg.get("email_info", {}).get("key", "email")
        email_val               = user_profile_cfg.get("email_info", {}).get("email", "unknown@email.com")

        # 이미지 정보
        image_info_key          = image_info_cfg.get("key", "image_info")
        image_name_key          = image_info_cfg.get("image_name_key", "image_name")
        image_path_key          = image_info_cfg.get("image_path_key", "image_path")
        image_hash_key          = image_info_cfg.get("image_hash_key", "image_hash")
        resolution_key          = image_info_cfg.get("resolution", {}).get("key", "resolution")
        width_key               = image_info_cfg.get("resolution", {}).get("width_key", "width")
        height_key              = image_info_cfg.get("resolution", {}).get("height_key", "height")
        channels_key            = image_info_cfg.get("resolution", {}).get("channels_key", "channels")

        # 객체 정보
        object_name_key         = object_info_cfg.get("object_name_key", "detected_obj")
        object_box_xyxy_key     = object_info_cfg.get("object_box_xyxy_key", "box_xyxy")
        object_box_xywh_key     = object_info_cfg.get("object_box_xywh_key", "box_xywh")
        object_confidence_key   = object_info_cfg.get("object_confidence_key", "confidence")
        object_class_id_key     = object_info_cfg.get("object_class_id_key", "class_id")
        object_class_name_key   = object_info_cfg.get("object_class_name_key", "class_name")
        object_label_key        = object_info_cfg.get("object_label_key", "label")
        object_index_key        = object_info_cfg.get("object_index_key", "index")

        # 얼굴 정보 키는 object_info_cfg 하위의 "face_info_key" 섹션에서 가져옵니다.
        face_info_key_cfg       = object_info_cfg.get("face_info_key", {}) # 이 변수를 detect_faces_in_crop_yolo_internal에 전달
        face_name_key           = face_info_key_cfg.get("face_name_key", "detected_face_crop") # 키 이름 예시 수정
        face_box_xyxy_key       = face_info_key_cfg.get("face_box_xyxy_key", "box_xyxy")
        face_confidence_key     = face_info_key_cfg.get("face_confidence_key", "confidence")
        face_class_id_key       = face_info_key_cfg.get("face_class_id_key", "class_id") # 키 이름 일관성
        face_class_name_key     = face_info_key_cfg.get("face_class_name_key", "class_name") # 키 이름 일관성
        face_label_key          = face_info_key_cfg.get("face_label_key", "label")
        face_embedding_key      = face_info_key_cfg.get("face_embedding_key", "embedding")

    except Exception as e:
        logger.error(f"detect_object: base_config에서 설정 키 로딩 실패 ('{input_path.name}'): {e}")
        return status

    # 1. 이미지 정보 load.(설정 기반으로 JSON 파일 읽기)
    # read_json_with_config_keys는 정규화된 키로 데이터를 반환합니다.
    # 예: "image_path", "image_hash", "resolution": {"width": W, "height": H}, "detected_objects"
    parsed_json_data = read_json_with_config_keys(config=base_config, json_path=input_path)
    if not parsed_json_data:
        status["error_input_file_read"]["value"] += 1
        try:
            if add_path_to_list_file(input_path, undetect_list_path):
                logger.warning(f"로드 실패/부적합 JSON 파일 경로 '{input_path}'를 '{undetect_list_path}'에 추가했습니다.")
            else:
                logger.warning(f"로드 실패/부적합 JSON 파일 경로 '{input_path}'를 '{undetect_list_path}'에 추가하는 데 실패했습니다.")
            status["error_copied_input_file"]["value"] += 1 # 읽는데 오류가 난 입력파일을 보관히가 위해 옮기다 오류  
        except Exception as e:
            logger.error(f"add_path_to_list_file 처리 오류: {undetect_list_path} {e}")

        logger.warning(f"read_json_with_config_keys를 사용하여 JSON 파일 읽기 실패: {input_path}. 건너뜁니다.")
        return status

    # 2. 읽어온 JSON 데이터에서 필요한 정보 추출
    # parsed_json_data는 이미 정규화된 키를 가지고 있을 것으로 예상됩니다.
    # 예: "user_name", "email", "image_name", "image_path", "image_hash", 
    #     "resolution": {"width": W, "height": H, "channels": C}, "detected_objects"

    # 사용자 프로필 정보 (정규화된 키 사용)
    # json_user_name_from_parsed = parsed_json_data.get("user_name") # 예시, 실제 키는 read_json_with_config_keys 반환값 확인
    # json_email_from_parsed = parsed_json_data.get("email")

    # 이미지 정보 (정규화된 키 사용)
    logger.debug(f"parsed_json_data: {parsed_json_data}")
    json_user_profile_data = parsed_json_data.get(user_profile_key, {})
    json_image_info_data = parsed_json_data.get(image_info_key, {})
    detected_objects_list = parsed_json_data.get(object_name_key, [])

    json_image_name_val         = json_image_info_data.get(image_name_key)
    json_image_path_val_str     = json_image_info_data.get(image_path_key) # read_json_with_config_keys가 반환하는 표준 키
    json_image_hash_val         = json_image_info_data.get(image_hash_key)
    json_resolution_name_val    = json_image_info_data.get(resolution_key, {})
    json_width_val              = json_image_info_data.get(resolution_key, {}).get(width_key, None) 
    json_height_val             = json_image_info_data.get(resolution_key, {}).get(height_key, None) # 오타 수정 및 height_key 사용
    json_channels_val           = json_image_info_data.get(resolution_key, {}).get(channels_key, None)

    logger.debug(f"json_user_profile_data: {json_user_profile_data}")
    logger.debug(f"json_image_info_data: {json_image_info_data}")
    logger.debug(f"detected_objects_list: {detected_objects_list}")
    # 객체 목록 (정규화된 키 사용)
    # detected_objects_list는 read_json_with_config_keys에 의해 "detected_objects" 키로 정규화되어 반환됨

    if not json_image_path_val_str:
        logger.error(f"이미지 파일 없음: {image_path_key} (JSON: {input_path.name} 내 image_path 참조) 건너뜁니다.")
        status["error_target_file_get"]["value"] += 1
        add_path_to_list_file(input_path, undetect_list_path)
        return status

    # 3. 원본 이미지 로드 및 정보 확인
    json_image_path = Path(json_image_path_val_str)
    if not json_image_path.is_file():
        logger.error(f"이미지 파일 없음: (JSON: {json_image_path_val_str} 참조) 건너뜁니다.")
        status["error_target_file_get"]["value"] += 1
        add_path_to_list_file(input_path, undetect_list_path)
        return status

    # 4. 이미지 파일 읽기 전에 이미지 회전
    if rotate_image_if_needed(str(json_image_path)) == False:
        logger.warning(f"이미지 파일 회전('rotate_image_if_needed')중 오류 발견. 건너뜁니다.")
        status["error_image_rotation"]["value"] += 1
        return status

    # 4.1. 이미지 파일 읽기
    actual_image = cv2.imread(str(json_image_path))
    if actual_image is None:
        status["read_input_files_error"]["value"] += 1
        logger.warning(f"이미지 파일 '{json_image_path}'을 읽을 수 없습니다. 건너뜁니다.")
        add_path_to_list_file(json_image_path, undetect_list_path) # 이미지 로드 실패도 기록
        return status
    else:
        logger.info(f"이미지 파일: {json_image_path}를 읽었습니다.")
        # 실제 이미지에서 너비, 높이, 채널 정보 가져오기 (JSON 정보가 없을 경우 대비)
        actual_height, actual_width, actual_channels = actual_image.shape

    logger.info("       actual_height:{actual_height},      actual_width:{actual_width},      actual_channels:{actual_channels}")
    # JSON에서 읽은 해상도 정보와 실제 이미지 해상도 중 유효한 값 사용
    # 해상도 검증 (JSON 값과 실제 이미지 값 비교)
    if json_width_val is not None and json_width_val != actual_width:
        logger.warning(f"JSON 너비({json_width_val})와 실제 이미지 너비({actual_width}) 불일치: {json_image_path.name}")
    if json_height_val is not None and json_height_val != actual_height: # 변수명 오타 수정
        logger.warning(f"JSON 높이({json_height_val})와 실제 이미지 높이({actual_height}) 불일치: {json_image_path.name}") # f-string 변수 수정
    if json_channels_val is not None and json_channels_val != actual_channels:
        logger.warning(f"JSON 채널({json_channels_val})과 실제 이미지 채널({actual_channels}) 불일치: {json_image_path.name}")
        return status
    logger.info(f"      json_height_val:{json_height_val},   json_width_val:{json_width_val},  json_channels_val:{json_channels_val}") # 로거 변수명 수정

    final_img_width = json_width_val if json_width_val is not None else actual_width
    final_img_height = json_height_val if json_height_val is not None else actual_height # 변수명 오타 수정
    final_img_channels = json_channels_val if json_channels_val is not None else actual_channels
    logger.info("    final_img_height:{final_img_height},final_img_width:{final_img_width},final_img_channels:{final_img_channels}")

    # 5. JSON에서 읽어온 객체 목록 처리
    status["total_object_count"]["value"] = len(detected_objects_list)
    if status["total_object_count"]["value"] > 0 :
        status["get_object_crop"]["value"] += 1 # 객체가 하나라도 있으면 카운트

    logger.info(f"처리 대상 JSON: {input_path.name}, 이미지: {json_image_path.name}")
    logger.info(f"검출된 객체 수 (JSON 기반): {status['total_object_count']['value']}")

    # 6. 각 객체별 얼굴 탐지
    updated_objects_data_list = []
    for obj_idx, obj_dict_data in enumerate(detected_objects_list):
        # obj_dict_data는 이미 정규화된 키("box_xyxy", "class_name" 등)를 가지고 있을 것으로 예상 (read_json_with_config_keys의 반환값에 따라 다름)
        # 하지만, object_utils.py의 read_json_with_config_keys는 detected_objects 내부의 각 객체는 원본 키 구조를 유지함.
        # 따라서 base_config에서 가져온 object_box_xyxy_key 등을 사용해야 함.
        obj_bbox_xyxy = obj_dict_data.get(object_box_xyxy_key)

        if not obj_bbox_xyxy or len(obj_bbox_xyxy) != 4:
            logger.warning(f"파일: {input_path.name}, 객체 ID: {obj_idx:3}의 bbox 좌표({object_box_xyxy_key})가 없거나 길이가 4가 아님: {obj_bbox_xyxy}")
            status["error_object_bbox_cnt"]["value"] += 1
            updated_objects_data_list.append(obj_dict_data) # 원본 데이터 유지
            continue

        try:
            x1, y1, x2, y2 = map(int, obj_bbox_xyxy)
        except (ValueError, TypeError) as e_map:
            logger.warning(f"파일: {input_path.name}, 객체 ID: {obj_idx:3}의 bbox 좌표 {obj_bbox_xyxy}를 정수로 변환 실패: {e_map}")
            status["error_object_bbox"]["value"] += 1
            updated_objects_data_list.append(obj_dict_data) # 원본 데이터 유지
            continue

        # 좌표 유효성 검사 및 크롭
        # obj_img_h, obj_img_w = actual_image.shape[:2] # 이미 위에서 actual_height, actual_width 가져옴
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(actual_width, x2), min(actual_height, y2)

        if x1_c >= x2_c or y1_c >= y2_c:
            logger.warning(f"객체 ID: {obj_idx:3}의 좌표 [{x1_c},{y1_c},{x2_c},{y2_c}]가 이미지 {json_image_path.name}에 대해 유효하지 않은 크롭 생성.")
            status["error_object_bbox_posit"]["value"] += 1
            updated_objects_data_list.append(obj_dict_data) # 원본 데이터 유지
            continue

        object_crop_image = actual_image[y1_c:y2_c, x1_c:x2_c]

        # 객체 크롭 내에서 얼굴 탐지
        faces_in_object = []
        if object_crop_image.size > 0: # 크롭된 이미지가 비어있지 않은 경우에만 탐지 시도
            try:
                faces_in_object = detect_faces_in_crop_yolo_internal(
                    output_face_info_config = face_info_key_cfg, # base_config에서 가져온 얼굴 정보 키 설정
                    image_crop=object_crop_image,
                    obj_bbox_origin_xyxy=[x1_c, y1_c, x2_c, y2_c],
                    yolo_face_model=yolo_face_model,
                    confidence_threshold=float(model_info["confidence_threshold"])
                )
            except Exception as e_face_detect: # 구체적인 예외 처리
                logger.error(f"객체 ID: {obj_idx:3}, detect_faces_in_crop_yolo_internal 호출 중 오류: {e_face_detect}", exc_info=True)
                status["error_faces_in_object"]["value"] += 1 # 얼굴 검출 내부 오류 카운트
        else:
            logger.warning(f"객체 ID: {obj_idx:3}, 크롭된 이미지가 비어있어 얼굴 탐지를 건너<0xEB><0><0x8F><0xBB>니다.")
            status["error_object_crop"]["value"] +=1

        # 탐지된 얼굴 정보 추가 (가장 신뢰도 높은 얼굴 1개)
        if faces_in_object:
            # detect_faces_in_crop_yolo_internal이 신뢰도 순으로 정렬된 리스트를 반환한다고 가정
            top_face = faces_in_object[0]
            # obj_dict_data는 현재 객체의 정보를 담고 있는 딕셔너리
            # 여기에 설정에서 정의된 face_name_key (예: "detected_face_crop")를 사용하여 얼굴 정보 추가
            obj_dict_data[face_name_key] = top_face
            status["detect_faces_in_object"]["value"] += 1
            logger.debug(f"객체 ID: {obj_idx:3}, 얼굴 찾음. 추가된 얼굴 정보 키: {face_name_key}, 내용: {top_face}")
        else:
            # 얼굴이 검출되지 않은 경우, 기존 obj_dict_data에 face_name_key가 없을 수 있음 (또는 빈 리스트/None으로 설정 가능)
            logger.debug(f"객체 ID: {obj_idx:3}, 얼굴이 검출되지 않았습니다.")

        updated_objects_data_list.append(obj_dict_data)
        logger.debug(f"객체 ID {obj_idx:3} 처리 완료. 현재까지 얼굴 검출 성공 객체 수: {status['detect_faces_in_object']['value']}")

    # 7. 최종 JSON 업데이트 및 저장
    # image_hash_value 결정
    final_image_hash = json_image_hash_val
    if not final_image_hash:
        logger.warning(f"JSON 파일 '{input_path.name}'에 이미지 해시가 없습니다. 다시 계산합니다.")
        if json_image is not None:
            final_image_hash = compute_sha256(actual_image)
        else: # 이미지가 로드되지 않은 극단적인 경우
            final_image_hash = "N/A"

    try:
        # 3. # write_json_from_config는 전체 JSON 구조를 base_config(json_keys)에 따라 새로 만듭니다.
        # user_profile 등의 정보도 base_config에 정의되어 있다면 함께 저장됩니다.
        # 이 함수는 parsed_json_data에서 user_profile 정보를 가져와서 사용할 수 있도록 수정되어야 할 수 있습니다.
        # 또는, write_json_from_config가 user_profile 정보를 직접 받도록 수정할 수 있습니다.
        # 현재는 user_profile 정보는 base_config의 기본값을 사용하게 됩니다.
        write_json_from_config(
            config=base_config, # json_keys 전체 설정
            image_path=json_image_path, # 원본 이미지 경로 (Path 객체)
            image_hash=final_image_hash,
            width=final_img_width,
            height=final_img_height,
            channels=final_img_channels,
            detected_objects=updated_objects_data_list, # 얼굴 탐지 결과가 포함된 최종 객체 목록
            json_path=input_path # 원본 JSON 파일 덮어쓰기
        )
        
        status["files_json_update"]["value"] += 1
        logger.info(f"업데이트된 JSON 파일 저장 성공 (덮어쓰기): {input_path}")
    except Exception as e_save_json:
        status["error_json_update"]["value"] += 1
        logger.error(f"write_json_from_config를 사용하여 업데이트된 JSON 파일 저장 실패: {input_path}, 오류: {e_save_json}", exc_info=True)

    return status

    # try:
    #     image_path = Path(image_path_str) # 이제 image_path_str는 유효한 문자열임이 보장됨
    #     if not image_path.is_file():
    #         logger.error(f"이미지 파일 없음: {image_path} (JSON: {input_path} 참조) 건너뜁니다.")
    #         status["error_target_file_get"]["value"] += 1 # 읽기 오류 파일 수 증가
    #         return  status # 파일 읽기 실패 시 다음 파일로 이동
    #
    # except Exception as e:
    #     logger.error(f"json파일에서 설정 값{image_path_key} 가져오기 오류: {e}")
    #     status["error_target_file_get"]["value"] += 1 # 또는 더 구체적인 오류 키 사용
    #     add_path_to_list_file(input_path, undetect_list_path) # 실패 목록에 추가 고려
    #     return status
    #
    # # 4.2. 이미지 파일 읽기 전에 이미지 회전
    # if rotate_image_if_needed(str(image_path)) == False: # input_path는 Path 객체이므로 문자열로 변환하여 전달
    #     logger.warning(f"이미지 파일 회전('rotate_image_if_needed')중 오률발견. 건너뜁니다.")
    #     status["error_image_rotation"]["value"] += 1
    #     return status # 파일 읽기 실패 시 다음 파일로 이동
    #
    # # 4.3. 이미지 파일 읽기 부분 ...
    # actual_image = cv2.imread(str(image_path))
    # if actual_image is None:
    #     status["read_input_files_error"]["value"] += 1 # 읽기 오류 파일 수 증가        
    #     logger.warning(f"이미지 파일 '{image_path}'을 읽을 수 없습니다. 건너뜁니다.")
    #     return  status # 파일 읽기 실패 시 다음 파일로 이동
    # else:
    #     logger.info(f"이미지 파일: {image_path}를 읽었습니다..")
    #     height, width, channels = actual_image.shape # 변수명 수정: frame -> actual_image
    #
    # # 5. json파일에서 읽어야 하는 object_list를 가저옴.
    # logger.info(f"json파일에서 찾을 값: {object_name_key}") # object_name_key 사용
    # try:
    #     # 객체 리스트 키
    #     # object_info_key_cfg 변수는 이미 위에서 base_config로부터 로드됨. 여기서는 json_file_cfg에서 객체 리스트를 가져와야 함.
    #     # object_name_key는 base_config에서 가져온, 실제 객체 리스트를 가리키는 키 이름임.
    #     detected_objects_list = json_file_cfg.get(object_name_key, None)
    #     if not isinstance(detected_objects_list, list):
    #         logger.error(f"JSON 파일 '{input_path.name}'에 '{object_name_key}' 섹션이 없거나 딕셔너리가 아닙니다. 데이터: {json_file_cfg}")
    #         status["error_target_file_get"]["value"] += 1
    #         add_path_to_list_file(input_path, undetect_list_path) # 실패 목록에 추가 고려
    #         return status
    #     logger.debug(f"detected_objects_list: {detected_objects_list}")
    # except Exception as e: # 이 try-except는 detected_objects_list를 가져오는 부분으로 옮겨지는 것이 더 적절해 보입니다.
    #     logger.error(f"json파일에서 객체 리스트 ('{object_name_key}') 가져오기 오류: {e}")
    #     return status # 파일 읽기 실패 시 다음 파일로 이동
    #
    # # 5. json파일에서 읽어야 하는 정보를 가저옴.
    # status["total_object_count"]["value"] = len(detected_objects_list)
    # status["get_object_crop"]["value"] += 1 # 읽기 오류 파일 수 증가        
    #
    # logger.info(f"검출대상: {object_name_key}이 {status['total_object_count']['value']}개 있습니다.")
    #
    # # 6. 처리대상 OBJECT는 하나씩 처리합니다.
    # logger.warning(f"==================================================================")
    # logger.warning(f"처리하는 파일 JSON: {input_path.name},이미지: {image_path.name}")
    # logger.warning(f"detected_objects_list: {detected_objects_list}\n")
    # logger.warning(f"==================================================================")
    #
    # updated_objects_data_list = []
    # for obj_idx, obj_dict_data in enumerate(detected_objects_list): # 객체 인덱스 추가 및 튜플 언패킹
    #     # obj_data는 원본 객체 하나의 정보(딕셔너리)를 담고 있습니다.
    #     # 예: {"box_xyxy": [...], "class_name": "person", ...}
    #     # `object_detector.py`는 객체 bbox를 "box_xyxy"로 저장함
    #
    #     try: # 수정: obj_dict_data 사용
    #         obj_bbox_xyxy = obj_dict_data.get(object_box_xyxy_key) # 수정: obj_dict_data 사용
    #         bbox_count = len(obj_bbox_xyxy)
    #     except (ValueError, TypeError) as e_map:
    #         logger.warning(f"파일:{input_path.name}, 객체 ID: {obj_idx:3}의 bbox 좌표 {obj_bbox_xyxy}를 정수로 만들지 못했습니다: {e_map}")
    #         status["error_object_bbox"]["value"] += 1 # 읽기 오류 파일 수 증가        
    #         continue
    #
    #     if not obj_bbox_xyxy or bbox_count != 4:
    #         logger.warning(f"파일:{input_path.name}, 객체 ID: {obj_idx:3}의 가저온 bbox 좌표수가 맞지 않습니다.")
    #         # 원본 객체 데이터를 그대로 updated_objects_data_list에 추가합니다.
    #         updated_objects_data_list.append(obj_dict_data) # 원본 객체 데이터 유지
    #         status["error_object_bbox_cnt"]["value"] += 1 # 읽기 오류 파일 수 증가        
    #         continue
    #
    #     try:
    #         # bbox 좌표를 정수형으로 변환
    #         x1, y1, x2, y2 = map(int, obj_bbox_xyxy)
    #     except (ValueError, TypeError) as e_map:
    #         logger.warning(f"파일:{input_path.name}, 객체 ID: {obj_idx:3}의 bbox 좌표를 정수로 만들지 못했습니다: {e_map}")
    #         updated_objects_data_list.append(obj_data) # 원본 객체 데이터 유지
    #         status["error_object_bbox_cnt"]["value"] += 1 # 읽기 오류 파일 수 증가        
    #         continue
    #
    #     # 좌표가 이미지 경계 내에 있고 유효한지 확인
    #     obj_img_h, obj_img_w = actual_image.shape[:2]
    #     logger.debug(f"원본이미지 크기 obj_img_h:{obj_img_h}, obj_img_w:{obj_img_w}, 검출된 object 수 :{bbox_count}, 정수 좌표 : {obj_bbox_xyxy}")
    #
    #     # 크롭 좌표를 이미지 경계 내로 제한
    #     x1_c, y1_c = max(0, x1), max(0, y1)
    #     x2_c, y2_c = min(obj_img_w, x2), min(obj_img_h, y2)
    #
    #     if x1_c >= x2_c or y1_c >= y2_c: # 크롭 영역이 유효하지 않은 경우
    #         logger.warning(f"객체 ID: {obj_idx:3}의 좌표[{x1_c},{y1_c},{x2_c},{y2_c}]가 이미지 {image_path.name}에 대해 맞지 않습니다.")
    #         updated_objects_data_list.append(obj_dict_data) # 원본 객체 데이터 유지
    #         status["error_object_bbox_posit"]["value"] += 1 # 읽기 오류 파일 수 증가        
    #         continue
    #
    #     # 원본 이미지에서 객체 영역 크롭
    #     object_crop_image = actual_image[y1_c:y2_c, x1_c:x2_c]
    #
    #     # 이 객체 크롭 내에서 얼굴 탐지
    #     try:
    #         faces_in_object = detect_faces_in_crop_yolo_internal(
    #             image_crop=object_crop_image,
    #             obj_bbox_origin_xyxy=[x1_c, y1_c, x2_c, y2_c], # 크롭 영역의 원본 이미지 내 시작 좌표 (x1, y1)
    #             yolo_face_model=yolo_face_model,
    #             confidence_threshold=float(model_info["confidence_threshold"])
    #         )
    #     except (ValueError, TypeError) as e:
    #         logger.warning(f"객체 ID: {obj_idx:3}, detect_faces_in_crop_yolo_internal 호출 중 오류: {e}")
    #
    #     # 2. 가장 신뢰도 높은 얼굴 1개만 사용 (있을 경우)
    #     if faces_in_object:
    #         logger.debug(f"객체 ID: {obj_idx:3}, 얼굴을 찾음. faces_in_object:{faces_in_object}")
    #         top_face = faces_in_object[0]  # 신뢰도 순으로 정렬된 결과이므로 첫 번째가 최고 신뢰도 # 수정: obj_dict_data 사용
    #         obj_dict_data[face_name_key] = top_face # 수정: obj_dict_data 사용
    #         status["detect_faces_in_object"]["value"] += 1
    #     else:
    #         status["error_faces_in_object"]["value"] += 1
    #         logger.warning(f"객체 ID: {obj_idx:3}, 얼굴이 검출되지 않았습니다.")
    #
    #     updated_objects_data_list.append(obj_dict_data) # 수정: obj_dict_data 사용
    #     logger.debug(f'객체 ID {obj_idx:3} 처리 완료, {status["detect_faces_in_object"]["value"]} / {status["total_object_count"]["value"]} 개 검출')
    #
    # # logger.debug(f'JSON 파일 "{input_path.name}"에서 총 {status["detect_faces_in_object_success"]["value"]} / {status['total_object_count']['value']} 개 검출')
    # #    actual_image.close()
    # logger.warning(f"json_file_cfg: {json_file_cfg}")
    # if status["detect_faces_in_object"]["value"] != status['total_object_count']['value']:
    #     status["unmatched_object_number"]["value"] += 1
    #     logger.warning(f"처리된 객체 수({status['get_object_crop']['value']})와 JSON 내 초기 객체 수({status['total_object_count']['value']})가 다릅니다.")
    #
    # # # 최종 JSON 업데이트 및 저장
    # json_file_cfg[object_name_key] = updated_objects_data_list # object_name_key 사용
    # logger.info(f"JSON 파일 '{input_path.name}'의 '{object_name_key}' 키를 업데이트된 객체 목록으로 교체했습니다.")
    # logger.warning(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # logger.warning(f"json_file_cfg: {json_file_cfg}")
    # logger.warning(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # logger.warning(f"******************************************************************")
    #
    # # 업데이트된 JSON 파일 저장
    # # 기존 json정보 가저오기
    # image_hash_from_json = json_file_cfg.get(image_info_key, {}).get(image_hash_key) # image_hash는 image_info 섹션 아래에 있음
    # if image_hash_from_json:
    #     image_hash_value = image_hash_from_json
    # else:
    #     logger.warning(f"JSON 파일 '{input_path.name}'에 '{image_hash_key}'가 없습니다. 이미지 해시를 다시 계산합니다.")
    #     image_hash_value = compute_sha256(actual_image) # Corrected: use actual_image
    #     json_file_cfg[image_hash_key] = image_hash_value # Optionally update json_file_cfg    
    #
    # try:
    #     """
    #     def save_object_json_with_polygon(
    #             image_path: Path,
    #             image_hash: str,
    #             ditec_obj: List[Dict[str, Any]], # ditec_obj 타입 힌트를 좀 더 구체적으로
    #             json_path: Path
    #         ) -> None:
    #     """
    #     save_object_json_with_polygon(
    #         image_path=image_path, # 원본 이미지 경로 (Path 객체)
    #         image_hash=image_hash_value,
    #         detected_objects=json_file_cfg[object_name_key], # 업데이트된 객체 목록, object_name_key 사용
    #         json_path=input_path # 원본 JSON 파일 덮어쓰기
    #         # base_cfg 인자는 save_object_json_with_polygon 함수 시그니처에 없으므로 제거, write_json_from_config 사용 시 필요
    #     ) # base_cfg는 save_object_json_with_polygon 함수 내부에서 JSON 구조 키를 참조하는데 사용됩니다.
    #     status["files_json_update"]["value"] += 1
    #     # save_object_json_with_polygon 함수 내부에 성공 로깅이 이미 있음
    # except Exception as e_save_json:
    #     status["error_json_update"]["value"] += 1 # 업데이트된 JSON 파일 저장 실패: /home/owner/SambaData/OwnerData/train/jsons/temp_1429591366524.json, 오류: name 'json_config' is not defined
    #     logger.error(f"업데이트된 JSON 파일 저장 실패: {input_path}, 오류: {e_save_json}")
    #
    # logger.debug(f"json_file_cfg: {json_file_cfg}")
    # return  status # 파일 읽기 실패 시 다음 파일로 이동

def run_main(cfg: configger):
    # 0. 메인 통계 정보를 담을 딕셔너리 초기화
    status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)
    logger.info("얼굴 탐지 프로세스 (JSON 객체 데이터 기반) 시작.")

    # 1. YOLO 모델 로딩 (기존 코드와 동일)
    try:
        model_info = model_load(cfg)
        # 반환된 튜플을 두 개의 변수로 언패킹
        yolo_model, config_data = model_info

    except Exception as e:
        logger.error(f"설정model_load 중 오류 발생: {e}")
        sys.exit(1)
    logger.debug(f"yolo_model 준비끝")

    # 2. JSON 파일 목록 가져오기 및 순회 처리
    # 2-1. 설정 값 가져오기-
    dataset_base_key_str = "project.paths.datasets"
    try:
        # object_detector.py의 IMAGE 출력물이 있는 디렉토리
        # json파일의 key를 만든다.
        cur_cfg = cfg.get_config(dataset_base_key_str)
        if cur_cfg is None:
            logger.error(f"'{dataset_base_key_str}' 설정 그룹을 찾을 수 없습니다.")
            sys.exit(1)

        input_dir_str = cur_cfg.get("raw_jsons_dir", "raw_jsons_dir")
        input_dir = Path(input_dir_str).expanduser() # 사용자 경로 확장 및 절대 경로로 변환
        logger.info(f"설정된 입력 JSON 디렉토리      (원시 문자열): '{input_dir_str}'")
        logger.info(f"확장 및 확인된 입력 JSON 디렉토리 (절대 경로): '{input_dir}'")
        if not input_dir.exists():
            logger.error(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
            sys.exit(1)

        """
        detected_dir_str = cur_cfg.get('detected_dir', None)
        detected_dir = Path(detected_dir_str).expanduser()
        detected_dir.mkdir(parents=True, exist_ok=True) 

        detected_objects_dir_str = cur_cfg.get('detected_objects_dir', None)
        detected_objects_dir = Path(detected_objects_dir_str).expanduser()
        detected_objects_dir.mkdir(parents=True, exist_ok=True) 
        logger.debug(f"detected_objects_dir: {detected_objects_dir}")
        """

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

    except Exception as e:
        logger.error(f"input_dir_key 가져오기 오류: {e}")
        sys.exit(1)

    # 2-2. 확장자 가저오기-
    supported_json_extensions = ['.json'] # 처리할 JSON 파일 확장자
    logger.info(f"입력 JSON 파일 확장자는: {supported_json_extensions}")

    # 2-5. 처리대상 json파일 갯수 계산하기
    input_file_iterator_for_counting = Path(input_dir).glob("**/*")
    status["total_input_found"]["value"] = sum(1 for p in input_file_iterator_for_counting if p.is_file() and p.suffix.lower() in supported_json_extensions)
    # 로그 메시지 개선: 어떤 디렉토리와 확장자를 검색했는지 명시
    logger.info(f"'{input_dir}' 디렉토리에서 다음 확장자를 가진 파일 검색: {supported_json_extensions}")
    logger.info(f"지원되는 JSON 파일 개수(total_input_found): {status['total_input_found']['value']}")

    if status["total_input_found"]["value"] == 0:
        logger.warning(f"'{input_dir}' 디렉토리에서 검출할 JSON 파일을 찾을 수 없습니다.")
        logger.info("✅ 최종 통계:")
        logger.info(f"   - 탐색된 JSON 파일 총 개수: {status['total_input_found']['value']}")
        logger.info(f"{Path(__file__).name} 정상 종료 (처리할 파일 없음).")
        sys.exit(0)

    # 2-6. 처리대상 json파일 갯수예따른 통계정보 표시할 자릿수 계산하기
    try:
        digit_width = calc_digit_number(status["total_input_found"]["value"])
    except Exception as e:
        digit_width = 0
        logger.error(f"calc_digit_number 값 가져오기 중 오류 발생: {e}")

    logger.info(f"✅ 검출할 json 파일 {status['total_input_found']['value']}개 발견. 숫자 수(digit_width) : {digit_width}")

    # 3. 처리할 JSON 파일 목록 가져오기
    # 3-1. glob 결과를 다시 생성 (실제 처리용)
    input_file_iterator_for_processing = input_dir.glob("**/*")

    # 3-2. json라일 하나씩 처리하기 시작한다.
    for input_path in input_file_iterator_for_processing: # 이터레이터를 순회
        if input_path.is_file(): # 파일인지 다시 확인 (glob 결과는 대부분 파일이지만 안전을 위해)
            status['req_process_count']['value'] += 1 # 처리 시작 파일 카운트 (로그용)
            logger.info(f"[{(status['req_process_count']['value']):{digit_width}}/{status['total_input_found']['value']}] JSON 파일 처리 시작: {input_path.name}")

            # 4. JSON정보중 object별로 얼굴을 검출하고 결과를 JSON하일에 추가한다.
            try:
                """
                def detect_face(
                    base_config: Dict[str, Any],
                    input_path: Path, 
                    undetect_objects_dir: Path, 
                    undetect_list_path: Path, 
                    yolo_face_model: YOLO, # YOLO 모델 객체를 파라미터로 추가
                    model_info: Dict
                    ):
                """
                ret = detect_face(
                    base_config = json_cfg,
                    input_path=input_path, 
                    undetect_objects_dir = undetect_objects_dir, 
                    undetect_list_path = undetect_list_path, 
                    yolo_face_model = yolo_model, # YOLO 모델 객체를 파라미터로 추가
                    model_info = config_data # 모델 관련 기타 설정 정보
                    )
            except Exception as e:
                logger.error(f"JSON 파일 '{input_path}' 처리 중 메인 오류 발생: {e}")
                status["files_processed_main_error"]["value"] += 1
                # 오류 발생 시 다음 파일로 이동하거나 종료할 수 있습니다.
                continue # 다음 파일로 이동

            # ret (detect_face의 결과 status)에서 main_status로 값 누적
            if ret: # ret이 None이 아닌 경우 (정상 반환된 경우)
                for stat_key in DEFAULT_STATUS_TEMPLATE.keys(): # DEFAULT_STATUS_TEMPLATE의 모든 키에 대해
                    if stat_key in ret and stat_key in status:
                        # ret와 main_status 모두 "value" 키를 가지고 숫자 값을 가진다고 가정
                        if "value" in ret[stat_key] and isinstance(ret[stat_key]["value"], (int, float)):
                            status[stat_key]["value"] += ret[stat_key]["value"]
            logger.info(f"[{(status['req_process_count']['value']):{digit_width}}/{status['total_input_found']['value']}] JSON 파일 처리 완료: {input_path.name}")

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

# 스크립트가 직접 실행될 때 run_object_detection 함수 호출
if __name__ == "__main__":
    """
    설정 파일을 로드하고 YOLO 객체 검출을 수행하는 함수에게 일을 시킴
    """
    # 0. 애플리케이션 아귀먼트 있으면 갖오기
    logger.info("얼굴 탐지기 (JSON 객체 데이터 기반) 애플리케이션 시작.")
    parsed_args = get_argument()
    logger.info(f"실행시 필요한 파라메터를 받았습니다.{parsed_args}")

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
        logger.info(f"로그경로를 결정했습니다.: {vars(parsed_args)}")
        logger.info(f"애플리케이션({script_name}) 시작")
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
    except Exception as e:
        logger.critical(f"치명적 오류: Configger 초기화 중 오류 발생 - {e}")
        sys.exit(1)

        # 3. 메인 얼굴 탐지 프로세스 실행
    try:
        run_main(cfg_object)

    except FileNotFoundError as e_fnf: # 설정 파일 관련 오류
        logger.error(f"설정 파일 오류: {e_fnf}")
    except Exception as e_main: # 그 외 메인 애플리케이션 흐름에서 발생한 예외
        logger.error(f"메인 애플리케이션 흐름에서 처리되지 않은 오류 발생: {e_main}")
    finally:
        logger.info("얼굴 탐지기 (JSON 객체 데이터 기반) 애플리케이션 종료.")
        # SimpleLogger의 종료 처리 (예: 비동기 핸들러 사용 시)
        if hasattr(logger, 'shutdown') and callable(logger.shutdown):
             logger.shutdown()
        exit(0)
