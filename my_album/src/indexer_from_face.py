# src/face_indexer_from_face.py

# 표준 라이브러리 임포트
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Set

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

# 외부 라이브러리 임포트
try:
    import cv2 # 메타데이터 추출 시 원본 이미지 경로 등을 참조할 수 있으므로 유지 (직접 사용 안 할 수도 있음)
    import dlib # dlib 모델 로드 부분은 유지 (향후 다른 기능에 필요할 수 있음)
    import numpy as np
    import faiss
except ImportError as e:
    logger.error(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
    logger.info("Poetry 환경에서 'poetry install'을 실행하여 필요한 라이브러리를 설치해주세요.")
    sys.exit(1)

# --- 전역 변수: 통계용 변수 사용을 위해해 ---
DEFAULT_STATUS_TEMPLATE  = {
    "total_input_found":         {"value": 0,  "msg": "총 입력 파일 수 (지원 확장자 기준)"},
    "error_input_file_read":        {"value": 0,  "msg": "입력 파일 읽기 오류 수"},
    "req_process_count":         {"value": 0,  "msg": "총 처리 요청 파일 수"},
    "error_extension":   {"value": 0,  "msg": "지원되지 않는 확장자로 건너뛴 파일 수"},
    "error_image_rotation":          {"value": 0,  "msg": "이미지 회전중 오류 발생 파일 수"},
    "error_target_file_get":        {"value": 0,  "msg": "처리대상(image or json) 파일 읽기 오류 수"},
    "error_input_file_process":        {"value": 0,  "msg": "입렫파일 읽기 오류 수"},
    "request_embedding":        {"value": 0,  "msg": "embdding 요청 수"},
    "error_embedding_empty_target":        {"value": 0,  "msg": "embdding 오류 수 - 처리대상이 없음"},
    "error_embeddings_array":        {"value": 0,  "msg": "embdding 오류 수 - 빈 임베딩인"},
    "error_embeddings_none_config":        {"value": 0,  "msg": "embdding 오류 수 - 설절값이 없음"},
    "error_embeddings_deff_config":        {"value": 0,  "msg": "embdding 오류 수 - 설절값과 다름"},
    "error_embeddings_deff_spec":        {"value": 0,  "msg": "embdding 오류 수 - 규격-shap이 다름"},
    "error_embeddings_none_learn":        {"value": 0,  "msg": "embdding 오류 수 - 학습값이 없음"},
    "error_embeddings_deff_drainage":        {"value": 0,  "msg": "embdding 오류 수 - 배수가 아님"},
    "error_embeddings_objebt_gen":        {"value": 0,  "msg": "embdding 오류 수 - 객체생성실패"},
    "error_embeddings_read_index":        {"value": 0,  "msg": "embdding 오류 수 - 객체생성실패"},
    "request_save_index":        {"value": 0,  "msg": "인덱스 저장 요청 수"},
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

# 이 스크립트에서는 직접 사용하지 않지만, 설정 파일 로직 등에서 참조할 수 있으므로 유지합니다.
# 또는, dlib 모델 로드 관련 부분을 완전히 제거해도 됩니다. 여기서는 일단 유지합니다.
def load_dlib_models(cfg_obj) -> dict | None:
    """
    dlib의 얼굴 인식 관련 모델 파일들을 로드하여 딕셔너리로 반환합니다.

    Args:
        cfg_obj (configger): 설정 파일 내용을 담고 있는 configger 객체.

    Returns:
        dict | None: 로드된 dlib 모델 객체들을 담은 딕셔너리 또는 로드 실패 시 None.
                     반환되는 딕셔너리 키: 'face_detector', 'shape_predictor', 'face_recognizer'
    """
    models_key_str = 'models'
    face_recognition_key_str = f'{models_key_str}.face_recognition'

    # 모델 디렉토리 경로 가져오기
    models_dir_key_str = f'{face_recognition_key_str}.models_dir'
    # get_path 내부에서 존재 여부 확인 및 로깅이 되므로, 반환 값이 None인지 여부만 체크합니다.
    models_dir_str = cfg_obj.get_path(models_dir_key_str, ensure_exists=True) # ensure_exists=True 전달
    models_dir = Path(models_dir_str)
    if models_dir is None:
        # get_path에서 이미 로깅했으므로 추가 로깅은 생략 가능
        # logger.warning(f"dlib 모델 디렉토리 '{models_dir_key_str}'를 가져오지 못했습니다.")
        return None
    logger.debug(f"모델이 있는 위치(models_dir): {models_dir}")

    # 얼굴 인식 모델 파일 이름 가져오기 및 경로 생성
    face_rec_model_name_key = f'{face_recognition_key_str}.face_rec_model_name'
    face_rec_model_name = cfg_obj.get_value(face_rec_model_name_key) # get_value에서 Path 객체 반환 가정
    if face_rec_model_name is None:
        logger.warning(f"dlib 얼굴 인식 모델 파일 이름 설정 값 '{face_rec_model_name_key}'을 가져오지 못했습니다.")
        return None
    # get_value에서 Path 객체를 반환하므로 / 연산자 사용 가능
    logger.debug(f"모딜 이름은(face_rec_model_name): {face_rec_model_name}")

    # get_value에서 Path 객체를 반환하므로 / 연산자 사용 가능
    landmark_model_name_key = f'{face_recognition_key_str}.landmark_model_name' # 예시 키 이름
    landmark_model_name = cfg_obj.get_value(landmark_model_name_key) # get_value에서 Path 객체 반환 가정
    if landmark_model_name is None:
        logger.warning(f"dlib 랜드마크 모델 파일 이름 설정 값 '{landmark_model_name_key}'을 가져오지 못했습니다.")
        return None

    # dlib 모델 로드 시도
    face_rec_model_path = models_dir / face_rec_model_name
    landmark_model_path = models_dir / landmark_model_name
    try:
        # 로드된 모델 객체를 함수 내 로컬 변수에 할당
        face_detector_dlib = dlib.get_frontal_face_detector()
        shape_predictor_obj = dlib.shape_predictor(str(landmark_model_path))
        face_recognizer_obj = dlib.face_recognition_model_v1(str(face_rec_model_path))

        logger.debug("dlib 얼굴 인식 관련 모델 로드 완료.")

        # 로드된 모델 객체들을 딕셔너리로 묶어 반환
        loaded_models = {
            'face_detector': face_detector_dlib,
            'shape_predictor': shape_predictor_obj,
            'face_recognizer': face_recognizer_obj
        }
        return loaded_models

    except Exception as e:
        # dlib 모델 로드 자체에서 오류 발생
        logger.error(f"dlib 모델 로드 중 오류 발생: {e}")
        return None # 로드 실패 시 None 반환
from typing import Callable

def get_all_face_data_from_json_batch(
    cfg: configger,
    json_file_path: Path,
    json_key_config: Dict[str, Any],
    process_func: Callable[[np.ndarray, Dict[str, Any]], None]
) -> int:
    """
    JSON 파일에서 얼굴 임베딩과 메타데이터를 하나씩 추출하며
    주어진 처리 함수(process_func)를 통해 외부 배치 컨트롤러에 전달합니다.

    Args:
        cfg: 설정 객체
        json_file_path: 대상 JSON 파일 경로
        json_key_config: JSON 구조 키 맵
        process_func: (임베딩, 메타데이터) → 처리 함수 (예: 배치 누적기)

    Returns:
        처리된 얼굴 수
    """
    logger.info(f"[배치용] JSON 파일 처리 시작: {json_file_path.name}")

    count = 0

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        detected_objects = data.get(
            json_key_config.get("detected_objects_list_key", "detected_obj")
        )
        if not isinstance(detected_objects, list):
            return 0

        # 키 맵 가져오기
        image_path = data.get(json_key_config.get("image_info_keys", {}).get("path", "image_path"))
        image_hash = data.get(json_key_config.get("image_info_keys", {}).get("hash", "image_hash"))

        for obj in detected_objects:
            faces = obj.get(
                json_key_config.get("object_keys", {}).get("face_crops_list_key", "detected_face_crop"), []
            )
            for face in faces:
                embedding_data = face.get(
                    json_key_config.get("face_keys", {}).get("embedding", "embedding")
                )
                if not embedding_data:
                    continue

                try:
                    embedding_np = np.array(embedding_data, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"임베딩 변환 실패: {e}")
                    continue

                metadata = {
                    "source_json_path": str(json_file_path),
                    "original_image_path": image_path,
                    "original_image_hash": image_hash,
                    "face_id": face.get("face_id"),
                    "face_bbox_in_obj": face.get("box"),
                    "embedding_score": face.get("score"),
                    "detected_face_bbox_xyxy": face.get("bbox_xyxy"),
                    "detected_face_confidence": face.get("confidence"),
                    "detected_face_label": face.get("label"),
                    "detected_object_class": obj.get("class_name"),
                    "detected_object_bbox_xyxy": obj.get("box_xyxy")
                }

                # ▶ 처리 함수에 전달 (ex. 배치 누적기)
                process_func(embedding_np, metadata)
                count += 1

        return count

    except Exception as e:
        logger.error(f"파일 처리 오류: {json_file_path.name} - {e}", exc_info=True)
        return 0

def get_all_face_data_from_json_alone(
    cfg:configger, 
    json_file_path: Path
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    주어진 JSON 파일에서 모든 얼굴의 임베딩과 메타데이터를 추출합니다.
    키 이름은 json_key_config 딕셔너리에서 가져옵니다.
    Args:
        json_file_path (Path): 얼굴 정보를 추출할 JSON 파일의 경로.
        json_key_config (Dict[str, Any]): 
                config.yaml의 data_structure_keys 섹션에서 로드한 설정 딕셔너리.
                YAML 구조에 따라 중첩된 딕셔너리일 수 있습니다.
    Returns:
        Tuple[List[np.ndarray], List[Dict[str, Any]]]:
            - 추출된 모든 얼굴 임베딩 리스트 (각 요소는 NumPy 배열).
            - 추출된 모든 얼굴 메타데이터 딕셔너리 리스트.
            - 오류 발생 또는 데이터 없음 시 ([], []) 반환.
    """
    # 0. 일준비
    # 0.1. 통계 정보를 담을 딕셔너리 초기화
    status = {k: {"value": v["value"], "msg": v["msg"]} for k, v in DEFAULT_STATUS_TEMPLATE.items()}

    # 0.1. 결과물을 임시로 담을 디렉토리 만들기 초기화
    embeddings_in_file: List[np.ndarray] = []
    metadatas_in_file: List[Dict[str, Any]] = []

    # # --- config에서 JSON 키 이름 가져오기 ---
    # # json_key_config의 구조에 맞춰 단계적으로 접근합니다.

    # # 최상위 'detected_obj' 리스트의 키 이름
    # 이미지 정보 관련 키 이름들을 가져오기 위한 설정 경로 문자열
    json_keys_str           = 'json_keys'
    image_info_key_str      = f'{json_keys_str}.image_info_key'
    logger.debug(f"image_info_key_str: {image_info_key_str}")

    # 설정에서 실제 키 값들을 가져옵니다.
    try:

        image_name_key_str      = f'{image_info_key_str}.name_key'
        image_name_key          = cfg.get_value(image_name_key_str)
        image_path_key_str      = f'{image_info_key_str}.path_key'
        image_path_key          = cfg.get_value(image_path_key_str)
        image_hash_key_str      = f'{image_info_key_str}.hash_key'
        image_hash_key          = cfg.get_value(image_hash_key_str)

        detected_obj_key_str    = f'{json_keys_str}.detected_obj_key'
        # object_name_key는 실제 JSON 파일 내 객체 리스트를 가리키는 키의 이름 (예: "detected_obj")
        object_list_key_name_str = f'{detected_obj_key_str}.object_name_key' # YAML 설정상 object_name_key가 객체 리스트의 키 이름
        object_list_key_val    = cfg.get_value(object_list_key_name_str)
        logger.debug(f"object_list_key_val (객체 리스트 키): {object_list_key_val}")

        object_info_key_str     = f'{detected_obj_key_str}.object_info_key'
        # 객체 내 클래스 이름과 바운딩 박스 키
        object_class_name_key_config_path = f'{object_info_key_str}.class_name_key'
        object_class_name_key_val = cfg.get_value(object_class_name_key_config_path)
        object_box_xyxy_key_config_path = f'{object_info_key_str}.box_xyxy_key'
        object_box_xyxy_key_val = cfg.get_value(object_box_xyxy_key_config_path)

        detected_face_key_str   = f'{object_info_key_str}.detected_face_key'
        # face_name_key는 객체 내의 얼굴 리스트를 가리키는 키의 이름 (예: "detected_face")
        face_name_key_str       = f'{detected_face_key_str}.face_name_key'
        face_list_in_obj_key_val = cfg.get_value(face_name_key_str) # 객체 내 얼굴 리스트 키
        logger.debug(f"face_list_in_obj_key_val (객체 내 얼굴 리스트 키): {face_list_in_obj_key_val}")

        detected_face_key_str   = f'json_keys.detected_obj_key.face_info_key'
        face_key_list = cfg.get_key_list(detected_face_key_str)

        face_info_key_str       = f'{detected_face_key_str}.face_info_key'
        face_bbox_xyxy_key_str  = f'{face_info_key_str}.bbox_xyxy_key'
        face_bbox_xyxy_key_val  = cfg.get_value(face_bbox_xyxy_key_str)
        face_confidence_key_str = f'{face_info_key_str}.confidence_key'
        face_confidence_key_val = cfg.get_value(face_confidence_key_str)
        face_label_key_str      = f'{face_info_key_str}.label_key'
        face_label_key_val      = cfg.get_value(face_label_key_str)
        face_embedding_key_str  = f'{face_info_key_str}.embedding_key'
        face_embedding_key_val  = cfg.get_value(face_embedding_key_str)
        face_face_id_key_str    = f'{face_info_key_str}.face_id_key'
        face_face_id_key_val    = cfg.get_value(face_face_id_key_str)
        face_box_key_str        = f'{face_info_key_str}.box_key'
        face_box_key_val        = cfg.get_value(face_box_key_str)
        face_score_key_str      = f'{face_info_key_str}.score_key'
        logger.debug(f"face_score_key_str: {face_score_key_str}")
        face_score_key_val      = cfg.get_value(face_score_key_str)

    except Exception as e:
        logger.error(f"설정 파일에서 JSON 키 값 로드 중 오류 발생 (파일: {json_file_path.name}): {e}", exc_info=True)
        return [], []
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # JSON 파일에서 최상위 객체 리스트를 가져옵니다.
        # object_list_key_val은 "detected_obj"와 같은 문자열 키 이름입니다.
        logger.debug(f"object_list_key_val: '{object_list_key_val}'")
        actual_objects_list = json_data.get(object_list_key_val)

        if not isinstance(actual_objects_list, list):
            logger.warning(f"JSON 파일 '{json_file_path.name}'에 '{object_list_key_val}' 키로 식별되는 리스트가 없거나 형식이 잘못되었습니다.")
            return [], []

        # 이미지 레벨 메타데이터 추출
        original_image_path = json_data.get(image_path_key) # image_path_key는 설정에서 가져온 값
        original_image_hash = json_data.get(image_hash_key) # image_hash_key는 설정에서 가져온 값
        for obj_entry in actual_objects_list: # 객체 리스트를 순회합니다.
            if not isinstance(obj_entry, dict): continue # 각 객체는 딕셔너리여야 합니다.

            # 현재 객체에서 얼굴 리스트를 가져옵니다. face_list_in_obj_key_val은 "detected_face"와 같은 문자열 키 이름입니다.
            faces_in_object_list = obj_entry.get(face_list_in_obj_key_val)
            if not isinstance(faces_in_object_list, list):
                continue

            for face_entry in faces_in_object_list: # 객체 내 얼굴 리스트를 순회합니다.
                if not isinstance(face_entry, dict): continue # 각 얼굴 항목은 딕셔너리여야 합니다.

                embedding_data = face_entry.get(face_embedding_key_val) # 설정에서 가져온 키 사용
                if embedding_data is None:
                    face_id_info = face_entry.get(face_face_id_key_val, 'N/A')
                    logger.debug(f"JSON 파일 '{json_file_path.name}'의 face_id '{face_id_info}'에 '{face_embedding_key_val}' 정보가 없습니다. 건너<0xEB><0><0x8E>니다.")
                    continue
                try:
                    embedding_np = np.array(embedding_data, dtype=np.float32)
                except Exception as e_np:
                    face_id_info = face_entry.get(face_face_id_key_val, 'N/A')
                    logger.warning(f"JSON 파일 '{json_file_path.name}'의 face_id '{face_id_info}' 임베딩 변환 중 오류: {e_np}. 건너<0xEB><0><0x8E>니다.")
                    continue

                metadata = {
                    "source_json_path": str(json_file_path),
                    "original_image_path": original_image_path,
                    "original_image_hash": original_image_hash,
                    "face_id": face_entry.get(face_face_id_key_val),
                    "face_bbox_in_obj": face_entry.get(face_box_key_val),
                    "embedding_score": face_entry.get(face_score_key_val),
                    "detected_face_bbox_xyxy": face_entry.get(face_bbox_xyxy_key_val),
                    "detected_face_confidence": face_entry.get(face_confidence_key_val),
                    "detected_face_label": face_entry.get(face_label_key_val),
                    "detected_object_class": obj_entry.get(object_class_name_key_val),
                    "detected_object_bbox_xyxy": obj_entry.get(object_box_xyxy_key_val)
                }
                embeddings_in_file.append(embedding_np)
                metadatas_in_file.append(metadata)

        return embeddings_in_file, metadatas_in_file

    except FileNotFoundError:
        logger.error(f"JSON 파일 찾기 오류: {json_file_path}", exc_info=True)
        return [], []
    except json.JSONDecodeError:
        logger.error(f"JSON 파일 파싱 오류: {json_file_path}", exc_info=True)
        return [], []
    except Exception as e:
        logger.error(f"JSON 파일 '{json_file_path.name}' 처리 중 예상치 못한 오류: {e}", exc_info=True)
        return [], []

#def add_embeddings_batch(embeddings: List[np.ndarray], metadatas: List[Dict[str, Any]], cfg_obj):
def add_embedding_to_index(embedding: np.ndarray, metadata: Dict[str, Any], cfg_obj):
    """
    단일 얼굴 임베딩과 메타데이터를 받아 FAISS 인덱스에 추가하고 저장합니다.
    저사양 환경에 적합한 방식 (IndexFlatL2 + 덧붙이기 저장)으로 동작합니다.
    """
    status = {k: {"value": v["value"], "msg": v["msg"]} for k, v in DEFAULT_STATUS_TEMPLATE.items()}

    try:
        embedding = embedding.astype('float32').reshape(1, -1)
        embedding_dim = embedding.shape[1]

        # 설정 불러오기
        index_file_path_str = cfg_obj.get_value('indexing.index_file_path')
        metadata_file_path_str = cfg_obj.get_value('indexing.metadata_path')

        if not index_file_path_str or not metadata_file_path_str:
            logger.critical("필수 설정(index_file_path 또는 metadata_path)이 누락되었습니다.")
            return

        index_file_path = Path(index_file_path_str)
        metadata_file_path = Path(metadata_file_path_str)

        # 디렉토리 생성
        index_file_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_file_path.parent.mkdir(parents=True, exist_ok=True)

        # 인덱스 파일이 존재하면 불러오고, 없으면 새로 생성
        if index_file_path.exists():
            index = faiss.read_index(str(index_file_path))
            if index.d != embedding_dim:
                status["error_embeddings_deff_drainage"]["value"] += 1
                logger.warning(f"임베딩 차원 불일치: 기존 인덱스({index.d}) vs 입력({embedding_dim})")
                return
        else:
            index = faiss.IndexFlatL2(embedding_dim)
            logger.info(f"새로운 IndexFlatL2 생성 (차원: {embedding_dim})")

        # 인덱스에 추가
        index.add(embedding)
        faiss.write_index(index, str(index_file_path))
        logger.info(f"인덱스에 1개 벡터 추가. 현재 총 벡터 수: {index.ntotal}")

        # 메타데이터 저장 (append)
        with open(metadata_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        logger.info("메타데이터 저장 완료.")

    except Exception as e:
        logger.critical(f"인덱스 추가 중 오류 발생: {e}", exc_info=True)

def build_and_save_index_alone(
        embeddings: List[np.ndarray], 
        metadatas: List[Dict[str, Any]], 
        cfg_obj: configger
    ):
    """
    수집된 얼굴 특징 벡터(embeddings)와 해당 메타데이터(metadatas)를 사용하여
    FAISS 인덱스를 구축하고, 인덱스 파일과 메타데이터 파일을 저장합니다.
    인덱싱 관련 설정(파일 경로, FAISS 인덱스 타입, 임베딩 차원 등)은
    설정 객체(cfg_obj)의 'indexing' 섹션에서 읽어옵니다.
    Args:
        embeddings (List[np.ndarray]): 인덱싱할 얼굴 특징 벡터들의 리스트. 각 요소는 NumPy 배열.
        metadatas (List[Dict[str, Any]]): 각 특징 벡터에 해당하는 메타데이터 딕셔너리들의 리스트.
                                         리스트 순서는 embeddings 리스트와 일치해야 합니다.
        cfg_obj (configger): 설정 파일 내용을 담고 있는 configger 객체.
    """
    status = {k: {"value": v["value"], "msg": v["msg"]} for k, v in DEFAULT_STATUS_TEMPLATE.items()}

    if not embeddings:
        # 인덱싱할 임베딩 데이터가 없으면, 정보 로깅 후 함수 종료
        logger.info("인덱싱할 얼굴 특징 벡터가 없습니다. FAISS 인덱스를 생성하지 않습니다.")
        status["error_embedding_empty_target"]["value"] += 1
        return

    try:
        # FAISS는 float32 타입의 NumPy 배열을 입력으로 사용합니다.
        embeddings_array = np.array(embeddings).astype('float32')
        if embeddings_array.ndim == 1: # 단일 임베딩만 있는 경우 2D로 변환
            if embeddings_array.size > 0:
                embeddings_array = embeddings_array.reshape(1, -1)
            else: # 빈 임베딩인 경우
                logger.info("빈 임베딩 배열입니다. FAISS 인덱스를 생성하지 않습니다.")
                status["error_embeddings_array"]["value"] += 1
                return

        embedding_dim = embeddings_array.shape[1]  # 특징 벡터의 차원 (예: dlib은 128차원)

        logger.info(f"총 {len(embeddings_array)}개의 얼굴 특징 벡터({embedding_dim} 차원) 수집 완료. FAISS 인덱스 구축 시작.")

        # --- FAISS 인덱스 구축 설정 (YAML 파일의 'indexing' 섹션에서 로드) ---
        index_file_path_str = cfg_obj.get_value('indexing.index_file_path')
        metadata_file_path_str = cfg_obj.get_value('indexing.metadata_path')
        faiss_index_type = cfg_obj.get_value('indexing.faiss_index_type', 'IndexFlatL2') # 기본값: IndexFlatL2
        
        # configured_embedding_dim은 JSON에서 직접 읽어온 임베딩의 차원을 사용하므로,
        # YAML 설정의 embedding_dim은 참고용 또는 검증용으로만 사용하거나 제거 가능.
        # 여기서는 YAML의 embedding_dim과 실제 데이터의 차원을 비교하는 로직을 유지합니다.
        configured_embedding_dim_yaml = cfg_obj.get_value('indexing.embedding_dim')
        if configured_embedding_dim_yaml is not None:
            configured_embedding_dim_yaml = int(configured_embedding_dim_yaml)
            if configured_embedding_dim_yaml != embedding_dim:
                status["error_embeddings_deff_config"]["value"] += 1
                logger.warning(
                    f"YAML에 설정된 embedding_dim ({configured_embedding_dim_yaml})과 "
                    f"실제 데이터의 특징 벡터 차원 ({embedding_dim})이 일치하지 않습니다. "
                    f"실제 데이터 차원인 {embedding_dim}을 사용합니다."
                )
        else:
            status["error_embeddings_deff_config"]["value"] += 1
            logger.info(f"YAML에 'indexing.embedding_dim'이 설정되지 않았습니다. 실제 데이터 차원 {embedding_dim}을 사용합니다.")


        # 필수 설정값들이 제대로 로드되었는지 확인
        if not index_file_path_str or not metadata_file_path_str:
            status["error_embeddings_deff_config"]["value"] += 1
            logger.critical("YAML 설정 파일의 'indexing' 섹션 또는 필수 키(index_file_path, metadata_path)가 누락되었습니다.")
            return

        index_file_path = Path(index_file_path_str)  # 인덱스 파일 저장 경로
        metadata_file_path = Path(metadata_file_path_str)  # 메타데이터 파일 저장 경로

        index = None  # FAISS 인덱스 객체 초기화

        # --- FAISS 인덱스 타입에 따른 인덱스 생성 ---
        logger.info(f"FAISS 인덱스 타입: '{faiss_index_type}' (임베딩 차원: {embedding_dim})")

        if faiss_index_type == 'IndexFlatL2':
            index = faiss.IndexFlatL2(embedding_dim)
        elif faiss_index_type == 'IndexFlatIP':
            index = faiss.IndexFlatIP(embedding_dim)
        elif faiss_index_type == 'IndexIVFFlat':
            nlist = int(cfg_obj.get_value('indexing.nlist', 100)) 
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)
            logger.info(f"  - IndexIVFFlat 파라미터: nlist={nlist}")
            if embeddings_array.shape[0] < nlist:
                status["error_embeddings_deff_spec"]["value"] += 1
                logger.warning(f"  경고: 학습 데이터 수({embeddings_array.shape[0]})가 nlist({nlist})보다 적습니다. IVFFlat 학습에 영향이 있을 수 있습니다.")
            if embeddings_array.shape[0] > 0: # 학습 데이터가 있어야 학습 가능
                logger.info("  IndexIVFFlat 학습 시작...")
                index.train(embeddings_array)
                logger.info("  IndexIVFFlat 학습 완료.")
            else:
                status["error_embeddings_none_learn"]["value"] += 1
                logger.warning("  학습 데이터가 없어 IndexIVFFlat 학습을 건너<0xEB><0><0x8E>니다.")
        elif faiss_index_type == 'IndexIVFPQ':
            nlist = int(cfg_obj.get_value('indexing.nlist', 100))
            M = int(cfg_obj.get_value('indexing.M', 8)) 
            nbits = int(cfg_obj.get_value('indexing.nbits', 8))
            logger.info(f"  - IndexIVFPQ 파라미터: nlist={nlist}, M={M}, nbits={nbits}")

            if embedding_dim % M != 0:
                status["error_embeddings_none_learn"]["value"] += 1
                logger.warning(f"  경고: IndexIVFPQ 사용 시 embedding_dim({embedding_dim})이 M({M})의 배수가 아닙니다. 성능 저하 또는 오류가 발생할 수 있습니다.")
            
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, M, nbits)
            if embeddings_array.shape[0] < nlist: 
                status["error_embeddings_deff_drainage"]["value"] += 1
                logger.warning(f"  경고: 학습 데이터 수({embeddings_array.shape[0]})가 nlist({nlist})보다 적습니다. IVFPQ 학습에 영향이 있을 수 있습니다.")
            if embeddings_array.shape[0] > 0: # 학습 데이터가 있어야 학습 가능
                logger.info("  IndexIVFPQ 학습 시작...")
                index.train(embeddings_array)
                logger.info("  IndexIVFPQ 학습 완료.")
            else:
                status["error_embeddings_none_learn"]["value"] += 1
                logger.warning("  학습 데이터가 없어 IndexIVFPQ 학습을 건너<0xEB><0><0x8E>니다.")
        else:
            status["error_embeddings_none_serport_type"]["value"] += 1
            logger.error(f"지원하지 않는 FAISS 인덱스 타입입니다: '{faiss_index_type}'.")
            logger.error("지원 타입: IndexFlatL2, IndexFlatIP, IndexIVFFlat, IndexIVFPQ")
            return

        if index is None:
            status["error_embeddings_objebt_gen"]["value"] += 1
            logger.critical("FAISS 인덱스 객체 생성에 실패했습니다. 설정을 확인해주세요.")
            return
        
        if embeddings_array.shape[0] > 0: # 추가할 데이터가 있을 때만 add 수행
            index.add(embeddings_array)
            logger.info(f"FAISS 인덱스에 총 {index.ntotal}개의 벡터 추가 완료.")
        else:
            status["error_embeddings_none_learn"]["value"] += 1
            logger.info("추가할 임베딩 데이터가 없어 FAISS 인덱스에 벡터를 추가하지 않았습니다.")


        # --- 인덱스 파일 및 메타데이터 파일 저장 ---
        index_file_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_file_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(index_file_path))
        logger.info(f"FAISS 인덱스 저장 완료: {index_file_path}")

        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            for meta_item in metadatas:
                f.write(json.dumps(meta_item, ensure_ascii=False) + '\n')
        logger.info(f"메타데이터 저장 완료: {metadata_file_path}")

    except KeyError as e:
        logger.critical(f"FAISS 인덱스 구축 설정 중 필수 키가 누락되었습니다: {e}")
        logger.critical("YAML 설정 파일의 'indexing' 섹션을 확인해주세요 (예: index_file_path, metadata_path, faiss_index_type 등).")
    except Exception as e:
        logger.critical(f"FAISS 인덱스 구축 및 저장 중 예상치 못한 오류 발생: {e}", exc_info=True)

def run_main(cfg: configger):
    # 0. 일 준비
    # 0.1. 통계 정보를 담을 딕셔너리 초기화
    status = {k: {"value": v["value"], "msg": v["msg"]} for k, v in DEFAULT_STATUS_TEMPLATE.items()}
    BATCH_SIZE = int(cfg.get_value("indexing.batch_size", 4))  # 설정값 또는 기본값

    # 0.2. dlib 모델 로드는 선택 사항이 되었으므로, 실패해도 치명적이지 않음
    loaded_dlib_models = load_dlib_models(cfg)

    if loaded_dlib_models is None:
        logger.warning("dlib 모델 로드에 실패")
        return None

    try:
        detector = loaded_dlib_models['face_detector']
        predictor = loaded_dlib_models['shape_predictor']
        recognizer = loaded_dlib_models['face_recognizer']
    except KeyError as e:
        logger.critical(f"dlib 모델 로딩 중 필수 키가 누락되었습니다: {e}")
        logger.critical("dlib 모델 로드, 필요한 키(face_detector, shape_predictor, face_recognizer)가 포함되어 있는지 확인하세요.")
        return None
    except Exception as e:
        logger.critical(f"dlib 모델 로딩 중 알 수 없는 오류 발생: {e}")
        return None
    logger.info("dlib 얼굴 인식 관련 모델 로드 완료와 각 값을 가저옴.")

    # 0.3. 입력 디렉토리 및 JSON 키 설정 가져오기
    dataset_base_key_str = "project.paths.datasets"
    input_dir_key = f"{dataset_base_key_str}.raw_jsons_dir"
    input_dir_str = cfg.get_value(input_dir_key)
    input_dir = Path(input_dir_str).expanduser().resolve()

    # 1. JSON 파일 목록 가져오기
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
    # json_files = [p for p in input_dir.glob("**/*.json") if p.is_file()]
    # total_input_found = len(json_files)    """

    # 1.1. JSON 파일 목록 가져오기 및 총 개수 세기 (메모리 효율적)
    logger.info(f"'{input_dir}' 디렉토리에서 JSON 파일 탐색 시작...")

    # glob 결과 이터레이터를 생성 (개수 세기용)
    # 이 이터레이터는 아래 sum() 함수에 의해 소모됩니다.
    json_file_iterator_for_counting = input_dir.glob("**/*.json")

    # is_file() 필터링을 적용하면서 개수 세기
    # sum(1 for ...) 구문은 이터레이터를 순회하며 각 요소에 대해 1을 더하여 개수를 셉니다.
    # 이 과정에서 전체 경로를 메모리에 리스트로 저장하지 않습니다.
    total_input_found = sum(1 for p in json_file_iterator_for_counting if p.is_file())

    if total_input_found == 0:
        logger.warning(f"'{input_dir}' 디렉토리에서 인덱싱할 JSON 파일을 찾을 수 없습니다.")
        logger.info("✅ 최종 통계:")
        logger.info(f"   - 탐색된 JSON 파일 총 개수: {total_input_found}")
        logger.info(f"   - 인덱싱된 총 얼굴 개수: 0")
        logger.info(f"{Path(__file__).name} 정상 종료 (처리할 파일 없음).")
        sys.exit(0)
    status["total_input_file"]["value"] = total_input_found

    logger.info(f'✅ 인덱싱할 JSON 파일 {status["total_input_file"]["value"]}개 발견.')

    digit_width = calc_digit_number(total_input_found)
    # 1.2. 모든 파일에서 얼굴 정보 누적 (메모리 효율적인 파일 순회)
    # glob 결과를 다시 생성 (실제 처리용)
    # 개수를 세느라 첫 번째 이터레이터가 소모되었으므로, 실제 처리를 위해서는 새로 만들어야 합니다.
    json_file_iterator_for_processing = input_dir.glob("**/*.json")

    all_embeddings: List[np.ndarray] = [] # 모든 파일의 임베딩을 누적할 리스트
    all_metadatas: List[Dict[str, Any]] = [] # 모든 파일의 메타데이터를 누적할 리스트
    total_faces_processed = 0 # 성공적으로 처리된 총 얼굴 개수
    batch_index = 0  # 배치 카운터
    # total_faces_failed 변수는 필요에 따라 추가

    logger.info("JSON 파일 내 얼굴 정보 수집 시작...")
    # enumerate를 사용하여 진행 상황을 표시하기 위해 리스트로 변환하는 대신,
    # total_input_found 활용하여 수동으로 카운트하며 진행 상황을 표시하는 것이 좋습니다.
    for json_file_path in json_file_iterator_for_processing: # 이터레이터를 순회
        if not json_file_path.is_file():
            status["error_input_file_read"]["value"] += 1
            continue

        status["req_process_count"]["value"] += 1
        logger.debug(f"[{status["req_process_count"]["value"]:>{digit_width}}/{status["total_input_file"]["value"]}] JSON 파일 처리 중: {json_file_path.name}")

        # 2. 현재 파일(json_file_path)에서 얼굴의 임베딩과 메타데이터 가져와서 indexing
        # json_key_config_data를 전달하도록 수정
        embeddings_from_file, metadatas_from_file = get_all_face_data_from_json_alone(
            cfg, 
            json_file_path
            )

        if embeddings_from_file: # 파일에서 유효한 임베딩이 추출된 경우
            all_embeddings.extend(embeddings_from_file) # 누적 리스트에 추가
            all_metadatas.extend(metadatas_from_file) # 누적 리스트에 추가
            total_faces_processed += len(embeddings_from_file) # 총 얼굴 개수 누적
            logger.info(f"  '{json_file_path.name}' 파일에서 얼굴 {len(embeddings_from_file)}개 정보 추출 완료. (누적 {total_faces_processed}개)")

            if len(all_embeddings) >= BATCH_SIZE:
                batch_index += 1
                logger.info(f"🔄 배치 {batch_index}: {len(all_embeddings)}개 얼굴 인덱싱 중...")

                build_and_save_index_alone(all_embeddings, all_metadatas, cfg) # build_and_save_index_alone 호출

                # 누적 데이터 초기화
                all_embeddings.clear()
                all_metadatas.clear()
        else:
            # 파일 내에 유효한 얼굴 정보가 없거나 오류 발생 시
            status["error_embedding"]["value"] += 1
            logger.warning(f"  '{json_file_path.name}' 파일에서 유효한 얼굴 정보를 추출하지 못했습니다.")
            # 실패 카운트 로직은 필요에 따라 추가

        if status["req_process_count"]["value"] > 2:
            break
    # 🔚 남은 데이터가 있을 경우 마지막 배치 처리
    if all_embeddings:
        batch_index += 1
        logger.info(f"🔄 마지막 배치 {batch_index}: {len(all_embeddings)}개 얼굴 인덱싱 중...")
        build_and_save_index_alone(all_embeddings, all_metadatas, cfg) # build_and_save_index_alone 호출

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

if __name__ == "__main__":
    # 0. 애플리케이션 아귀먼트 있으면 갖오기
    logger.info(f"애플리케이션 시작")
    parsed_args = get_argument()

    # 1. logger 일할준비 시키기기
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

    # 2. configger 일할 준비 시키기기
    # configger는 이제 위에서 설정된 공유 logger를 사용합니다.
    # root_dir과 config_path는 실제 프로젝트에 맞게 설정해야 합니다.
    # 설정 파일이 실제로 존재하는지 확인
    logger.info(f"Configger 초기화 시도: root_dir='{parsed_args.root_dir}', config_path='{parsed_args.config_path}'")

    try:
        cfg_object = configger(root_dir=parsed_args.root_dir, config_path=parsed_args.config_path)
        logger.info(f"Configger 초기화 끝")
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}")

    # 3. 본 프로그램 시작작
    logger.info(f" 이제 일하자 ")
    try:
        run_main(cfg_object)
    except KeyError as e:
        logger.critical(f"설정 파일에서 필수 경로 키를 찾을 수 없습니다: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"설정 파일에서 경로 정보 처리 중 오류 발생: {e}")
        sys.exit(1)
    finally:
        # 애플리케이션 종료 시 로거 정리 (특히 비동기 사용 시 중요)
        logger.info(f"{script_name} 애플리케이션 종료")
        logger.shutdown()
        exit(0)

    print("모든 JSON 파일 경로 처리가 완료되었습니다.")


