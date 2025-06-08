# src/face_indexer_from_face.py

# 표준 라이브러리 임포트
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Set
import traceback

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
    "total_input_found":         {"value": 0,  "msg": "총 입력 파일 수 (지원 확장자 기준)"}, # 찾은 총 입력 파일 수
    "error_input_file_read":        {"value": 0,  "msg": "입력 파일 읽기 오류 수"}, # 입력 파일 읽기 실패 수
    "req_process_count":         {"value": 0,  "msg": "총 처리 시도 파일 수"}, # 처리 시도한 총 파일 수
    "error_extension":   {"value": 0,  "msg": "지원되지 않는 확장자로 건너뛴 파일 수"}, # 확장자 오류로 건너뛴 파일 수
    "error_image_rotation":          {"value": 0,  "msg": "이미지 회전중 오류 발생 파일 수"}, # 이미지 회전 중 오류 발생 수
    "error_target_file_get":        {"value": 0,  "msg": "처리대상(image or json) 파일 읽기 오류 수"}, # 대상 파일(이미지 또는 JSON) 읽기 오류 수
    "error_input_file_process":        {"value": 0,  "msg": "입력파일 처리 중 오류 수"}, # 입력 파일 처리 중 발생한 오류 수
    "request_embedding_processing":        {"value": 0,  "msg": "임베딩 처리 요청 수"}, # 임베딩 처리 요청된 횟수
    "error_embedding_empty_target":        {"value": 0,  "msg": "임베딩 오류 - 처리 대상 없음"}, # 임베딩 대상이 없는 경우의 오류 수
    "error_embedding_array_empty":        {"value": 0,  "msg": "임베딩 오류 - 빈 임베딩 배열"}, # 임베딩 배열이 비어있는 경우의 오류 수
    "error_embedding_config_missing":        {"value": 0,  "msg": "임베딩 오류 - 설정 값 없음"}, # 임베딩 관련 설정 값이 누락된 경우
    "error_embedding_config_mismatch":        {"value": 0,  "msg": "임베딩 오류 - 설정 값과 차원 불일치"}, # 설정된 임베딩 차원과 실제 데이터 차원이 다른 경우
    "error_embedding_data_shape_mismatch": {"value": 0,  "msg": "임베딩 오류 - 데이터 규격(shape) 불일치"}, # 임베딩 데이터의 형태(shape)가 예상과 다른 경우
    "error_embedding_training_data_missing": {"value": 0,  "msg": "임베딩 오류 - 학습 데이터 없음 (IVF 계열)"}, # IVF 계열 인덱스 학습 데이터가 없는 경우
    "error_embedding_dimension_m_mismatch":  {"value": 0,  "msg": "임베딩 오류 - 차원이 M의 배수 아님 (IVFPQ)"}, # IVFPQ 인덱스에서 차원이 M의 배수가 아닌 경우
    "error_embedding_object_creation_failed": {"value": 0,  "msg": "임베딩 오류 - FAISS 인덱스 객체 생성 실패"}, # FAISS 인덱스 객체 생성 실패 수
    "error_embedding_index_read_failed":      {"value": 0,  "msg": "임베딩 오류 - FAISS 인덱스 파일 읽기 실패"}, # FAISS 인덱스 파일 읽기 실패 수
    "error_embedding_general":           {"value": 0,  "msg": "임베딩 추출/처리 중 일반 오류 수"}, # 임베딩 관련 일반 오류 수
    "request_save_index":        {"value": 0,  "msg": "인덱스 저장 요청 수"}, # 인덱스 저장 요청 횟수
    "total_object_count":      {"value": 0,  "msg": "검출된 총 객체 수"}, # 검출된 전체 객체 수
    "files_with_detected_objects":     {"value": 0,  "msg": "객체가 1개 이상 검출된 파일 수"}, # 하나 이상의 객체가 검출된 파일 수
    "get_object_crop":           {"value": 0,  "msg": "객체가 검출된 객체 수"}, # 성공적으로 크롭된 객체 수
    "error_object_crop":           {"value": 0,  "msg": "객체 크롭(crop) 처리 오류 수"}, # 객체 크롭 중 오류 발생 수
    "error_object_bbox_format":           {"value": 0,  "msg": "객체 바운딩 박스 형식 오류 수"}, # 객체 바운딩 박스 형식이 잘못된 경우
    "error_object_bbox_count_mismatch":    {"value": 0,  "msg": "객체 바운딩 박스 개수 불일치 오류 수"}, # 바운딩 박스 개수가 예상과 다른 경우
    "error_object_bbox_position":           {"value": 0,  "msg": "객체 바운딩 박스 좌표 오류 수"}, # 바운딩 박스 좌표값이 잘못된 경우
    "undetection_object":   {"value": 0,  "msg": "객체가 검출되지 않은 파일 수"}, # 객체가 검출되지 않은 파일 수
    "error_copied_input_file": {"value": 0, "msg": "오류 발생 입력파일 보관 실패 수"}, # 오류 발생 시 입력 파일 백업 실패 수
    "detect_faces_in_object":    {"value": 0,  "msg": "객체에서 얼굴검출을 성공한 수"}, # 객체 내에서 얼굴 검출 성공 수
    "error_faces_in_object":    {"value": 0,  "msg": "객체에서 얼굴검출 실패 수"}, # 객체 내에서 얼굴 검출 실패 수
    "unmatched_object_number":   {"value": 0,  "msg": "검출 대상 object수와 검출한 object의 수가 다른 파일수"}, # 예상 객체 수와 실제 검출된 객체 수가 다른 파일 수
    "total_output_files":        {"value": 0,  "msg": "총 출력 파일수"}, # 생성된 총 출력 파일 수
    "read_input_files_success":          {"value": 0,  "msg": "읽은 입력 파일 수 (detect_object 기준)"}, # (detect_object 기준) 성공적으로 읽은 입력 파일 수
    "read_input_files_error":          {"value": 0,  "msg": "읽은 입력 파일 수 (detect_object 기준)"}, # (detect_object 기준) 읽기 실패한 입력 파일 수
    "files_json_load":           {"value": 0,  "msg": "JSON 정보 읽은 파일 수"}, # JSON 정보를 성공적으로 읽은 파일 수
    "files_json_update":         {"value": 0,  "msg": "JSON 파일 덧씌우기 성공 파일 수"}, # JSON 파일 업데이트 성공 수
    "error_json_update":         {"value": 0,  "msg": "JSON 파일 덧씌우기 실패 수"}, # JSON 파일 업데이트 실패 수
    "get_image_path_in_json":    {"value": 0,  "msg": "IMAGE 파일 경로 가져온 파일 수"}, # JSON에서 이미지 경로를 성공적으로 가져온 파일 수
    "undetected_image_copied_success": {"value": 0, "msg": "미검출 이미지 복사 성공 수"}, # 미검출 이미지를 성공적으로 복사한 수
    "undetected_image_copied_error": {"value": 0, "msg": "미검출 이미지 복사 실패 수"}, # 미검출 이미지 복사 실패 수
    "undetection_object_file":   {"value": 0,  "msg": "객체가 검출되지 않은 파일 수"}, # (중복된 의미, 위 undetection_object와 통합 가능성 검토)
    "num_detected_objects":      {"value": 0,  "msg": "검출된 총 객체 수"}, # (중복된 의미, 위 total_object_count와 통합 가능성 검토)
    "files_object_crop":         {"value": 0,  "msg": "객체가 있는 파일 수"}, # (중복된 의미, 위 files_with_detected_objects와 통합 가능성 검토)
    "error_faild_file_backup":        {"value": 0,  "msg": "읽을때 오류가 난 입력 파일을 보관하는데 오류발생 수"}, # 읽기 오류난 입력 파일 백업 실패 수
    "files_skipped_extension":   {"value": 0,  "msg": "지원되지 않는 확장자로 건너뛴 파일 수"}, # (중복된 의미, 위 error_extension과 통합 가능성 검토)
    "files_processed_for_log":   {"value": 0,  "msg": "로그용으로 처리 시도한 파일 수"}, # 최종 통계에는 보통 표시되지 않음
    "files_processed_main_error":{"value": 0,  "msg": "메인 루프에서 처리 중 오류 발생 파일 수"} # 메인 처리 루프에서 오류가 발생한 파일 수
} 

# 이 스크립트에서는 직접 사용하지 않지만, 설정 파일 로직 등에서 참조할 수 있으므로 유지합니다.
# 또는, dlib 모델 로드 관련 부분을 완전히 제거해도 됩니다. 여기서는 일단 유지합니다.
def load_dlib_models(cfg_obj: configger) -> dict | None:
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
    models_dir_key_str = f'{face_recognition_key_str}.models_dir' # YAML 파일 내 모델 디렉토리 경로 키
    # get_path 함수는 경로 존재 여부를 확인하고 로깅하므로, 반환 값이 None인지 확인합니다.
    models_dir_path_obj = cfg_obj.get_path(models_dir_key_str, ensure_exists=True) # ensure_exists=True로 경로 존재 확인
    if models_dir_path_obj is None:
        # get_path에서 이미 로깅했을 것이므로 추가 로깅은 생략
        return None
    logger.debug(f"모델이 있는 위치(models_dir_path_obj): {models_dir_path_obj}")

    # 얼굴 인식 모델 파일 이름 가져오기 및 경로 생성
    face_rec_model_name_key = f'{face_recognition_key_str}.face_rec_model_name'
    face_rec_model_name = cfg_obj.get_value(face_rec_model_name_key) # 설정에서 얼굴 인식 모델 파일 이름 가져오기
    if face_rec_model_name is None:
        logger.warning(f"dlib 얼굴 인식 모델 파일 이름 설정 값 '{face_rec_model_name_key}'을 가져오지 못했습니다.")
        return None
    logger.debug(f"모딜 이름은(face_rec_model_name): {face_rec_model_name}")

    # 랜드마크 모델 파일 이름 가져오기 및 경로 생성
    landmark_model_name_key = f'{face_recognition_key_str}.landmark_model_name' # YAML 파일 내 랜드마크 모델 파일 이름 키
    landmark_model_name = cfg_obj.get_value(landmark_model_name_key) # 설정에서 랜드마크 모델 파일 이름 가져오기
    if landmark_model_name is None:
        logger.warning(f"dlib 랜드마크 모델 파일 이름 설정 값 '{landmark_model_name_key}'을 가져오지 못했습니다.")
        return None

    # Path 객체를 사용하여 경로 결합
    face_rec_model_path = models_dir_path_obj / face_rec_model_name
    landmark_model_path = models_dir_path_obj / landmark_model_name
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

def get_all_face_data_from_json_alone(
    cfg:configger, 
    json_file_path: Path,
    json_handler: JsonConfigHandler # JsonConfigHandler 인스턴스 추가
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    주어진 JSON 파일에서 모든 얼굴의 임베딩과 메타데이터를 추출합니다.
    키 이름은 JsonConfigHandler 인스턴스를 통해 접근합니다.
    Args:
        cfg (configger): 설정 객체. (현재 함수 내에서 직접 사용되지는 않지만, 향후 확장성을 위해 유지)
        json_file_path (Path): 얼굴 정보를 추출할 JSON 파일의 경로.
    Returns:
        Tuple[List[np.ndarray], List[Dict[str, Any]]]:
            - 추출된 모든 얼굴 임베딩 리스트 (각 요소는 NumPy 배열).
            - 추출된 모든 얼굴 메타데이터 딕셔너리 리스트.
            - 오류 발생 또는 데이터 없음 시 ([], []) 반환.
    """
    embeddings_in_file: List[np.ndarray] = []
    metadatas_in_file: List[Dict[str, Any]] = []

    # 0. JsonConfigHandler의 read_json 메소드를 사용하여 JSON 데이터 로드
    json_data = json_handler.read_json(json_file_path)
    if json_data is None:
        # json_handler.read_json 내부에서 이미 오류 로깅이 수행됨
        logger.critical(f"0. '{json_file_path.name}' 파일에서 JSON 데이터를 읽지 못했습니다 (json_handler.read_json 반환값 None).") # read_json에서 로깅하므로 중복 로깅 방지
        return [], []

    try:
        # 1. JSON 파일에서 최상위 객체 리스트를 가져옵니다.
        logger.debug(f"객체 리스트 키 이름 사용: '{json_handler.object_info_key}' (파일: {json_file_path.name})")
        json_objects_list = json_data.get(json_handler.object_info_key)
        logger.debug(f"[{json_file_path.name}] Found object list for key '{json_handler.object_info_key}': {type(json_objects_list)} with length {len(json_objects_list) if isinstance(json_objects_list, list) else 'N/A'}")
        if not isinstance(json_objects_list, list):
            logger.error(f"1. JSON 파일 '{json_file_path.name}'에 '{json_handler.object_info_key}' 키로 식별되는 리스트가 없거나 형식이 잘못되었습니다.")
            return [], []

        if not json_objects_list:
            logger.error(f"1. [{json_file_path.name}] Object list for key '{json_handler.object_info_key}' is empty.")
        
        # 2. 이미지 레벨 메타데이터 추출
        # 2.1. image_info_key로 이미지 정보 딕셔너리를 먼저 가져옵니다.
        image_info_data = json_data.get(json_handler.image_info_key, {})
        json_image_path = image_info_data.get(json_handler.image_path_key)
        json_image_hash = image_info_data.get(json_handler.image_hash_key)

        for objt_indx, obj_entry in enumerate(json_objects_list): # 객체 리스트를 순회합니다.
            if not isinstance(obj_entry, dict):
                logger.error(f"2.1. [{json_file_path.name}] - 인덱스 {objt_indx}의 객체가 딕셔너리가 아닙니다. 건너뜁니다.")
                continue # 각 객체는 딕셔너리여야 합니다.

            # 2.2. 현재 객체에서 얼굴 리스트를 가져옵니다.
            json_face_list = obj_entry.get(json_handler.face_info_key)
            # 이 데이터는 단일 얼굴의 dict이거나, 여러 얼굴의 dict 리스트일 수 있습니다.
            json_face_list = obj_entry.get(json_handler.face_info_key)
            logger.debug(f"2.2. [{json_file_path.name}] Object {objt_indx}: Raw face data for key '{json_handler.face_info_key}': type={type(json_face_list)}, length={len(json_face_list) if isinstance(json_face_list, list) else ('1' if isinstance(json_face_list, dict) else 'N/A')}")

            # 2.3. 얼굴 리스트의 Type를 확인합니다.
            faces_to_iterate = []
            if isinstance(json_face_list, dict):
                # 단일 얼굴 dict인 경우, 리스트에 추가하여 일관되게 처리
                faces_to_iterate.append(json_face_list)
            elif isinstance(json_face_list, list):
                # 여러 얼굴의 dict 리스트인 경우, 그대로 사용
                faces_to_iterate = json_face_list
            else:
                if json_face_list is None:
                    logger.warning(f"2.3. [{json_file_path.name}] - 객체 {objt_indx}: 키 '{json_handler.face_info_key}'에 얼굴 데이터가 없습니다. 이 객체의 얼굴 정보를 건너뜁니다.")
                else:
                    logger.error(f"2.3. [{json_file_path.name}] - 객체 {objt_indx}: 키 '{json_handler.face_info_key}'의 얼굴 데이터 타입이 예상과 다릅니다 (타입: {type(json_face_list)}). 건너뜁니다.")
                continue

            if not faces_to_iterate:
                logger.warning(f"2.3. [{json_file_path.name}] Object {objt_indx}: No faces to iterate for key '{json_handler.face_info_key}'.")
                continue # 이미 위에서 처리되었거나, 아래 루프가 빈 리스트를 처리함

            # 3. 얼굴 리스트의 각 정보를 처리합니다.
            logger.debug(f"[{json_file_path.name}] Object {objt_indx}: Normalized {len(faces_to_iterate)} face(s) to process.")
            for face_indx, face_entry in enumerate(faces_to_iterate): # 객체 내 얼굴 리스트를 순회합니다.
                if not isinstance(face_entry, dict):
                    logger.warning(f"3. [{json_file_path.name}] - 객체 {objt_indx}, 얼굴 인덱스 {face_indx}의 항목이 딕셔너리가 아닙니다 (타입: {type(face_entry)}). 건너뜁니다.")
                    continue # 각 얼굴 항목은 딕셔너리여야 합니다.

                logger.debug(f"[{json_file_path.name}] Object {objt_indx}, Face {face_indx}: Processing face entry. Looking for embedding key '{json_handler.face_embedding_key}'.")
                embedding_data = face_entry.get(json_handler.face_embedding_key)
                if embedding_data is None:
                    face_id_val = face_entry.get(json_handler.face_id_key, "N/A") # 얼굴 ID가 없을 경우 "N/A"
                    logger.warning(f"3. [{json_file_path.name}] - 객체 {objt_indx}, 얼굴 {face_indx} (ID: {face_id_val}): 키 '{json_handler.face_embedding_key}'의 임베딩 데이터가 없습니다. 건너뜁니다.")
                    continue
                try:
                    embedding_np = np.array(embedding_data, dtype=np.float32)
                except Exception as e_np:
                    face_id_val = face_entry.get(json_handler.face_id_key, "N/A")
                    logger.warning(f"3. JSON 파일 '{json_file_path.name}'의 face_id '{face_id_val}' (객체 {objt_indx}, 얼굴 {face_indx}) 임베딩 NumPy 변환 중 오류: {e_np}. 건너뜁니다.")
                    continue

                metadata = {
                    "source_json_path": str(json_file_path),
                    json_handler.image_path_key: json_image_path,
                    json_handler.image_hash_key: json_image_hash,
                    json_handler.face_box_xyxy_key: face_entry.get(json_handler.face_box_xyxy_key),
                    json_handler.face_confidence_key: face_entry.get(json_handler.face_confidence_key),
                    json_handler.face_class_id_key: face_entry.get(json_handler.face_class_id_key),
                    json_handler.face_class_name_key: face_entry.get(json_handler.face_class_name_key),
                    json_handler.face_label_key: face_entry.get(json_handler.face_label_key),
                    json_handler.face_id_key: face_entry.get(json_handler.face_id_key),
                    json_handler.face_box_key: face_entry.get(json_handler.face_box_key), # YAML 설정에 따라 'box' 또는 다른 키
                    # 객체 정보 추가
                    json_handler.object_class_name_key: obj_entry.get(json_handler.object_class_name_key),
                    json_handler.object_box_xyxy_key: obj_entry.get(json_handler.object_box_xyxy_key),
                    json_handler.object_index_key: objt_indx, # 객체의 인덱스
                    "face_index_in_object": face_indx, # 객체 내 얼굴의 인덱스
                }
                embeddings_in_file.append(embedding_np)
                metadatas_in_file.append(metadata)

        return embeddings_in_file, metadatas_in_file

    except Exception as e:
        logger.critical(f"인덱스 추가 중 오류 발생: {e}")
        return [], []

#def add_embeddings_batch(embeddings: List[np.ndarray], metadatas: List[Dict[str, Any]], cfg_obj):
def add_embedding_to_index(embedding: np.ndarray, metadata: Dict[str, Any], cfg_obj):
    """
    단일 얼굴 임베딩과 메타데이터를 FAISS 인덱스에 추가하고 저장합니다.
    참고: 이 함수는 현재 `run_main`의 주 실행 흐름에서는 사용되지 않습니다.
    # 단일 임베딩을 점진적으로 추가하는 시나리오에 사용될 수 있습니다.
    """
    # 이 함수는 현재 메인 흐름에서 사용되지 않으므로 status 관련 로직은 포함하지 않습니다.

    try:
        embedding = embedding.astype('float32').reshape(1, -1)
        embedding_dim = embedding.shape[1]

        # 설정 값 불러오기
        index_file_path_str = cfg_obj.get_value('indexing.index_file_path')
        metadata_file_path_str = cfg_obj.get_value('indexing.metadata_path')

        if not index_file_path_str or not metadata_file_path_str: # 필수 경로 설정 확인
            logger.critical("필수 설정(index_file_path 또는 metadata_path)이 누락되었습니다.")
            return

        index_file_path = Path(index_file_path_str)
        metadata_file_path = Path(metadata_file_path_str)

        # 저장 경로의 부모 디렉토리 생성 (존재하지 않을 경우)
        index_file_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_file_path.parent.mkdir(parents=True, exist_ok=True)

        # 인덱스 파일이 존재하면 불러오고, 없으면 새로 생성
        if index_file_path.exists():
            index = faiss.read_index(str(index_file_path))
            if index.d != embedding_dim:
                logger.warning(f"임베딩 차원 불일치: 기존 인덱스 차원({index.d}) vs 현재 입력 차원({embedding_dim})")
                return
        else:
            index = faiss.IndexFlatL2(embedding_dim)
            logger.info(f"새로운 IndexFlatL2 생성 (차원: {embedding_dim})")

        # 인덱스에 임베딩 추가
        index.add(embedding)
        faiss.write_index(index, str(index_file_path))
        logger.info(f"인덱스에 1개 벡터 추가. 현재 총 벡터 수: {index.ntotal}")

        # 메타데이터 저장 (기존 파일에 추가 'a' 모드)
        with open(metadata_file_path, 'a', encoding='utf-8') as f: # 'a'는 append 모드
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
    FAISS 인덱스를 구축하고 인덱스 파일과 메타데이터 파일을 저장합니다.
    인덱싱 관련 설정(파일 경로, FAISS 인덱스 타입, 임베딩 차원 등)은
    설정 객체(cfg_obj)의 'indexing' 섹션에서 읽어옵니다.
    Args:
        embeddings (List[np.ndarray]): 인덱싱할 얼굴 특징 벡터들의 리스트. 각 요소는 NumPy 배열.
        metadatas (List[Dict[str, Any]]): 각 특징 벡터에 해당하는 메타데이터 딕셔너리들의 리스트.
                                         리스트 순서는 embeddings 리스트와 일치해야 합니다.
        cfg_obj (configger): 설정 파일 내용을 담고 있는 configger 객체.
    """

    if not embeddings:
        logger.info("인덱싱할 얼굴 특징 벡터가 없습니다. FAISS 인덱스를 생성하지 않습니다.")
        # 호출하는 쪽(예: run_main)에서 status["error_embedding_empty_target"]를 업데이트할 수 있습니다.
        return

    try:
        # FAISS는 float32 타입의 NumPy 배열을 입력으로 사용합니다.
        embeddings_array = np.array(embeddings).astype('float32')
        if embeddings_array.ndim == 1: # 단일 임베딩만 있는 경우 (1D 배열), 2D 배열로 변환
            if embeddings_array.size > 0:
                embeddings_array = embeddings_array.reshape(1, -1)
            else: # 비어있는 1D 배열인 경우
                logger.info("빈 임베딩 배열입니다. FAISS 인덱스를 생성하지 않습니다.")
                return

        embedding_dim = embeddings_array.shape[1]  # 특징 벡터의 차원 (예: dlib은 128차원)

        logger.info(f"총 {len(embeddings_array)}개의 얼굴 특징 벡터({embedding_dim} 차원) 수집 완료. FAISS 인덱스 구축 시작.")
        
        datasets_dir_cfg = cfg_obj.get_config('project.paths.datasets', {})
        undetect_objects_dir_str = datasets_dir_cfg.get("undetect_objects_dir",  None) # 파일이므로 ensure_exists=False 또는 부모 디렉토리만 생성
        undetect_objects_dir = Path(undetect_objects_dir_str)# 파일이므로 ensure_exists=False 또는 부모 디렉토리만 생성
        undetect_objects_dir.mkdir(parents=True, exist_ok=True)

        undetect_list_file_path_str = datasets_dir_cfg.get("undetect_list_path",  None) # 파일이므로 ensure_exists=False 또는 부모 디렉토리만 생성
        undetect_list_file_path = Path(undetect_list_file_path_str)

        # --- FAISS 인덱스 구축 설정 (YAML 파일의 'indexing' 섹션에서 로드) ---
        indexing_cfg = cfg_obj.config('indexing', {})
        index_file_path_str = indexing_cfg.get('index_file_path',None) # YAML 설정에서_file_path')
        index_file_path = Path(index_file_path_str)

        metadata_file_path_str = indexing_cfg.get('metadata_path', None)
        metadata_file_path = Path(metadata_file_path_str)

        faiss_index_type = indexing_cfg.get('faiss_index_type', 'IndexFlatL2') # 기본값: IndexFlatL2

        cfg_embedding_dim = indexing_cfg.get('embedding_dim',  None)
        # YAML 설정의 embedding_dim과 실제 데이터의 차원을 비교합니다.
        # JSON에서 직접 읽어온 임베딩의 차원을 사용하므로, YAML 설정은 검증용으로 사용됩니다.
        if cfg_embedding_dim is not None:
            cfg_embedding_dim = int(cfg_embedding_dim)
            if cfg_embedding_dim != embedding_dim:
                logger.warning(
                    f"YAML에 설정된 embedding_dim ({cfg_embedding_dim})과 "
                    f"실제 데이터의 특징 벡터 차원 ({embedding_dim})이 일치하지 않습니다. "
                    f"실제 데이터 차원인 {embedding_dim}을 사용합니다."
                )
        else:
            logger.info(f"YAML에 'indexing.embedding_dim'이 설정되지 않았습니다. 실제 데이터 차원 {embedding_dim}을 사용합니다.")

        if not index_file_path_str or not metadata_file_path_str:
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
            logger.info(f"  - IndexIVFFlat 파라미터: nlist={nlist} (클러스터 수)")
            if embeddings_array.shape[0] < nlist:
                logger.warning(f"  경고: 학습 데이터 수({embeddings_array.shape[0]})가 nlist({nlist})보다 적습니다. IVFFlat 학습에 영향이 있을 수 있습니다.")
            if embeddings_array.shape[0] > 0: # 학습 데이터가 있을 경우에만 학습 수행
                logger.info("  IndexIVFFlat 학습 시작...")
                index.train(embeddings_array)
                logger.info("  IndexIVFFlat 학습 완료.")
            else: # 학습 데이터가 없을 경우
                logger.warning("  학습 데이터가 없어 IndexIVFFlat 학습을 건너뜁니다.")
        elif faiss_index_type == 'IndexIVFPQ':
            nlist = int(cfg_obj.get_value('indexing.nlist', 100))
            M = int(cfg_obj.get_value('indexing.M', 8)) 
            nbits = int(cfg_obj.get_value('indexing.nbits', 8))
            logger.info(f"  - IndexIVFPQ 파라미터: nlist={nlist}, M={M}, nbits={nbits}")

            if embedding_dim % M != 0:
                logger.warning(f"  경고: IndexIVFPQ 사용 시 임베딩 차원({embedding_dim})이 M({M})의 배수가 아닙니다. 성능 저하 또는 오류가 발생할 수 있습니다.")
            
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, M, nbits)
            if embeddings_array.shape[0] < nlist: 
                logger.warning(f"  경고: 학습 데이터 수({embeddings_array.shape[0]})가 nlist({nlist})보다 적습니다. IVFPQ 학습에 영향이 있을 수 있습니다.")
            if embeddings_array.shape[0] > 0: # 학습 데이터가 있을 경우에만 학습 수행
                logger.info("  IndexIVFPQ 학습 시작...")
                index.train(embeddings_array)
                logger.info("  IndexIVFPQ 학습 완료.")
            else: # 학습 데이터가 없을 경우
                logger.warning("  학습 데이터가 없어 IndexIVFPQ 학습을 건너뜁니다.")
        else: # 지원하지 않는 인덱스 타입
            logger.error(f"지원하지 않는 FAISS 인덱스 타입입니다: '{faiss_index_type}'.")
            logger.error("지원 타입: IndexFlatL2, IndexFlatIP, IndexIVFFlat, IndexIVFPQ")
            return

        if index is None:
            logger.critical("FAISS 인덱스 객체 생성에 실패했습니다. 설정을 확인해주세요.")
            return
        
        if embeddings_array.shape[0] > 0: # 추가할 임베딩 데이터가 있을 경우에만 인덱스에 추가
            index.add(embeddings_array)
            logger.info(f"FAISS 인덱스에 총 {index.ntotal}개의 벡터 추가 완료.")
        else:
            logger.info("추가할 임베딩 데이터가 없어 FAISS 인덱스에 벡터를 추가하지 않았습니다.")


        # --- 인덱스 파일 및 메타데이터 파일 저장 ---
        # 저장 경로의 부모 디렉토리 생성 (존재하지 않을 경우)
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
        logger.critical(f"FAISS 인덱스 구축 및 저장 중 예상치 못한 오류 발생: {e}")

def run_main(cfg: configger):
    # 0. 초기 설정 및 준비 단계
    # 0.1. 통계 정보를 담을 딕셔너리 초기화
    status = {k: {"value": v["value"], "msg": v["msg"]} for k, v in DEFAULT_STATUS_TEMPLATE.items()}
    # BATCH_SIZE = int(cfg.get_value("indexing.batch_size", 1000)) # 현재 로직에서는 모든 데이터를 모아 한번에 처리

    # 0.2. dlib 모델 로드 (선택 사항, 현재 인덱싱 로직에서 직접 사용되지 않음)
    loaded_dlib_models = load_dlib_models(cfg)

    # dlib 모델 로드 결과 처리 (향후 dlib 기반 기능 추가 시 활용 가능)
    # 향후 dlib 기반 기능을 추가할 경우 이 모델들을 활용할 수 있습니다.
    if loaded_dlib_models is None:
        logger.warning("dlib 모델 로드에 실패했습니다. (이 스크립트의 주 기능에는 영향 없음)")
        # dlib 모델 로드가 필수는 아니므로, 여기서 프로그램을 종료하지 않습니다.
        # 필요에 따라 로드 실패 시 처리를 추가할 수 있습니다.

    else:
        # loaded_dlib_models가 None이 아닐 때만 내부 키에 접근
        try:
            detector = loaded_dlib_models['face_detector']
            predictor = loaded_dlib_models['shape_predictor']
            recognizer = loaded_dlib_models['face_recognizer']
            logger.info("dlib 얼굴 인식 관련 모델 로드 및 변수 할당 완료 (현재 인덱싱 로직에서 직접 사용되지 않음).")
        except KeyError as e:
            logger.critical(f"dlib 모델 딕셔너리에서 필수 키를 찾을 수 없습니다: {e}. 'face_detector', 'shape_predictor', 'face_recognizer' 키가 필요합니다.")
            # 필요하다면 여기서 run_main을 중단시킬 수 있습니다. 예: return
        except Exception as e: # TypeError 등 다른 예외 처리
            logger.critical(f"dlib 모델 변수 할당 중 예상치 못한 오류 발생: {e}")
            # 필요하다면 여기서 run_main을 중단시킬 수 있습니다. 예: return

    # 0.3. 입력 디렉토리 및 JSON 키 설정 가져오기
    dataset_base_key_str = "project.paths.datasets" # YAML 내 데이터셋 경로의 기본 키
    input_dir_key = f"{dataset_base_key_str}.raw_jsons_dir" # 처리할 원본 JSON 파일들이 있는 디렉토리 키
    input_dir_str = cfg.get_value(input_dir_key) # 설정에서 경로 문자열 가져오기
    input_dir = Path(input_dir_str).expanduser().resolve() # Path 객체로 변환 및 절대 경로화

    # JSON 키 설정을 configger에서 가져옵니다. 'json_keys'는 YAML 파일 내의 관련 설정 섹션 이름입니다.
    json_key_config_data = cfg.get_value('json_keys')
    if not json_key_config_data or not isinstance(json_key_config_data, dict):
        logger.critical("YAML 설정 파일에서 'json_keys' 섹션을 찾을 수 없거나 형식이 잘못되었습니다.")
        sys.exit(1)

    # JsonConfigHandler 인스턴스 생성
    try:
        json_handler = JsonConfigHandler(json_key_config_data)
    except Exception as e_json_handler:
        logger.error(f"JsonConfigHandler 초기화 중 오류 발생: {e_json_handler}")
        sys.exit(1)

    # 1. 입력 JSON 파일 목록 가져오기 및 처리

    # 1.1. JSON 파일 목록 가져오기 및 총 개수 세기 (메모리 효율적)
    logger.info(f"'{input_dir}' 디렉토리에서 JSON 파일 탐색 시작...")

    # glob 결과 이터레이터를 생성 (개수 세기용)
    # 이 이터레이터는 아래 sum() 함수에 의해 소모됩니다.
    json_file_iterator_for_counting = input_dir.glob("**/*.json")

    # 1.2. is_file() 필터링을 적용하면서 개수를 셉니다.
    # sum(1 for ...) 구문은 이터레이터를 순회하며 각 요소에 대해 1을 더하여 총 개수를 계산합니다.
    # 이 과정에서 전체 경로를 메모리에 리스트로 저장하지 않습니다.
    total_input_found = sum(1 for p in json_file_iterator_for_counting if p.is_file())

    if total_input_found == 0:
        logger.warning(f"1.2. '{input_dir}' 디렉토리에서 인덱싱할 JSON 파일을 찾을 수 없습니다.")
        logger.info("✅ 최종 통계:")
        logger.info(f"   - 탐색된 JSON 파일 총 개수: {total_input_found}")
        logger.info(f"   - 인덱싱된 총 얼굴 개수: 0")
        logger.info(f"{Path(__file__).name} 정상 종료 (처리할 파일 없음).")
        sys.exit(0)
    status["total_input_found"]["value"] = total_input_found

    logger.info(f'✅ 인덱싱할 JSON 파일 {status["total_input_found"]["value"]}개 발견.')

    digit_width = calc_digit_number(total_input_found) # 로그 출력 시 숫자 너비 계산
    # 1.2. 모든 파일에서 얼굴 정보 누적 (메모리 효율적인 파일 순회)
    # glob 결과를 다시 생성 (실제 처리용)
    json_file_iterator_for_processing = input_dir.glob("**/*.json")

    all_embeddings: List[np.ndarray] = [] # 모든 파일의 임베딩을 누적할 리스트
    all_metadatas: List[Dict[str, Any]] = [] # 모든 파일의 메타데이터를 누적할 리스트
    total_faces_processed = 0 # 성공적으로 처리된 총 얼굴 개수
    # batch_index = 0  # 배치 처리 시 사용 (현재는 모든 데이터 수집 후 일괄 처리)
    # total_faces_failed 변수는 필요에 따라 추가

     # 1.3. JSON 파일 내 얼굴 정보 수집 시작...
    logger.info("JSON 파일 내 얼굴 정보 수집 시작...")
    for json_file_path in json_file_iterator_for_processing: # 이터레이터를 순회
        if not json_file_path.is_file():
            status["error_input_file_read"]["value"] += 1 # 파일이 아닌 경우 오류 카운트
            continue

        status["req_process_count"]["value"] += 1
        logger.debug(f"1.3. [{status['req_process_count']['value']:>{digit_width}}/{status['total_input_found']['value']}] JSON 파일 처리 중: {json_file_path.name}")

        # 2. 현재 파일(json_file_path)에서 얼굴의 임베딩과 메타데이터 가져와서 indexing
        # json_key_config_data를 전달하도록 수정
        embeddings_from_file, metadatas_from_file = get_all_face_data_from_json_alone(
            cfg, 
            json_file_path,
            json_handler           # JsonConfigHandler 인스턴스 전달
            )

        if embeddings_from_file: # 파일에서 유효한 임베딩이 추출된 경우
            all_embeddings.extend(embeddings_from_file) # 누적 리스트에 추가
            all_metadatas.extend(metadatas_from_file) # 누적 리스트에 추가
            total_faces_processed += len(embeddings_from_file) # 총 얼굴 개수 누적
            logger.info(f"  '{json_file_path.name}' 파일에서 얼굴 {len(embeddings_from_file)}개 정보 추출 완료. (누적 {total_faces_processed}개)")

        else:
            # 파일 내에 유효한 얼굴 정보가 없거나 오류 발생 시
            status["error_embedding_general"]["value"] += 1 # 보다 일반적인 에러 카운터 사용
            logger.warning(f"  '{json_file_path.name}' 파일에서 유효한 얼굴 정보를 추출하지 못했습니다.")

        # 테스트 목적으로 특정 파일 개수만 처리 후 중단 (개발 시 유용)
        # if status["req_process_count"]["value"] > 2:
        #     break

    # 3. 모든 파일 처리 후, 수집된 전체 임베딩으로 FAISS 인덱스 구축
    if not all_embeddings:
        logger.warning("3. 수집된 얼굴 임베딩이 없어 FAISS 인덱스를 생성하지 않습니다.")
        status["error_embedding_empty_target"]["value"] +=1
    else:
        logger.info(f"총 {len(all_embeddings)}개의 얼굴 임베딩을 사용하여 FAISS 인덱스 구축 및 저장 시작...")
        build_and_save_index_alone(all_embeddings, all_metadatas, cfg)

    # 9. 모든 이미지 처리 완료 또는 중단 후 자원 해제
    # 9-1. 통계 결과 출력 ---
    logger.info("--- JSON 파일 처리 및 인덱싱 통계 ---") # 헤더 메시지 변경
    # 통계 메시지 중 가장 긴 것을 기준으로 출력 너비 조절 (visual_length 사용)
    max_visual_msg_len = 0
    if DEFAULT_STATUS_TEMPLATE: # DEFAULT_STATUS_TEMPLATE이 비어있지 않은 경우에만 실행
        # status 딕셔너리에 있는 키들 중 DEFAULT_STATUS_TEMPLATE에도 있는 키의 메시지만 고려
        valid_msgs = [
            DEFAULT_STATUS_TEMPLATE[key]["msg"]
            for key in status.keys()
            if key in DEFAULT_STATUS_TEMPLATE and "msg" in DEFAULT_STATUS_TEMPLATE[key]
        ]
        if valid_msgs: # 유효한 메시지가 있을 경우에만 max 계산
            max_visual_msg_len = max(visual_length(msg) for msg in valid_msgs)

    # 통계 출력 시 사용할 숫자 너비 계산 (가장 큰 값 기준)
    # status 딕셔너리의 모든 value들을 가져와서 그 중 최대값을 기준으로 너비 계산
    all_values = [data["value"] for data in status.values() if isinstance(data.get("value"), int)]
    max_val_for_width = max(all_values) if all_values else 0 # 모든 값이 0이거나 없을 경우 대비

    # 로그 출력 시 숫자 너비는 위에서 total_input_found 기준으로 계산된 digit_width를 사용합니다.
    # 여기서는 run_main 초반에 계산된 digit_width를 그대로 사용합니다.

    fill_char = '-' # 채움 문자 변경
    logger.info("--- 이미지 파일 처리 통계 ---")
    for key, data in status.items():
        # DEFAULT_STATUS_TEMPLATE에 해당 키가 있을 경우에만 메시지를 가져오고, 없으면 키 이름을 사용
        msg = DEFAULT_STATUS_TEMPLATE.get(key, {}).get("msg", key)
        value = data["value"]
        # f-string의 기본 정렬은 문자 개수 기준이므로, visual_length에 맞춰 수동으로 패딩 추가
        padding_spaces = max(0, max_visual_msg_len - visual_length(msg))
        logger.info(f"{msg}{fill_char * padding_spaces} : {value:>{digit_width}}") # 기존 digit_width 사용
    logger.info("------------------------------------") # 구분선 길이 조정
    # --- 통계 결과 출력 끝 ---

if __name__ == "__main__":
    # 0. 애플리케이션 시작 및 인자 파싱
    logger.info(f"애플리케이션 시작")
    parsed_args = get_argument()

    # 1. 로거 설정
    script_name = Path(__file__).stem # 로깅 및 파일 이름에 사용할 스크립트 이름
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

    # 2. 설정(configger) 초기화
    # configger는 위에서 설정된 로거를 내부적으로 사용할 수 있습니다.
    logger.info(f"Configger 초기화 시도: root_dir='{parsed_args.root_dir}', config_path='{parsed_args.config_path}'")

    try:
        cfg_object = configger(root_dir=parsed_args.root_dir, config_path=parsed_args.config_path)
        logger.info(f"Configger 초기화 완료")
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}")
        sys.exit(1) # 설정 로드 실패 시 종료

    # 3. 메인 로직 실행
    logger.info(f"메인 처리 로직 시작...")
    try:
        run_main(cfg_object) # 설정 객체를 run_main 함수에 전달
    except KeyError as e:
        logger.critical(f"설정 파일에서 필수 경로 키를 찾을 수 없습니다: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"설정 파일에서 경로 정보 처리 중 오류 발생: {e}")
        sys.exit(1)
    finally:
        # 애플리케이션 종료 시 로거 정리 (필요시)
        logger.info(f"{script_name} 애플리케이션 종료")
        logger.shutdown()
        exit(0)

    # 최종 print 문은 로깅으로 대체되었으므로 제거하거나 주석 처리합니다.
    # print("모든 JSON 파일 경로 처리가 완료되었습니다.")


