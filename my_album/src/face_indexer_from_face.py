# src/face_indexer_from_face.py

# 표준 라이브러리 임포트
import os
import sys
import json
from pathlib import Path
import datetime
from typing import Dict, Any, List, Tuple, Set
import argparse

# 사용자 정의 유틸리티 모듈 임포트
try:
    from my_utils.config_utils.SimpleLogger import logger, get_log_cfg_argument
    from my_utils.config_utils.configger import configger
    from my_utils.photo_utils.object_utils import compute_sha256, load_json, save_object_json_with_polygon, save_cropped_face_image
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
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

# --- 전역 변수: dlib 모델 (선택적 유지) ---
# 이 스크립트에서는 직접 사용하지 않지만, 설정 파일 로직 등에서 참조할 수 있으므로 유지합니다.
# 또는, dlib 모델 로드 관련 부분을 완전히 제거해도 됩니다. 여기서는 일단 유지합니다.
face_detector_dlib = None
shape_predictor = None
face_recognizer = None

def load_dlib_models(cfg_obj: configger) -> bool:
    """
    dlib의 얼굴 인식 관련 모델 파일들을 로드하여 전역 변수에 할당합니다.
    (이 스크립트에서는 직접 사용하지 않지만, 호환성 또는 향후 사용을 위해 유지)

    Args:
        cfg_obj (configger): 설정 파일 내용을 담고 있는 configger 객체.

    Returns:
        bool: 모델 로드 성공 여부.
    """
    global face_detector_dlib, shape_predictor, face_recognizer

    try:
        dlib_models_dir_str = cfg_obj.get_value('models.face_recognition.dlib_models_dir')
        if not dlib_models_dir_str:
            logger.warning("YAML 설정 파일에 'models.face_recognition.dlib_models_dir' 경로가 없어 dlib 모델을 로드하지 않습니다.")
            return True # dlib 모델 로드가 필수는 아니므로 True 반환

        dlib_models_dir = Path(dlib_models_dir_str)

        if not dlib_models_dir.exists():
            logger.warning(f"dlib 모델 디렉토리 '{dlib_models_dir}'가 존재하지 않아 모델을 로드할 수 없습니다.")
            # 필요시 생성 로직 추가 가능: dlib_models_dir.mkdir(parents=True, exist_ok=True)
            return True
        elif not dlib_models_dir.is_dir():
            logger.warning(f"dlib 모델 경로 '{dlib_models_dir}'는 디렉토리가 아닙니다.")
            return True

        face_rec_model_name = cfg_obj.get_value('models.face_recognition.face_rec_model_name', "dlib_face_recognition_resnet_model_v1.dat")
        landmark_model_name = cfg_obj.get_value('models.face_recognition.landmark_model_name', "shape_predictor_68_face_landmarks.dat")
        
        face_rec_model_path = dlib_models_dir / face_rec_model_name
        landmark_model_path = dlib_models_dir / landmark_model_name

        if not face_rec_model_path.exists() or not landmark_model_path.exists():
            logger.warning(f"dlib 모델 파일 일부 또는 전체를 찾을 수 없습니다. ({face_rec_model_path.name}, {landmark_model_path.name})")
            return True # 필수가 아니므로 계속 진행

        # import 된 dlib 객체 
        face_detector_dlib = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(str(landmark_model_path))
        face_recognizer = dlib.face_recognition_model_v1(str(face_rec_model_path))
        logger.info("참고: dlib 얼굴 인식 관련 모델 로드 완료 (이 스크립트에서 직접 사용되지 않을 수 있음).")
        return True

    except Exception as e:
        logger.warning(f"dlib 모델 로드 중 오류 발생 (무시하고 진행): {e}", exc_info=True)
        return True # 오류 발생 시에도 계속 진행

def get_all_face_data_from_json(json_file_path: Path, json_key_config: Dict[str, Any]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    주어진 JSON 파일에서 모든 얼굴의 임베딩과 메타데이터를 추출합니다.
    키 이름은 config.yaml에서 로드한 json_key_config 딕셔너리에서 가져옵니다.

    Args:
        json_file_path (Path): 얼굴 정보를 추출할 JSON 파일의 경로.
        json_key_config (Dict[str, Any]): config.yaml의 data_structure_keys 섹션에서 로드한 설정 딕셔너리.
                                          YAML 구조에 따라 중첩된 딕셔너리일 수 있습니다.

    Returns:
        Tuple[List[np.ndarray], List[Dict[str, Any]]]:
            - 추출된 모든 얼굴 임베딩 리스트 (각 요소는 NumPy 배열).
            - 추출된 모든 얼굴 메타데이터 딕셔너리 리스트.
            - 오류 발생 또는 데이터 없음 시 ([], []) 반환.
    """
    embeddings_in_file: List[np.ndarray] = []
    metadatas_in_file: List[Dict[str, Any]] = []

    # --- config에서 JSON 키 이름 가져오기 ---
    # json_key_config의 구조에 맞춰 단계적으로 접근합니다.
    # get() 메서드를 사용하여 키가 없을 경우를 대비하고, 기본값을 설정합니다.

    # 최상위 'detected_obj' 리스트의 키 이름
    detected_objects_list_key_name = json_key_config.get('detected_objects_list_key', 'detected_obj')

    # 이미지 정보 관련 키 이름들을 담고 있는 하위 딕셔너리 가져오기
    image_info_keys_map = json_key_config.get('image_info_keys', {})
    image_path_key_name = image_info_keys_map.get('path', 'image_path')
    image_hash_key_name = image_info_keys_map.get('hash', 'image_hash')
    # 이미지 크기 키 (config에 추가했다면)
    # image_width_key_name = image_info_keys_map.get('width', 'image_width')
    # image_height_key_name = image_info_keys_map.get('height', 'image_height')


    # 객체 정보 관련 키 이름들을 담고 있는 하위 딕셔너리 가져오기
    object_keys_map = json_key_config.get('object_keys', {})
    # 각 객체 안의 얼굴 목록 ('detected_face_crop') 키 이름
    face_crops_list_key_name = object_keys_map.get('face_crops_list_key', 'detected_face_crop')
    # 객체 메타데이터에 사용할 키 이름들 (필요하다면)
    object_class_name_key = object_keys_map.get('class_name', 'class_name')
    object_box_xyxy_key = object_keys_map.get('box_xyxy', 'box_xyxy')


    # 얼굴 정보 관련 키 이름들을 담고 있는 하위 딕셔너리 가져오기
    face_keys_map = json_key_config.get('face_keys', {})
    # 얼굴 임베딩 및 메타데이터 추출에 필요한 키 이름들
    embedding_key_name = face_keys_map.get('embedding', 'embedding')
    face_id_key_name = face_keys_map.get('face_id', 'face_id')
    face_box_key_name = face_keys_map.get('box', 'box') # 함수 설명의 'box' 키
    embedding_score_key_name = face_keys_map.get('score', 'score') # 함수 설명의 'score' 키
    # JSON 예시에 있는 얼굴 감지 결과 키들 (메타데이터에 포함)
    detected_face_bbox_key = face_keys_map.get('bbox_xyxy', 'bbox_xyxy')
    detected_face_confidence_key = face_keys_map.get('confidence', 'confidence')
    detected_face_label_key = face_keys_map.get('label', 'label')

    # --- JSON 파일 로드 및 데이터 추출 ---
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # config에서 가져온 최상위 키 변수 사용: 'detected_obj' 리스트에 접근
        detected_objects = data.get(detected_objects_list_key_name)
        if not detected_objects or not isinstance(detected_objects, list):
            logger.warning(f"JSON 파일 '{json_file_path.name}'에 '{detected_objects_list_key_name}' 리스트가 없거나 형식이 잘못되었습니다.")
            return [], []

        # 'detected_obj' 리스트 순회
        for obj_entry in detected_objects:
             # config에서 가져온 객체 내 얼굴 목록 키 변수 사용: 'detected_face_crop' 리스트에 접근
            face_crops_in_obj = obj_entry.get(face_crops_list_key_name)

            # 해당 객체에 얼굴 정보가 없거나 형식이 잘못된 경우 다음 객체로 넘어감
            if not face_crops_in_obj or not isinstance(face_crops_in_obj, list):
                 continue

            # 'detected_face_crop' 리스트 순회 (실제 얼굴 정보 항목들)
            for face_entry in face_crops_in_obj:
                # --- 각 얼굴 항목에서 임베딩 및 메타데이터 추출 ---

                # config에서 가져온 임베딩 키 변수 사용: 'embedding' 데이터에 접근
                embedding_data = face_entry.get(embedding_key_name)

                # 임베딩 데이터가 없거나 형식이 잘못된 경우 건너뛰기
                if embedding_data is None:
                    # config에서 가져온 face_id 키 변수 사용 (로그 메시지용)
                    face_id_info = face_entry.get(face_id_key_name, 'N/A')
                    logger.debug(f"JSON 파일 '{json_file_path.name}'의 face_id '{face_id_info}'에 임베딩 정보가 없습니다. 건너<0xEB><0><0x8E>니다.")
                    continue

                try:
                    # 임베딩 데이터를 NumPy 배열로 변환 (float32 타입으로)
                    embedding_np = np.array(embedding_data, dtype=np.float32)
                    # 필요시 임베딩 차원 검증 로직 추가 (config에서 embedding_dim 가져와서 비교)
                except Exception as e_np:
                    face_id_info = face_entry.get(face_id_key_name, 'N/A')
                    logger.warning(f"JSON 파일 '{json_file_path.name}'의 face_id '{face_id_info}' 임베딩 변환 중 오류: {e_np}. 건너<0xEB><0><0x8E>니다.")
                    continue

                # 메타데이터 구성 (config에서 가져온 키 변수들 사용)
                metadata = {
                    "source_json_path": str(json_file_path), # 현재 JSON 파일 경로
                    "original_image_path": data.get(image_path_key_name), # 원본 이미지 경로 키 사용
                    "original_image_hash": data.get(image_hash_key_name), # 원본 이미지 해시 키 사용
                    # 원본 이미지 크기 정보 (config에 추가하고 JSON에도 있다면 가져오기)
                    # "original_image_width": data.get(image_width_key_name),
                    # "original_image_height": data.get(image_height_key_name),
                    "face_id": face_entry.get(face_id_key_name), # 얼굴 ID 키 사용
                    "face_bbox_in_obj": face_entry.get(face_box_key_name), # 함수 설명의 얼굴 box 키 사용
                    "embedding_score": face_entry.get(embedding_score_key_name), # 함수 설명의 임베딩 스코어 키 사용

                    # JSON 예시에 있던 얼굴 감지 결과 키들 (config에 추가했다면 config 키 사용)
                    "detected_face_bbox_xyxy": face_entry.get(detected_face_bbox_key),
                    "detected_face_confidence": face_entry.get(detected_face_confidence_key),
                    "detected_face_label": face_entry.get(detected_face_label_key),

                    # 얼굴을 포함하는 상위 객체 정보 (config에 추가했다면 config 키 사용)
                    "detected_object_class": obj_entry.get(object_class_name_key),
                    "detected_object_bbox_xyxy": obj_entry.get(object_box_xyxy_key)
                }

                # 추출된 임베딩과 메타데이터를 리스트에 추가
                embeddings_in_file.append(embedding_np)
                metadatas_in_file.append(metadata)

        # 모든 객체와 그 안의 모든 얼굴 정보를 순회한 후 결과 반환
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

# 함수 호출 시 config에서 로드한 json_key_config 딕셔너리를 전달해야 합니다.
# embeddings_from_file, metadatas_from_file = get_all_face_data_from_json(json_file_path, json_key_config)


# def get_all_face_data_from_json(json_file_path: Path, key:str) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
#     """
#     주어진 JSON 파일에서 모든 얼굴의 임베딩과 메타데이터를 추출합니다.

#     가정:
#     - JSON 파일 내 key 키는 얼굴 정보 딕셔너리 리스트를 값으로 가집니다.
#     - 각 얼굴 정보 딕셔너리에는 'embedding' (얼굴 특징 벡터), 'face_id', 'box', 'score' 등의 키가 있습니다.
#     - 'embedding'은 리스트 또는 NumPy 배열 형태로 저장되어 있습니다.

#     Args:
#         json_file_path (Path): 얼굴 정보를 추출할 JSON 파일의 경로.

#     Returns:
#         Tuple[List[np.ndarray], List[Dict[str, Any]]]:
#             - 추출된 모든 얼굴 임베딩 리스트 (각 요소는 NumPy 배열).
#             - 추출된 모든 얼굴 메타데이터 딕셔너리 리스트.
#             - 오류 발생 또는 데이터 없음 시 ([], []) 반환.
#     """
#     embeddings_in_file: List[np.ndarray] = []
#     metadatas_in_file: List[Dict[str, Any]] = []

#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)

#         if key not in data or not isinstance(data[key], list):
#             logger.warning(f"JSON 파일 '{json_file_path.name}'에 key 리스트가 없거나 형식이 잘못되었습니다.")
#             return [], []

#         for face_entry in data[key]:
#             # JSON에서 임베딩 추출 (키 이름은 실제 데이터에 맞게 조정 필요)
#             embedding_data = face_entry.get('embedding') # 또는 'face_embedding', 'vector' 등
#             if embedding_data is None:
#                 logger.debug(f"JSON 파일 '{json_file_path.name}'의 face_id '{face_entry.get('face_id')}'에 임베딩 정보가 없습니다. 건너<0xEB><0><0x8E>니다.")
#                 continue
            
#             try:
#                 # 임베딩 데이터를 NumPy 배열로 변환 (float32 타입으로)
#                 embedding_np = np.array(embedding_data, dtype=np.float32)
#                 # 필요시 차원 검증 로직 추가: if embedding_np.shape[0] != 128: ...
#             except Exception as e_np:
#                 logger.warning(f"JSON 파일 '{json_file_path.name}'의 face_id '{face_entry.get('face_id')}' 임베딩 변환 중 오류: {e_np}. 건너<0xEB><0><0x8E>니다.")
#                 continue

#             # 메타데이터 구성 (필요한 정보만 선택적으로 포함)
#             # 'face_image_path'는 더 이상 개별 얼굴 이미지 파일이 없으므로,
#             # 원본 이미지 경로 또는 JSON 파일 경로 등으로 대체하거나 제거합니다.
#             metadata = {
#                 "source_json_path": str(json_file_path), # 어떤 JSON 파일에서 왔는지
#                 "original_image_path": data.get('image_path'),
#                 "original_image_hash": data.get('image_hash'),
#                 "face_id_in_original": face_entry.get('face_id'),
#                 "original_bbox": face_entry.get('box'),
#                 "score": face_entry.get('score'),
#                 # 필요에 따라 face_entry의 다른 정보 추가
#             }
            
#             embeddings_in_file.append(embedding_np)
#             metadatas_in_file.append(metadata)

#         return embeddings_in_file, metadatas_in_file

#     except json.JSONDecodeError:
#         logger.error(f"JSON 파일 파싱 오류: {json_file_path}", exc_info=True)
#         return [], []
#     except Exception as e:
#         logger.error(f"JSON 파일 '{json_file_path.name}' 처리 중 예상치 못한 오류: {e}", exc_info=True)
#         return [], []


def build_and_save_index(embeddings: List[np.ndarray], metadatas: List[Dict[str, Any]], cfg_obj: configger):
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
    if not embeddings:
        # 인덱싱할 임베딩 데이터가 없으면, 정보 로깅 후 함수 종료
        logger.info("인덱싱할 얼굴 특징 벡터가 없습니다. FAISS 인덱스를 생성하지 않습니다.")
        return

    try:
        # FAISS는 float32 타입의 NumPy 배열을 입력으로 사용합니다.
        embeddings_array = np.array(embeddings).astype('float32')
        if embeddings_array.ndim == 1: # 단일 임베딩만 있는 경우 2D로 변환
            if embeddings_array.size > 0:
                embeddings_array = embeddings_array.reshape(1, -1)
            else: # 빈 임베딩인 경우
                logger.info("빈 임베딩 배열입니다. FAISS 인덱스를 생성하지 않습니다.")
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
                logger.warning(
                    f"YAML에 설정된 embedding_dim ({configured_embedding_dim_yaml})과 "
                    f"실제 데이터의 특징 벡터 차원 ({embedding_dim})이 일치하지 않습니다. "
                    f"실제 데이터 차원인 {embedding_dim}을 사용합니다."
                )
        else:
            logger.info(f"YAML에 'indexing.embedding_dim'이 설정되지 않았습니다. 실제 데이터 차원 {embedding_dim}을 사용합니다.")


        # 필수 설정값들이 제대로 로드되었는지 확인
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
            logger.info(f"  - IndexIVFFlat 파라미터: nlist={nlist}")
            if embeddings_array.shape[0] < nlist:
                logger.warning(f"  경고: 학습 데이터 수({embeddings_array.shape[0]})가 nlist({nlist})보다 적습니다. IVFFlat 학습에 영향이 있을 수 있습니다.")
            if embeddings_array.shape[0] > 0: # 학습 데이터가 있어야 학습 가능
                logger.info("  IndexIVFFlat 학습 시작...")
                index.train(embeddings_array)
                logger.info("  IndexIVFFlat 학습 완료.")
            else:
                logger.warning("  학습 데이터가 없어 IndexIVFFlat 학습을 건너<0xEB><0><0x8E>니다.")
        elif faiss_index_type == 'IndexIVFPQ':
            nlist = int(cfg_obj.get_value('indexing.nlist', 100))
            M = int(cfg_obj.get_value('indexing.M', 8)) 
            nbits = int(cfg_obj.get_value('indexing.nbits', 8))
            logger.info(f"  - IndexIVFPQ 파라미터: nlist={nlist}, M={M}, nbits={nbits}")

            if embedding_dim % M != 0:
                logger.warning(f"  경고: IndexIVFPQ 사용 시 embedding_dim({embedding_dim})이 M({M})의 배수가 아닙니다. 성능 저하 또는 오류가 발생할 수 있습니다.")
            
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, M, nbits)
            if embeddings_array.shape[0] < nlist: 
                logger.warning(f"  경고: 학습 데이터 수({embeddings_array.shape[0]})가 nlist({nlist})보다 적습니다. IVFPQ 학습에 영향이 있을 수 있습니다.")
            if embeddings_array.shape[0] > 0: # 학습 데이터가 있어야 학습 가능
                logger.info("  IndexIVFPQ 학습 시작...")
                index.train(embeddings_array)
                logger.info("  IndexIVFPQ 학습 완료.")
            else:
                logger.warning("  학습 데이터가 없어 IndexIVFPQ 학습을 건너<0xEB><0><0x8E>니다.")
        else:
            logger.error(f"지원하지 않는 FAISS 인덱스 타입입니다: '{faiss_index_type}'.")
            logger.error("지원 타입: IndexFlatL2, IndexFlatIP, IndexIVFFlat, IndexIVFPQ")
            return

        if index is None:
            logger.critical("FAISS 인덱스 객체 생성에 실패했습니다. 설정을 확인해주세요.")
            return
        
        if embeddings_array.shape[0] > 0: # 추가할 데이터가 있을 때만 add 수행
            index.add(embeddings_array)
            logger.info(f"FAISS 인덱스에 총 {index.ntotal}개의 벡터 추가 완료.")
        else:
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




if __name__ == "__main__":
    parsed_args = get_log_cfg_argument()

    try:
        script_name = Path(__file__).stem
        date_str = datetime.datetime.now().strftime("%y%m%d")
        log_file_name = f"{script_name}_{date_str}.log"
        full_log_path = Path(parsed_args.log_dir) / log_file_name
        logger.setup(
            logger_path=str(full_log_path),
            min_level='DEBUG',
            include_function_name=True,
            pretty_print=True
        )
        logger.info(f"JSON 기반 얼굴 인덱싱 애플리케이션 ({Path(__file__).name}) 시작.") # 애플리케이션 이름 변경
        logger.info(f"명령줄 인자로 결정된 경로: {vars(parsed_args)}")
    except Exception as e:
        print(f"치명적 오류: 로거 설정 중 오류 발생 - {e}", file=sys.stderr)
        sys.exit(1)

    cfg_object = None
    try:
        cfg_object = configger(root_dir=parsed_args.root_dir, config_path=parsed_args.config_path)
        logger.info(f"설정 파일 로드 완료: {parsed_args.config_path}")
    except FileNotFoundError:
        logger.critical(f"치명적 오류: 설정 파일({parsed_args.config_path})을 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"치명적 오류: 설정 파일 로드 중 오류 발생 - {e}", exc_info=True)
        sys.exit(1)

    try:
        # dlib 모델 로드는 선택 사항이 되었으므로, 실패해도 치명적이지 않음
        load_dlib_models(cfg_object)

        # 이제 input_json_dir로 detected_face_images_dir이 필요합니다.
        # detected_face_json_dir_key = 'project.paths.datasets.raw_jsons_dir'
        detected_face_json_dir_key = 'project.paths.datasets.detected_face_json_dir'
        logger.debug(f"인덱싱 대상 원본 JSON 디렉토리를 찾을 열쇠:{detected_face_json_dir_key}")
        input_json_dir_str = cfg_object.get_value(detected_face_json_dir_key)
        if not input_json_dir_str:
            logger.critical(f"YAML 설정 파일에 {detected_face_json_dir_key} 경로가 누락되었습니다.")
            sys.exit(1)
        logger.debug(f"인덱싱 대상 원본 JSON 디렉토리를 찾은값:   {input_json_dir_str}")
        input_json_dir = Path(input_json_dir_str).expanduser()
        logger.debug(f"인덱싱 대상 원본 JSON 디렉토리:            {input_json_dir}")
    except KeyError as e:
        logger.critical(f"설정 파일에서 필수 경로 키를 찾을 수 없습니다: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"설정 파일에서 경로 정보 처리 중 오류 발생: {e}")
        sys.exit(1)

    if not input_json_dir.is_dir():
        logger.critical(f"지정된 원본 JSON 디렉토리가 존재하지 않거나 디렉토리가 아닙니다: {input_json_dir}")
        sys.exit(1)

    # 프로그램 시작 부분에서 ConfigManager 객체 생성 및 설정 로드 후
    json_key_config_data = cfg_object.get_json_key_config()

    # JSON 파일 목록 가져오기
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
    # json_files = [p for p in input_json_dir.glob("**/*.json") if p.is_file()]
    # total_json_files_found = len(json_files)    """

    # 1. JSON 파일 목록 가져오기 및 총 개수 세기 (메모리 효율적)
    logger.info(f"'{input_json_dir}' 디렉토리에서 JSON 파일 탐색 시작...")

    # glob 결과 이터레이터를 생성 (개수 세기용)
    # 이 이터레이터는 아래 sum() 함수에 의해 소모됩니다.
    json_file_iterator_for_counting = input_json_dir.glob("**/*.json")

    # is_file() 필터링을 적용하면서 개수 세기
    # sum(1 for ...) 구문은 이터레이터를 순회하며 각 요소에 대해 1을 더하여 개수를 셉니다.
    # 이 과정에서 전체 경로를 메모리에 리스트로 저장하지 않습니다.
    total_json_files_found = sum(1 for p in json_file_iterator_for_counting if p.is_file())

    if total_json_files_found == 0:
        logger.warning(f"'{input_json_dir}' 디렉토리에서 인덱싱할 JSON 파일을 찾을 수 없습니다.")
        logger.info("✅ 최종 통계:")
        logger.info(f"   - 탐색된 JSON 파일 총 개수: {total_json_files_found}")
        logger.info(f"   - 인덱싱된 총 얼굴 개수: 0")
        logger.info(f"{Path(__file__).name} 정상 종료 (처리할 파일 없음).")
        sys.exit(0)

    logger.info(f"✅ 인덱싱할 JSON 파일 {total_json_files_found}개 발견.")

    # 2. 모든 파일에서 얼굴 정보 누적 (메모리 효율적인 파일 순회)
    # glob 결과를 다시 생성 (실제 처리용)
    # 개수를 세느라 첫 번째 이터레이터가 소모되었으므로, 실제 처리를 위해서는 새로 만들어야 합니다.
    json_file_iterator_for_processing = input_json_dir.glob("**/*.json")

    all_embeddings: List[np.ndarray] = [] # 모든 파일의 임베딩을 누적할 리스트
    all_metadatas: List[Dict[str, Any]] = [] # 모든 파일의 메타데이터를 누적할 리스트
    total_faces_processed = 0 # 성공적으로 처리된 총 얼굴 개수
    # total_faces_failed 변수는 필요에 따라 추가

    logger.info("JSON 파일 내 얼굴 정보 수집 시작...")
    # enumerate를 사용하여 진행 상황을 표시하기 위해 리스트로 변환하는 대신,
    # total_json_files_found를 활용하여 수동으로 카운트하며 진행 상황을 표시하는 것이 좋습니다.
    processed_file_count = 0
    for json_file_path in json_file_iterator_for_processing: # 이터레이터를 순회
        if json_file_path.is_file(): # 파일인지 다시 확인 (glob 결과는 대부분 파일이지만 안전을 위해)
            processed_file_count += 1 # 처리 시작 파일 카운트
            logger.debug(f"[{processed_file_count}/{total_json_files_found}] JSON 파일 처리 중: {json_file_path.name}")

            # 현재 파일(json_file_path)에서 얼굴의 임베딩과 메타데이터 가져오기
            # get_all_face_data_from_json 함수 호출 시 config에서 가져온 키 설정 전달
            embeddings_from_file, metadatas_from_file = get_all_face_data_from_json(json_file_path, json_key_config_data)

            if embeddings_from_file: # 파일에서 유효한 임베딩이 추출된 경우
                all_embeddings.extend(embeddings_from_file) # 누적 리스트에 추가
                all_metadatas.extend(metadatas_from_file) # 누적 리스트에 추가
                total_faces_processed += len(embeddings_from_file) # 총 얼굴 개수 누적
                logger.info(f"  '{json_file_path.name}' 파일에서 얼굴 {len(embeddings_from_file)}개 정보 추출 완료. (누적 {total_faces_processed}개)")
            else:
                # 파일 내에 유효한 얼굴 정보가 없거나 오류 발생 시
                logger.warning(f"  '{json_file_path.name}' 파일에서 유효한 얼굴 정보를 추출하지 못했습니다.")
                # 실패 카운트 로직은 필요에 따라 추가
    # 모든 파일 처리가 완료된 후 누적된 정보 요약
    logger.info("\n=== JSON 파일 내 얼굴 정보 수집 완료 ===")
    logger.info(f"  총 탐색된 JSON 파일 개수: {total_json_files_found}")
    logger.info(f"  성공적으로 추출된 총 얼굴 임베딩/메타데이터 개수: {total_faces_processed}")
    logger.info("========================================\n")


    # 3. FAISS 인덱스 구축 및 저장 (모든 정보 수집 후 한 번 실행)
    try:
        # 누적된 모든 임베딩과 메타데이터를 사용하여 인덱스 구축 및 저장
        build_and_save_index(all_embeddings, all_metadatas, cfg_object)

    except FileNotFoundError as e_fnf:
        logger.error(f"FAISS 인덱스 구축 중 설정 파일 관련 오류 발생: {e_fnf}")
        sys.exit(1) # 오류 발생 시 종료

    except Exception as e_main:
        logger.error(f"FAISS 인덱스 구축 및 저장 중 처리되지 않은 오류 발생: {e_main}", exc_info=True)
        sys.exit(1) # 오류 발생 시 종료

    finally:
        # 정상 종료 또는 오류 종료 시 항상 실행될 정리 작업
        logger.info(f"JSON 기반 얼굴 인덱싱 애플리케이션 ({Path(__file__).name}) 종료.")
        if hasattr(logger, 'shutdown') and callable(logger.shutdown):
            logger.shutdown()

    # --- 실제 적용 코드 끝 ---

    # 이 print문은 logger.info의 final 메시지 이후에 실행되므로 필요 없거나 위치 조정 필요
    # print("모든 JSON 파일 경로 처리가 완료되었습니다.")
    print("모든 JSON 파일 경로 처리가 완료되었습니다.")


