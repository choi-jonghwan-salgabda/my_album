import os
import sys     # 표준 출력 스트림 사용을 위해 필요
from pathlib import Path # 경로 관리를 위해 필요
from typing import List, Dict, Any, Tuple, Optional
import json
import base64 # search_by_face_capture에서 사용하므로 최상단으로 이동
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import traceback # 초기 오류 로깅에 사용
# 실제 얼굴 검출 및 특징 추출을 위한 dlib 임포트 (예시)
import dlib
import numpy as np
import cv2 # 이미지 처리를 위해 OpenCV 임포트
import faiss # FAISS 임포트

# shared_utils 패키지에서 configger 클래스 가져오기
# 사용자 정의 유틸리티 모듈 임포트
try:
    from datetime import datetime
    from my_utils.config_utils.SimpleLogger import logger, get_argument
    from my_utils.config_utils.configger import configger
    from my_utils.object_utils.data_loading_helpers import load_all_json_from_list_file, load_json_batch_from_list_file # data_loading_helpers에서 필요한 함수 임포트
    from my_utils.object_utils.photo_utils import JsonConfigHandler # JsonConfigHandler 임포트
    # 얼굴 검출 및 특징 추출, 검색 관련 모듈 임포트 (가상)
    # from my_album.src.face_utils import detect_and_extract_features  # 예시: 얼굴 검출 및 특징 추출 함수
    # from my_album.src.search_utils import search_similar_faces       # 예시: 유사 얼굴 검색 함수
except ImportError as e: # logger 사용하도록 변경 (main에서 초기화 후)
    # logger가 초기화되기 전이므로 print 사용
    # 실제 발생한 예외 e를 출력하여 원인 파악
    # 이 부분은 logger.setup 이후에 logger를 사용하도록 변경하는 것이 좋습니다.
    # 다만, 현재 코드는 logger.setup이 main 블록 안에 있어, 모듈 로딩 시점에는 logger 사용이 어려울 수 있습니다.
    # 따라서 초기 임포트 오류는 print로 남기고, traceback을 사용하여 상세 정보를 출력합니다.
    # 실제 발생한 예외 e를 출력하여 원인 파악
    print(f"모듈 임포트 중 오류 발생: {e}")
    print(f"자세한 오류 정보:")
    traceback.print_exc() # 전체 트레이스백 출력 (개발 단계에서 유용)
    sys.exit(1)

# --- Gunicorn 환경과 직접 실행 환경을 위한 설정 값 로드 ---
# Gunicorn으로 실행될 때는 환경 변수나 기본값을 사용하고,
# 직접 실행될 때는 get_argument()로 명령줄 인자를 파싱합니다.

IS_GUNICORN = "gunicorn" in sys.argv[0]

if not IS_GUNICORN:
    # 직접 실행 시에만 명령줄 인자 파싱
    parsed_args = get_argument()
    ROOT_DIR_FOR_CONFIG = parsed_args.root_dir
    CONFIG_PATH_FOR_CONFIG = parsed_args.config_path
    LOG_DIR_FOR_LOGGER = parsed_args.log_dir
    LOG_LEVEL_FOR_LOGGER = parsed_args.log_level
else:
    # Gunicorn으로 실행될 때의 기본값 또는 환경 변수 사용
    # 예: 환경 변수에서 루트 디렉토리, 설정 파일 경로, 로그 디렉토리 등을 가져옵니다.
    # 이 값들은 Gunicorn 실행 스크립트나 .env 파일 등을 통해 설정할 수 있습니다.
    # 여기서는 예시로 하드코딩된 기본값을 사용하지만, 실제로는 환경 변수 사용을 권장합니다.
    # 프로젝트 루트를 기준으로 상대 경로를 사용하거나, 절대 경로를 환경 변수로 전달합니다.
    # 현재 app.py 위치: app.py
    # 프로젝트 루트: /home/owner/SambaData/Backup/FastCamp/Myproject/
    # my_utils 루트: /home/owner/SambaData/Backup/FastCamp/Myproject/my_utils/
    # web_service 루트: /home/owner/SambaData/Backup/FastCamp/Myproject/web_service/
    # 현재 pyproject.toml 위치: /home/owner/SambaData/Backup/FastCamp/Myproject/web_service/my_album_app/

    # Gunicorn 실행 시 프로젝트 루트를 기준으로 경로 설정
    # 이 경로는 Gunicorn 실행 위치에 따라 달라질 수 있으므로 주의해야 합니다.
    # 가장 안전한 방법은 환경 변수를 사용하는 것입니다.
    # 예: os.environ.get("MY_ALBUM_ROOT_DIR", "/default/path/to/root")
    # 여기서는 app.py 파일 위치를 기준으로 상대 경로를 계산해봅니다.
    _current_file_path = Path(__file__).resolve()
    # app.py -> my_album_labeling_app -> src -> my_album_app -> web_service -> Myproject
    _project_root_from_app = _current_file_path.parent.parent.parent.parent.parent 

    ROOT_DIR_FOR_CONFIG = os.environ.get("MY_ALBUM_ROOT_DIR", str(_project_root_from_app))
    CONFIG_PATH_FOR_CONFIG = os.environ.get("MY_ALBUM_CONFIG_PATH", str(Path(ROOT_DIR_FOR_CONFIG) / "config" / "photo_album.yaml"))
    LOG_DIR_FOR_LOGGER = os.environ.get("MY_ALBUM_LOG_DIR", str(Path(ROOT_DIR_FOR_CONFIG) / "logs" / "my_album_app"))
    LOG_LEVEL_FOR_LOGGER = os.environ.get("MY_ALBUM_LOG_LEVEL", "INFO")

    # Gunicorn 환경에서는 SimpleLogger의 print 출력을 최소화하기 위해 로거를 먼저 설정합니다.
    try:
        _log_file_name = f"{Path(__file__).stem}_{datetime.now().strftime('%y%m%d')}_gunicorn.log"
        _full_log_path = Path(LOG_DIR_FOR_LOGGER) / _log_file_name
        Path(LOG_DIR_FOR_LOGGER).mkdir(parents=True, exist_ok=True) # 로그 디렉토리 생성
        logger.setup(
            logger_path=str(_full_log_path),
            min_level=LOG_LEVEL_FOR_LOGGER,
            include_function_name=True,
            pretty_print=True
        )
        logger.info(f"Gunicorn 환경 로거 초기화 완료. 로그 경로: {_full_log_path}")
    except Exception as e_log_setup:
        print(f"Gunicorn 로거 설정 중 오류: {e_log_setup}", file=sys.stderr)
        # Gunicorn 환경에서는 sys.exit(1)이 워커를 종료시킬 수 있으므로 주의

logger.info(f"ROOT_DIR_FOR_CONFIG:      {ROOT_DIR_FOR_CONFIG}")
logger.info(f"CONFIG_PATH_FOR_CONFIG:   {CONFIG_PATH_FOR_CONFIG}")
logger.info(f"LOG_DIR_FOR_LOGGER:       {LOG_DIR_FOR_LOGGER}")
logger.info(f"LOG_LEVEL_FOR_LOGGER:     {LOG_LEVEL_FOR_LOGGER}")

# --- 2. Configger 초기화 ---
try:
    global_cfg_object = configger(root_dir=ROOT_DIR_FOR_CONFIG, config_path=CONFIG_PATH_FOR_CONFIG)
except Exception as e_cfg:
    # configger 초기화 실패 시, 로거가 파일에 기록하도록 설정되지 않았을 수 있으므로 stderr에 직접 출력
    logger.critical(f"Configger 초기화 실패: {e_cfg}. Traceback: {traceback.format_exc()}", file=sys.stderr)
    sys.exit(1)

# --- 3. 애플리케이션 전역 변수 및 설정 (configger 사용) ---
logger.info("애플리케이션 전역 설정 로드 시작...")
try:
    # JSON 키 설정을 configger에서 가져옵니다.
    json_key_cfg = global_cfg_object.get_config('json_keys')
    if not json_key_cfg or not isinstance(json_key_cfg, dict):
        logger.critical("YAML 설정 파일에서 'json_keys' 섹션을 찾을 수 없거나 형식이 잘못되었습니다.")
        sys.exit(1)
    try:
        global_json_handler = JsonConfigHandler(json_key_cfg)
    except Exception as e_json_handler:
        logger.error(f"JsonConfigHandler 초기화 중 오류 발생: {e_json_handler}", exc_info=True)
        sys.exit(1)

    # datasets관련 설정
    datasets_cfg = global_cfg_object.get_config('project.paths.datasets')
    if not isinstance(datasets_cfg, dict):
        logger.error("datasets_cfg 설정을 불러오지 못했습니다.")
        sys.exit(1)

    # 여러 JSON 파일 로드용 디렉토리 경로. YAML 키: project.paths.datasets.raw_jsons_dir
    # 이미지 소스 디렉토리.  # YAML 키: project.paths.datasets.raw_image_dir
    global_raw_image_dir_str = datasets_cfg.get('raw_image_dir')
    if not global_raw_image_dir_str:
        logger.critical("필수 설정 누락: 'project.paths.datasets.raw_image_dir'이 설정 파일에 없거나 비어있습니다.")
        sys.exit(1)

    global_raw_jsons_dir_str = datasets_cfg.get('raw_jsons_dir')
    if not global_raw_jsons_dir_str:
        logger.warning("'project.paths.datasets.raw_jsons_dir'이 설정 파일에 없거나 비어있습니다. 관련 기능이 제한될 수 있습니다.")
        sys.exit(1)
    
    # JSON 파일 목록을 담은 .lst 파일 경로
    global_json_list_file_path_str = datasets_cfg.get('json_list_file_path') # YAML 키: project.paths.datasets.json_file_list_path
    if not global_json_list_file_path_str:
        logger.warning("'project.paths.datasets.json_file_list_path'가 설정 파일에 없거나 비어있습니다. 데이터 로딩 방식에 영향을 줄 수 있습니다.")

    # 인덱싱 및 배치 로딩을 위한 배치 크기
    indexing_cfg = global_cfg_object.get_config('indexing')
    global_batch_size_for_indexing = indexing_cfg.get('batch_sizeL', 10) # 기본값 10으로 설정
    if not isinstance(global_batch_size_for_indexing, int) or global_batch_size_for_indexing <= 0:
        logger.warning(f"'indexing.batch_sizeL' 설정이 유효하지 않거나 없습니다. 기본값 10을 사용합니다. 현재 값: {global_batch_size_for_indexing}")
        global_batch_size_for_indexing = 10

    # Web Server관련 경로 설정 (json_save_path 관련 부분 제거)
    web_service_cfg = global_cfg_object.get_config('project.source.web_service')
    if not isinstance(web_service_cfg, dict):
        logger.error("web_service 설정을 불러오지 못했습니다.")
        sys.exit(1)

    # flask_labeling_app_cfg는 현재 직접 사용되지 않으므로, 필요시 로드합니다.
    # flask_labeling_app_cfg = web_service_cfg.get('flask_labeling_app')
    # if not isinstance(flask_labeling_app_cfg, dict):
    #     logger.error("flask_labeling_app_cfg 설정을 불러오지 못했습니다.")
    #     sys.exit(1)

    # 템플릿 및 정적 파일 폴더
    # YAML 키: project.source.web_service.flask_labeling_app.template_folder 등
    global_templates_dir_str = web_service_cfg.get('templates_dir')
    if not global_templates_dir_str:
        logger.critical("필수 설정 누락: 'project.source.web_service.templates_dir'이 설정 파일에 없거나 비어있습니다.")
        sys.exit(1)

    global_static_folder_str = web_service_cfg.get('static_dir')
    if not global_static_folder_str:
        logger.critical("필수 설정 누락: 'project.source.web_service.static_dir'이 설정 파일에 없거나 비어있습니다.")
        sys.exit(1)

    global_image_serve_prefix_str = web_service_cfg.get('image_serve_prefix')
    if not global_image_serve_prefix_str: # 값이 없는 경우
        logger.warning("'project.source.web_service.templates.image_serve_prefix'가 설정 파일에 없거나 비어있습니다. 이미지 제공 URL이 예상과 다를 수 있습니다.")
        # image_serve_prefix_str = "/default_images_prefix" # 필요시 기본값 설정
    elif not global_image_serve_prefix_str.startswith('/'): # 슬래시로 시작하지 않는 경우
        global_image_serve_prefix_str = '/' + global_image_serve_prefix_str
        logger.debug(f"image_serve_prefix에 슬래시를 추가했습니다: {global_image_serve_prefix_str}")

    # 로깅: 설정 값들을 명확히 보여줌
    root_dir = global_cfg_object.get_value('project.root_dir')
    logger.info(f"project.root:{root_dir}")
    source_dir = global_cfg_object.get_value('project.source.src_dir')
    logger.info(f"project.source_dir:{source_dir}")
    web_service_dir = global_cfg_object.get_value('project.source.web_service.web_service_dir')
    logger.info(f"project.source.web_service_dir:{web_service_dir}")
    logger.info(f"  다중 JSON 로드 디렉토리: {global_raw_jsons_dir_str if global_raw_jsons_dir_str else '설정되지 않음'}")
    logger.info(f"  JSON 목록 파일 경로: {global_json_list_file_path_str if global_json_list_file_path_str else '설정되지 않음'}")
    logger.info(f"  인덱싱/배치 로드 크기 (batch_sizeL): {global_batch_size_for_indexing}")
    logger.info(f"  이미지 소스 디렉토리: {global_raw_image_dir_str}")
    logger.info(f"  템플릿(templates_dir) 경로: {global_templates_dir_str}")
    logger.info(f"  정적 폴더: {global_static_folder_str}")
    logger.info(f"  이미지 제공 접두사: {global_image_serve_prefix_str if global_image_serve_prefix_str else '설정되지 않음'}")

except KeyError as e_key:
    # global_cfg_object.get_config 또는 dict.get 을 사용하면 일반적으로 KeyError가 발생하지 않지만,
    # 만약 다른 방식으로 설정에 접근하다가 발생할 경우를 대비합니다.
    logger.critical(f"설정 파일에서 키를 찾는 중 오류 발생: {e_key}. Traceback: {traceback.format_exc()}")
    sys.exit(1)
except TypeError as e_type:
    # 주로 Path() 등에 None과 같은 부적절한 타입의 인자가 전달될 때 발생합니다.
    # 위의 None 또는 빈 문자열 체크로 대부분 방지되지만, 예기치 않은 상황을 대비합니다.
    logger.critical(f"설정 값 처리 중 타입 오류 발생 (예: 경로 관련 설정이 None): {e_type}. Traceback: {traceback.format_exc()}")
    sys.exit(1)
except Exception as e_global_setup:
    logger.critical(f"애플리케이션 전역 설정 로드 중 예상치 못한 오류: {e_global_setup}. Traceback: {traceback.format_exc()}")
    sys.exit(1)

app = Flask(__name__, template_folder=global_templates_dir_str, static_folder=global_static_folder_str) # 변수명 오타 수정
app.config['global_raw_jsons_dir_str'] = global_raw_jsons_dir_str
app.config['global_json_list_file_path_str'] = global_json_list_file_path_str # JSON 목록 파일 경로 저장
app.config['global_raw_image_dir_str'] = global_raw_image_dir_str
app.config['global_batch_size_for_indexing'] = global_batch_size_for_indexing # 배치 크기 저장

# --- 사진 경로와 ID 매핑 생성 (Gunicorn 환경에서도 실행되도록 모듈 레벨에서) ---
# 이 맵은 load_photo_data_from_sources 호출 시마다 새로 생성되는 photos_data에 대한 인덱스 매핑입니다.
# 만약 photos_data가 자주 변경되지 않는다면, 시작 시 한 번 로드하여 전역 변수에 저장하고
# 필요에 따라 업데이트하는 것이 효율적일 수 있습니다.
# 현재는 요청 시마다 load_photo_data_from_sources가 호출되므로, 맵도 요청 처리의 일부로 생성됩니다.
# search_by_face_capture에서 이 맵을 사용하므로, 해당 함수가 호출될 때 맵이 최신 상태여야 합니다.
# 여기서는 모듈 레벨에서 초기화하지 않고, 필요한 라우트 함수 내에서 데이터를 로드하고 맵을 생성하는 방식을 유지합니다.
# (또는 @app.before_request 등을 사용하여 요청 시작 시 데이터를 로드하고 맵을 생성할 수 있습니다.)
# global_photo_path_to_id_map 변수는 search_by_face_capture 함수 내에서만 사용되므로,
# 해당 함수 내에서 데이터를 로드하고 맵을 생성하는 것이 더 명확할 수 있습니다.

# --- 얼굴 검색을 위한 전역 변수 (애플리케이션 시작 시 로드) ---
# 실제 모델 및 인덱스 객체로 초기화 필요
global_face_detector_model = None    # 예: YOLO 모델
global_face_embedder_model = None    # 예: dlib ResNet 모델
global_faiss_index = None            # FAISS 인덱스 객체
global_face_metadata_list = []       # FAISS 인덱스에 대한 메타데이터 리스트

# --- 모델 및 FAISS 인덱스 로드 (Gunicorn 환경에서도 실행되도록 모듈 레벨에서) ---
try:
    logger.info("얼굴 검색 모델 및 FAISS 인덱스 로드 시도...")
    # 실제 모델 로드 로직은 여기에 구현되어야 합니다.
    # 예시: global_face_detector_model = load_my_detector(cfg_oglobal_cfg_objectbject.get_path('models...'))
    
    indexing_cfg = global_cfg_object.get_config('indexing')
    if not isinstance(indexing_cfg, dict):
        logger.error("web_indexing_cfgservice 설정을 불러오지 못했습니다.")
        sys.exit(1)

    faiss_index_path_str = indexing_cfg.get('index_file_path')
    if faiss_index_path_str and Path(faiss_index_path_str).exists():
        import faiss # 실제 사용 시 주석 해제
        global_faiss_index = faiss.read_index(faiss_index_path_str)
        logger.info(f"FAISS 인덱스 로드 완료: {faiss_index_path_str}")
    else:
        logger.warning(f"FAISS 인덱스 파일을 찾을 수 없거나 경로가 설정되지 않았습니다: {faiss_index_path_str}")

    metadata_path_str = indexing_cfg.get('metadata_path')
    if metadata_path_str and Path(metadata_path_str).exists():
        with open(metadata_path_str, 'r', encoding='utf-8') as f_meta:
            for line in f_meta:
                global_face_metadata_list.append(json.loads(line))
        logger.info(f"FAISS 메타데이터 로드 완료: {metadata_path_str}. 항목 수: {len(global_face_metadata_list)}")
    else:
        logger.warning(f"FAISS 메타데이터 파일을 찾을 수 없거나 경로가 설정되지 않았습니다: {metadata_path_str}")
except Exception as e_model_load:
    logger.error(f"모델 또는 FAISS 인덱스 로드 중 오류: {e_model_load}", exc_info=True)

# --- Dlib 모델 로드 (Gunicorn 환경에서도 실행되도록 모듈 레벨로 이동) ---
try:
    logger.info("Dlib 모델 로드 시도 (모듈 레벨)...")
    # global_cfg_object는 이미 모듈 레벨에서 초기화됨
    face_recognition_cfg = global_cfg_object.get_config('models.face_recognition')
    if not isinstance(face_recognition_cfg, dict):
        logger.error("face_recognition_cfg 설정을 불러오지 못했습니다.")
        sys.exit(1)

    landmark_model_filename = face_recognition_cfg.get('landmark_model_path') # YAML 키 수정됨
    face_rec_model_filename = face_recognition_cfg.get('face_rec_model_path') # YAML 키 수정됨

    if landmark_model_filename and face_rec_model_filename:
        global_face_detector_model = dlib.get_frontal_face_detector()
        # landmark_model_path = models_dir_path / landmark_model_filename # get_value가 전체 경로를 반환하도록 YAML 수정됨
        # face_rec_model_path = models_dir_path / face_rec_model_filename   # get_value가 전체 경로를 반환하도록 YAML 수정됨
        landmark_model_path = Path(landmark_model_filename) # YAML에서 이미 전체 경로로 가정
        face_rec_model_path = Path(face_rec_model_filename)   # YAML에서 이미 전체 경로로 가정

        if landmark_model_path.exists() and face_rec_model_path.exists():
            global_face_embedder_model = {
                'shape_predictor': dlib.shape_predictor(str(landmark_model_path)),
                'face_recognizer': dlib.face_recognition_model_v1(str(face_rec_model_path))
            }
            logger.info(f"Dlib 모델 로드 완료: Detector, Shape Predictor({landmark_model_path.name}), Face Recognizer({face_rec_model_path.name})")
        else:
            logger.error(f"Dlib 모델 파일 경로를 찾을 수 없습니다. Predictor: '{landmark_model_path}', Recognizer: '{face_rec_model_path}'")
    else:
        logger.error("Dlib 모델 로드를 위한 설정(models_dir, landmark_model_name, face_rec_model_name)이 YAML에 충분하지 않습니다.")
except Exception as e_dlib_load_module:
    logger.error(f"모듈 레벨 Dlib 모델 로드 중 오류: {e_dlib_load_module}", exc_info=True)

# --- 실제 얼굴 검출, 특징 추출, 검색 함수 (플레이스홀더) ---
def detect_and_extract_features_dlib(image_bytes: bytes, 
                                     detector: dlib.fhog_object_detector, 
                                     shape_predictor: dlib.shape_predictor, 
                                     face_recognizer: dlib.face_recognition_model_v1) -> Optional[np.ndarray]:
    """
    Dlib을 사용하여 주어진 이미지 바이트에서 얼굴을 검출하고 특징 벡터를 추출합니다.
    가장 큰 얼굴 하나만 처리합니다.

    Args:
        image_bytes (bytes): 이미지 파일의 바이트 데이터.
        detector (dlib.fhog_object_detector): Dlib HOG 얼굴 검출기.
        shape_predictor (dlib.shape_predictor): Dlib 랜드마크 예측기.
        face_recognizer (dlib.face_recognition_model_v1): Dlib 얼굴 인식 모델.

    Returns:
        Optional[np.ndarray]: 추출된 128차원 얼굴 특징 벡터 (NumPy 배열). 얼굴을 찾지 못하면 None.
    """
    if not image_bytes:
        logger.warning("입력된 이미지 바이트가 없습니다.")
        return None
    if not all([detector, shape_predictor, face_recognizer]):
        logger.error("Dlib 모델(detector, shape_predictor, face_recognizer) 중 하나 이상이 로드되지 않았습니다.")
        return None

    try:
        # 바이트 데이터를 OpenCV 이미지로 디코딩
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("이미지 바이트를 디코딩할 수 없습니다.")
            return None

        # Dlib은 RGB 이미지를 사용하므로 변환 (OpenCV는 기본적으로 BGR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 얼굴 검출
        dets = detector(rgb_img, 1) # Upsample 1 time
        if not dets:
            logger.info("이미지에서 얼굴을 찾지 못했습니다.")
            return None

        # 여러 얼굴 중 가장 큰 얼굴 선택 (간단한 예시)
        # 실제로는 더 정교한 선택 로직이 필요할 수 있습니다.
        det = max(dets, key=lambda rect: rect.width() * rect.height())
        shape = shape_predictor(rgb_img, det)
        face_descriptor = face_recognizer.compute_face_descriptor(rgb_img, shape)
        return np.array(face_descriptor, dtype=np.float32)

    except Exception as e:
        logger.error(f"얼굴 검출 및 특징 추출 중 오류 발생: {e}", exc_info=True)
        return None

def search_similar_faces_faiss(query_features: np.ndarray,
                               index: Optional[faiss.Index],
                               metadata_list: List[Dict[str, Any]],
                               top_k: int) -> List[Dict[str, Any]]:
    """
    FAISS 인덱스를 사용하여 유사한 얼굴을 검색합니다.
    """
    results = []
    if query_features is None or index is None or not metadata_list:
        logger.warning("유사 얼굴 검색을 위한 입력값이 충분하지 않습니다 (query_features, index, metadata_list).")
        return results

    try:
        query_features_np = np.array([query_features]).astype('float32')
        distances, indices = index.search(query_features_np, top_k)

        if indices.size == 0:
            logger.info("FAISS 검색 결과가 없습니다.")
            return results

        for i in range(indices.shape[1]):
            idx = indices[0, i]
            dist = distances[0, i]

            if idx < 0 or idx >= len(metadata_list):
                logger.warning(f"FAISS 검색 결과 인덱스 {idx}가 유효하지 않습니다. 건너<0xEB><0><0x8E>니다.")
                continue
            
            entry = metadata_list[idx]
            similarity_score = 1.0 / (1.0 + dist) # 예시 변환

            # global_json_handler는 모듈 레벨에서 접근 가능해야 합니다.
            # 또는 이 함수에 인자로 전달할 수 있습니다.
            # 여기서는 global_json_handler가 모듈 레벨에 있다고 가정합니다.
            image_path_key = global_json_handler.image_path_key if global_json_handler else "image_path"
            face_id_key = global_json_handler.face_id_key if global_json_handler else "face_id"
            
            results.append({
                "image_path": entry.get(image_path_key, f"unknown_image_{idx}.jpg"),
                "similarity": similarity_score,
                "face_id_in_source": entry.get(face_id_key),
                "source_json_path": entry.get("source_json_path"),
            })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)

    except AttributeError as e_attr: # global_json_handler 등이 None일 경우 대비
        logger.error(f"FAISS 검색 중 속성 오류 (global_json_handler 등이 초기화되지 않았을 수 있음): {e_attr}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"FAISS 유사 얼굴 검색 중 오류 발생: {e}", exc_info=True)
        return []

    return results

# --- 개별 JSON 파일 저장 함수 ---
def save_individual_photo_data(single_photo_data: Dict[str, Any], target_json_path_str: str):
    """단일 사진 데이터를 지정된 개별 JSON 파일에 저장합니다."""
    try:
        if not target_json_path_str:
            logger.error("개별 JSON 저장 경로가 제공되지 않았습니다. 저장을 건너뜁니다.")
            return

        target_path_obj = Path(target_json_path_str)
        target_path_obj.parent.mkdir(parents=True, exist_ok=True) # 필요한 경우 부모 디렉토리 생성
        
        # '_source_json_path' 키는 저장할 필요 없으므로, 저장 전에 제거하거나 복사본을 만듭니다.
        data_to_save = {k: v for k, v in single_photo_data.items() if k != '_source_json_path'}

        logger.info(f"개별 사진 데이터를 파일 '{target_path_obj}'에 저장합니다.")
        with open(target_path_obj, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"데이터 저장 중 오류 발생: {e}", exc_info=True)

# --- 사진 데이터 로딩 헬퍼 함수 ---
def load_current_photos_data(page: Optional[int] = None, per_page: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    사진 데이터를 로드합니다.
    - page와 per_page가 제공되면: .lst 파일에서 해당 배치(batch)를 로드합니다.
    - page와 per_page가 없으면: .lst 파일에서 모든 데이터를 로드하려고 시도합니다.
    반환값: (사진 데이터 리스트, 전체 항목 수)
    """
    photos_data: List[Dict[str, Any]] = []
    total_items: int = 0
    
    list_file_path_from_config = app.config.get('global_json_list_file_path_str')
    raw_jsons_base_dir = app.config.get('global_raw_jsons_dir_str') # .lst 파일 내 상대경로의 기준
    base_dir_for_loader = Path(raw_jsons_base_dir) if raw_jsons_base_dir else None
    
    # per_page가 None이면 설정된 기본 배치 크기를 사용
    effective_per_page = per_page if per_page is not None else app.config.get('global_batch_size_for_indexing', 10)

    if list_file_path_from_config and page is not None: # page 정보가 있으면 배치 로드 시도
        logger.info(f".lst 파일에서 사진 데이터 배치 로드 시도: {list_file_path_from_config}, 페이지: {page}, 페이지당 항목 수: {effective_per_page}")
        photos_data, total_items = load_json_batch_from_list_file(
            list_file_path_from_config,
            global_json_handler.read_json,
            page,
            effective_per_page,
            base_dir_for_loader
        )
    elif list_file_path_from_config: # 페이지 정보 없이 .lst 파일 경로만 있는 경우 -> 전체 로드
        logger.info(f".lst 파일에서 모든 사진 데이터 로드 시도: {list_file_path_from_config}")
        photos_data = load_all_json_from_list_file(
            list_file_path_from_config,
            global_json_handler.read_json,
            base_dir_for_loader
        )
        total_items = len(photos_data)
    
    # .lst 파일 로드에 실패했거나, .lst 파일이 지정되지 않은 경우 (대체 로딩 로직 제거)
    if not list_file_path_from_config:
        logger.error(".lst 파일 경로(global_json_list_file_path_str)가 설정되지 않았습니다. 데이터를 로드할 수 없습니다.")
        # photos_data와 total_items는 이미 빈 리스트와 0으로 초기화되어 있음
            
    if not photos_data and total_items > 0 and page is not None:
        # 배치 로드를 시도했고, 전체 아이템은 있으나 현재 페이지에 데이터가 없는 경우 (예: 마지막 페이지를 넘어선 요청)
        logger.info(f"요청된 페이지({page})에 대한 데이터가 없습니다. 전체 항목 수: {total_items}.")
    elif not photos_data and total_items == 0 :
        logger.warning("로드된 사진 데이터가 없습니다.")
        
    return photos_data, total_items

@app.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    # 페이지당 항목 수: URL 파라미터 > 설정(indexing.batch_sizeL)
    default_per_page = app.config.get('global_batch_size_for_indexing', 10)
    per_page = request.args.get('per_page', default_per_page, type=int)
    
    photos_data_batch, total_photos = load_current_photos_data(page=page, per_page=per_page)
    
    # 'id'는 전체 데이터셋에서의 인덱스를 의미하도록 계산 (페이지네이션 시 일관성 유지)
    photos_with_ids = [
        {**photo, 
         'id': (page - 1) * per_page + i, 
         'image_url': url_for('serve_image', filename=photo.get(global_json_handler.image_path_key if global_json_handler else "image_path", 'unknown.jpg'))}
        for i, photo in enumerate(photos_data_batch)
    ]
    total_pages = (total_photos + per_page - 1) // per_page if per_page > 0 else 0
    
    list_file_path_for_template = app.config.get('global_json_list_file_path_str')

    return render_template('index.html', 
                           photos=photos_with_ids, 
                           json_list_file_path=list_file_path_for_template, # 현재 사용 중인 .lst 파일 경로
                           current_page=page,
                           total_pages=total_pages,
                           per_page=per_page)

@app.route('/image/<int:image_id>')
def label_image(image_id):
    # 이 라우트는 특정 ID의 이미지를 로드해야 하므로, 전체 데이터를 로드하거나
    # ID 기반으로 단일 항목을 로드하는 로직이 필요합니다.
    # 여기서는 일단 전체 데이터를 로드하는 방식을 유지합니다 (load_current_photos_data에 page 정보 없이 호출).
    # TODO: image_id에 해당하는 파일만 직접 로드하도록 최적화 (json_files.lst와 image_id를 사용)
    photos_data, total_photos = load_current_photos_data() # 페이지 정보 없이 호출 -> 전체 로드 시도

    if 0 <= image_id < len(photos_data):
        image_data = photos_data[image_id]
        # 이미지 URL 생성
        image_url = url_for('serve_image', filename=image_data['image_path'])
        return render_template('label_image.html', image_data=image_data, image_id=image_id, image_url=image_url)
    return "이미지를 찾을 수 없습니다.", 404

@app.route('/save_labels/<int:image_id>', methods=['POST'])
def save_labels(image_id):
    # 1. 현재 페이지의 사진 데이터 로드 (또는 image_id에 해당하는 단일 데이터 로드 최적화)
    #    load_current_photos_data는 _source_json_path를 포함한 데이터를 반환해야 합니다.
    photos_data, total_photos = load_current_photos_data() # 페이지 정보 없이 호출 -> 전체 로드 시도
    
    if not (0 <= image_id < len(photos_data)):
        logger.error(f"잘못된 image_id({image_id}) 또는 로드된 데이터가 없습니다. 라벨을 저장할 수 없습니다.")
        return "데이터 파일을 찾을 수 없어 라벨을 저장할 수 없습니다.", 500

    image_data_to_update = photos_data[image_id] # 메모리 상의 데이터 복사본
    original_json_path_str = image_data_to_update.get('_source_json_path')

    if not original_json_path_str:
        logger.error(f"image_id {image_id}에 대한 원본 JSON 파일 경로(_source_json_path)를 찾을 수 없습니다. 저장을 건너뜁니다.")
        # 이 경우, 저장을 수행할 수 없으므로 오류를 반환합니다.
        return "원본 파일 정보를 찾을 수 없어 저장할 수 없습니다.", 500

    # 2. 원본 개별 JSON 파일 읽기 (안전하게 작업하기 위해)
    #    또는 image_data_to_update (메모리 복사본)를 직접 수정하고 저장할 수도 있습니다.
    #    여기서는 메모리 복사본을 수정하고, 그 전체를 원본 파일에 덮어쓰는 방식을 사용합니다.
    #    더 안전한 방법은 원본 파일을 읽고, 그 내용을 수정한 뒤 저장하는 것입니다.
    #    현재는 load_current_photos_data가 이미 파일 내용을 로드했으므로 image_data_to_update를 사용합니다.

    # 3. 폼 데이터로 얼굴 이름 업데이트 (메모리 상의 데이터에)
    #    global_json_handler.face_info_key는 'detected_face' 또는 설정된 키여야 합니다.
    faces_key = global_json_handler.face_info_key if global_json_handler else 'faces' # 'faces'는 템플릿과 일치해야 함
    if faces_key in image_data_to_update and isinstance(image_data_to_update[faces_key], list):
        for i, face_entry in enumerate(image_data_to_update[faces_key]):
            face_name = request.form.get(f'face_{i}_name')
            if face_name is not None:
                # global_json_handler.face_label_key는 'label' 또는 설정된 키여야 합니다.
                label_key_for_face = global_json_handler.face_label_key if global_json_handler else 'name' # 'name'은 템플릿과 일치해야 함
                face_entry[label_key_for_face] = face_name.strip()
    
    # 4. 수정된 단일 사진 데이터를 원본 개별 JSON 파일에 저장
    save_individual_photo_data(image_data_to_update, original_json_path_str)

    return redirect(url_for('label_image', image_id=image_id))

# 설정된 경로에서 이미지 파일 제공
@app.route(f'{global_image_serve_prefix_str}/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config.get('global_raw_image_dir_str', ''), filename)

@app.route('/search_face_capture', methods=['POST'])
def search_by_face_capture():
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"error": "이미지 데이터가 없습니다."}), 400

        image_data_url = data['image_data']
        # Base64 데이터에서 실제 이미지 데이터 부분만 추출
        # format: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
        header, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # 1. 수신된 이미지로부터 얼굴 특징 추출
        #    실제 구현에서는 로드된 모델(global_face_detector_model, global_face_embedder_model)을 사용해야 합니다.
        #    global_face_embedder_model은 shape_predictor와 face_recognizer를 포함하는 딕셔너리 또는 객체일 수 있습니다.
        #    여기서는 dlib 모델들이 개별적으로 로드되었다고 가정하고, shape_predictor와 face_recognizer를 전달합니다.
        #    실제 모델 로드 방식에 따라 인자 전달 방식을 조정해야 합니다.
        target_features = detect_and_extract_features_dlib(
            image_bytes, 
            global_face_detector_model, # dlib.get_frontal_face_detector()
            global_face_embedder_model.get('shape_predictor') if isinstance(global_face_embedder_model, dict) else None, # dlib.shape_predictor
            global_face_embedder_model.get('face_recognizer') if isinstance(global_face_embedder_model, dict) else None # dlib.face_recognition_model_v1
        )
        
        if not target_features:
            logger.info("캡처된 이미지에서 얼굴 특징을 추출할 수 없습니다.")
            return jsonify({"message": "이미지에서 얼굴을 찾을 수 없거나 특징을 추출할 수 없습니다."}), 200 # 400 Bad Request도 고려 가능

        # 2. FAISS 인덱스에서 유사 얼굴 검색
        #    global_cfg_object main 블록에서 초기화되므로, 여기서 직접 접근은 어렵습니다.
        #    top_k 값은 애플리케이션 설정에서 가져오거나 상수로 정의해야 합니다.
        #    global_cfg_object 모듈 레벨에서 초기화되었으므로 라우트 함수에서 접근 가능합니다.
        top_k_results = global_cfg_object.get_value('indexing.top_k_search_results', 10)
        
        # 플레이스홀더 대신 실제 FAISS 검색 함수 호출
        search_results_raw = search_similar_faces_faiss(
            target_features,
            global_faiss_index,
            global_face_metadata_list,
            top_k_results
        )

        # 3. 검색 결과를 프론트엔드 형식에 맞게 가공
        processed_results = []
        
        # 얼굴 검색은 전체 데이터에 대한 검색이므로, 모든 사진 데이터를 로드합니다.
        photos_data_for_map, _ = load_current_photos_data() # 전체 데이터 로드

        global_photo_path_to_id_map = {photo['image_path']: idx for idx, photo in enumerate(photos_data_for_map)}

        if not global_photo_path_to_id_map: # 맵이 비어있으면 경고
            logger.warning("photo_path_to_id_map이 비어있습니다. 검색 결과를 ID에 매핑할 수 없습니다.")
        
        for res in search_results_raw:
            image_path = res.get("image_path")
            similarity = res.get("similarity", 0.0) # 기본값 설정

            if image_path and image_path in global_photo_path_to_id_map:
                photo_id = global_photo_path_to_id_map[image_path]
                # 유사도 임계값 적용 (예시)
                MIN_SIMILARITY_THRESHOLD = global_cfg_object.get_value('indexing.min_similarity_threshold', 0.5)
                if similarity >= MIN_SIMILARITY_THRESHOLD:
                    processed_results.append({
                        "image_id": photo_id, 
                        "image_url": url_for('serve_image', filename=image_path),
                        "label_url": url_for('label_image', image_id=photo_id),
                        "similarity": round(similarity, 4) # 소수점 4자리까지 표시
                    })
            else:
                logger.warning(f"검색된 이미지 경로 '{image_path}'를 global_photo_path_to_id_map 찾을 수 없습니다.")

        if not processed_results:
            logger.info("유사한 얼굴을 찾지 못했습니다.")

        return jsonify(processed_results)

    except Exception as e:
        logger.error(f"얼굴 검색 API 오류: {e}", exc_info=True)
        return jsonify({"error": "얼굴 검색 중 서버 오류 발생"}), 500

if __name__ == '__main__':
    # 이 블록은 `python src/my_album_labeling_app/app.py`로 직접 실행될 때만 호출됩니다.
    # Gunicorn으로 실행될 때는 이 블록이 실행되지 않습니다.
    # 모듈 레벨에서 이미 parsed_args, global_cfg_object, logger 등이 초기화되었습니다.
    # (단, IS_GUNICORN 플래그에 따라 parsed_args는 직접 실행 시에만 사용됨)

    # 직접 실행 시 로거 설정 (Gunicorn 환경에서는 이미 위에서 설정됨)
    if not IS_GUNICORN:
        try:
            _log_file_name = f"{Path(__file__).stem}_{datetime.now().strftime('%y%m%d')}.log"
            _full_log_path = Path(LOG_DIR_FOR_LOGGER) / _log_file_name # LOG_DIR_FOR_LOGGER는 parsed_args.log_dir에서 옴
            Path(LOG_DIR_FOR_LOGGER).mkdir(parents=True, exist_ok=True)
            logger.setup(
                logger_path=str(_full_log_path),
                min_level=LOG_LEVEL_FOR_LOGGER, # LOG_LEVEL_FOR_LOGGER는 parsed_args.log_level에서 옴
                include_function_name=True,
                pretty_print=True
            )
            logger.info(f"직접 실행 환경 로거 초기화 완료. 로그 경로: {_full_log_path}")
            logger.info(f"명령줄 인자로 결정된 경로: root_dir='{ROOT_DIR_FOR_CONFIG}', config_path='{CONFIG_PATH_FOR_CONFIG}', log_dir='{LOG_DIR_FOR_LOGGER}', log_level='{LOG_LEVEL_FOR_LOGGER}'")
        except Exception as e_log_setup_main:
            print(f"직접 실행 로거 설정 중 오류: {e_log_setup_main}", file=sys.stderr)
            sys.exit(1)

    # Dlib 모델 로드는 모듈 레벨에서 이미 시도되었으므로, 여기서는 추가 로드 로직이 필요 없습니다.
    # 다만, 로드 성공 여부를 확인하고 싶다면 global 변수들을 체크할 수 있습니다.
    if global_face_detector_model is None or global_face_embedder_model is None:
        logger.warning("Dlib 모델이 (if __name__ == '__main__') 블록에서 로드되지 않았거나 실패했습니다. 모듈 레벨 로드를 확인하세요.")


    # 이 블록은 주로 Flask 개발 서버를 실행하는 데 사용됩니다.
    logger.info(f"Flask 개발 서버 실행 (if __name__ == '__main__')")

    try:
        # Flask 앱 실행을 위한 호스트 및 포트 설정 (global_cfg_object 사용)
        host = global_cfg_object.get_value('project.source.web_service.host', '0.0.0.0')
        port = int(global_cfg_object.get_value('project.source.web_service.port', 5001))
        debug_mode = global_cfg_object.get_value('project.source.web_service.debug', True)
    except Exception as e:
        logger.error(f"Flask 실행 설정 로드 중 오류 발생: {e}", exc_info=True)
        # 기본값으로 실행 시도 또는 종료
        host, port, debug_mode = '0.0.0.0', 5001, True 

    app.run(host=host, port=port, debug=debug_mode)
    
    # app.run()이 종료된 후에 실행될 코드 (일반적으로 Ctrl+C로 서버 중단 시)
    logger.info(f"Flask 웹 애플리케이션({Path(__file__).stem}) 종료 (if __name__ == '__main__')")
    if hasattr(logger, 'shutdown') and callable(logger.shutdown):
        logger.shutdown()
        sys.exit(0) # 정상 종료 명시

        # 애플리케이션 종료 시 로거 정리 (특히 비동기 사용 시 중요)
        logger.info(f"Flask 웹 애플리케이션({Path(__file__).stem}) 종료")
        if hasattr(logger, 'shutdown') and callable(logger.shutdown):
            logger.shutdown()
        exit(0)
    app.run(host=host, port=port, debug=debug_mode) # try 블록 밖으로 이동
