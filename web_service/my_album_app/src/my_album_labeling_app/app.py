import os
import sys     # 표준 출력 스트림 사용을 위해 필요
from pathlib import Path # 경로 관리를 위해 필요
from typing import List, Dict, Any, Tuple, Optional, Union # Union 추가
import json
import base64 # search_by_face_capture에서 사용하므로 최상단으로 이동
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import copy # for deepcopy in save_labels if needed, or for modifying global data carefully
import traceback # 초기 오류 로깅에 사용
# 실제 얼굴 검출 및 특징 추출을 위한 dlib 임포트 (예시)
import dlib
import numpy as np
import cv2 # 이미지 처리를 위해 OpenCV 임포트
import faiss # FAISS 임포트
import threading # 데이터 로딩 동기화를 위한 Lock

# shared_utils 패키지에서 configger 클래스 가져오기
# 사용자 정의 유틸리티 모듈 임포트
try:
    from datetime import datetime
    from my_utils.config_utils.SimpleLogger import logger, get_argument
    from my_utils.config_utils.configger import configger
    from my_utils.config_utils.JsonManager import JsonManager
    from my_utils.object_utils.data_loading_helpers import load_all_json_from_list_file, load_json_batch_from_list_file
    from my_utils.object_utils.photo_utils import crop_image_from_path_to_buffer
    # `label_image`와 `save_labels`에서 얼굴 이미지를 동적으로 자르기 위해 필요합니다.
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
        _log_file_name = f"{Path(__file__).stem}_{datetime.now().strftime('%y%m%d_%H%M%S')}_gunicorn.log"
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
    # CONFIG 파일(photo_album.yaml) 1차 키 설정을 가져옵니다.
    project_cfg     = global_cfg_object.get_config('project')
    models_cfg      = global_cfg_object.get_config('models')
    processing_cfg  = global_cfg_object.get_config('processing')
    json_keys_cfg   = global_cfg_object.get_config('json_keys')
    indexing_cfg    = global_cfg_object.get_config('indexing')
    # Check if all primary config sections were loaded successfully and are dictionaries
    configs_map = {
        "project": project_cfg,
        "models": models_cfg,
        "processing": processing_cfg,
        "json_keys": json_keys_cfg,
        "indexing": indexing_cfg
    }
    
    all_loaded_correctly = True
    for name, cfg_item in configs_map.items():
        if not isinstance(cfg_item, dict):
            # Log which specific config failed and its type
            logger.error(f"1차_cfg 설정 '{name}'을(를) 불러오지 못했거나 dict 타입이 아닙니다 (실제 타입: {type(cfg_item)}). YAML 파일을 확인하세요.")
            all_loaded_correctly = False
            
    if not all_loaded_correctly:
        logger.error("필수 1차 설정(project, models, processing, json_keys, indexing) 중 일부를 불러오지 못했습니다. 애플리케이션을 종료합니다.")
        sys.exit(1)
    
    # datasets관련 설정
    datasets_cfg = project_cfg.get('paths', {}).get('datasets', {})
    if not isinstance(datasets_cfg, dict):
        logger.error("datasets_cfg 설정을 불러오지 못했습니다.")
        sys.exit(1)

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

    # outputs 관련 설정 (labeled_face_crop_dir 접근을 위해)
    outputs_cfg = project_cfg.get('paths', {}).get('outputs', {})
    if not isinstance(outputs_cfg, dict):
        logger.error("outputs_cfg 설정을 불러오지 못했습니다. (project.paths.outputs)")
        sys.exit(1)
    global_labeled_face_crop_dir_str = outputs_cfg.get('labeled_face_crop_dir')
    if not global_labeled_face_crop_dir_str:
        logger.critical("필수 설정 누락: 'project.paths.outputs.labeled_face_crop_dir'이 설정 파일에 없거나 비어있습니다.")
        sys.exit(1)
    
    # 디렉토리 경로를 Path 객체로 변환하고 존재 여부 확인 및 생성
    labeled_face_crop_dir_path = Path(global_labeled_face_crop_dir_str)
    if not labeled_face_crop_dir_path.exists():
        try:
            labeled_face_crop_dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"디렉토리 생성됨: {labeled_face_crop_dir_path}")
        except OSError as e:
            logger.error(f"디렉토리 '{labeled_face_crop_dir_path}' 생성 실패: {e}")
            sys.exit(1) # 디렉토리 생성 실패 시 애플리케이션 종료
    else:
        logger.info(f"디렉토리 이미 존재함: {labeled_face_crop_dir_path}")

    web_service_cfg = project_cfg.get('source', {}).get('web_service', {})
    if not isinstance(web_service_cfg, dict):
        logger.error("web_service_cfg 설정을 불러오지 못했습니다.")
        sys.exit(1)

    web_service_dir = web_service_cfg.get('web_service_dir')
    if not web_service_dir:
        logger.error("web_service_dir 값을 불러오지 못했습니다.")
        sys.exit(1)

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

    global_image_serve_prefix_str = web_service_cfg.get('image_serve_prefix','my_original_images')
    if not global_image_serve_prefix_str: # 값이 없는 경우
        # The YAML key is 'project.source.web_service.image_serve_prefix', not with '.templates'
        logger.warning("'project.source.web_service.image_serve_prefix'가 설정 파일에 없거나 비어있습니다. 기본값 '/my_original_images'를 사용합니다.")
        global_image_serve_prefix_str = "/my_original_images" # Assign a default prefix
    elif not global_image_serve_prefix_str.startswith('/'): # 슬래시로 시작하지 않는 경우
        global_image_serve_prefix_str = '/' + global_image_serve_prefix_str
        logger.debug(f"image_serve_prefix에 슬래시를 추가했습니다: {global_image_serve_prefix_str}")

    global_face_crops_serve_prefix_str = web_service_cfg.get('face_crops_serve_prefix', 'face_crops_images')
    if not global_face_crops_serve_prefix_str:
        logger.warning("'project.source.web_service.face_crops_serve_prefix'가 설정 파일에 없거나 비어있습니다. 기본값 '/face_crops_images'를 사용합니다.")
        global_face_crops_serve_prefix_str = "/face_crops_images"
    elif not global_face_crops_serve_prefix_str.startswith('/'):
        global_face_crops_serve_prefix_str = '/' + global_face_crops_serve_prefix_str
        logger.debug(f"face_crops_serve_prefix에 슬래시를 추가했습니다: {global_face_crops_serve_prefix_str}")

    # JSON 키 설정을 configger에서 가져옵니다.
    try:
        global_json_handler = JsonManager(json_keys_cfg)
    except Exception as e_json_handler:
        logger.error(f"JsonManager 초기화 중 오류 발생: {e_json_handler}", exc_info=True)
        sys.exit(1)

    # 여러 JSON 파일 로드용 디렉토리 경로. YAML 키: project.paths.datasets.raw_jsons_dir

    # 인덱싱 및 배치 로딩을 위한 배치 크기
    global_batch_size_for_indexing = indexing_cfg.get('batch_sizeL', 10) # 기본값 10으로 설정
    if not isinstance(global_batch_size_for_indexing, int) or global_batch_size_for_indexing <= 0:
        logger.warning(f"'indexing.batch_sizeL' 설정이 유효하지 않거나 없습니다. 기본값 10을 사용합니다. 현재 값: {global_batch_size_for_indexing}")
        global_batch_size_for_indexing = 10

    # flask_labeling_app_cfg는 현재 직접 사용되지 않으므로, 필요시 로드합니다.
    # flask_labeling_app_cfg = web_service_cfg.get('flask_labeling_app')
    # if not isinstance(flask_labeling_app_cfg, dict):
    #     logger.error("flask_labeling_app_cfg 설정을 불러오지 못했습니다.")
    #     sys.exit(1)

    # 로깅: 설정 값들을 명확히 보여줌
    root_dir = project_cfg.get('root_dir')
    logger.info(f"project.root:{root_dir}")
    logger.info(f"  JSONS 파일이 있는곳:    {global_raw_jsons_dir_str if global_raw_jsons_dir_str else '설정되지 않음'}")
    logger.info(f"  JSONS 목록이 있는곳:    {global_json_list_file_path_str if global_json_list_file_path_str else '설정되지 않음'}")
    logger.info(f"  이미지 파일들이 있는곳: {global_raw_image_dir_str}")
    logger.info(f"  이름진 얼굴이 있는 곳:   {global_labeled_face_crop_dir_str}") # 로깅 추가
    logger.info(f"  웹서버 소스가 있는곳:    {web_service_dir}")
    logger.info(f"  템플릿(.html) 있는곳:   {global_templates_dir_str}")
    logger.info(f"  정적값(.css)가 있는곳:  {global_static_folder_str}")
    logger.info(f"  인덱싱 / 배치크기:       {global_batch_size_for_indexing}")
    logger.info(f"  원본 이미지  접두사:    {global_image_serve_prefix_str if global_image_serve_prefix_str else '설정되지 않음'}")
    logger.info(f"  크롭 이미지  접두사: {global_face_crops_serve_prefix_str if global_face_crops_serve_prefix_str else '설정되지 않음'}")

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

# --- 유틸리티 함수: 중첩된 딕셔너리에서 값 가져오기 ---
def get_value_from_json(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    점(.)으로 구분된 경로를 사용하여 중첩된 딕셔너리에서 값을 검색합니다.
    예: get_value_from_json(data, "image_info.filename")
    """
    keys = path.split('.')
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list): # 경로에 리스트 인덱스가 포함된 경우 처리 (예: "detected_obj.0.name")
            try:
                idx = int(key)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return default
            except ValueError: # 키가 정수 인덱스가 아님
                return default
        else:
            return default
    return current

def _resolve_image_path_to_url_filename(path_from_json_str: str, base_image_dir_str: str) -> str:
    """
    JSON/메타데이터의 이미지 경로 문자열을 'serve_raw_image' 엔드포인트 및
    url_for와 함께 사용하기에 적합한 상대 경로로 해석합니다.
    """
    if not path_from_json_str:
        return 'unknown.jpg'

    base_image_dir = Path(base_image_dir_str)
    image_path_to_resolve = Path(path_from_json_str)
    
    absolute_image_path_obj = None
    
    if image_path_to_resolve.is_absolute():
        absolute_image_path_obj = image_path_to_resolve
    elif any(path_from_json_str.startswith(p) for p in ["home/", "mnt/", "var/", "opt/", "srv/", "usr/"]):
        absolute_image_path_obj = Path("/") / image_path_to_resolve
    else:
        # URL에 사용하기에 적합한 상대 경로로 가정합니다.
        return path_from_json_str

    if absolute_image_path_obj:
        try:
            # URL을 위해 서빙 디렉토리에 대한 상대 경로를 계산합니다.
            return str(absolute_image_path_obj.relative_to(base_image_dir))
        except ValueError:
            logger.warning(
                f"이미지 경로 '{absolute_image_path_obj}' (원본: '{path_from_json_str}')가 "
                f"기본 디렉토리 '{base_image_dir}' 내에 없습니다. 파일 이름으로 대체합니다."
            )
            return absolute_image_path_obj.name
            
    return 'unknown.jpg'

app = Flask(__name__, template_folder=global_templates_dir_str, static_folder=global_static_folder_str) # 변수명 오타 수정
app.config['global_raw_jsons_dir_str'] = global_raw_jsons_dir_str
app.config['global_json_list_file_path_str'] = global_json_list_file_path_str # JSON 목록 파일 경로 저장
app.config['global_raw_image_dir_str'] = global_raw_image_dir_str
app.config['global_labeled_face_crop_dir_str'] = global_labeled_face_crop_dir_str # 새로 추가된 설정 저장
app.config['global_batch_size_for_indexing'] = global_batch_size_for_indexing # 배치 크기 저장

# --- 애플리케이션 전역 데이터 (시작 시 로드) ---
GLOBAL_PHOTOS_DATA: List[Dict[str, Any]] = []
GLOBAL_TOTAL_PHOTOS: int = 0
GLOBAL_PHOTO_PATH_TO_ID_MAP: Dict[str, int] = {}
GLOBAL_DATA_LOAD_LOCK = threading.Lock() # 데이터 로딩 동기화를 위한 Lock

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
    face_recognition_cfg = models_cfg.get('face_recognition')
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
                                     detector: Optional[dlib.fhog_object_detector], 
                                     shape_predictor: Optional[dlib.shape_predictor], 
                                     face_recognizer: Optional[dlib.face_recognition_model_v1]) -> Optional[np.ndarray]:
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
        logger.error("Dlib 모델(detector, shape_predictor, face_recognizer) 중 하나 이상이 로드되지 않았습니다. 얼굴 특징 추출 불가.")
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
            # FAISS 메타데이터에 저장된 image_path는 단순 키 'image_path'일 가능성이 높음. 확인 필요.
            # 여기서는 FAISS 메타데이터 생성 시 'image_path' 키로 이미지 경로를 저장했다고 가정.
            
            results.append({
                "image_path": entry.get("image_path"), # FAISS 메타데이터의 키
                "similarity": similarity_score,
                "face_id_in_source": entry.get(global_json_handler.face_id_key if global_json_handler else "face_id"), # FAISS 메타데이터의 키
                "source_json_path": entry.get("source_json_path"), # FAISS 메타데이터의 키
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

def _load_all_photo_data_into_globals():
    """
    모든 사진 데이터를 파일 시스템에서 로드하여 전역 변수에 저장하고,
    사진 경로와 ID 매핑을 생성합니다.
    이 함수는 GLOBAL_DATA_LOAD_LOCK에 의해 동기화되어야 합니다.
    """
    global GLOBAL_PHOTOS_DATA, GLOBAL_TOTAL_PHOTOS, GLOBAL_PHOTO_PATH_TO_ID_MAP
    logger.info("전역 사진 데이터 로드 시작...")

    photos_data_temp: List[Dict[str, Any]] = []
    total_items_temp: int = 0
    
    list_file_path = app.config.get('global_json_list_file_path_str')
    raw_jsons_base_dir_str = app.config.get('global_raw_jsons_dir_str')
    base_dir_for_loader = Path(raw_jsons_base_dir_str) if raw_jsons_base_dir_str else None

    if list_file_path:
        logger.info(f".lst 파일에서 모든 사진 데이터 로드 시도: {list_file_path}")
        # global_json_handler.read_json은 부울 값을 반환하고 인스턴스 상태를 변경하므로,
        # 여러 파일을 로드하는 데 적합하지 않습니다. 상태 비저장(stateless) 정적 메서드를 사용합니다.
        loaded_data_with_nones = load_all_json_from_list_file(
            list_file_path,
            JsonManager.load_json_from_path, # 상태를 변경하지 않는 정적 메서드 사용
            base_dir_for_loader
        )
        # 로드 중 오류가 발생하여 None이 반환된 항목들을 필터링합니다.
        photos_data_temp = [item for item in loaded_data_with_nones if item is not None]
        
        if len(loaded_data_with_nones) != len(photos_data_temp):
            logger.warning(f"{len(loaded_data_with_nones) - len(photos_data_temp)}개의 JSON 파일을 로드하는 데 실패했습니다.")

    elif raw_jsons_base_dir_str: # .lst 파일이 없고 raw_jsons_dir만 있는 경우 (대안)
        logger.warning(f".lst 파일 경로가 제공되지 않았습니다. '{raw_jsons_base_dir_str}' 디렉토리에서 직접 JSON 파일 로드를 시도합니다.")
        # 이 부분은 data_loading_helpers에 load_all_json_from_directory 와 같은 함수가 필요합니다.
        # 현재는 load_all_json_from_list_file만 있으므로, 이 경로는 제한적으로 동작하거나 실패할 수 있습니다.
        # 여기서는 예시로 비워두거나, 해당 기능을 구현해야 합니다.
        # 지금은 .lst 파일이 필수라고 가정하고, 이 로직은 경고만 남깁니다.
        logger.error("raw_jsons_dir에서 직접 로드하는 기능은 현재 구현되지 않았습니다. .lst 파일을 사용하세요.")
    else:
        logger.error(".lst 파일 경로(global_json_list_file_path_str)와 raw_jsons_dir 모두 설정되지 않았습니다. 데이터를 로드할 수 없습니다.")

    if not photos_data_temp:
        logger.warning("로드된 사진 데이터가 없습니다.")
    total_items_temp = len(photos_data_temp)
    GLOBAL_PHOTOS_DATA = photos_data_temp
    GLOBAL_TOTAL_PHOTOS = total_items_temp
    logger.info(f"전역 사진 데이터 로드 완료. 총 {GLOBAL_TOTAL_PHOTOS}개 항목 로드됨.")

    # GLOBAL_PHOTO_PATH_TO_ID_MAP 생성
    temp_map: Dict[str, int] = {}
    # global_json_handler는 모듈 레벨에서 초기화되어 있어야 함
    key_for_image_path = f"{global_json_handler.image_info_key}.{global_json_handler.image_path_key}"
    base_image_dir_path = Path(app.config.get('global_raw_image_dir_str', ''))

    if not GLOBAL_PHOTOS_DATA:
        logger.warning("GLOBAL_PHOTOS_DATA가 비어있어 GLOBAL_PHOTO_PATH_TO_ID_MAP을 생성할 수 없습니다.")
    else:
        logger.info(f"GLOBAL_PHOTO_PATH_TO_ID_MAP 생성을 위해 {len(GLOBAL_PHOTOS_DATA)}개의 사진 데이터 처리 시작...")
        for idx, photo_item in enumerate(GLOBAL_PHOTOS_DATA):
            img_path_in_json = get_value_from_json(photo_item, key_for_image_path)
            if img_path_in_json:
                path_obj = Path(img_path_in_json)
                # JSON 내 경로가 상대 경로이면 base_image_dir_path를 기준으로 절대 경로 생성
                # JSON 내 경로가 이미 절대 경로이면 그대로 사용
                abs_path_str = str((base_image_dir_path / path_obj).resolve()) if not path_obj.is_absolute() else str(path_obj.resolve())
                
                # FAISS 메타데이터의 image_path와 형식을 일치시켜야 합니다.
                # FAISS 인덱싱 시 사용된 경로 형식(예: 절대 경로, 특정 기준 디렉토리로부터의 상대 경로)을 확인해야 합니다.
                # 여기서는 FAISS 메타데이터가 절대 경로를 저장한다고 가정합니다.
                temp_map[abs_path_str] = idx
            # else: logger.debug(f"인덱스 {idx}의 사진 데이터에 유효한 이미지 경로('{key_for_image_path}')가 없습니다.") # 로그가 너무 많을 수 있음

    GLOBAL_PHOTO_PATH_TO_ID_MAP = temp_map
    if not GLOBAL_PHOTO_PATH_TO_ID_MAP and GLOBAL_PHOTOS_DATA: # 데이터는 있는데 맵이 비었을 경우
         logger.warning(f"GLOBAL_PHOTO_PATH_TO_ID_MAP이 비어있습니다. 원인 확인 필요 (이미지 경로 키, 기본 이미지 디렉토리 설정, FAISS 메타데이터 경로 형식 등).")
    logger.info(f"GLOBAL_PHOTO_PATH_TO_ID_MAP 생성 완료. {len(GLOBAL_PHOTO_PATH_TO_ID_MAP)}개 매핑됨.")


def ensure_global_data_loaded():
    """
    전역 사진 데이터가 로드되지 않았으면 로드합니다.
    여러 스레드/워커가 동시에 로드하는 것을 방지하기 위해 Lock을 사용합니다.
    """
    global GLOBAL_PHOTOS_DATA # 상태 확인을 위해 필요
    if not GLOBAL_PHOTOS_DATA: # 데이터가 비어있을 때만 로드 시도
        with GLOBAL_DATA_LOAD_LOCK:
            if not GLOBAL_PHOTOS_DATA: # Lock 획득 후 다시 한번 확인 (double-checked locking)
                _load_all_photo_data_into_globals()

# --- 사진 데이터 로딩 헬퍼 함수 ---
def load_current_photos_data(page: Optional[int] = None, per_page: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    전역으로 로드된 사진 데이터에서 페이징 처리하여 반환합니다.
    """
    ensure_global_data_loaded() # 데이터가 로드되었는지 확인 (이미 로드되었다면 아무것도 안 함)

    global GLOBAL_PHOTOS_DATA, GLOBAL_TOTAL_PHOTOS, global_json_handler

    if page is not None and per_page is not None and per_page > 0:
        start_index = (page - 1) * per_page
        end_index = start_index + per_page
        paginated_data = GLOBAL_PHOTOS_DATA[start_index:end_index]
        # logger.debug(f"페이지 {page}, 페이지당 {per_page}개 요청. 데이터 슬라이싱: [{start_index}:{end_index}]")
        return paginated_data, GLOBAL_TOTAL_PHOTOS
    else: # 페이지 정보가 없으면 전체 데이터 반환 (또는 첫 페이지만 반환하도록 정책 수정 가능)
        # logger.debug("페이지 정보 없이 전체 데이터 요청.")
        return GLOBAL_PHOTOS_DATA, GLOBAL_TOTAL_PHOTOS

# 애플리케이션 시작 시 (또는 첫 요청 전) 데이터 로드
# Gunicorn의 --preload 옵션과 함께 사용하면 마스터 프로세스에서 한 번 로드됩니다.
ensure_global_data_loaded()

@app.route('/')
def index():        #client 첫 접속
    logger.info(f"app.route('/')")
    
    page = request.args.get('page', 1, type=int)
    # 페이지당 항목 수: URL 파라미터 > 설정(indexing.batch_sizeL)
    default_per_page = app.config.get('global_batch_size_for_indexing', 10)
    per_page = request.args.get('per_page', default_per_page, type=int)
    
    photos_data_batch, total_photos = load_current_photos_data(page=page, per_page=per_page)
    
    # YAML 설정에서 가져온 이미지 경로 키 (점(.)으로 구분된 전체 경로일 수 있음)
    actual_image_path_key = f"{global_json_handler.image_info_key}.{global_json_handler.image_path_key}" if global_json_handler else "image_info_key.image_path_key" # 기본값 # global_json_handler는 모듈 레벨에서 초기화
                
    photos_with_ids = []
    for i, photo_data in enumerate(photos_data_batch):
        # 전역 데이터는 이미 로드 시 _source_json_path를 포함할 수 있음 (load_all_json_from_list_file의 동작에 따라)
        # ID는 (page-1)*per_page + i 로 계산하여 현재 페이지 내에서의 상대적 ID가 아닌, 전체 데이터셋에서의 ID를 부여
        # 단, GLOBAL_PHOTOS_DATA가 정렬되어 있다는 가정 하에 이 ID가 유효함.
        # 만약 GLOBAL_PHOTOS_DATA의 순서가 원본 .lst 파일과 같다면, 이 ID는 해당 파일 내 인덱스와 유사하게 사용 가능.
        # 여기서는 image_id를 라우트에서 사용할 때 GLOBAL_PHOTOS_DATA의 인덱스로 직접 사용하므로,
        # 템플릿에 전달하는 id는 표시용 또는 클라이언트 측 식별용으로 사용.
        original_index = (page - 1) * per_page + i # GLOBAL_PHOTOS_DATA에서의 실제 인덱스

        path_from_json_str = get_value_from_json(photo_data, actual_image_path_key, 'unknown.jpg')
        image_url_filename = _resolve_image_path_to_url_filename(
            path_from_json_str,
            app.config.get('global_raw_image_dir_str')
        )
        
        photos_with_ids.append({
            **photo_data, 
            'id': original_index, # GLOBAL_PHOTOS_DATA에서의 인덱스
            'image_url': url_for('serve_raw_image', filename=image_url_filename) # 'serve_image'를 'serve_raw_image'로 수정
        })

    total_pages = (total_photos + per_page - 1) // per_page if per_page > 0 else 0
    
    list_file_path_for_template = app.config.get('global_json_list_file_path_str')

    return render_template('index.html', 
                           photos=photos_with_ids, 
                           json_list_file_path=list_file_path_for_template, # 현재 사용 중인 .lst 파일 경로
                           current_page=page,
                           total_pages=total_pages,
                           per_page=per_page)

@app.route('/image/<int:image_id>')
def label_image(image_id):  # 선택한 사진의 얼굴을  client에게 보내줌
    logger.info(f"app.route('/image/<int:image_id>'): image_id={image_id}")
    ensure_global_data_loaded()

    global GLOBAL_PHOTOS_DATA, GLOBAL_TOTAL_PHOTOS

    if 0 <= image_id < GLOBAL_TOTAL_PHOTOS:
        json_data_from_global = GLOBAL_PHOTOS_DATA[image_id]

        # 1. json정보에서 image_path를 가저온다.
        image_path_key = f"{global_json_handler.image_info_key}.{global_json_handler.image_path_key}" if global_json_handler else "image_info_key.image_path_key"
        logger.debug(f"JSON에서 이미지 경로를 가져오기 위한 키 경로: '{image_path_key}'")

        # image_path_key 사용하여 json_data_from_global에서 실제 이미지 경로 값을 가져옵니다.
        image_path_val = get_value_from_json(json_data_from_global, image_path_key)

        image_url_filename = _resolve_image_path_to_url_filename(
            image_path_val,
            app.config.get('global_raw_image_dir_str')
        )
        
        if image_url_filename == 'unknown.jpg':
            logger.error(f"image_id {image_id}에 대한 이미지 파일명을 최종적으로 결정할 수 없습니다. JSON 내 경로 값: {image_path_val}")
            return f"이미지 파일 경로를 확인할 수 없습니다 (ID: {image_id}). JSON 데이터 또는 설정을 확인하세요.", 404
        
        # 2. json정보에서 image_path를 가저온다.
        # 템플릿에 전달할 데이터 복사본 생성 및 크롭 이미지 URL 추가
        json_data_for_template = copy.deepcopy(json_data_from_global)
        # logger.debug(f"json_data_for_template: {json_data_for_template}") # 너무 길 수 있으므로 필요시 주석 해제

        # global_json_handler가 유효한 경우, 얼굴 목록 키와 크롭 파일명 키를 가져와 처리합니다.
        if global_json_handler:
            # # 템플릿으로 전달되는 face_list_key와 동일한 키를 사용하여 Python 내부에서 얼굴 목록에 접근합니다.
            # # 이 키는 global_json_handler.face_info_key에서 직접 가져옵니다 (예: "detected_face").
            # key_for_object_list: JSON 데이터에서 실제 객체 목록을 담고 있는 키 (예: "detected_obj")
            try:
                key_for_object_list = global_json_handler.object_info_key
                key_for_face_dict_in_object = global_json_handler.face_info_key # 객체 내 얼굴 딕셔너리 키 (예: "detected_face")
            except AttributeError:
                logger.error(
                    "Configuration Error: 'JsonManager' is missing 'object_info_key' or 'face_info_key' attribute. "
                    "Please check: \n"
                    "1. Your 'json_keys' section in 'photo_album.yaml' (for object_info_lst.key and object_info_lst.face_info_lst.key).\n"
                    "2. The 'JsonManager' class implementation.\n"
                    "Cropped images may not be displayed correctly until this is resolved."
                )
                key_for_object_list = None
                key_for_face_dict_in_object = None

            if key_for_object_list: # 객체 목록 키가 유효한 경우
                object_list_from_data = json_data_for_template.get(key_for_object_list)
                # logger.info(f"object_list_from_data (using key '{key_for_object_list}'): {object_list_from_data}")

                if isinstance(object_list_from_data, list):
                    for obj_idx, object_entry in enumerate(object_list_from_data): # 각 객체 (예: "person" 딕셔너리) 순회
                        # logger.debug(f"Processing object_entry: {object_entry}")

                        if isinstance(object_entry, dict):
                            if key_for_face_dict_in_object: # 얼굴 딕셔너리 키가 유효한 경우
                                face_info_dict = object_entry.get(key_for_face_dict_in_object)
                                if isinstance(face_info_dict, dict):

                                    # Base64 인코딩된 이미지 추가
                                    face_box_xyxy = face_info_dict.get(global_json_handler.face_box_xyxy_key) # face_box_xyxy
                                    if face_box_xyxy and image_path_val:
                                        cropped_image_buffer = crop_image_from_path_to_buffer(image_path_val, face_box_xyxy)
                                        if cropped_image_buffer:
                                            img_bytes = cropped_image_buffer.getvalue()
                                            base64_encoded_data = base64.b64encode(img_bytes).decode('utf-8')
                                            face_info_dict['cropped_image_base64'] = f"data:image/jpeg;base64,{base64_encoded_data}"
                                        else:
                                            face_info_dict['cropped_image_base64'] = None
                                            logger.warning(f"image_id {image_id}, obj {obj_idx}: 얼굴 이미지 자르기 실패 (bbox: {face_box_xyxy}).")
                                    else:
                                        face_info_dict['cropped_image_base64'] = None
                                        if not face_box_xyxy: logger.warning(f"image_id {image_id}, obj {obj_idx}: bbox 정보 없음.")
                                        if not image_path_val: logger.warning(f"image_id {image_id}, obj {obj_idx}: 원본 이미지 경로 없음.")

                                    # 저장된 크롭 이미지 URL 추가
                                    saved_crop_filename = face_info_dict.get('face_crop_filename') # BUGFIX: 이름(label) 대신 크롭 파일명 키 사용
                                    if saved_crop_filename:
                                        try:
                                            face_info_dict['saved_cropped_image_url'] = url_for('serve_cropped_image', filename=saved_crop_filename)
                                        except Exception as e_url_crop:
                                            logger.error(f"Error generating URL for saved_cropped_image '{saved_crop_filename}' for image_id {image_id}: {e_url_crop}", exc_info=True)
                                            face_info_dict['saved_cropped_image_url'] = None
                                    else:
                                        face_info_dict['saved_cropped_image_url'] = None
                                    
                                    # 템플릿에서 얼굴 식별을 위한 인덱스 추가
                                    face_info_dict['object_index'] = obj_idx
                                    # 만약 face_info_dict 리스트 내의 단일 얼굴이 아니라,
                                    # 객체 내 유일한 얼굴 정보를 담는 딕셔너리라면 face_index_in_object는 0 또는 불필요.
                                    # 현재 JSON 구조상 detected_face는 객체 당 하나이므로 face_index_in_object는 0으로 간주.
                                    face_info_dict['face_index_in_object'] = 0 
                                else:
                                    logger.warning(f"Value for key '{key_for_face_dict_in_object}' in object_entry is not a dict: {face_info_dict}")
                        # elseif isinstance(object_entry, list):
                        # else:
                            # else: logger.warning("key_for_face_dict_in_object is not configured.")
                # elseif isinstance(object_list_from_data, list):
                # else:
                
        return render_template('label_image.html',
                               image_data=json_data_for_template, # 수정된 데이터 전달
                               image_id=image_id,
                               passed_face_dict_key=key_for_face_dict_in_object, # 객체 내 얼굴 딕셔너리 키 전달 (예: "detected_face")
                               face_list_key=key_for_object_list, # 템플릿에 실제 객체 목록 키 전달 (예: "detected_obj")
                               face_label_key=global_json_handler.face_label_key if global_json_handler else "name", # name 키 (YAML에서 설정한대로)
                               # face_crop_filename_key는 템플릿에서 직접 파일명을 가져올 때 사용 (현재는 Python에서 URL을 만들어 전달)
                               # face_crop_filename_key=key_for_crop_filename_in_face_dict, 
                               # 템플릿에서는 image_data[face_list_key]로 객체 목록 접근 후,
                               # 각 object_item[passed_face_dict_key].cropped_image_url 사용
                               )
    
    logger.warning(f"label_image: 유효하지 않은 image_id({image_id}) 또는 사진 데이터를 찾을 수 없습니다.")
    return "이미지를 찾을 수 없습니다.", 404

@app.route('/save_labels/<int:image_id>', methods=['POST'])
def save_labels(image_id):  # 입력한 사진의 얼굴의 이름을 저장함
    logger.info(f"app.route('/save_labels/<int:image_id>', methods=['POST']'): image_id={image_id}")
    ensure_global_data_loaded()
    global GLOBAL_PHOTOS_DATA, GLOBAL_TOTAL_PHOTOS # 전역 변수 사용 명시
    
    if not (0 <= image_id < GLOBAL_TOTAL_PHOTOS):
        logger.error(f"잘못된 image_id({image_id}) 또는 로드된 데이터가 없습니다. 라벨을 저장할 수 없습니다.")
        return "데이터 파일을 찾을 수 없어 라벨을 저장할 수 없습니다.", 500

    json_data_to_update = GLOBAL_PHOTOS_DATA[image_id] # 수정 제안
    original_json_path_str = json_data_to_update.get('_source_json_path')

    if not original_json_path_str:
        logger.error(f"image_id {image_id}에 대한 원본 JSON 파일 경로(_source_json_path)를 찾을 수 없습니다. 데이터: {json_data_to_update}")
        # 이 경우, 저장을 수행할 수 없으므로 오류를 반환합니다.
        return "원본 파일 정보를 찾을 수 없어 저장할 수 없습니다.", 500

    # 폼 데이터로 얼굴 이름 업데이트 및 크롭 이미지 저장
    if global_json_handler:
        object_list_key = global_json_handler.object_info_key         # 예: "detected_obj"
        face_info_key = global_json_handler.face_info_key   # 예: "detected_face"
        face_label_key = global_json_handler.face_label_key   # 예: "name" 또는 "label"
        face_bbox_key = global_json_handler.face_box_xyxy_key # bbox 키

        # 원본 이미지의 실제 파일 경로 가져오기 (크롭을 위해)
        image_path_key = f"{global_json_handler.image_info_key}.{global_json_handler.image_path_key}"
        image_path_val = get_value_from_json(json_data_to_update, image_path_key)

        if not image_path_val:
            logger.error(f"image_id {image_id}의 원본 이미지 경로를 JSON에서 찾을 수 없습니다. 크롭 이미지 저장을 건너<0xEB><0><0x8E>니다.")
            # 라벨만 저장하고 리다이렉트하거나, 오류 반환 결정 필요
            # 여기서는 라벨만 저장 시도

        if object_list_key in json_data_to_update and isinstance(json_data_to_update[object_list_key], list):
            object_list = json_data_to_update[object_list_key]
            for obj_idx, object_entry in enumerate(object_list): # 각 객체 순회
                # 템플릿에서 face_obj{{obj_idx}}_face{{face_idx}}_name 형태로 이름을 전송했다고 가정
                # 현재 구조상 객체당 얼굴은 하나이므로 face_idx는 0으로 간주
                face_idx_in_obj = 0 
                face_name_from_form = request.form.get(f'face_obj{obj_idx}_face{face_idx_in_obj}_name')
                if face_name_from_form is not None and isinstance(object_entry, dict):
                    if face_info_key in object_entry:
                        face_info_dict = object_entry.get(face_info_key)
                        if isinstance(face_info_dict, dict):
                            # 1. 라벨 업데이트
                            face_info_dict[face_label_key] = face_name_from_form.strip()
                            logger.debug(f"Updated label for image_id {image_id}, obj_idx {obj_idx}: '{face_name_from_form.strip()}'")

                            # 2. 얼굴 이미지 크롭 및 저장, JSON에 파일명 기록
                            if image_path_val: # 원본 이미지 경로가 있을 때만 크롭 시도
                                face_bbox_val = face_info_dict.get(face_bbox_key)
                                if face_bbox_val:
                                    cropped_buffer = crop_image_from_path_to_buffer(image_path_val, face_bbox_val, output_format='.jpg')
                                    if cropped_buffer:
                                        # 고유 파일명 생성 (예: 원본JSON명_obj인덱스_face인덱스.jpg)
                                        original_json_stem = Path(original_json_path_str).stem
                                        crop_filename = f"{original_json_stem}_obj{obj_idx}_face{face_idx_in_obj}.jpg"
                                        crop_save_path = Path(app.config.get('global_labeled_face_crop_dir_str')) / crop_filename
                                        
                                        try:
                                            with open(crop_save_path, 'wb') as f_crop:
                                                f_crop.write(cropped_buffer.getvalue())
                                            # BUGFIX: 이름(label)을 덮어쓰지 않고 별도 키에 크롭 파일명 저장
                                            face_info_dict['face_crop_filename'] = crop_filename
                                            logger.info(f"Cropped face for image_id {image_id}, obj_idx {obj_idx} saved to: {crop_save_path}")
                                        except Exception as e_crop_save:
                                            logger.error(f"Failed to save cropped face for image_id {image_id}, obj_idx {obj_idx} to {crop_save_path}: {e_crop_save}")
                                    else: logger.warning(f"Failed to crop face for image_id {image_id}, obj_idx {obj_idx} (bbox: {face_bbox_val}).")
                                else: logger.warning(f"No bbox found for face in image_id {image_id}, obj_idx {obj_idx}. Cannot crop.")
                        else:
                            logger.warning(f"Face data for obj_idx {obj_idx} in image_id {image_id} is not a dictionary.")
                    else:
                        logger.warning(f"Object_entry {i} in image_id {image_id} does not contain key '{face_info_key}'.")
        else:
                        logger.warning(f"Object_entry {obj_idx} in image_id {image_id} does not contain key '{face_info_key}'.")

    # 수정된 단일 사진 데이터를 원본 개별 JSON 파일에 저장
    save_individual_photo_data(json_data_to_update, original_json_path_str)
    logger.info(f"image_id {image_id}에 대한 GLOBAL_PHOTOS_DATA 업데이트: {json_data_to_update}") # 저장 후 데이터 로깅
    GLOBAL_PHOTOS_DATA[image_id] = json_data_to_update #  GLOBAL_PHOTOS_DATA 업데이트
    
    # 중요: GLOBAL_PHOTOS_DATA[image_id]는 이미 위에서 직접 수정되었음.
    # Gunicorn 환경에서는 이 변경이 해당 요청을 처리한 워커의 메모리에만 반영됩니다.
    # 다른 워커는 여전히 이전 데이터를 가지고 있을 수 있습니다.
    # 이 문제를 해결하려면 외부 공유 저장소(Redis 등)를 사용하거나,
    # 워커 간 메시징을 통해 데이터 리프레시를 트리거해야 합니다.
    # 현재는 리다이렉트 후 같은 워커가 요청을 받는다면 수정된 내용을 보게 됩니다.
    logger.info(f"image_id {image_id}의 라벨이 파일과 현재 워커의 메모리에 업데이트되었습니다.")

    return redirect(url_for('label_image', image_id=image_id))

# 1. 원본 이미지 제공 라우트 (기존 global_image_serve_prefix_str 사용)
@app.route(f'{global_image_serve_prefix_str}/<path:filename>') # 기존 접두사 사용
def serve_raw_image(filename):
    logger.info(f"app.route({global_image_serve_prefix_str}/<path:filename>) for raw image: {filename}")
    source_directory = app.config.get('global_raw_image_dir_str')
    if not source_directory:
        logger.error(f"global_raw_image_dir_str: {global_raw_image_dir_str} 설정이 없습니다.")
        return "설정 오류", 500
    return send_from_directory(source_directory, filename)

# 2. 크롭된 얼굴 이미지 제공 라우트 (새로운 접두사 및 labeled_face_crop_dir 사용)

@app.route(f'{global_face_crops_serve_prefix_str}/<path:filename>') # 새 접두사 사용
def serve_cropped_image(filename): # 함수 이름 변경 (serve_image -> serve_cropped_image)
    logger.info(f"app.route({global_face_crops_serve_prefix_str}/<path:filename>): {filename}")
    # 이미지 소스 디렉토리를 global_labeled_face_crop_dir_str 로 변경
    source_directory = app.config.get('global_labeled_face_crop_dir_str')
    if not source_directory:
        logger.error(f"global_labeled_face_crop_dir_str: {global_labeled_face_crop_dir_str} 설정이 없습니다.")
        return "설정 오류", 500
    return send_from_directory(source_directory, filename)

@app.route('/search_face_capture', methods=['POST'])
def search_by_face_capture():   # 카메라로 찍은 사진을 받아 처리
    logger.info(f"app.route('/search_face_capture', methods=['POST'])")
    try:
        ensure_global_data_loaded() # GLOBAL_PHOTO_PATH_TO_ID_MAP 사용 전 로드 확인
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
        
        if target_features is None:
            logger.info("캡처된 이미지에서 얼굴 특징을 추출할 수 없습니다.")
            return jsonify({"message": "이미지에서 얼굴을 찾을 수 없거나 특징을 추출할 수 없습니다."}), 200 # 400 Bad Request도 고려 가능

        logger.info(f"추출된 얼굴 특징 벡터 (shape: {target_features.shape}): {target_features[:5]}...") # 처음 5개 값만 로깅

        # 2. FAISS 인덱스에서 유사 얼굴 검색
        #    global_cfg_object main 블록에서 초기화되므로, 여기서 직접 접근은 어렵습니다.
        #    top_k 값은 애플리케이션 설정에서 가져오거나 상수로 정의해야 합니다.
        #    global_cfg_object 모듈 레벨에서 초기화되었으므로 라우트 함수에서 접근 가능합니다.
        top_k_results = indexing_cfg.get('top_k_search_results', 10)
        
        # 플레이스홀더 대신 실제 FAISS 검색 함수 호출
        search_results_raw = search_similar_faces_faiss(
            target_features,
            global_faiss_index,
            global_face_metadata_list,
            top_k_results
        )

        logger.info(f"FAISS 검색 결과 (상위 {len(search_results_raw)}개):")
        for i, res in enumerate(search_results_raw[:3]): # 상위 3개 결과만 로깅
            logger.info(f"  - 결과 {i+1}: 유사도={res.get('similarity'):.4f}, 경로='{res.get('image_path')}'")
        if len(search_results_raw) > 3:
            logger.info("  - ... (추가 결과 생략)")

        # 3. 검색 결과를 프론트엔드 형식에 맞게 가공
        processed_results = []
        processed_photo_ids = set() # 검색 결과에 중복된 사진이 들어가지 않도록 ID를 추적

        # GLOBAL_PHOTO_PATH_TO_ID_MAP은 ensure_global_data_loaded()에 의해 이미 채워져 있어야 함
        if not GLOBAL_PHOTO_PATH_TO_ID_MAP and GLOBAL_TOTAL_PHOTOS > 0 : # 데이터는 있는데 맵이 비어있는 경우 경고
             logger.warning("GLOBAL_PHOTO_PATH_TO_ID_MAP이 비어있지만 사진 데이터는 존재합니다. _load_all_photo_data_into_globals의 맵 생성 로직 및 FAISS 메타데이터의 경로 형식을 확인하세요.")
                
        for res in search_results_raw:
            image_path_from_faiss = res.get("image_path") 
            similarity = res.get("similarity", 0.0) # 기본값 설정

            if image_path_from_faiss and image_path_from_faiss in GLOBAL_PHOTO_PATH_TO_ID_MAP:
                photo_id = GLOBAL_PHOTO_PATH_TO_ID_MAP[image_path_from_faiss]
                
                if photo_id in processed_photo_ids:
                    continue # 이미 처리된 사진이므로 건너<0xEB><0><0x8E>니다.

                # 유사도 임계값 적용 (예시)
                MIN_SIMILARITY_THRESHOLD = indexing_cfg.get('min_similarity_threshold', 0.5)
                if similarity >= MIN_SIMILARITY_THRESHOLD:
                    image_filename_for_url = _resolve_image_path_to_url_filename(
                        image_path_from_faiss,
                        app.config.get('global_raw_image_dir_str')
                    )

                    processed_results.append({
                        "image_id": photo_id,
                        "image_url": url_for('serve_raw_image', filename=image_filename_for_url), # 원본 이미지 URL
                        "label_url": url_for('label_image', image_id=photo_id),
                        "similarity": float(round(similarity, 4)) # Ensure it's a Python float
                    })
                    processed_photo_ids.add(photo_id) # 처리된 사진 ID 추가
            else:
                logger.warning(
                    f"FAISS 검색 결과의 이미지 경로를 맵에서 찾을 수 없습니다. "
                    f"경로 불일치 가능성이 높습니다.\n"
                    f"  - FAISS 메타데이터 경로: '{image_path_from_faiss}'\n"
                    f"  - 맵에 존재하는 경로 예시: '{next(iter(GLOBAL_PHOTO_PATH_TO_ID_MAP.keys())) if GLOBAL_PHOTO_PATH_TO_ID_MAP else '맵이 비어있음'}'"
                )

        if not processed_results:
            logger.info("유사한 얼굴을 찾지 못했습니다.")

        logger.info(f"최종 처리된 검색 결과 ({len(processed_results)}개): {processed_results}")

        response = jsonify(processed_results)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    except Exception as e:
        logger.error(f"얼굴 검색 API 오류: {e}", exc_info=True)
        return jsonify({"error": "얼굴 검색 중 서버 오류 발생"}), 500 # 여기도 필요하면 캐시 제어 헤더 추가 가능

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

    if global_face_detector_model is None or global_face_embedder_model is None:
        logger.warning("Dlib 모델이 (if __name__ == '__main__') 블록에서 로드되지 않았거나 실패했습니다. 모듈 레벨 로드를 확인하세요.")

    # ensure_global_data_loaded()는 이미 모듈 레벨에서 호출되었으므로, 여기서 다시 호출할 필요는 없음.
    if not GLOBAL_PHOTOS_DATA and not IS_GUNICORN: # 직접 실행 시 데이터 로드 확인
        logger.warning("전역 사진 데이터(GLOBAL_PHOTOS_DATA)가 비어있습니다. _load_all_photo_data_into_globals() 호출 결과를 확인하세요.")

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
