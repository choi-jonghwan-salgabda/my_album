import os
import sys     # 표준 출력 스트림 사용을 위해 필요
from pathlib import Path
from typing import Any, List, Callable, Optional, Tuple, Dict # Dict 추가
import traceback # 초기 오류 로깅에 사용

# shared_utils 패키지에서 configger 클래스 가져오기
# 사용자 정의 유틸리티 모듈 임포트
try:
    from my_utils.config_utils.SimpleLogger import logger, get_argument
    from my_utils.config_utils.configger import configger
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


# JSON 읽기 함수를 위한 타입 별칭 정의 (명확성을 위해)
JsonReadFunction = Callable[[Path], Any]

def load_all_json_from_directory(
    directory_path_str: str,
    json_read_func: JsonReadFunction
) -> List[Any]:
    """
    Reads    지정된 디렉토리에서 제공된 읽기 함수를 사용하여 모든 JSON 파일을 읽고
    그 데이터를 리스트로 집계합니다.

    각 JSON 파일은 반환될 리스트의 요소가 될 데이터(예: 딕셔너리 또는 리스트)를
    포함하고 있을 것으로 예상됩니다.

    Args:
        directory_path_str (str): JSON 파일들을 포함하는 디렉토리의 경로 문자열입니다.
                                  홈 디렉토리를 위한 물결표시(~)를 지원합니다.
        json_read_func (JsonReadFunction): JSON 파일에 대한 Path 객체를 받아
                                           파싱된 내용을 반환하는 함수입니다.

    Returns:
        List[Any]: 각 객체가 JSON 파일의 내용인 데이터 객체들의 리스트입니다.
                   디렉토리가 존재하지 않거나, 디렉토리가 아니거나,
                   성공적으로 읽은 JSON 파일이 없는 경우 빈 리스트를 반환합니다.
    """
    all_data: List[Any] = []
    
    resolved_dir_path = Path(directory_path_str).expanduser()

    if not resolved_dir_path.exists():
        logger.error(f"디렉토리가 존재하지 않습니다: {resolved_dir_path}")
        return all_data
    
    if not resolved_dir_path.is_dir():
        logger.error(f"경로가 디렉토리가 아닙니다: {resolved_dir_path}")
        return all_data

    json_files_found = False
    for json_file_path in resolved_dir_path.glob("*.json"):
        json_files_found = True
        try:
            data = json_read_func(json_file_path)
            if data is not None:
                all_data.append(data)
            else:
                logger.warning(f"JSON 읽기 함수가 파일에 대해 None을 반환했습니다: {json_file_path}")
        except Exception as e:
            logger.error(f"JSON 파일 처리 중 오류 발생 {json_file_path}: {e}", exc_info=True)
    
    if not json_files_found:
        logger.warning(f"디렉토리에서 *.json 파일을 찾을 수 없습니다: {resolved_dir_path}")
    elif not all_data and json_files_found:
        logger.warning(f"{resolved_dir_path} 에서 JSON 파일을 찾았지만, 성공적으로 로드된 데이터가 없습니다.")
        
    return all_data

def load_all_json_from_list_file(
    list_file_path_str: Optional[str],
    json_read_func: JsonReadFunction,
    base_dir_for_relative_paths: Optional[Path] = None
) -> List[Any]:
    """
    JSON 파일 경로들을 포함하는 목록 파일을 읽은 다음, 각 JSON 파일을
    제공된 읽기 함수를 사용하여 읽고 그 데이터를 리스트로 집계합니다.

    목록에 있는 각 JSON 파일은 반환될 리스트의 요소가 될 데이터(예: 단일 사진의
    메타데이터를 나타내는 딕셔너리)를 포함하고 있을 것으로 예상됩니다.

    Args:
        list_file_path_str (Optional[str]): JSON 파일 경로 목록(한 줄에 하나의 경로)을
                                            포함하는 파일의 경로 문자열입니다.
                                            홈 디렉토리를 위한 물결표시(~)를 지원합니다.
        json_read_func (JsonReadFunction): JSON 파일에 대한 Path 객체를 받아
                                           파싱된 내용을 반환하는 함수입니다.
        base_dir_for_relative_paths (Optional[Path]): 목록 파일에서 발견된 상대 경로를
                                                      해결하기 위한 기본 디렉토리입니다.
                                                      None이면, 경로는 목록 파일의 디렉토리를 기준으로 해결됩니다.

    Returns:
        List[Any]: 각 객체가 JSON 파일의 내용인 데이터 객체들의 리스트입니다.
                   목록 파일 경로가 제공되지 않았거나, 목록 파일이 존재하지 않거나,
                   성공적으로 읽은 JSON 파일이 없는 경우 빈 리스트를 반환합니다.
    """
    all_data: List[Any] = []
    if not list_file_path_str:
        logger.warning("JSON 목록 파일 경로가 제공되지 않았습니다.")
        return all_data

    list_file_path = Path(list_file_path_str).expanduser().resolve()
    if not list_file_path.exists():
        logger.warning(f"JSON 목록 파일을 찾을 수 없습니다: {list_file_path}")
        return all_data

    try:
        with open(list_file_path, 'r', encoding='utf-8') as f_list:
            json_file_paths_in_list = [line.strip() for line in f_list if line.strip() and not line.strip().startswith('#')]
    except Exception as e:
        logger.error(f"JSON 목록 파일 읽기 중 오류 발생 {list_file_path}: {e}", exc_info=True)
        return all_data

    if not json_file_paths_in_list:
        logger.warning(f"No JSON file paths found in list file: {list_file_path}")
        return all_data

    effective_base_dir = base_dir_for_relative_paths if base_dir_for_relative_paths else list_file_path.parent

    for path_str_from_list in json_file_paths_in_list:
        json_file_path = Path(path_str_from_list)
        if not json_file_path.is_absolute():
            json_file_path = (effective_base_dir / json_file_path).resolve()
        
        if not json_file_path.exists():
            logger.warning(f"{list_file_path} 에 나열된 JSON 파일을 찾을 수 없습니다: {json_file_path}")
            continue
        
        try:
            data = json_read_func(json_file_path)
            if data is not None:
                if isinstance(data, dict): # 데이터가 딕셔너리인지 확인하여 소스 경로 추가
                    data['_source_json_path'] = str(json_file_path) # 원본 JSON 파일 경로 추가
                all_data.append(data)
            else:
                logger.warning(f"JSON 읽기 함수가 파일에 대해 None을 반환했습니다: {json_file_path}")
        except Exception as e:
            logger.error(f"JSON 파일 처리 중 오류 발생 {json_file_path} (목록 {list_file_path} 에서): {e}", exc_info=True)
            
    if not all_data and json_file_paths_in_list:
        logger.warning(f"{list_file_path} 에 JSON 파일들이 나열되었지만, 성공적으로 로드된 데이터가 없습니다.")
        
    return all_data

def load_json_batch_from_list_file(
    list_file_path_str: Optional[str],
    json_read_func: JsonReadFunction,
    page: int,
    per_page: int,
    base_dir_for_relative_paths: Optional[Path] = None
) -> Tuple[List[Any], int]:
    """
    JSON 파일 경로들을 포함하는 목록 파일을 읽은 다음, page 및 per_page 매개변수를
    기반으로 특정 배치의 JSON 파일들을 읽습니다.

    Args:
        list_file_path_str (Optional[str]): JSON 파일 경로들을 나열하는 파일의 경로입니다.
        json_read_func (JsonReadFunction): 단일 JSON 파일을 파싱하는 함수입니다.
        page (int): 페이지 번호입니다 (1부터 시작).
        per_page (int): 페이지당 항목 수입니다.
        base_dir_for_relative_paths (Optional[Path]): 상대 경로를 위한 기본 디렉토리입니다.

    Returns:
        Tuple[List[Any], int]: 다음을 포함하는 튜플입니다:
            - 현재 배치를 위한 데이터 객체들의 리스트.
            - 목록 파일에서 찾은 총 JSON 파일 경로 수.
        오류가 발생하거나 로드된 데이터가 없으면 ([], 0)을 반환합니다.
    """
    batch_data: List[Any] = []
    total_items = 0

    if not list_file_path_str:
        logger.warning("배치 로딩을 위한 JSON 목록 파일 경로가 제공되지 않았습니다.")
        return batch_data, total_items
    if page < 1:
        logger.warning(f"페이지 번호는 1 이상이어야 합니다, 현재 값: {page}.")
        return batch_data, total_items
    if per_page <= 0:
        logger.warning(f"Per_page는 0보다 커야 합니다, 현재 값: {per_page}.")
        return batch_data, total_items

    list_file_path = Path(list_file_path_str).expanduser().resolve()
    if not list_file_path.exists():
        logger.warning(f"JSON 목록 파일을 찾을 수 없습니다: {list_file_path}")
        return batch_data, total_items

    try:
        with open(list_file_path, 'r', encoding='utf-8') as f_list:
            all_json_file_paths = [line.strip() for line in f_list if line.strip() and not line.strip().startswith('#')]
    except Exception as e:
        logger.error(f"JSON 목록 파일 읽기 중 오류 발생 {list_file_path}: {e}", exc_info=True)
        return batch_data, total_items

    total_items = len(all_json_file_paths)
    if total_items == 0:
        logger.warning(f"목록 파일에서 JSON 파일 경로를 찾을 수 없습니다: {list_file_path}")
        return batch_data, total_items

    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paths_for_batch = all_json_file_paths[start_index:end_index]

    effective_base_dir = base_dir_for_relative_paths if base_dir_for_relative_paths else list_file_path.parent
    for path_str_from_list in paths_for_batch:
        json_file_path = Path(path_str_from_list)
        if not json_file_path.is_absolute():
            json_file_path = (effective_base_dir / json_file_path).resolve()
        
        if not json_file_path.exists():
            logger.warning(f"{list_file_path} 에 나열된 JSON 파일을 찾을 수 없습니다: {json_file_path}")
            continue
        try:
            data = json_read_func(json_file_path)
            if data is not None:
                if isinstance(data, dict): # 데이터가 딕셔너리인지 확인하여 소스 경로 추가
                    data['_source_json_path'] = str(json_file_path) # 원본 JSON 파일 경로 추가
                batch_data.append(data)
            else:
                logger.warning(f"JSON 읽기 함수가 파일에 대해 None을 반환했습니다: {json_file_path}")
        except Exception as e:
            logger.error(f"JSON 파일 처리 중 오류 발생 {json_file_path} (목록 {list_file_path} 에서): {e}", exc_info=True)
            
    return batch_data, total_items
