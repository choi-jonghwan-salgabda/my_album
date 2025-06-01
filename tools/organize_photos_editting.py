# -*- coding: utf-8 -*-
import os
import sys
import hashlib
import shutil
import argparse
import logging
import json # JSON 처리 라이브러리
from collections import defaultdict
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, List, Set, Any # 타입 힌트

# --- 로거 설정 ---
# 기본 로거 설정 (파일 핸들러는 main 블록에서 추가)
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)-6s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # 로그를 화면에도 출력
logger = logging.getLogger(__name__) # 로거 객체 생성

# --- 지원할 이미지 확장자 ---
IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif'}

# ==============================================================================
# 파일 처리 함수들
# ==============================================================================

def calculate_hash(filepath: str, chunk_size: int = 8192) -> Optional[str]:
    """파일의 SHA256 해시 값을 계산합니다."""
    hasher = hashlib.sha256() # SHA256 해시 객체 생성
    try:
        # 파일을 바이너리 읽기 모드('rb')로 열기
        with open(filepath, 'rb') as file:
            # 파일을 chunk_size 만큼씩 읽어 해시 업데이트
            while chunk := file.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest() # 계산된 해시 값을 16진수 문자열로 반환
    except IOError as e:
        # 파일 읽기/쓰기 오류 발생 시 로깅
        logger.error(f"파일 읽기 오류 '{filepath}': {e}")
        return None
    except FileNotFoundError:
        # 해시 계산 중 파일이 사라진 경우 로깅
        logger.warning(f"해시 계산 중 파일을 찾을 수 없음 (이미 이동/삭제됨?): '{filepath}'")
        return None
    except Exception as e:
        # 기타 예상치 못한 오류 발생 시 로깅
        logger.error(f"해시 계산 중 예상치 못한 오류 발생 '{filepath}': {e}")
        return None

def find_image_files_generator(search_dir: str, exclude_dirs: Optional[List[str]] = None) -> Generator[Tuple[str, str], None, None]:
    """
    지정된 디렉토리와 하위 디렉토리에서 이미지 파일을 찾아 (해시, 경로) 튜플을 생성(yield)합니다.
    exclude_dirs에 포함된 디렉토리는 검색에서 제외합니다.
    """
    total_files_scanned = 0 # 스캔한 총 파일 수
    image_files_found = 0 # 찾은 이미지 파일 수
    # 제외할 디렉토리 목록을 절대 경로 set으로 변환 (중복 제거 및 빠른 검색)
    excluded_paths_abs: Set[str] = set()
    if exclude_dirs:
        try:
            # os.path.normpath: 경로 구분자 등을 운영체제에 맞게 정규화 (예: /a/b/../c -> /a/c)
            excluded_paths_abs = {os.path.normpath(os.path.abspath(d)) for d in exclude_dirs}
            logger.info(f"다음 디렉토리를 검색에서 제외합니다: {excluded_paths_abs}")
        except Exception as e:
            logger.error(f"제외 디렉토리 경로 처리 중 오류 발생: {e}")
            # 오류 발생해도 계속 진행 (제외 기능만 실패)

    logger.info(f"'{search_dir}' 디렉토리에서 이미지 파일 검색 시작 (배치 처리 모드)...")

    # os.walk를 사용하여 디렉토리 순회
    # topdown=True: 상위 디렉토리부터 방문 (dirs 리스트 수정 가능)
    # onerror: 디렉토리 접근 오류 발생 시 호출될 함수 지정 (람다 함수로 간단히 로깅)
    try:
        for root, dirs, files in os.walk(search_dir, topdown=True, onerror=lambda err: logger.error(f"디렉토리 접근 오류: {err}")):
            # --- 제외 디렉토리 처리 ---
            try:
                # 현재 탐색 중인 디렉토리의 절대 경로 정규화
                current_root_abs = os.path.normpath(os.path.abspath(root))
                # 현재 경로가 제외 목록의 경로로 시작하는지 확인
                if excluded_paths_abs and any(current_root_abs.startswith(excluded_path) for excluded_path in excluded_paths_abs):
                    logger.debug(f"제외된 디렉토리 건너뛰기: {root}")
                    dirs[:] = [] # 현재 디렉토리의 하위 디렉토리 탐색 중단 (dirs 리스트를 비움)
                    continue # 다음 디렉토리로 이동
            except Exception as e:
                logger.error(f"제외 디렉토리 확인 중 오류 발생 (경로: {root}): {e}")
                continue # 문제가 있는 디렉토리는 건너뛰기

            # --- 현재 디렉토리의 파일 처리 ---
            for filename in files:
                total_files_scanned += 1 # 스캔 카운트 증가
                try:
                    file_path = os.path.join(root, filename) # 파일 전체 경로 생성
                    # 파일 확장자를 소문자로 변경하여 지원하는 확장자인지 확인
                    if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                        # 실제 파일인지 확인 (디렉토리나 심볼릭 링크 등 제외)
                        if not os.path.isfile(file_path):
                            logger.debug(f"이미지 확장자이나 파일이 아님 (건너뛰기): {file_path}")
                            continue

                        image_files_found += 1 # 찾은 이미지 카운트 증가
                        logger.debug(f"이미지 파일 발견: {file_path}")
                        file_hash = calculate_hash(file_path) # 파일 해시 계산
                        if file_hash:
                            # 해시 계산 성공 시 (해시, 파일 경로) 튜플을 yield (제너레이터 반환)
                            yield file_hash, file_path
                            logger.debug(f"  - 해시: {file_hash} (생성됨)")
                        else:
                            # 해시 계산 실패 로그는 calculate_hash 함수에서 처리됨
                            pass
                except Exception as e:
                    # 개별 파일 처리 중 오류 발생 시 로깅하고 계속 진행
                    logger.error(f"파일 처리 중 오류 발생 (파일: {filename} in {root}): {e}")

                # 스캔 진행 상황 로깅 (지정한 개수마다)
                if total_files_scanned % 5000 == 0: # 예: 5000개 파일마다 로그 출력
                     logger.info(f"... {total_files_scanned}개 파일 스캔 중 ...")

    except Exception as e:
        # 파일 시스템 탐색 중 예상치 못한 오류 발생 시 로깅 (exc_info=True: 스택 트레이스 포함)
        logger.error(f"파일 시스템 탐색 중 예상치 못한 오류 발생 (시작 경로: {search_dir}): {e}", exc_info=True)

    # 스캔 완료 로그
    logger.info(f"총 {total_files_scanned}개 파일 스캔 시도 완료.")
    logger.info(f"총 {image_files_found}개의 이미지 파일을 찾았습니다 (생성 완료).")


def safe_move_file(src_path: str, dest_dir: str) -> Optional[str]:
    """
    파일을 대상 디렉토리로 안전하게 이동합니다. 이름 충돌 시 숫자를 붙입니다.
    대상 디렉토리가 없으면 생성합니다.
    """
    # 이동할 원본 파일이 실제로 존재하는지 확인
    if not os.path.exists(src_path):
        logger.warning(f"이동할 원본 파일 없음 (이미 처리됨?): {src_path}")
        return None # 이동 불가

    # 대상 디렉토리가 존재하지 않으면 생성 (exist_ok=True: 이미 있어도 오류 아님)
    if not os.path.isdir(dest_dir):
        try:
            os.makedirs(dest_dir, exist_ok=True)
            logger.info(f"대상 디렉토리 생성: {dest_dir}")
        except OSError as e:
            logger.error(f"대상 하위 디렉토리 생성 실패 '{dest_dir}': {e}")
            return None # 디렉토리 생성 실패 시 이동 불가

    # 대상 파일 경로 생성 (원본 파일 이름 사용)
    dest_path = os.path.join(dest_dir, os.path.basename(src_path))
    counter = 1 # 이름 충돌 시 붙일 숫자 카운터
    # 파일 이름과 확장자 분리
    original_name, ext = os.path.splitext(os.path.basename(src_path))
    # 대상 경로에 파일이 이미 존재하면 이름 변경 시도
    while os.path.exists(dest_path):
        # 파일명 뒤에 _숫자 추가 (예: image_1.jpg, image_2.jpg)
        dest_path = os.path.join(dest_dir, f"{original_name}_{counter}{ext}")
        counter += 1
        # 무한 루프 방지 (같은 이름의 파일이 1000개 이상 있는 극히 드문 경우)
        if counter > 1000:
             logger.error(f"파일명 충돌 해결 시도 1000회 초과: {src_path} -> {dest_dir}")
             return None # 이름 충돌 해결 불가

    # 파일 이동 실행 (shutil.move 사용)
    try:
        shutil.move(src_path, dest_path)
        logger.info(f"이동: '{src_path}' -> '{dest_path}'")
        return dest_path # 성공 시 최종 이동된 경로 반환
    except (IOError, OSError) as e:
        # 파일 이동 중 입출력/운영체제 오류 발생 시 로깅
        logger.error(f"파일 이동 오류 '{src_path}' -> '{dest_path}': {e}")
        return None # 이동 실패
    except Exception as e:
        # 기타 예상치 못한 오류 발생 시 로깅
        logger.error(f"파일 이동 중 예상치 못한 오류 발생 '{src_path}': {e}")
        return None # 이동 실패

# ==============================================================================
# 배치 처리 및 맵 업데이트 함수
# ==============================================================================

def update_map_data(map_data: Dict[str, Dict[str, Any]], file_hash: str, src_path: str, moved_path: str):
    """메모리 내의 맵 데이터를 업데이트합니다."""
    # 해당 해시 키가 맵 데이터에 없으면 새로 생성
    if file_hash not in map_data:
        map_data[file_hash] = {"originals": {}}
    # 'originals' 딕셔너리에 원본 경로를 key로, 이동된 경로를 value로 저장
    map_data[file_hash]["originals"][src_path] = moved_path

def save_map_file(map_data: Dict[str, Dict[str, Any]], map_file_path: str):
    """맵 데이터를 JSON 파일로 안전하게 저장합니다."""
    try:
        # 임시 파일에 먼저 저장 (저장 중 오류 발생 시 원본 파일 보호)
        temp_map_file_path = map_file_path + ".tmp"
        # utf-8 인코딩, 보기 좋게 들여쓰기(indent=2), 한글 깨짐 방지(ensure_ascii=False)
        with open(temp_map_file_path, 'w', encoding='utf-8') as f:
            json.dump(map_data, f, ensure_ascii=False, indent=2)
        # 임시 파일 저장이 성공하면, 원본 파일과 교체 (os.replace는 원자적 연산 시도)
        os.replace(temp_map_file_path, map_file_path)
        logger.debug(f"맵 파일 저장 완료: {map_file_path}")
    except IOError as e:
        logger.error(f"맵 파일 쓰기 오류 '{map_file_path}': {e}")
    except Exception as e:
        logger.error(f"맵 파일 저장 중 예상치 못한 오류 발생 '{map_file_path}': {e}")
        # 오류 발생 시 생성된 임시 파일 삭제 시도
        if os.path.exists(temp_map_file_path):
            try:
                os.remove(temp_map_file_path)
            except OSError:
                logger.warning(f"임시 맵 파일 삭제 실패: {temp_map_file_path}")

def load_map_file(map_file_path: str) -> Dict[str, Dict[str, Any]]:
    """JSON 맵 파일을 로드합니다. 파일이 없거나 손상되었으면 빈 딕셔너리를 반환합니다."""
    # 맵 파일이 존재하는 경우
    if os.path.exists(map_file_path):
        try:
            # 파일을 읽기 모드('r')로 열기 (utf-8 인코딩)
            with open(map_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f) # JSON 데이터 로드
                # 로드된 데이터가 딕셔너리 형태인지 간단히 확인
                if isinstance(data, dict):
                    logger.info(f"기존 맵 파일 로드 완료: {map_file_path}")
                    return data
                else:
                    # 유효한 JSON 객체가 아닌 경우 오류 로깅 및 빈 딕셔너리 반환
                    logger.error(f"맵 파일 내용이 유효한 JSON 객체가 아닙니다. 새 맵을 생성합니다: {map_file_path}")
                    return {}
        except json.JSONDecodeError:
            # JSON 파싱 오류 (파일 내용 손상)
            logger.error(f"맵 파일 형식이 잘못되었습니다. 새 맵을 생성합니다: {map_file_path}")
            # 손상된 파일을 백업 (선택적)
            try:
                shutil.copyfile(map_file_path, map_file_path + ".corrupted")
                logger.info(f"손상된 맵 파일을 백업했습니다: {map_file_path}.corrupted")
            except Exception as backup_e:
                logger.error(f"손상된 맵 파일 백업 실패: {backup_e}")
            return {} # 빈 딕셔너리 반환
        except IOError as e:
            # 파일 읽기 오류
            logger.error(f"맵 파일 읽기 오류 '{map_file_path}': {e}")
            return {} # 빈 딕셔너리 반환
        except Exception as e:
            # 기타 예상치 못한 오류
            logger.error(f"맵 파일 로드 중 예상치 못한 오류 발생 '{map_file_path}': {e}")
            return {} # 빈 딕셔너리 반환
    else:
        # 맵 파일이 존재하지 않으면 새로 시작
        logger.info(f"기존 맵 파일을 찾을 수 없습니다. 새로 생성합니다: {map_file_path}")
        return {} # 빈 딕셔너리 반환


def process_batch(
    batch_hashes: Dict[str, List[str]], # 처리할 (해시: [경로 리스트]) 딕셔너리
    organized_dir: str, # 고유 파일 저장 경로
    duplicates_dir: str, # 중복 파일 저장 경로
    map_data: Dict[str, Dict[str, Any]] # 업데이트할 전체 맵 데이터 (메모리 내 객체)
) -> Tuple[int, int, int, int]:
    """
    주어진 해시 배치를 처리하여 파일을 이동하고 통계를 반환합니다.
    성공적으로 이동된 파일 정보를 map_data에 직접 업데이트합니다.
    """
    unique_count = 0 # 이번 배치에서 이동된 고유 파일 수
    duplicate_set_count = 0 # 이번 배치에서 발견된 중복 파일 세트 수
    moved_duplicates_count = 0 # 이번 배치에서 이동된 중복 파일 수
    move_errors = 0 # 이번 배치에서 발생한 이동 오류 수

    logger.info(f"배치 처리 시작: {len(batch_hashes)}개의 고유 해시 포함")

    # 배치 내 각 해시와 경로 리스트 순회
    for file_hash, paths in batch_hashes.items():
        # 현재 실제로 존재하는 파일 경로만 필터링 (이전 배치나 다른 프로세스에서 이동/삭제되었을 수 있음)
        existing_paths = [p for p in paths if os.path.exists(p)]

        # 해당 해시에 대해 처리할 파일이 없으면 건너뛰기
        if not existing_paths:
            logger.debug(f"해시 {file_hash}에 대한 모든 파일이 이미 처리됨, 건너뛰기.")
            continue

        moved_path: Optional[str] = None # 이동 성공 시 경로 저장 변수

        # 존재하는 파일이 하나뿐이면 고유 파일로 간주
        if len(existing_paths) == 1:
            src_path = existing_paths[0]
            # 맵 데이터에 이 원본 파일 정보가 이미 있는지 확인 (이전 실행에서 처리되었을 수 있음)
            # map_data.get(file_hash, {}): 해시 키가 없으면 빈 딕셔너리 반환
            # .get("originals", {}): 'originals' 키가 없으면 빈 딕셔너리 반환
            if src_path not in map_data.get(file_hash, {}).get("originals", {}):
                # 맵에 없으면 이동 시도
                moved_path = safe_move_file(src_path, organized_dir)
                if moved_path:
                    # 이동 성공 시 카운트 증가 및 맵 데이터 업데이트
                    unique_count += 1
                    update_map_data(map_data, file_hash, src_path, moved_path)
                else:
                    # 이동 실패 시 오류 카운트 증가 (safe_move_file 내부에서 오류 로깅됨)
                    move_errors += 1
            else:
                # 맵에 이미 존재하면 건너뛰기 (중복 처리 방지)
                logger.debug(f"파일이 이미 맵에 존재하여 건너<0xEB><0><0x8F><0xBC>니다 (이전 처리됨?): {src_path}")

        # 존재하는 파일이 여러 개이면 중복 파일로 간주
        else:
            duplicate_set_count += 1 # 중복 세트 발견 카운트 증가
            logger.warning(f"중복 발견 (해시: {file_hash}): {existing_paths}")

            # 중복 파일을 저장할 하위 디렉토리 이름 결정 (첫 번째 파일 이름 기준)
            base_filename = os.path.basename(existing_paths[0])
            duplicate_sub_dir_name = os.path.splitext(base_filename)[0] # 확장자 제외
            duplicate_target_dir = os.path.join(duplicates_dir, duplicate_sub_dir_name)

            # 존재하는 모든 중복 파일을 해당 하위 디렉토리로 이동
            for src_path in existing_paths:
                # 맵 데이터에 이 원본 파일 정보가 이미 있는지 확인
                if src_path not in map_data.get(file_hash, {}).get("originals", {}):
                    # 맵에 없으면 이동 시도
                    moved_path = safe_move_file(src_path, duplicate_target_dir)
                    if moved_path:
                        # 이동 성공 시 카운트 증가 및 맵 데이터 업데이트
                        moved_duplicates_count += 1
                        update_map_data(map_data, file_hash, src_path, moved_path)
                    else:
                        # 이동 실패 시 오류 카운트 증가
                        move_errors += 1
                else:
                    # 맵에 이미 존재하면 건너뛰기
                     logger.debug(f"파일이 이미 맵에 존재하여 건너<0xEB><0><0x8F><0xBC>니다 (이전 처리됨?): {src_path}")

    # 배치 처리 결과 로깅
    logger.info(f"배치 처리 완료: 고유 {unique_count}, 중복 세트 {duplicate_set_count}, 이동된 중복 {moved_duplicates_count}, 오류 {move_errors}")
    # 배치 처리 통계 반환
    return unique_count, duplicate_set_count, moved_duplicates_count, move_errors

# ==============================================================================
# 메인 정리 로직 함수
# ==============================================================================

def organize_photos(
    search_dir: str, # 검색 시작 경로
    organized_dir: str, # 고유 파일 저장 경로
    duplicates_dir: str, # 중복 파일 저장 경로
    map_file_path: str, # 맵 파일 경로
    exclude_dirs: Optional[List[str]] = None, # 제외할 디렉토리 목록
    batch_size: int = 5000 # 배치 처리 단위 크기
):
    """
    사진을 배치 단위로 검색하고 고유/중복 파일을 분류하여 이동합니다.
    처리 결과를 map_file_path에 저장합니다.
    """
    # --- 경로 유효성 검사 및 생성 ---
    # 검색 디렉토리가 존재하는지 확인
    if not os.path.isdir(search_dir):
        logger.error(f"오류: 검색 디렉토리를 찾을 수 없습니다 - {search_dir}")
        return # 함수 종료
    try:
        # 대상 디렉토리(고유, 중복) 생성 (exist_ok=True: 이미 존재해도 오류 아님)
        os.makedirs(organized_dir, exist_ok=True)
        os.makedirs(duplicates_dir, exist_ok=True)
        logger.info(f"대상 디렉토리 확인/생성 완료:")
        logger.info(f"  - 선별된 사진: {organized_dir}")
        logger.info(f"  - 겹치는 사진: {duplicates_dir}")
    except OSError as e:
        logger.error(f"대상 디렉토리 생성 실패: {e}")
        return # 함수 종료

    # --- 맵 데이터 로드 ---
    # 스크립트 시작 시 맵 파일을 로드하여 메모리에 유지
    map_data = load_map_file(map_file_path)

    # --- 이번 실행에 대한 통계 변수 초기화 ---
    total_unique_count = 0 # 이동된 고유 파일 수
    total_duplicate_set_count = 0 # 발견된 중복 세트 수
    total_moved_duplicates_count = 0 # 이동된 중복 파일 수
    total_move_errors = 0 # 발생한 오류 수
    total_processed_hashes_in_run = 0 # 처리된 고유 해시 수
    total_processed_files_in_run = 0 # 처리 시도된 파일 수
    batch_num = 0 # 처리된 배치 번호

    # --- 파일 검색 제너레이터 생성 ---
    image_generator = find_image_files_generator(search_dir, exclude_dirs)
    # 현재 배치의 파일 정보를 담을 딕셔너리 (해시: [경로 리스트])
    batch_hashes: Dict[str, List[str]] = defaultdict(list)
    files_in_current_batch = 0 # 현재 배치에 포함된 파일 수

    logger.info(f"배치 크기 {batch_size}로 파일 처리 시작...")

    # --- 메인 처리 루프 ---
    while True:
        try:
            # 제너레이터로부터 다음 (해시, 파일 경로) 가져오기
            file_hash, file_path = next(image_generator)

            # 맵 데이터에 이 원본 파일 정보가 이미 있는지 확인 (중복 스캔/처리 방지)
            if file_path in map_data.get(file_hash, {}).get("originals", {}):
                 logger.debug(f"파일이 이미 맵에 존재하여 스캔/배치에서 제외: {file_path}")
                 continue # 이미 처리된 파일이므로 다음 파일로 넘어감

            # 현재 배치에 파일 정보 추가
            batch_hashes[file_hash].append(file_path)
            files_in_current_batch += 1

            # 현재 배치 크기가 지정된 크기에 도달하면 배치 처리 실행
            if files_in_current_batch >= batch_size:
                batch_num += 1 # 배치 번호 증가
                logger.info(f"--- 배치 {batch_num} 처리 시작 ({len(batch_hashes)}개 고유 해시, {files_in_current_batch}개 파일) ---")
                # 배치 처리 함수 호출 (map_data를 직접 수정)
                unique_count, duplicate_set_count, moved_duplicates_count, move_errors = process_batch(
                    batch_hashes, organized_dir, duplicates_dir, map_data
                )
                # 이번 실행 통계 업데이트
                total_unique_count += unique_count
                total_duplicate_set_count += duplicate_set_count
                total_moved_duplicates_count += moved_duplicates_count
                total_move_errors += move_errors
                total_processed_hashes_in_run += len(batch_hashes)
                total_processed_files_in_run += files_in_current_batch

                # --- 배치 처리 후 맵 파일 저장 (중간 저장) ---
                save_map_file(map_data, map_file_path)
                logger.info(f"--- 배치 {batch_num} 처리 완료 (맵 파일 업데이트됨) ---")

                # 다음 배치를 위해 배치 정보 초기화
                batch_hashes = defaultdict(list)
                files_in_current_batch = 0

        except StopIteration:
            # 제너레이터가 모든 파일을 반환하고 종료된 경우
            # 마지막으로 남아있는 배치 처리
            if batch_hashes:
                batch_num += 1
                logger.info(f"--- 마지막 배치 {batch_num} 처리 시작 ({len(batch_hashes)}개 고유 해시, {files_in_current_batch}개 파일) ---")
                unique_count, duplicate_set_count, moved_duplicates_count, move_errors = process_batch(
                    batch_hashes, organized_dir, duplicates_dir, map_data
                )
                # 이번 실행 통계 업데이트
                total_unique_count += unique_count
                total_duplicate_set_count += duplicate_set_count
                total_moved_duplicates_count += moved_duplicates_count
                total_move_errors += move_errors
                total_processed_hashes_in_run += len(batch_hashes)
                total_processed_files_in_run += files_in_current_batch

                # --- 마지막 배치 처리 후 최종 맵 파일 저장 ---
                save_map_file(map_data, map_file_path)
                logger.info(f"--- 마지막 배치 {batch_num} 처리 완료 (맵 파일 업데이트됨) ---")
            break # 메인 루프 종료

        except Exception as e:
            # 파일 처리 루프 중 예상치 못한 오류 발생 시 로깅
            logger.error(f"파일 처리 루프 중 예외 발생: {e}", exc_info=True)
            logger.warning("오류 발생 후 다음 파일 처리 계속 시도...")
            # 오류 카운트 증가
            total_move_errors += 1
            # 필요시 현재 배치를 비우고 계속 진행할 수 있음
            # batch_hashes = defaultdict(list)
            # files_in_current_batch = 0

    # --- 최종 작업 요약 로그 출력 ---
    logger.info("=" * 30 + " 이번 실행 작업 요약 " + "=" * 30)
    logger.info(f"이번 실행에서 이동된 고유 파일 수    : {total_unique_count}")
    logger.info(f"이번 실행에서 발견된 중복 파일 세트 수: {total_duplicate_set_count}")
    logger.info(f"이번 실행에서 이동된 중복 파일 수    : {total_moved_duplicates_count}")
    logger.info(f"이번 실행 중 발생한 오류 수          : {total_move_errors}")
    logger.info(f"이번 실행에서 처리된 고유 해시 수    : {total_processed_hashes_in_run}")
    logger.info(f"이번 실행에서 처리 시도된 파일 수    : {total_processed_files_in_run}")
    logger.info(f"이번 실행에서 처리된 배치 수         : {batch_num}")
    logger.info(f"최종 맵 파일 위치                  : {map_file_path}") # 최종 맵 파일 경로 명시
    logger.info("=" * 70)
    logger.info("사진 정리 작업 완료.")

# ==============================================================================
# 메인 실행 블록 (`if __name__ == "__main__":`)
# 스크립트가 직접 실행될 때만 이 코드 블록이 실행됨
# ==============================================================================

if __name__ == "__main__":
    # --- 명령줄 인자 파서 설정 ---
    parser = argparse.ArgumentParser(
        description="디렉토리 내 이미지 파일을 배치 단위로 스캔하여 고유 파일과 중복 파일을 분리하고 이동 결과를 맵 파일에 저장합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 도움말 형식 개선 (기본값 표시)
    )
    # --- 필수 경로 인자 (옵션 형태로 지정, required=True) ---
    parser.add_argument("--target_dir", type=str, required=True,
                        help="사진을 검색할 시작 디렉토리 경로 (필수)")
    parser.add_argument("--orgniz_dir", type=str, required=True,
                        help="고유한 사진들을 이동시킬 대상 디렉토리 경로 (필수)")
    parser.add_argument("--duplic_dir", type=str, required=True,
                        help="중복된 사진들을 이동시킬 대상 디렉토리 경로 (필수)")
    # --- 선택적 옵션 인자들 ---
    parser.add_argument("--exclud_dirs", type=str, nargs='*', default=[],
                        help="검색에서 제외할 디렉토리 경로 목록 (공백으로 구분)")
    parser.add_argument("--batch_size", type=int, default=5000,
                        help="한 번에 처리할 이미지 파일 수 (메모리 사용량 조절)")
    parser.add_argument("--log_file", type=str, default="organize_photos.log", # 로그 파일 이름 기본값 변경
                        help="로그를 기록할 파일 이름")
    parser.add_argument("--map_file", type=str, default=None,
                        help="파일 이동 정보를 저장/로드할 맵 파일 경로 (기본값: 로그파일이름_map.json)")
    parser.add_argument("--debug", action="store_true",
                        help="디버그 레벨 로그를 활성화합니다 (더 상세한 로그 출력).")

    # 명령줄 인자 파싱
    args = parser.parse_args()

    # --- 로그 파일 핸들러 설정 ---
    log_file_path = os.path.abspath(args.log_file) # 로그 파일 절대 경로
    # 스크립트 재실행 시 로그 핸들러가 중복 추가되는 것을 방지하기 위해 기존 핸들러 제거
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler): # 파일 핸들러인 경우
            try:
                handler.close() # 핸들러 닫기
                logger.removeHandler(handler) # 로거에서 핸들러 제거
            except Exception as e:
                print(f"기존 로그 핸들러 제거 중 오류: {e}") # 로거 설정 전이므로 print 사용

    # 파일 핸들러 추가 (로그 파일에 기록)
    try:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8') # utf-8 인코딩
        # 로그 형식 설정 (시간 - 로그레벨 - 메시지)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler) # 로거에 파일 핸들러 추가
    except Exception as e:
        print(f"로그 파일 핸들러 설정 오류: {e}")
        # 파일 로깅 없이 계속 진행하거나 여기서 종료할 수 있음
        # sys.exit(1)

    # --- 로그 레벨 설정 ---
    # --debug 옵션이 있으면 DEBUG 레벨, 없으면 INFO 레벨
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level) # 로거의 기본 레벨 설정
    # 모든 핸들러(화면 출력, 파일 출력)의 레벨도 동일하게 설정
    for handler in logger.handlers:
        handler.setLevel(log_level)

    if args.debug:
        logger.info("디버그 로깅 활성화됨.")

    # --- 경로 인자 처리 (절대 경로 변환 및 사용자 홈 디렉토리 확장 '~') ---
    try:
        target_dir_abs = os.path.abspath(os.path.expanduser(args.target_dir))
        orgniz_dir_abs = os.path.abspath(os.path.expanduser(args.orgniz_dir))
        duplic_dir_abs = os.path.abspath(os.path.expanduser(args.duplic_dir))
        exclud_dirs_abs = [os.path.abspath(os.path.expanduser(d)) for d in args.exclud_dirs]
    except Exception as e:
        logger.error(f"경로 처리 중 오류 발생: {e}")
        sys.exit(1) # 경로 처리 오류 시 스크립트 종료

    # --- 맵 파일 경로 결정 ---
    if args.map_file:
        # 사용자가 --map_file 옵션으로 경로를 지정한 경우
        map_file_abs = os.path.abspath(os.path.expanduser(args.map_file))
    else:
        # 지정하지 않은 경우, 로그 파일 경로에서 확장자를 제외하고 '_map.json' 추가
        map_file_abs = os.path.splitext(log_file_path)[0] + "_map.json"

    # --- 스크립트 시작 정보 로깅 ---
    logger.info("="*20 + " 사진 정리 시작 (배치 모드) " + "="*20)
    logger.info(f"검색 디렉토리        : {target_dir_abs}")
    logger.info(f"선별된 사진 저장 위치: {orgniz_dir_abs}")
    logger.info(f"겹치는 사진 저장 위치: {duplic_dir_abs}")
    if exclud_dirs_abs:
        logger.info(f"제외할 디렉토리      : {exclud_dirs_abs}")
    logger.info(f"배치 크기            : {args.batch_size}")
    logger.info(f"로그 파일 위치       : {log_file_path}")
    logger.info(f"맵 파일 위치         : {map_file_abs}") # 맵 파일 경로 로깅
    logger.info("=" * 70 )

    # --- 메인 사진 정리 함수 호출 ---
    try:
        organize_photos(
            target_dir_abs,
            orgniz_dir_abs,
            duplic_dir_abs,
            map_file_path=map_file_abs, # 결정된 맵 파일 경로 전달
            exclude_dirs=exclud_dirs_abs,
            batch_size=args.batch_size
        )
    except Exception as e:
        # 예상치 못한 심각한 오류 발생 시 로깅 후 종료
        logger.critical(f"스크립트 실행 중 치명적인 오류 발생: {e}", exc_info=True)
        sys.exit(1)

