import os
import sys
import hashlib
import shutil
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, List, Set # 타입 힌트 추가

# --- 로거 설정 ---
# 기본 로거 설정 (파일 핸들러는 main 블록에서 추가)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- 지원할 이미지 확장자 ---
IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif'}

def calculate_hash(filepath: str, chunk_size: int = 8192) -> Optional[str]:
    """파일의 SHA256 해시 값을 계산합니다."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as file:
            while chunk := file.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except IOError as e:
        logger.error(f"파일 읽기 오류 '{filepath}': {e}")
        return None
    except Exception as e:
        logger.error(f"해시 계산 중 예상치 못한 오류 발생 '{filepath}': {e}")
        return None

# --- find_image_files를 제너레이터로 변경 ---
def find_image_files_generator(search_dir: str, exclude_dirs: Optional[List[str]] = None) -> Generator[Tuple[str, str], None, None]:
    """
    지정된 디렉토리와 하위 디렉토리에서 이미지 파일을 찾아 (해시, 경로) 튜플을 생성(yield)합니다.
    exclude_dirs에 포함된 디렉토리는 검색에서 제외합니다.
    """
    total_files_scanned = 0
    image_files_found = 0
    # 제외할 디렉토리 목록을 절대 경로 set으로 변환
    excluded_paths_abs: Set[str] = set()
    if exclude_dirs:
        try:
            excluded_paths_abs = {os.path.abspath(d) for d in exclude_dirs}
            logger.info(f"다음 디렉토리를 검색에서 제외합니다: {excluded_paths_abs}")
        except Exception as e:
            logger.error(f"제외 디렉토리 경로 처리 중 오류 발생: {e}")
            # 오류 발생 시 제외 없이 진행하거나, 여기서 중단할 수 있음
            # return # 또는 raise

    logger.info(f"'{search_dir}' 디렉토리에서 이미지 파일 검색 시작 (배치 처리 모드)...")

    # os.walk 에러 핸들링 추가
    try:
        for root, dirs, files in os.walk(search_dir, topdown=True, onerror=lambda err: logger.error(f"디렉토리 접근 오류: {err}")):
            # --- 제외 디렉토리 처리 ---
            try:
                current_root_abs = os.path.abspath(root)
                # 제외 목록에 있는 경로로 시작하는지 확인 (하위 폴더 포함 제외)
                if excluded_paths_abs and any(current_root_abs.startswith(excluded_path) for excluded_path in excluded_paths_abs):
                    logger.debug(f"제외된 디렉토리 건너뛰기: {root}")
                    dirs[:] = [] # 이 디렉토리의 하위 탐색 중단
                    continue # 다음 root로 이동
            except Exception as e:
                logger.error(f"제외 디렉토리 확인 중 오류 발생 (경로: {root}): {e}")
                continue # 문제가 있는 디렉토리는 건너뛰기

            # --- 기존 파일 처리 로직 ---
            for filename in files:
                total_files_scanned += 1
                try:
                    file_path = os.path.join(root, filename)
                    # 파일 확장자를 소문자로 변환하여 확인
                    if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                        image_files_found += 1
                        logger.debug(f"이미지 파일 발견: {file_path}")
                        file_hash = calculate_hash(file_path)
                        if file_hash:
                            # 해시와 파일 경로를 yield
                            yield file_hash, file_path
                            logger.debug(f"  - 해시: {file_hash} (생성됨)")
                        else:
                            logger.warning(f"해시 계산 실패: {file_path}")

                except Exception as e:
                    logger.error(f"파일 처리 중 오류 발생 (파일: {filename} in {root}): {e}")
                    # 개별 파일 오류는 로깅하고 계속 진행

                # 스캔 진행 상황 로깅 (선택적)
                if total_files_scanned % 1000 == 0: # 스캔 로그는 유지
                     logger.info(f"... {total_files_scanned}개 파일 스캔 중 ...")

    except Exception as e:
        logger.error(f"파일 시스템 탐색 중 예상치 못한 오류 발생 (시작 경로: {search_dir}): {e}", exc_info=True)
        # 심각한 오류 발생 시 여기서 중단될 수 있음

    logger.info(f"총 {total_files_scanned}개 파일 스캔 시도 완료.")
    logger.info(f"총 {image_files_found}개의 이미지 파일을 찾았습니다 (생성 완료).")


def safe_move_file(src_path: str, dest_dir: str) -> Optional[str]:
    """
    파일을 대상 디렉토리로 안전하게 이동합니다.
    이름 충돌 시 파일명 뒤에 숫자를 붙여 처리합니다.
    """
    if not os.path.exists(src_path):
        # 파일이 이미 이동되었거나 삭제된 경우, 경고 없이 None 반환 (배치 처리 중 흔히 발생 가능)
        # logger.warning(f"이동할 원본 파일 없음 (이미 처리됨?): {src_path}")
        return None # 이동 실패 또는 불필요

    # 대상 디렉토리 존재 확인 및 생성 시도 (필요한 경우)
    if not os.path.isdir(dest_dir):
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"대상 하위 디렉토리 생성 실패 '{dest_dir}': {e}")
            return None

    dest_path = os.path.join(dest_dir, os.path.basename(src_path))
    counter = 1
    # 파일 이름 충돌 해결
    original_name, ext = os.path.splitext(os.path.basename(src_path))
    while os.path.exists(dest_path):
        dest_path = os.path.join(dest_dir, f"{original_name}_{counter}{ext}")
        counter += 1
        if counter > 1000: # 무한 루프 방지
             logger.error(f"파일명 충돌 해결 시도 1000회 초과: {src_path} -> {dest_dir}")
             return None

    try:
        shutil.move(src_path, dest_path)
        logger.info(f"이동: '{src_path}' -> '{dest_path}'")
        return dest_path # 성공 시 이동된 경로 반환
    except (IOError, OSError) as e:
        logger.error(f"파일 이동 오류 '{src_path}' -> '{dest_path}': {e}")
        return None # 이동 실패
    except Exception as e:
        logger.error(f"파일 이동 중 예상치 못한 오류 발생 '{src_path}': {e}")
        return None # 이동 실패


# --- 배치 처리 함수 추가 ---
def process_batch(batch_hashes: Dict[str, List[str]], organized_dir: str, duplicates_dir: str) -> Tuple[int, int, int, int]:
    """주어진 해시 배치를 처리하여 파일을 이동하고 통계를 반환합니다."""
    unique_count = 0
    duplicate_set_count = 0
    moved_duplicates_count = 0
    move_errors = 0

    logger.info(f"배치 처리 시작: {len(batch_hashes)}개의 고유 해시 포함")

    for file_hash, paths in batch_hashes.items():
        # 실제 존재하는 파일 경로만 필터링 (이전 배치에서 이동되었을 수 있음)
        existing_paths = [p for p in paths if os.path.exists(p)]

        if not existing_paths:
            # 이 해시에 해당하는 모든 파일이 이미 처리됨
            logger.debug(f"해시 {file_hash}에 대한 모든 파일이 이미 처리됨, 건너뛰기.")
            continue

        if len(existing_paths) == 1:
            # 고유 파일 처리 (현재 배치 기준)
            src_path = existing_paths[0]
            if safe_move_file(src_path, organized_dir):
                unique_count += 1
            else:
                # safe_move_file 내부에서 오류 로깅됨
                move_errors += 1
        else:
            # 중복 파일 처리 (현재 배치 기준)
            duplicate_set_count += 1
            logger.warning(f"중복 발견 (해시: {file_hash}): {existing_paths}")

            # 첫 번째 존재하는 파일을 기준으로 중복 폴더 이름 결정
            base_filename = os.path.basename(existing_paths[0])
            duplicate_sub_dir_name = os.path.splitext(base_filename)[0]
            duplicate_target_dir = os.path.join(duplicates_dir, duplicate_sub_dir_name)

            # 대상 하위 폴더 생성은 safe_move_file 내부에서 처리됨

            # 존재하는 모든 중복 파일을 해당 하위 폴더로 이동
            for src_path in existing_paths:
                if safe_move_file(src_path, duplicate_target_dir):
                    moved_duplicates_count += 1
                else:
                    # safe_move_file 내부에서 오류 로깅됨
                    move_errors += 1

    logger.info(f"배치 처리 완료: 고유 {unique_count}, 중복 세트 {duplicate_set_count}, 이동된 중복 {moved_duplicates_count}, 오류 {move_errors}")
    return unique_count, duplicate_set_count, moved_duplicates_count, move_errors


# --- organize_photos 함수 수정 ---
def organize_photos(search_dir: str, organized_dir: str, duplicates_dir: str, exclude_dirs: Optional[List[str]] = None, batch_size: int = 5000):
    """
    사진을 배치 단위로 검색하고 고유/중복 파일을 분류하여 이동합니다.
    """
    # --- 경로 유효성 검사 및 생성 ---
    if not os.path.isdir(search_dir):
        logger.error(f"오류: 검색 디렉토리를 찾을 수 없습니다 - {search_dir}")
        return
    try:
        os.makedirs(organized_dir, exist_ok=True)
        os.makedirs(duplicates_dir, exist_ok=True)
        logger.info(f"대상 디렉토리 확인/생성 완료:")
        logger.info(f"  - 선별된 사진: {organized_dir}")
        logger.info(f"  - 겹치는 사진: {duplicates_dir}")
    except OSError as e:
        logger.error(f"대상 디렉토리 생성 실패: {e}")
        return

    # --- 전체 통계 변수 ---
    total_unique_count = 0
    total_duplicate_set_count = 0
    total_moved_duplicates_count = 0
    total_move_errors = 0
    total_processed_hashes = 0 # 처리된 고유 해시 수
    total_processed_files = 0 # 배치에 포함되어 처리 시도된 파일 수
    batch_num = 0

    # --- 파일 검색 및 배치 처리 ---
    image_generator = find_image_files_generator(search_dir, exclude_dirs)
    batch_hashes: Dict[str, List[str]] = defaultdict(list)
    files_in_current_batch = 0

    logger.info(f"배치 크기 {batch_size}로 파일 처리 시작...")

    while True:
        try:
            # 제너레이터에서 다음 (해시, 경로) 가져오기
            file_hash, file_path = next(image_generator)
            batch_hashes[file_hash].append(file_path)
            files_in_current_batch += 1

            # 현재 배치 크기가 지정된 크기에 도달하면 처리
            if files_in_current_batch >= batch_size:
                batch_num += 1
                logger.info(f"--- 배치 {batch_num} 처리 시작 ({len(batch_hashes)}개 고유 해시, {files_in_current_batch}개 파일) ---")
                # 현재 배치 처리
                unique_count, duplicate_set_count, moved_duplicates_count, move_errors = process_batch(
                    batch_hashes, organized_dir, duplicates_dir
                )
                # 전체 통계 업데이트
                total_unique_count += unique_count
                total_duplicate_set_count += duplicate_set_count
                total_moved_duplicates_count += moved_duplicates_count
                total_move_errors += move_errors
                total_processed_hashes += len(batch_hashes) # 처리된 고유 해시 수
                total_processed_files += files_in_current_batch # 처리 시도된 파일 수

                logger.info(f"--- 배치 {batch_num} 처리 완료 ---")
                # 다음 배치를 위해 초기화
                batch_hashes = defaultdict(list)
                files_in_current_batch = 0

        except StopIteration:
            # 제너레이터 종료 (모든 파일 스캔 완료)
            # 마지막 남은 배치 처리
            if batch_hashes:
                batch_num += 1
                logger.info(f"--- 마지막 배치 {batch_num} 처리 시작 ({len(batch_hashes)}개 고유 해시, {files_in_current_batch}개 파일) ---")
                unique_count, duplicate_set_count, moved_duplicates_count, move_errors = process_batch(
                    batch_hashes, organized_dir, duplicates_dir
                )
                # 전체 통계 업데이트
                total_unique_count += unique_count
                total_duplicate_set_count += duplicate_set_count
                total_moved_duplicates_count += moved_duplicates_count
                total_move_errors += move_errors
                total_processed_hashes += len(batch_hashes)
                total_processed_files += files_in_current_batch

                logger.info(f"--- 마지막 배치 {batch_num} 처리 완료 ---")
            break # while 루프 종료
        except Exception as e:
            logger.error(f"파일 처리 루프 중 예외 발생: {e}", exc_info=True)
            # 오류 발생 시 현재 배치를 건너뛰거나 중단할 수 있음
            # 여기서는 다음 파일/배치로 계속 진행하도록 함
            # 문제가 계속되면 수동 중단 필요
            logger.warning("오류 발생 후 다음 파일 처리 계속 시도...")
            # 현재 배치를 비우고 계속할 수 있음 (선택적)
            # batch_hashes = defaultdict(list)
            # files_in_current_batch = 0
            total_move_errors += 1 # 오류 카운트 증가


    # --- 최종 작업 요약 ---
    logger.info("=" * 30 + " 최종 작업 요약 " + "=" * 30)
    logger.info(f"총 이동된 고유 파일 수             : {total_unique_count}")
    logger.info(f"총 발견된 중복 파일 세트 수        : {total_duplicate_set_count}")
    logger.info(f"총 이동된 중복 파일 수             : {total_moved_duplicates_count}")
    logger.info(f"총 파일 이동/처리 중 발생한 오류 수: {total_move_errors}")
    logger.info(f"총 처리된 고유 콘텐츠(해시) 수     : {total_processed_hashes}")
    logger.info(f"총 처리 시도된 파일 수 (배치 합계) : {total_processed_files}")
    logger.info(f"총 처리된 배치 수                  : {batch_num}")
    logger.info("=" * 70)
    logger.info("사진 정리 작업 완료.")


# --- 메인 실행 블록 수정 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="디렉토리 내 이미지 파일을 배치 단위로 스캔하여 고유 파일과 중복 파일을 분리하여 이동합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 위치 인수로 변경 (필수)
    parser.add_argument("--target_dir", type=str,
                        help="사진을 검색할 시작 디렉토리 경로")
    parser.add_argument("--orgniz_dir", type=str,
                        help="고유한 사진들을 이동시킬 대상 디렉토리 경로")
    parser.add_argument("--duplic_dir", type=str,
                        help="중복된 사진들을 이동시킬 대상 디렉토리 경로")
    # 옵션 인자들
    parser.add_argument("--exclud_dir", type=str, nargs='*', default=[],
                        help="검색에서 제외할 디렉토리 경로 목록 (공백으로 구분)")
    parser.add_argument("--batch_size", type=int, default=5000, # 배치 크기 인자 추가
                        help="한 번에 처리할 이미지 파일 수 (메모리 사용량 조절)")
    parser.add_argument("--log_file", type=str, default="photo_organizer.log",
                        help="로그를 기록할 파일 이름")
    parser.add_argument("--debug", action="store_true",
                        help="디버그 레벨 로그를 활성화합니다.")

    args = parser.parse_args()

    # --- 로그 파일 핸들러 추가 ---
    log_file_path = os.path.abspath(args.log_file)
    # 기존 파일 핸들러 제거 (스크립트 재실행 시 중복 로깅 방지)
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            try:
                handler.close()
                logger.removeHandler(handler)
            except Exception as e:
                print(f"기존 로그 핸들러 제거 중 오류: {e}") # 로거 설정 전이므로 print 사용

    try:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"로그 파일 핸들러 설정 오류: {e}")
        # 파일 로깅 없이 계속 진행하거나 종료할 수 있음
        # sys.exit(1)

    # --- 디버그 레벨 설정 ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level)
    # 모든 핸들러의 레벨도 동일하게 설정
    for handler in logger.handlers:
        handler.setLevel(log_level)

    if args.debug:
        logger.info("디버그 로깅 활성화됨.")

    # --- 절대 경로 변환 및 검증 ---
    try:
        target_dir_abs = os.path.abspath(os.path.expanduser(args.target_dir))
        orgniz_dir_abs = os.path.abspath(os.path.expanduser(args.orgniz_dir))
        duplic_dir_abs = os.path.abspath(os.path.expanduser(args.duplic_dir))
        exclud_dir_abs = [os.path.abspath(os.path.expanduser(d)) for d in args.exclud_dir]
    except Exception as e:
        logger.error(f"경로 처리 중 오류 발생: {e}")
        sys.exit(1)

    # --- 시작 로그 ---
    logger.info("="*20 + " 사진 정리 시작 (배치 모드) " + "="*20)
    logger.info(f"검색 디렉토리        : {target_dir_abs}")
    logger.info(f"선별된 사진 저장 위치: {orgniz_dir_abs}")
    logger.info(f"겹치는 사진 저장 위치: {duplic_dir_abs}")
    if exclud_dir_abs:
        logger.info(f"제외할 디렉토리      : {exclud_dir_abs}")
    logger.info(f"배치 크기            : {args.batch_size}")
    logger.info(f"로그 파일 위치       : {log_file_path}")
    logger.info("=" * 70 )


    # --- 사진 정리 함수 호출 ---
    try:
        organize_photos(
            target_dir_abs,
            orgniz_dir_abs,
            duplic_dir_abs,
            exclude_dirs=exclud_dir_abs,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.critical(f"스크립트 실행 중 치명적인 오류 발생: {e}", exc_info=True)
        sys.exit(1)
