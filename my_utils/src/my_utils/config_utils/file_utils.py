# file_utils.py
"""
Provides safe file operation wrappers that integrate with the project's
shared logger, including pre-emptive disk space checks.
"""

import os
import shutil
import errno
import re
from pathlib import Path
from typing import Generator, List, Tuple, Optional


try:
    from my_utils.config_utils.SimpleLogger import logger
except ImportError:
    # Fallback for standalone execution or if logger is not found
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DiskFullError(OSError):
    """Exception raised when there is no space left on device."""
    pass


def load_extensions_from_file(file_path: Path) -> set[str]:
    """
    지정된 파일에서 확장자 목록을 읽어 세트로 반환합니다.

    - 각 줄은 하나의 확장자로 간주합니다 (예: .jpg).
    - 확장자는 소문자로 변환됩니다.
    - '#'으로 시작하는 줄(주석)과 빈 줄은 무시됩니다.
    - '.'으로 시작하지 않는 유효하지 않은 형식의 항목은 경고를 기록하고 무시합니다.

    Args:
        file_path (Path): 확장자 목록이 포함된 텍스트 파일의 경로.

    Returns:
        set[str]: 읽어온 확장자들의 세트. 파일 읽기 실패 시 빈 세트를 반환합니다.
    """
    if not file_path.is_file():
        logger.error(f"확장자 파일 '{file_path}'을(를) 찾을 수 없거나 파일이 아닙니다.")
        return set()

    extensions = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                ext = line.strip()
                if not ext or ext.startswith('#'):
                    continue

                ext = ext.lower()
                if ext.startswith('.'):
                    extensions.add(ext)
                else:
                    logger.warning(f"'{file_path.name}' 파일 {line_num}번째 줄의 '{ext}'는 "
                                   f"유효한 확장자 형식(예: .jpg)이 아닐 수 있습니다. 무시합니다.")
    except IOError as e:
        logger.error(f"확장자 파일 '{file_path}'을(를) 읽는 중 오류 발생: {e}")
        return set()
    return extensions


def check_disk_space(path: Path, min_free_bytes: int = 1 * 1024**3) -> bool:
    """Checks if the disk containing the path has at least min_free_bytes of free space."""
    try:
        total, used, free = shutil.disk_usage(path)
        return free >= min_free_bytes
    except Exception as e:
        logger.error(f"[check_disk_space] Failed to check disk space for {path}: {e}")
        return False

def get_original_filename(path: Path) -> str:
    """
    파일명에서 일반적인 버전 관리 또는 복사본 패턴을 제거하여 원본 파일명을 추정합니다.
    
    예시:
        'photo (1).jpg'      → 'photo.jpg'
        'image_v2.png'       → 'image.png'
        'holiday-3.jpg'      → 'holiday.jpg'
        'report-Copy.docx'   → 'report.docx'
        '사본 - my_image.jpg' → 'my_image.jpg'
        '일괄편집_my_image.jpg' → 'my_image.jpg'

        확장자를 제외한 파일명(stem)에서 다음 패턴을 제거합니다:
        - ' (숫자)'               -> 예: 'photo (1)'
        - '-숫자' 또는 '_숫자'      -> 예: 'photo-1', 'photo_1'
        - '_v'와 숫자             -> 예: 'photo_v2'
        - '-Copy' 또는 '_Copy'    (대소문자 무관) -> 예: 'photo-Copy'
        - '사본 -', '일괄편집_', '포맷변환_' 등 접두사

    Parameters:
        path (Path): 분석할 파일 경로

    Returns:
        str: 정제된 원본 파일명
    """
    stem = path.stem
    suffix = path.suffix

    stem = re.sub(r'\s*\(\d+\)$', '', stem)
    # 2. '-숫자' 또는 '_숫자' 형태 제거: 예) 'image-1', 'file_3' → 'image', 'file'
    stem = re.sub(r'[-_]\d+$', '', stem)
    # 3. '_v숫자' 형태 제거: 예) 'photo_v2' → 'photo'
    stem = re.sub(r'[_]v\d+$', '', stem)
    # 4. '-Copy' 또는 '_Copy' 제거 (대소문자 무시): 예) 'file-Copy' → 'file'
    stem = re.sub(r'[-_][Cc]opy$', '', stem)
    # 5. '-복사본' 제거 (대소문자 무시): 예) 'file-복사본' → 'file'
    stem = re.sub(r'[-_]복사본$', '', stem)
    # 6. '-사본' 제거 (대소문자 무시): 예) 'file-복사본' → 'file'
    stem = re.sub(r'[-_]사본$', '', stem)
    # 7. '사본 -' 제거 (대소문자 무시): 예) 'file-복사본' → 'file'
    stem = re.sub(r'사본[ -_]$', '', stem)
    # 8. '일괄편집 -' 제거 (대소문자 무시): 예) 'file-복사본' → 'file'
    stem = re.sub(r'일괄편집[ -_]$', '', stem)
    # 8. '포맷변환 -' 제거 (대소문자 무시): 예) 'file-복사본' → 'file'
    stem = re.sub(r'포맷변환[ -_]$', '', stem)
    
    # 결과 반환: 정리된 이름 + 확장자
    return stem.strip() + suffix

def get_unique_path(directory: Path, filename: str) -> Path:
    """
    주어진 디렉토리 내에서 파일에 대한 고유한 경로를 생성합니다.
    동일한 이름의 파일이 이미 존재하는 경우, ' (숫자)' 형식의 버전 꼬리표를 추가합니다.
    예: 'image.jpg'가 존재하면 'image (1).jpg'를 반환합니다.

    Args:
        directory (Path): 대상 디렉토리.
        filename (str): 원본 파일 이름.

    Returns:
        Path: 디렉토리 내에서 고유성이 보장된 파일 경로.
    """
    destination = directory / filename
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        new_filename = f"{stem} ({counter}){suffix}"
        new_path = directory / new_filename
        if not new_path.exists():
            return new_path
        counter += 1

def safe_move(src: str, dst: str):
    """
    A wrapper for shutil.move that uses the shared logger's disk space monitor.
    Raises DiskFullError if there's not enough space.
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src}")

        # Check if move is across different filesystems
        src_dev = os.stat(src_path).st_dev

        # To get the destination device, we need an existing path on it.
        if hasattr(logger, '_disk_get_device_info'):
            dst_dev, _ = logger._disk_get_device_info(dst_path)
            if src_dev != dst_dev:
                file_size = src_path.stat().st_size
                logger.disk_pre_write_check(dst_path, file_size)

        shutil.move(src, dst)
    except FileNotFoundError:
        logger.error(f"Source file for move not found: {src}")
        raise
    except OSError as e:
        if e.errno == errno.ENOSPC:
            raise DiskFullError(f"No space left on device to move to '{dst}'") from e
        raise

def safe_copy(src: str, dst: str):
    """
    A wrapper for shutil.copy2 that uses the shared logger's disk space monitor.
    Raises DiskFullError if there's not enough space.
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
             raise FileNotFoundError(f"Source file not found: {src}")

        file_size = src_path.stat().st_size
        if hasattr(logger, 'disk_pre_write_check'):
            logger.disk_pre_write_check(dst_path, file_size)

        shutil.copy2(src, dst)
    except FileNotFoundError:
        logger.error(f"Source file for copy not found: {src}")
        raise
    except OSError as e:
        if e.errno == errno.ENOSPC:
            raise DiskFullError(f"No space left on device to copy to '{dst}'") from e
        raise

def scan_files_in_batches(
    root_dir: Path,
    batch_size: int = 100,
    allowed_extensions: Optional[List[str]] = None
) -> Tuple[int, Generator[List[Path], None, None]]:
    """
    조건에 맞는 파일들을 재귀적으로 탐색하고, 총 개수와 배치 단위 리스트를 제너레이터로 반환.
    이 함수는 먼저 모든 파일 경로를 수집하여 정확한 총 개수를 계산한 후,
    메모리에 있는 리스트로부터 배치(batch)를 생성하는 제너레이터를 반환합니다.

    Args:
        root_dir (Path): 탐색 시작 디렉토리
        batch_size (int): 한 번에 반환할 파일 경로 수
        allowed_extensions (List[str], optional): 허용된 확장자 목록 (예: ['.jpg', '.png'])

    Returns:
        Tuple[int, Generator[List[Path]]]: (전체 파일 수, 배치 제너레이터)
    """
    if not root_dir.is_dir():
        logger.error(f"탐색할 디렉토리를 찾을 수 없습니다: {root_dir}")
        return 0, (i for i in [])  # 0과 빈 제너레이터 반환

    if allowed_extensions:
        # 확장자 비교를 위해 소문자 set으로 변환하여 성능 향상
        allowed_exts = {ext.lower() for ext in allowed_extensions}
    else:
        allowed_exts = None  # 모든 파일 허용

    # 1. Path.rglob를 사용하여 조건에 맞는 모든 파일 경로를 리스트에 수집합니다.
    logger.info(f"'{root_dir}'에서 파일 스캔을 시작합니다...")
    matching_files = [
        p for p in root_dir.rglob("*")
        if p.is_file() and (allowed_exts is None or p.suffix.lower() in allowed_exts)
    ]

    total_count = len(matching_files)
    logger.info(f"총 {total_count}개의 파일을 찾았습니다. {batch_size}개씩 배치로 처리합니다.")

    # 2. 수집된 리스트에서 배치를 생성하는 제너레이터를 정의합니다.
    def batch_generator() -> Generator[List[Path], None, None]:
        for i in range(0, total_count, batch_size):
            yield matching_files[i:i + batch_size]

    return total_count, batch_generator()

import tempfile
import time

if __name__ == "__main__":
    if hasattr(logger, "setup"):
        logger.setup(
            console_min_level="INFO",
            file_min_level="DEBUG",
            logger_path=f"./file_utils_test_{time.strftime('%y%m%d')}.log"
        )
    else:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    """
    scan_files_in_batches 함수를 테스트하기 위한 메인 함수입니다.
    임시 디렉토리와 파일을 생성하고, 다양한 조건으로 함수를 호출하여 결과를 검증한 후,
    생성된 임시 파일들을 정리합니다.
    """
    temp_dir = None
    try:
        # 1. 테스트용 임시 디렉토리 및 파일 구조 생성
        temp_dir = Path(tempfile.mkdtemp(prefix="scantest_"))
        logger.info(f"--- 테스트 시작: 임시 디렉토리 생성됨: {temp_dir} ---")

        # 파일 구조 생성
        (temp_dir / "sub1").mkdir()
        (temp_dir / "sub1" / "file1.txt").touch()
        (temp_dir / "sub1" / "image1.jpg").touch()

        (temp_dir / "sub2").mkdir()
        (temp_dir / "sub2" / "file2.log").touch()
        (temp_dir / "sub2" / "image2.png").touch()
        (temp_dir / "sub2" / "image3.jpg").touch()

        (temp_dir / "file3.txt").touch()
        (temp_dir / "image4.JPG").touch() # 대문자 확장자 테스트

        logger.info("테스트 파일 구조 생성 완료.")
        time.sleep(0.1) # 파일 시스템이 변경사항을 인지할 시간을 줌

        # 2. 테스트 케이스 1: 특정 확장자, 배치 크기 2
        logger.info("\n--- 테스트 1: 특정 확장자 (.jpg, .png) / 배치 크기 2 ---")
        extensions_to_find = ['.jpg', '.png']
        batch_size = 2
        total_count, batch_gen = scan_files_in_batches(
            root_dir=temp_dir,
            batch_size=batch_size,
            allowed_extensions=extensions_to_find
        )

        logger.info(f"예상 파일 수: 3, 실제 반환된 총 파일 수: {total_count}")
        if total_count == 3:
            logger.info("[성공] 총 파일 수가 정확합니다.")
        else:
            logger.error(f"[실패] 총 파일 수가 일치하지 않습니다. (예상: 3, 실제: {total_count})")

        batch_num = 0
        for batch in batch_gen:
            batch_num += 1
            logger.info(f"  배치 {batch_num}:")
            for file_path in batch:
                logger.info(f"    - {file_path.relative_to(temp_dir)}")
        
        expected_batch_count = (total_count + batch_size - 1) // batch_size
        if batch_num == expected_batch_count:
             logger.info(f"[성공] 배치 개수가 정확합니다. (예상: {expected_batch_count}, 실제: {batch_num})")
        else:
             logger.error(f"[실패] 배치 개수가 일치하지 않습니다. (예상: {expected_batch_count}, 실제: {batch_num})")


        # 3. 테스트 케이스 2: 모든 파일, 배치 크기 3
        logger.info("\n--- 테스트 2: 모든 파일 / 배치 크기 3 ---")
        total_count_all, batch_gen_all = scan_files_in_batches(
            root_dir=temp_dir,
            batch_size=3,
            allowed_extensions=None # 모든 파일
        )

        logger.info(f"예상 파일 수: 5, 실제 반환된 총 파일 수: {total_count_all}")
        if total_count_all == 5:
            logger.info("[성공] 총 파일 수가 정확합니다.")
        else:
            logger.error(f"[실패] 총 파일 수가 일치하지 않습니다. (예상: 5, 실제: {total_count_all})")

        batch_num_all = 0
        for batch in batch_gen_all:
            batch_num_all += 1
            logger.info(f"  배치 {batch_num_all}:")
            for file_path in batch:
                logger.info(f"    - {file_path.relative_to(temp_dir)}")

    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}", exc_info=True)
    finally:
        # 4. 테스트 종료 후 임시 디렉토리 정리
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"\n--- 테스트 종료: 임시 디렉토리 삭제됨: {temp_dir} ---")
            except Exception as e:
                logger.error(f"임시 디렉토리 삭제 중 오류 발생 {temp_dir}: {e}")

def get_all_dirs(path: Path) -> list[Path]:
    """
    os.walk를 사용하여 깊이 우선 후위 순회 방식으로 하위 디렉토리 경로를 효율적으로 수집합니다.
    이 방식은 pathlib.rglob을 사용하는 것보다 일반적으로 더 빠릅니다.
    """
    # os.walk는 제너레이터를 반환하므로, 리스트로 변환합니다.
    # topdown=False는 가장 깊은 디렉토리부터 순회하도록 보장하므로, 별도의 정렬이 필요 없습니다.
    # 최상위 디렉토리(path)는 결과에서 제외합니다.
    all_subdirs = []
    for dirpath, _, _ in os.walk(path, topdown=False):
        current_path = Path(dirpath)
        if current_path != path:
            all_subdirs.append(current_path)
    return all_subdirs
