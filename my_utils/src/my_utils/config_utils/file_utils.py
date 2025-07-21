# file_utils.py
"""
Provides safe file operation wrappers that integrate with the project's
shared logger, including pre-emptive disk space checks.
"""

import os
import shutil
import errno
import hashlib
import re
from pathlib import Path
from typing import Optional
from PIL import Image
from PIL.ExifTags import TAGS

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

def check_disk_space(path: Path, min_free_bytes: int = 1 * 1024**3) -> bool:
    """Checks if the disk containing the path has at least min_free_bytes of free space."""
    try:
        total, used, free = shutil.disk_usage(path)
        return free >= min_free_bytes
    except Exception as e:
        logger.error(f"[check_disk_space] Failed to check disk space for {path}: {e}")
        return False

def calculate_sha256(file_path: Path) -> Optional[str]:
    """
    주어진 파일 경로에 대해 SHA256 해시 값을 계산하여 문자열로 반환합니다.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        logger.error(f"파일 읽기 오류 {file_path}: {e}")
        return None


def get_exif_date_taken(image_path: Path) -> Optional[str]:
    """
    이미지 파일에서 EXIF 'DateTimeOriginal' 태그를 읽어 촬영 날짜를 'YYYY-MM-DD' 형식으로 반환합니다.
    EXIF 데이터가 없거나 날짜 정보가 없으면 None을 반환합니다.
    이 함수를 사용하려면 'Pillow' 라이브러리가 설치되어 있어야 합니다 (pip install Pillow).
    """
    try:
        with Image.open(image_path) as img:
            # exif 데이터가 없는 경우도 있으므로 _getexif() 사용
            exif_data = img._getexif()
            if not exif_data:
                return None

            # 'DateTimeOriginal' 태그 (Tag ID 36867)를 찾습니다.
            date_taken_str = exif_data.get(36867)
            if date_taken_str:
                # 값은 'YYYY:MM:DD HH:MM:SS' 형식이므로, 날짜 부분만 추출하여 포맷 변경
                return date_taken_str.split(' ')[0].replace(':', '-')
    except Exception as e:
        # Pillow가 처리할 수 없는 파일이거나 다른 EXIF 관련 오류 발생 시
        logger.debug(f"EXIF 날짜를 읽는 중 오류 발생 '{image_path.name}': {e}")
        return "ERROR"
    return None

def get_original_filename(path: Path) -> str:
    """
    파일명에서 일반적인 버전 관리 패턴을 제거하여 원본 파일명을 반환합니다.
    예: 'photo (1).jpg' -> 'photo.jpg', 'image_v2.png' -> 'image.png'

    확장자를 제외한 파일명(stem)에서 다음 패턴을 제거합니다:
    - ' (숫자)' -> 예: 'photo (1)'
    - '-숫자' 또는 '_숫자' -> 예: 'photo-1', 'photo_1'
    - '_v'와 숫자 -> 예: 'photo_v2'
    - '-Copy' 또는 '_Copy' (대소문자 무관) -> 예: 'photo-Copy'
    """
    stem = path.stem
    suffix = path.suffix

    # ' (숫자)' 패턴 제거, 예: 'image (1)'
    stem = re.sub(r'\s*\(\d+\)$', '', stem)
    # '-숫자' 또는 '_숫자' 패턴 제거, 예: 'image-1' 또는 'image_1'
    stem = re.sub(r'[-_]\d+$', '', stem)
    # '_v' + 숫자 패턴 제거, 예: 'image_v2'
    stem = re.sub(r'[_]v\d+$', '', stem)
    # '-Copy' 또는 '_Copy' 패턴 제거 (대소문자 무관)
    stem = re.sub(r'[-_][Cc]opy$', '', stem)
    
    return stem.strip() + suffix



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
