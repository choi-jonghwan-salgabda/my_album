# file_utils.py
"""
Provides safe file operation wrappers that integrate with the project's
shared logger, including pre-emptive disk space checks.
"""

import os
import shutil
import errno
from pathlib import Path

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