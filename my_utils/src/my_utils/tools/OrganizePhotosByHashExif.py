# OrganizePhotosByHash.py
"""
이 스크립트는 지정된 소스 디렉토리의 모든 이미지 파일을 스캔하여, 파일 내용(해시)과
촬영 날짜(EXIF)를 기반으로 체계적으로 정리하는 도구입니다.

주요 기능 및 특징:
1.  콘텐츠 기반 정리:
    각 이미지 파일의 내용에 기반한 SHA256 해시 값을 계산하여, 파일명이 다르더라도
    내용이 동일한 파일을 식별하고 그룹화합니다.

2.  날짜 기반 하위 폴더 생성:
    이미지의 EXIF 메타데이터에서 '촬영 날짜'를 읽어 `YYYY-MM-DD` 형식의
    하위 폴더를 생성합니다. 이를 통해 동일한 날짜에 찍은 사진들을 쉽게 찾을 수 있습니다.
    - EXIF 정보가 없거나 날짜 태그가 없는 경우: `unknown_date` 폴더에 저장됩니다.
    - EXIF 데이터가 손상되어 읽기 오류가 발생하는 경우: `exif_error` 폴더에 저장되어
      잠재적으로 손상된 파일을 식별할 수 있습니다.

3.  유연한 파일 처리 (`--action`):
    - `move` (기본값): 원본 이미지 파일을 대상 디렉토리로 이동시킵니다.
    - `copy`: 원본 파일을 유지한 채 대상 디렉토리로 복사합니다.

4.  지능적인 중복 처리:
    대상 디렉토리에 이미 동일한 해시와 파일명을 가진 파일이 존재할 경우,
    불필요한 작업을 방지합니다.
    - `move` 모드: 중복된 원본 파일을 삭제하여 저장 공간을 절약합니다.
    - `copy` 모드: 중복 복사를 건너뛰어 작업 시간을 단축합니다.

5.  안전한 테스트 실행 (`--dry-run`):
    이 옵션을 사용하면 실제 파일 이동, 복사, 삭제 없이 어떤 작업이 수행될지
    미리 로그로 확인할 수 있어 안전합니다.

6.  손상된 파일 처리:
    Pillow 라이브러리를 통해 이미지 파일의 유효성을 검사하여, 손상되었거나
    열 수 없는 파일은 별도로 처리하고 건너뜁니다.

7. 최종 디렉토리 구조 예시:
    <destination_dir>/
    └── a1b2c3d4.../  (파일 해시)
        ├── 2023-10-26/
        │   └── photo1.jpg
        ├── unknown_date/
        │   └── screenshot.png
        └── exif_error/
            └── corrupted_exif.jpg
"""

import os
import sys
from pathlib import Path
import shutil
import argparse
from datetime import datetime
import io
import copy
import hashlib
import concurrent.futures
import multiprocessing
import pprint
import threading
import functools
from tqdm import tqdm
from typing import Optional, Dict, List, Any, Tuple
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError

try:
    from my_utils.config_utils.arg_utils import get_argument
    from my_utils.config_utils.SimpleLogger import logger
    from my_utils.config_utils.configger import configger
    from my_utils.config_utils.file_utils import safe_move, safe_copy, DiskFullError, get_original_filename, get_unique_path    
    from my_utils.config_utils.display_utils import calc_digit_number, scan_files_and_get_display_width, truncate_string, visual_length, with_progress_bar, create_dynamic_description_func
    from my_utils.object_utils.photo_utils import calculate_sha256, get_exif_date_taken, is_image_valid_debug, ExifReadError
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

# 처리 상태를 기록하기 위한 기본 템플릿
DEFAULT_STATUS_TEMPLATE = {
    "images_scanned":               {"value": 0, "msg": "소스 디렉토리에서 스캔된 이미지 수"},
    "hashes_calculated":            {"value": 0, "msg": "계산된 해시 수"},
    "exif_dates_found":             {"value": 0, "msg": "EXIF에서 촬영 날짜를 찾은 이미지 수"},
    "exif_dates_missing":           {"value": 0, "msg": "EXIF 데이터가 없거나 촬영 날짜 태그가 없는 이미지 수"},
    "images_processed":             {"value": 0, "msg": "이동 또는 복사된 이미지 수"},
    "exif_read_errors":             {"value": 0, "msg": "손상 등의 이유로 EXIF 데이터를 읽지 못한 이미지 수"},
    "images_deleted_as_duplicate":  {"value": 0, "msg": "중복으로 인해 삭제된 이미지 수 (이동 모드)"},
    "images_skipped_as_duplicate":  {"value": 0, "msg": "중복으로 인해 건너뛴 이미지 수 (복사 모드)"},
    "corrupted_images_skipped":     {"value": 0, "msg": "손상되어 건너뛴 이미지 수"},
    "images_quarantined":           {"value": 0, "msg": "오류로 인해 격리된 파일 수"},
    "quarantine_errors":            {"value": 0, "msg": "파일 격리 중 발생한 오류 수"},
    "move_errors":                  {"value": 0, "msg": "파일 처리 중 발생한 오류 수"},
    "subdirectories_created":       {"value": 0, "msg": "생성된 해시 하위 디렉토리 수"},
}

def _quarantine_file(img_path: Path, quarantine_dir: Optional[Path], dry_run: bool, action: str, status: dict) -> bool:
    """
    오류가 발생한 파일을 격리 디렉토리로 이동 또는 복사합니다.
    
    - `get_unique_path`를 사용하여 격리 폴더에 동일한 이름의 파일이 이미 존재할 경우,
      ' (1)', ' (2)'와 같은 꼬리표를 붙여 덮어쓰기를 방지합니다.
    - `dry_run` 모드에서는 실제 파일 작업을 수행하지 않고 로그만 남깁니다.
    - `action` 값('move' 또는 'copy')에 따라 이동 또는 복사를 수행합니다.
    """
    if not quarantine_dir:
        return False
    
    try:
        # 격리 폴더에 동일한 이름의 파일이 있을 경우 덮어쓰지 않고 버전 꼬리표를 답니다.
        dest_path = get_unique_path(quarantine_dir, img_path.name)
        log_prefix = "(Dry Run) " if dry_run else ""
        action_verb = "이동" if action == 'move' else "복사"
        logger.warning(f"{log_prefix}오류 파일을 격리 폴더로 {action_verb}: '{img_path}' -> '{dest_path}'")
        if not dry_run:
            if action == 'move':
                safe_move(str(img_path), str(dest_path))
            else: # action == 'copy'
                safe_copy(str(img_path), str(dest_path))
        return True
    except (DiskFullError, OSError, Exception) as e:
        logger.error(f"파일 격리 실패: '{img_path}'. 오류: {e}")
        status["quarantine_errors"]["value"] += 1
        return False

def _process_file_for_parallel(img_path: Path) -> Dict[str, Any]:
    """
    병렬 처리를 위해 단일 파일을 처리하는 워커 함수.
    파일을 읽고, 유효성을 검사하며, 해시와 EXIF 날짜를 추출합니다.
    - EXIF 날짜 정보가 없거나 읽기 오류가 발생해도 처리를 중단하지 않고, 'unknown_date' 또는 'exif_error'로 분류하여 정상적으로 결과를 반환합니다.
    - 결과는 공유 상태(전역 변수 등)를 직접 수정하는 대신, 처리 결과를 담은 딕셔너리로 반환하여 데이터 경쟁을 방지합니다.
    """
    # --- I/O 최적화 ---
    # 각 파일을 디스크에서 딱 한 번만 읽어 메모리에 올린 후(io.BytesIO),
    # 유효성 검사, 해시 계산, EXIF 추출 등 모든 작업을 메모리 내에서 처리합니다.
    # 이는 동일한 파일을 여러 번 읽는 것을 방지하여 디스크 I/O를 최소화하고 성능을 향상시킵니다.
    #
    # --- 상태 비공유(Stateless) 설계 ---
    # 이 함수는 병렬 처리를 위해 여러 프로세스에서 독립적으로 실행됩니다.
    # 각 워커는 공유된 상태(예: 전역 변수, 상태 딕셔너리)를 직접 수정하지 않고,
    # 처리 결과를 담은 딕셔너리를 반환합니다. 이는 복잡한 잠금(Lock) 메커니즘 없이도
    # 데이터 경쟁(Race Condition)을 방지하는 안전하고 효율적인 방법입니다.
    try:

        img_path = Path(str(img_path).replace('\0', ''))        # 경로명에서 \0(null) 없애기

        file_content = img_path.read_bytes()
        if not file_content:
            logger.warning(f"사진 읽기 실패: err_msg: 사진의 크기가 0 입니다. path:'{img_path}'")
            return {"path": img_path, "err_type": "zero_byte", "err_msg":"사진의 크기가 0 입니다."}
        
        file_stream = io.BytesIO(file_content)

        # is_image_valid_debug 함수는 이미지 전체를 로드하여 더 엄격하게 검사합니다.
        valid, err_msg = is_image_valid_debug(file_stream)
        if not valid:
            logger.warning(f"사진 검증 실패: err_msg: '{err_msg}', path:'{img_path}'")
            return {"path": img_path, "err_type": "corrupted_image", "err_msg": err_msg}

        # is_image_valid_debug가 스트림을 읽었으므로, 해시 계산 전에 스트림 위치를 처음으로 되돌립니다.
        file_stream.seek(0)
        file_hash, hash_err_msg = calculate_sha256(file_stream)
        if not file_hash:
            logger.warning(f"사진 Hash만들기 실패: err_msg: '{hash_err_msg}', path:'{img_path}'")
            return {"path": img_path, "err_type": "hash_failed", "err_msg": hash_err_msg}

        # EXIF 데이터를 읽기 전에 스트림 위치를 다시 처음으로 되돌립니다.
        # 각 함수가 스트림을 독립적으로 사용할 수 있도록 보장하는 것이 안전합니다.
        file_stream.seek(0)
        date_str, exif_err_msg = get_exif_date_taken(file_stream)

        if date_str:
            # EXIF 날짜를 성공적으로 가져온 경우, 해당 날짜 문자열을 사용합니다.
            final_date_str = date_str
        elif exif_err_msg:
            # EXIF 날짜를 가져오지 못하고 오류 메시지가 있는 경우, 'exif_error'로 분류합니다.
            logger.warning(f"사진 찍은 날짜 실패: {exif_err_msg}, 경로:'{img_path}'")
            final_date_str = "exif_error"
        else:
            # 그 외의 경우 (예: EXIF 데이터는 있으나 날짜 태그가 전혀 없는 경우) 'unknown_date'로 분류합니다.
            logger.warning(f"사진 찍은 날짜 실패: 원인 불명, 경로:'{img_path}'")
            final_date_str = "unknown_date"
        
        original_name = get_original_filename(img_path)

        # --- Null Byte 제거 (오류 방지) ---
        # 파일 시스템에서 사용할 수 없는 NULL 바이트(\0)를 제거하여 'embedded null byte' 오류를 방지합니다.
        # EXIF 데이터나 파일명에 예기치 않게 포함될 수 있습니다.
        safe_date_str = final_date_str.replace('\0', '')        # 경로명에서 \0(null) 없애기
        safe_original_name = original_name.replace('\0', '')        # 경로명에서 \0(null) 없애기

        return {
            "path": img_path,
            "hash": file_hash,
            "date_str": safe_date_str,
            "original_name": safe_original_name,
            "err_type": None,
            "err_msg": None
        }
    except FileNotFoundError:
        logger.warning(f"file_not_found: err_msg: '파일을 찾을 수 없습니다.', path:'{img_path}'")
        return {"path": img_path, "err_type": "file_not_found", "err_msg": "파일을 찾을 수 없습니다."}
    except (IOError, OSError) as e:
        logger.warning(f"read_error: err_msg: '파일 읽기 오류: {e}', path:'{img_path}'")
        return {"path": img_path, "err_type": "read_error", "err_msg": f"파일 읽기 오류: {e}"}
    except Exception as e:
        logger.warning(f"unknown: err_msg: '알 수 없는 오류: {e}', path:'{img_path}'")
        return {"path": img_path, "err_type": "unknown", "err_msg": f"알 수 없는 오류: {e}"}

def log_listener_process(queue: multiprocessing.Queue):
    """
    [병렬 처리] 멀티프로세싱 로그 리스너 프로세스(또는 스레드).
    
    - 역할: 여러 워커 프로세스에서 발생하는 로그 메시지를 중앙에서 처리하기 위한 목적.
    - 동작: 공유된 `log_queue`에서 로그 레코드(레벨, 메시지)를 계속 가져와,
            메인 프로세스의 `logger` 인스턴스를 사용하여 파일 및 콘솔에 기록합니다.
    - 이유: 각 워커 프로세스가 직접 파일에 쓰면 파일 잠금 문제(Race Condition)가 발생하므로,
            하나의 전담 리스너가 모든 로그를 순차적으로 처리하여 문제를 해결합니다.
    """
    while True:
        try:
            record = queue.get()
            if record is None:  # 종료 신호
                break
            level, message = record
            logger.log(level=level, message=message)
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception:
            import traceback
            print(f"로그 리스너 오류:\n{traceback.format_exc()}", file=sys.stderr)

def init_worker(queue: multiprocessing.Queue):
    """
    [병렬 처리] 각 워커 프로세스 초기화 함수.
    
    - 역할: `ProcessPoolExecutor`가 새로운 워커 프로세스를 생성할 때마다 호출됩니다.
    - 동작: 각 워커 프로세스 내의 전역 `logger` 인스턴스를 설정하여,
            모든 로그 메시지(예: logger.info, logger.warning)가 파일이나 콘솔에 직접 기록되지 않고,
            대신 공유 `log_queue`로 보내지도록 설정합니다.
    """
    logger.setup(mp_log_queue=queue)

def _process_hash_group_parallel(args: Tuple[str, List[Dict[str, Any]], Path, str, bool]) -> Dict[str, Any]:
    """
    [병렬 처리] 단일 해시 그룹을 처리하는 워커 함수.
    
    - 작업 단위: 동일한 해시값을 가진 파일들의 묶음(그룹).
    - 역할: 한 워커가 하나의 해시 그룹 전체를 책임지고 처리합니다.
            그룹에 속한 모든 파일에 대해 실제 이동/복사/삭제 작업을 수행합니다.
    - 반환: 이 워커에서 수행된 작업의 결과를 담은 상태 딕셔너리를 반환합니다.
            메인 프로세스는 모든 워커로부터 이 결과들을 받아 최종 통계를 집계합니다.
    - 이유: 관련된 파일(동일 해시)들을 하나의 워커가 처리함으로써 디렉토리 생성 등의 작업을 효율적으로 관리합니다.
    """
    hash_value, tasks, destination_dir, action, dry_run = args
    
    worker_status = {
        "images_processed": 0,
        "images_deleted_as_duplicate": 0,
        "images_skipped_as_duplicate": 0,
        "move_errors": 0,
        "subdirectories_created": 0,
        "error_details": [] # 오류 상세 정보를 담을 리스트
    }
    created_dirs_in_worker = set()

    for task_data in tasks:
        img_path = Path(str(task_data["path"]).replace('\0', ''))       # 경로명에서 \0(null) 없애기
        safe_name = str(task_data["original_name"]).replace('\0', '')       # 경로명에서 \0(null) 없애기
        dest_file_path = destination_dir / hash_value / task_data["date_str"] / safe_name
        
        try:
            if dest_file_path.exists():
                if action == 'move':
                    if not dry_run:
                        img_path.unlink()
                    worker_status["images_deleted_as_duplicate"] += 1
                else: # copy
                    worker_status["images_skipped_as_duplicate"] += 1
            else:
                dest_parent_dir = dest_file_path.parent
                is_new_dir = not dest_parent_dir.exists() and dest_parent_dir not in created_dirs_in_worker

                if is_new_dir:
                    created_dirs_in_worker.add(dest_parent_dir)
                    if not dry_run:
                        dest_parent_dir.mkdir(parents=True, exist_ok=True)
                
                if not dry_run:
                    (safe_move if action == 'move' else safe_copy)(str(img_path), str(dest_file_path))
                worker_status["images_processed"] += 1
        except Exception as e:
            # 워커 프로세스에서는 직접 로깅하는 대신 오류 정보를 집계하여 반환합니다.
            worker_status["move_errors"] += 1
            import traceback
            error_info = {
                "file": str(img_path),
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            worker_status["error_details"].append(error_info)

    worker_status["subdirectories_created"] = len(created_dirs_in_worker)
    return worker_status


def organize_photos_by_hash_logic(
    source_dir: Path,
    destination_dir: Path,
    allowed_extensions: Optional[set[str]] = None,
    quarantine_dir: Optional[Path] = None,
    dry_run: bool = True,
    action: str = 'copy',
    parallel: bool = False,
    max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
    """
    해시값과 EXIF 날짜 정보를 기준으로 이미지 파일을 정리하는 핵심 로직.
    이 함수는 여러 단계(Phase)로 구성되며, 각 단계는 특정 작업을 수행합니다.
    병렬 처리 옵션이 활성화되면, 각 단계의 특성에 맞춰 최적화된 병렬 처리 전략을 사용합니다.
    
    Parameters:
        source_dir (Path): 원본 이미지 파일들이 있는 디렉토리 경로
        destination_dir (Path): 정리된 이미지들이 저장될 대상 디렉토리
        quarantine_dir (Optional[Path]): 오류 파일을 이동할 격리 디렉토리.
        dry_run (bool): 실제 파일을 이동/복사하지 않고 경로만 시뮬레이션할지 여부 (기본값: False)
        action (str): 'move' 또는 'copy' 중 선택, 파일을 이동할지 복사할지 결정 (기본값: 'move')

    Returns:
        dict: 처리 상태를 담은 상태 딕셔너리 (스캔한 이미지 수, 오류 수, 생성된 하위 디렉토리 수 등)
    Returns:
        dict: 처리 상태를 담은 상태 딕셔너리 (스캔한 이미지 수, 오류 수, 생성된 하위 디렉토리 수 등)
    """
    # DecompressionBombError 방지를 위해 Pillow의 이미지 픽셀 수 제한을 해제합니다.
    Image.MAX_IMAGE_PIXELS = None
    status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)

    if not source_dir.is_dir():
        logger.error(f"소스 디렉토리를 찾을 수 없습니다: {source_dir}")
        return status
    logger.crt_view(f"소스 디렉토리: {source_dir}")

    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        logger.crt_view(f"대상 디렉토리: {destination_dir}")
    except OSError as e:
        logger.error(f"대상 디렉토리를 생성할 수 없습니다 ({destination_dir}): {e}")
        return status

    # ======================================================================================
    # Phase 1: 파일 목록 스캔
    # ======================================================================================
    # - 작업 내용: 소스 디렉토리 내의 모든 이미지 파일 경로를 수집합니다.
    # - 병렬 처리 전략: 하위 디렉토리 단위 분배.
    #   - 소스 디렉토리 바로 아래의 하위 디렉토리들을 각 워커에게 할당합니다.
    #   - 하위 디렉토리 수가 적고 특정 디렉토리에 파일이 몰려있으면 비효율적일 수 있으므로,
    #     하위 디렉토리 수가 CPU 코어 수에 비해 충분히 많을 때만 병렬 스캔을 사용하고,
    #     그렇지 않으면 더 빠른 단일 스레드 순차 스캔으로 전환합니다.
    logger.crt_view(f"'{source_dir}'에서 처리할 이미지 파일 목록 스캔을 시작합니다...")

    all_found_files = []
    visual_width = 40  # 기본 너비
    scan_message = ""

    num_cores = os.cpu_count() or 1
    sub_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    
    if parallel and len(sub_dirs) > num_cores / 2:
        logger.info(f"하위 디렉토리({len(sub_dirs)}개)가 충분하여 병렬 스캔을 사용합니다 (전략: 디렉토리 단위 분배).")
        all_found_files, visual_width, scan_message = scan_files_and_get_display_width(
            source_dir=source_dir,
            extensions=allowed_extensions,
            buffer_ratio=0.25,
            min_width=20,
            max_width=50
        )
    else:
        if parallel:
            logger.info(f"하위 디렉토리({len(sub_dirs)}개)가 적어 더 효율적인 순차 스캔을 사용합니다.")
        
        with tqdm(desc="파일 스캔 중 (순차)", unit="개") as pbar:
            for p in source_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in allowed_extensions:
                    all_found_files.append(p)
                    pbar.update(1)
        scan_message = f"순차 스캔 완료. 총 {len(all_found_files)}개 파일 발견."

    if all_found_files is None:
        logger.error(f"파일 스캔 중 심각한 오류가 발생하여 작업을 중단합니다. 메시지: {scan_message}")
        return status
    logger.crt_view(f"파일 스캔 결과: {scan_message}")

    image_files = []
    # 격리 디렉토리가 소스 디렉토리 내부에 있을 경우, 해당 디렉토리의 파일은 처리 대상에서 제외합니다.
    # 이는 격리된 파일을 다시 스캔하여 무한 루프에 빠지는 것을 방지합니다.
    if quarantine_dir and source_dir in quarantine_dir.parents:
        resolved_quarantine_dir = quarantine_dir.resolve()
        image_files = [f for f in all_found_files if not str(f.resolve()).startswith(str(resolved_quarantine_dir))]
    else:
        image_files = all_found_files

    image_number = len(image_files)
    status["images_scanned"]["value"] = image_number
    logger.crt_view(f"{source_dir} 에서 {image_number}개의 이미지 파일을 찾았습니다.")
    if not image_files:
        logger.crt_view("처리할 이미지가 없습니다.")
        return status

    # --- Phase 2: 정보 수집 (병렬 또는 순차) ---
    logger.crt_view("이미지 정보(해시, EXIF) 수집을 시작합니다...")
    results = []
    if parallel:
        # --- 병렬 처리 및 로깅 큐 설정 ---
        manager = multiprocessing.Manager()
        log_queue = manager.Queue()

        # 로그 리스너 스레드 시작
        listener_thread = threading.Thread(target=log_listener_process, args=(log_queue,))
        listener_thread.start()

        # 워커 초기화 함수 설정
        initializer = functools.partial(init_worker, log_queue)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=initializer) as executor:
            results = list(tqdm(executor.map(_process_file_for_parallel, image_files), total=image_number, desc="정보 수집 중 (병렬)", unit="파일"))

        log_queue.put(None)  # 리스너 스레드에 종료 신호 전송
        listener_thread.join() # 리스너 스레드 종료 대기
    else:
        # 순차 처리
        for img_path in tqdm(image_files, desc="정보 수집 중 (순차)", unit="파일"):
            results.append(_process_file_for_parallel(img_path))

    # ======================================================================================
    # Phase 3: 결과 분석 및 오류 처리
    # ======================================================================================
    # - 작업 내용: 병렬/순차 처리된 결과(딕셔너리 리스트)를 순회하며,
    #   성공한 작업과 실패한 작업을 분리하고 통계를 업데이트합니다.
    #   오류가 발생한 파일은 격리 처리합니다.
    successful_tasks = []
    for res in results:
        if res["err_type"]:
            error_type = res["err_type"]
            img_path = res["path"]
            if error_type == "zero_byte" or error_type == "corrupted_image":
                status["corrupted_images_skipped"]["value"] += 1
            elif error_type == "exif_error":
                status["exif_read_errors"]["value"] += 1
            else:
                status["move_errors"]["value"] += 1
            
            logger.warning(f"정보 수집 오류 ({error_type}): '{img_path}'")
            if _quarantine_file(img_path, quarantine_dir, dry_run, action, status):
                status["images_quarantined"]["value"] += 1
        else:
            successful_tasks.append(res)
            status["hashes_calculated"]["value"] += 1
            if res["date_str"] not in ["unknown_date", "exif_error"]:
                status["exif_dates_found"]["value"] += 1
            elif res["date_str"] == "unknown_date":
                status["exif_dates_missing"]["value"] += 1
            elif res["date_str"] == "exif_error":
                status["exif_read_errors"]["value"] += 1
            
    # ======================================================================================
    # Phase 4: 작업 그룹화
    # ======================================================================================
    # - 작업 내용: 성공적으로 처리된 파일 정보들을 해시값 기준으로 그룹화합니다.
    #   `defaultdict(list)`를 사용하여 동일한 해시값을 가진 파일 정보들을 하나의 리스트로 묶습니다.
    from collections import defaultdict
    logger.crt_view("처리할 파일들을 해시값 기준으로 그룹화합니다...")
    grouped_tasks = defaultdict(list)
    for task in successful_tasks:
        grouped_tasks[task['hash']].append(task)
    logger.crt_view(f"{len(grouped_tasks)}개의 해시 그룹으로 작업을 그룹화했습니다.")

    # ======================================================================================
    # Phase 5: 파일 작업 (이동/복사)
    # ======================================================================================
    # - 작업 내용: 그룹화된 파일들에 대해 실제 이동/복사 작업을 수행합니다. 이 단계는 I/O 집약적입니다.
    # - 병렬 처리 전략: 해시 그룹 단위 분배.
    #   - 각 워커는 하나의 해시 그룹 전체(동일한 파일의 중복본들)를 처리합니다.
    #   - 이렇게 하면 동일한 해시 디렉토리를 생성하는 작업이 여러 워커에서 동시에 발생하는 것을
    #     방지하고, 관련된 파일 I/O를 한 워커가 전담하여 처리 흐름을 단순화합니다.
    if grouped_tasks:
        logger.crt_view(f"파일 이동/복사 작업을 시작합니다...")
        
        # 병렬 처리를 위한 인자 리스트 준비
        processing_args = [
            (hash_val, tasks, destination_dir, action, dry_run) 
            for hash_val, tasks in grouped_tasks.items()
        ]

        group_results = []
        if parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results_iterator = executor.map(_process_hash_group_parallel, processing_args)
                group_results = list(tqdm(results_iterator, total=len(grouped_tasks), desc="파일 그룹 처리 중 (병렬)", unit="그룹"))
        else:
            # 순차 처리
            for args_tuple in tqdm(processing_args, desc="파일 그룹 처리 중 (순차)", unit="그룹"):
                group_results.append(_process_hash_group_parallel(args_tuple))

        # ======================================================================================
        # Phase 6: 최종 통계 집계
        # ======================================================================================
        # - 작업 내용: 각 워커(또는 순차 처리)로부터 반환된 상태 딕셔너리들을 합산하여 최종 통계를 만듭니다.
        for group_status in group_results:
            for key, value in group_status.items():
                if key == "error_details":
                    for error_detail in value:
                        logger.error(f"워커 오류 발생:\n  파일: {error_detail['file']}\n  오류: {error_detail['error']}\n{error_detail['traceback']}")
                elif key in status:
                    status[key]["value"] += value

    return status

if __name__ == "__main__":
    """
    스크립트의 메인 실행 함수입니다.
    """
    # -src와 -dst만 필수로 지정하고, -qrt(격리폴더) 등은 선택사항으로 변경합니다.
    # -qrt를 지정하지 않으면 소스 디렉토리 하위에 'quarantine' 폴더가 자동으로 사용됩니다.
    supported_args_for_script = [
        # 이 스크립트가 지원하는 인자 목록을 명시적으로 정의합니다.
        'source_dir', 
        'destination_dir', 
        'quarantine_dir', 
        'log_mode',
        'dry_run', 
        'action', 
        'parallel', 
        'max_workers'
    ]
    parsed_args = get_argument(
        required_args=['-src', '-dst'],
        supported_args=supported_args_for_script
    )
    script_name = Path(__file__).stem
    
    if hasattr(logger, "setup"):
        date_str = datetime.now().strftime("%y%m%d_%H%M")
        log_file_name = f"{script_name}_{date_str}.log"
        full_log_path = Path(parsed_args.log_dir) / log_file_name
        # 로거 설정: 콘솔에는 CRT_VIEW 레벨 이상만, 파일에는 사용자가 지정한 레벨 이상을 기록
        logger.setup(
            logger_path=full_log_path,
            console_min_level = "crt_view",
            file_min_level=parsed_args.log_level.upper(),
            include_function_name=True,
            pretty_print=True,
            async_file_writing=(parsed_args.log_mode == 'async')
        )
    
    logger.crt_view(f"애플리케이션 ({script_name}) 시작")

    try:
        # 설정 파일(YAML) 로드
        config_manager = configger(root_dir=parsed_args.root_dir, config_path=parsed_args.config_path)
        logger.debug("Configger 초기화 완료.")
        # YAML에서 리스트를 가져와 소문자 set으로 변환
        extensions_from_config = config_manager.get_value("processing.supported_image_extensions", default=[])
        allowed_extensions = {ext.lower() for ext in extensions_from_config} if extensions_from_config else set()
    except Exception as e:
        logger.error(f"Configger 초기화 또는 설정 로드 중 오류 발생: {e}", exc_info=True)
        allowed_extensions = set() # 오류 발생 시 빈 세트로 초기화

    # 명령줄 인자로부터 경로 및 옵션 설정
    input_dir_path = parsed_args.source_dir

    source_dir = Path(input_dir_path).expanduser().resolve()
    destination_dir = Path(parsed_args.destination_dir).expanduser().resolve()
    dry_run_mode = parsed_args.dry_run
    action_mode = parsed_args.action # argparse에 정의된 기본값('copy')을 사용합니다.

    # 격리 디렉토리 경로 결정
    quarantine_dir: Path
    if parsed_args.quarantine_dir:
        # 사용자가 명시적으로 경로를 지정한 경우
        quarantine_dir = Path(parsed_args.quarantine_dir).expanduser().resolve()
        logger.crt_view(f"사용자 지정 격리 디렉토리 사용: '{quarantine_dir}'")
    else:
        # 기본값: 소스 디렉토리 아래 'quarantine' 폴더
        quarantine_dir = source_dir / 'quarantine'
        logger.crt_view(f"기본 격리 디렉토리 사용: '{quarantine_dir}'")

    if not dry_run_mode: # 실제 실행 모드일 때만 디렉토리 생성
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    if source_dir == destination_dir:
        logger.error("소스 디렉토리와 대상 디렉토리는 같을 수 없습니다. 스크립트를 종료합니다.")
        sys.exit(1)

    try:
        # 메인 로직 함수 호출
        logger.crt_view("+++++++++++++++++++++++++++++++++++++")
        logger.crt_view("--- 사진 해시 기반 정리 처리 작업 시작 ---")
        logger.crt_view("-------------------------------------")
        # organize_photos_by_hash_logic 함수에 action_mode 전달
        final_status = organize_photos_by_hash_logic(
            source_dir=source_dir,
            allowed_extensions=allowed_extensions,
            destination_dir=destination_dir,
            quarantine_dir=quarantine_dir,
            dry_run=dry_run_mode,
            action=action_mode,
            parallel=parsed_args.parallel,
            max_workers=parsed_args.max_workers
        )

        # 최종 통계 출력
        logger.crt_view("--- 사진 해시 기반 정리 처리 작업 통계 ---")
        # Get all messages and find the maximum visual length for alignment
        all_msgs = [v.get("msg", k) for k, v in DEFAULT_STATUS_TEMPLATE.items()]
        max_visual_msg_len = max(visual_length(msg) for msg in all_msgs) if all_msgs else 20

        # Get all values to determine the required width for the numbers
        all_values = [s_item["value"] for s_item in final_status.values()]
        max_val_for_width = max(all_values) if all_values else 0
        digit_width_stats = calc_digit_number(max_val_for_width)
        
        for key, data in final_status.items():
            msg = DEFAULT_STATUS_TEMPLATE.get(key, {}).get("msg", key.replace("_", " ").capitalize())
            value = data["value"]
            padding = ' ' * (max_visual_msg_len - visual_length(msg))
            logger.crt_view(f"  {msg}{padding} : {value:>{digit_width_stats}}")
        logger.crt_view("-------------------------------------")
        logger.crt_view("--- 사진 해시 기반 정리 처리 작업 끝 ----")
        logger.crt_view("+++++++++++++++++++++++++++++++++++++")

    except KeyboardInterrupt:
        logger.warning("\n사용자에 의해 작업이 중단되었습니다.")
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 최상위 오류 발생: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.crt_view(f"애플리케이션 ({script_name}) 종료{ ' (Dry Run 모드)' if dry_run_mode else ''}")
        if hasattr(logger, "shutdown"):
            logger.shutdown()
