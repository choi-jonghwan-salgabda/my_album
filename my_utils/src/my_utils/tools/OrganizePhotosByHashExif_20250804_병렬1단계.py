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
from tqdm import tqdm # tqdm 임포트
from typing import Optional, Dict, List, Any 
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
    """오류가 발생한 파일을 격리 디렉토리로 이동 또는 복사합니다."""
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

def _execute_file_action(img_path: Path, dest_file_path: Path, action: str, dry_run: bool, status: dict, created_dirs: set):
    """
    계산된 정보를 바탕으로 실제 파일 이동, 복사, 삭제 또는 건너뛰기 작업을 수행합니다.
    """
    if dest_file_path.exists():
        if action == 'move':
            if dry_run:
                logger.info(f"(Dry Run) 중복 파일 삭제 예정 (소스): '{img_path}'")
            else:
                img_path.unlink()
                logger.info(f"중복 파일 삭제 (소스): '{img_path}'")
            status["images_deleted_as_duplicate"]["value"] += 1
        elif action == 'copy':
            logger.info(f"중복 파일 건너뛰기 (소스): '{img_path}'")
            status["images_skipped_as_duplicate"]["value"] += 1
    else:
        dest_parent_dir = dest_file_path.parent
        is_new_dir = not dest_parent_dir.exists() and dest_parent_dir not in created_dirs

        if is_new_dir:
            created_dirs.add(dest_parent_dir)
            if not dry_run:
                dest_parent_dir.mkdir(parents=True, exist_ok=True)

        # 로그 메시지 생성 및 출력
        log_prefix = "(Dry Run) " if dry_run else ""
        action_verb = "이동" if action == 'move' else "복사"
        log_msg = f"{log_prefix}{action_verb} 예정: '{img_path}' -> '{dest_file_path}'"
        if is_new_dir and dry_run:
            log_msg += f" (새 디렉토리 '{dest_parent_dir.name}' 생성 예정)"
        logger.info(log_msg)

        if not dry_run:
            (safe_move if action == 'move' else safe_copy)(str(img_path), str(dest_file_path))
        status["images_processed"]["value"] += 1

def _process_file_for_parallel(img_path: Path) -> Dict[str, Any]:
    """
    병렬 처리를 위해 단일 파일을 처리하는 워커 함수.
    파일을 읽고, 유효성을 검사하며, 해시와 EXIF 날짜를 추출합니다.
    결과는 공유 상태를 수정하는 대신 딕셔너리로 반환됩니다.
    """
    try:
        file_content = img_path.read_bytes()
        if not file_content:
            return {"path": img_path, "error": "zero_byte"}
        
        file_stream = io.BytesIO(file_content)

        if not is_image_valid_debug(file_stream):
            return {"path": img_path, "error": "corrupted_image"}

        file_hash = calculate_sha256(file_stream)
        if not file_hash:
            return {"path": img_path, "error": "hash_failed"}

        try:
            date_str = get_exif_date_taken(file_stream) or "unknown_date"
        except ExifReadError:
            date_str = "exif_error"
        
        original_name = get_original_filename(img_path)
        
        return {
            "path": img_path,
            "hash": file_hash,
            "date_str": date_str,
            "original_name": original_name,
            "error": None
        }
    except FileNotFoundError:
        return {"path": img_path, "error": "file_not_found"}
    except (IOError, OSError) as e:
        return {"path": img_path, "error": f"read_error: {e}"}
    except Exception as e:
        return {"path": img_path, "error": f"unknown: {e}"}

def organize_photos_by_hash_logic(
    source_dir: Path,
    destination_dir: Path,
    allowed_extensions: Optional[set[str]] = None,
    quarantine_dir: Optional[Path] = None,
    dry_run: bool = True,
    action: str = 'copy',
    parallel: bool = False,
    max_workers: Optional[int] = None
    ):
    """
    해시값과 EXIF 날짜 정보를 기준으로 이미지 파일을 정리하는 핵심 로직 함수입니다.

    Parameters:
        source_dir (Path): 원본 이미지 파일들이 있는 디렉토리 경로
        destination_dir (Path): 정리된 이미지들이 저장될 대상 디렉토리
        quarantine_dir (Optional[Path]): 오류 파일을 이동할 격리 디렉토리.
        dry_run (bool): 실제 파일을 이동/복사하지 않고 경로만 시뮬레이션할지 여부 (기본값: False)
        action (str): 'move' 또는 'copy' 중 선택, 파일을 이동할지 복사할지 결정 (기본값: 'move')

    Returns:
        dict: 처리 상태를 담은 상태 딕셔너리 (스캔한 이미지 수, 오류 수, 생성된 하위 디렉토리 수 등)
    """
    # DecompressionBombError 방지를 위해 Pillow의 이미지 픽셀 수 제한을 해제합니다.
    Image.MAX_IMAGE_PIXELS = None
    status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)

    if not source_dir.is_dir():
        logger.error(f"소스 디렉토리를 찾을 수 없습니다: {source_dir}")
        return status

    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"대상 디렉토리: {destination_dir}")
    except OSError as e:
        logger.error(f"대상 디렉토리를 생성할 수 없습니다 ({destination_dir}): {e}")
        return status

    # --- Phase 1: 파일 목록 스캔 ---
    logger.info(f"'{source_dir}' 에서 처리할 이미지 파일 목록을 스캔합니다. 파일 수에 따라 시간이 걸릴 수 있습니다...")
    all_found_files, visual_width = scan_files_and_get_display_width(
        source_dir = source_dir,
        extensions = allowed_extensions,
        buffer_ratio = 0.25,
        min_width = 20,
        max_width = 50
    )

    image_files = []
    if quarantine_dir and source_dir in quarantine_dir.parents:
        resolved_quarantine_dir = quarantine_dir.resolve()
        image_files = [f for f in all_found_files if not str(f.resolve()).startswith(str(resolved_quarantine_dir))]
    else:
        image_files = all_found_files

    image_number = len(image_files)
    status["images_scanned"]["value"] = image_number
    logger.info(f"{source_dir} 에서 {image_number}개의 이미지 파일을 찾았습니다.")
    if not image_files:
        logger.info("처리할 이미지가 없습니다.")
        return status

    # --- Phase 2: 정보 수집 (병렬 또는 순차) ---
    logger.info("이미지 정보(해시, EXIF) 수집을 시작합니다...")
    results = []
    if parallel:
        # 병렬 처리
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # tqdm을 ProcessPoolExecutor와 함께 사용
            results_iterator = executor.map(_process_file_for_parallel, image_files)
            results = list(tqdm(results_iterator, total=image_number, desc="정보 수집 중 (병렬)", unit="파일"))
    else:
        # 순차 처리
        for img_path in tqdm(image_files, desc="정보 수집 중 (순차)", unit="파일"):
            results.append(_process_file_for_parallel(img_path))

    # --- Phase 3: 결과 분석 및 처리 대상 선별 ---
    successful_tasks = []
    for res in results:
        if res["error"]:
            error_type = res["error"]
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

    # --- Phase 4: 파일 작업 (순차 실행) ---
    logger.info(f"정보 수집 완료. {len(successful_tasks)}개의 파일에 대한 작업을 시작합니다...")
    created_dirs_in_this_run = set()
    
    description_func = create_dynamic_description_func(
        prefix="처리 중: ",
        width=visual_width
    )

    def task_function(task_data: dict, idx: int):
        img_path = task_data["path"]
        dest_file_path = destination_dir / task_data["hash"] / task_data["date_str"] / task_data["original_name"]
        _execute_file_action(img_path, dest_file_path, action, dry_run, status, created_dirs_in_this_run)

    try:
        with_progress_bar(
            items=successful_tasks,
            task_func=task_function,
            desc="파일 처리 중",
            unit="파일",
            description_func=lambda item: description_func(item['path'])
        )
    except DiskFullError as e:
        logger.critical(f"디스크 공간 부족으로 작업을 중단합니다. 오류: {e}")

    status["subdirectories_created"]["value"] = len(created_dirs_in_this_run)
    return status


if __name__ == "__main__":
    """
    스크립트의 메인 실행 함수입니다.
    """
    # -src와 -dst만 필수로 지정하고, -qrt(격리폴더) 등은 선택사항으로 변경합니다.
    # -qrt를 지정하지 않으면 소스 디렉토리 하위에 'quarantine' 폴더가 자동으로 사용됩니다.
    supported_args_for_script = [
        'source_dir', 
        'destination_dir', 
        'quarantine_dir', 
        'dry_run', 
        'action', 
        'parallel', 
        'max_workers'
    ]
    parsed_args = get_argument(
        required_args=['-src', '-dst', '-qrt'],
        supported_args=supported_args_for_script
    )
    script_name = Path(__file__).stem
    
    if hasattr(logger, "setup"):
        date_str = datetime.now().strftime("%y%m%d_%H%M")
        log_file_name = f"{script_name}_{date_str}.log"
        full_log_path = Path(parsed_args.log_dir) / log_file_name
        logger.setup(
            logger_path=full_log_path,
            console_min_level = "info",
            file_min_level=parsed_args.log_level.upper(),
            include_function_name=True,
            pretty_print=True
        )
    
    logger.info(f"애플리케이션 ({script_name}) 시작")

    try:
        config_manager = configger(root_dir=parsed_args.root_dir, config_path=parsed_args.config_path)
        logger.debug("Configger 초기화 완료.")
        # YAML에서 리스트를 가져와 소문자 set으로 변환
        extensions_from_config = config_manager.get_value("processing.supported_image_extensions", default=[])
        allowed_extensions = {ext.lower() for ext in extensions_from_config} if extensions_from_config else set()
    except Exception as e:
        logger.error(f"Configger 초기화 또는 설정 로드 중 오류 발생: {e}", exc_info=True)
        allowed_extensions = set() # 오류 발생 시 빈 세트로 초기화

    # --source-dir 또는 --target-dir을 소스 디렉토리로 사용
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
        logger.info(f"사용자 지정 격리 디렉토리 사용: '{quarantine_dir}'")
    else:
        # 기본값: 소스 디렉토리 아래 'quarantine' 폴더
        quarantine_dir = source_dir / 'quarantine'
        logger.info(f"기본 격리 디렉토리 사용: '{quarantine_dir}'")

    if not dry_run_mode: # 실제 실행 모드일 때만 디렉토리 생성
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    if source_dir == destination_dir:
        logger.error("소스 디렉토리와 대상 디렉토리는 같을 수 없습니다. 스크립트를 종료합니다.")
        sys.exit(1)

    try:
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

        logger.warning("--- 사진 해시 기반 정리 처리 통계 ---")
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
            logger.warning(f"  {msg}{padding} : {value:>{digit_width_stats}}")
        logger.warning("------------------------------------")

    except KeyboardInterrupt:
        logger.warning("\n사용자에 의해 작업이 중단되었습니다.")
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 최상위 오류 발생: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"애플리케이션 ({script_name}) 종료{ ' (Dry Run 모드)' if dry_run_mode else ''}")
        if hasattr(logger, "shutdown"):
            logger.shutdown()
