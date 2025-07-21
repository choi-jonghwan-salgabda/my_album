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
import uuid
import copy
import hashlib
from tqdm import tqdm # tqdm 임포트
from typing import Optional, Dict, List, Any 
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError

try:
    from my_utils.config_utils.file_utils import calculate_sha256, safe_move, safe_copy, DiskFullError, get_exif_date_taken, get_original_filename
    from my_utils.config_utils.arg_utils import get_argument
    from my_utils.config_utils.SimpleLogger import logger
    from my_utils.config_utils.display_utils import calc_digit_number, get_display_width, truncate_string, visual_length
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

# 지원하는 이미지 확장자 목록 (소문자로 통일)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic"}

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
    "move_errors":                  {"value": 0, "msg": "파일 처리 중 발생한 오류 수"},
    "subdirectories_created":       {"value": 0, "msg": "생성된 해시 하위 디렉토리 수"},
}

def _process_single_file(img_path: Path, status: dict) -> Optional[tuple]:
    """
    단일 이미지 파일의 유효성을 검사하고, 해시 및 EXIF 날짜를 추출합니다.

    Args:
        img_path (Path): 처리할 이미지 파일 경로.
        status (dict): 통계 업데이트를 위한 상태 딕셔너리.

    Returns:
        Optional[tuple]: (file_hash, date_str, original_name) 튜플. 처리 실패 시 None.
    """
    # 이미지 무결성 검사 (Pillow 사용)
    try:
        with Image.open(img_path) as img:
            img.verify()  # 이미지 헤더 및 데이터 구조 검사
    except (IOError, UnidentifiedImageError, SyntaxError) as e:
        logger.warning(f"  손상된 이미지 파일: '{img_path}'. 오류: {e}. 건너뜁니다.")
        status["corrupted_images_skipped"]["value"] += 1
        return None
    except DecompressionBombError as e:
        logger.warning(f"  이미지 크기 초과 (Decompression Bomb): '{img_path}'. 오류: {e}. 건너뜁니다.")
        status["corrupted_images_skipped"]["value"] += 1
        return None

    file_hash = calculate_sha256(img_path)
    if not file_hash:
        logger.warning(f"  해시 계산 실패 (I/O 오류 가능성): {img_path}. 건너뜁니다.")
        status["move_errors"]["value"] += 1
        return None
    status["hashes_calculated"]["value"] += 1

    # EXIF 날짜 읽기 및 통계 업데이트
    raw_date_str = get_exif_date_taken(img_path)
    if raw_date_str == "ERROR":
        status["exif_read_errors"]["value"] += 1
        date_str = "exif_error"
    elif raw_date_str:
        status["exif_dates_found"]["value"] += 1
        date_str = raw_date_str
    else:
        status["exif_dates_missing"]["value"] += 1
        date_str = "unknown_date"

    original_name = get_original_filename(img_path)
    return file_hash, date_str, original_name

def _execute_file_action(img_path: Path, dest_file_path: Path, action: str, dry_run: bool, status: dict, created_dirs: set):
    """
    계산된 정보를 바탕으로 실제 파일 이동, 복사, 삭제 또는 건너뛰기 작업을 수행합니다.
    """
    hash_subdir = dest_file_path.parent

    if dest_file_path.exists():
        if action == 'move':
            if dry_run:
                logger.info(f"(Dry Run) 중복 파일 삭제 예정 (소스): '{img_path}'")
            else:
                img_path.unlink()
                logger.info(f"  중복 파일 삭제 (소스): '{img_path}'")
            status["images_deleted_as_duplicate"]["value"] += 1
        elif action == 'copy':
            logger.info(f"  중복 파일 건너뛰기 (소스): '{img_path}'")
            status["images_skipped_as_duplicate"]["value"] += 1
    else:
        if not dry_run and not hash_subdir.exists():
            hash_subdir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(hash_subdir)
        elif dry_run and not hash_subdir.exists() and hash_subdir not in created_dirs:
            # Dry-run 모드에서도 통계 예측을 위해 생성될 디렉토리를 세트에 추가합니다.
            created_dirs.add(hash_subdir)

        log_prefix = "(Dry Run) " if dry_run else ""
        action_verb = "이동" if action == 'move' else "복사"
        log_msg = f"{log_prefix}{action_verb} 예정: '{img_path}' -> '{dest_file_path}'"

        # Dry-run 모드에서 새 디렉토리 생성을 로그에 명시적으로 표시합니다.
        if dry_run and not dest_file_path.parent.exists() and dest_file_path.parent in created_dirs:
            log_msg += " (새 디렉토리 생성 예정)"
        logger.info(log_msg)

        if not dry_run:
            (safe_move if action == 'move' else safe_copy)(str(img_path), str(dest_file_path))
        status["images_processed"]["value"] += 1

def organize_photos_by_hash_logic(source_dir: Path, destination_dir: Path, dry_run: bool = False, action: str = 'move'):
    """
    소스 디렉토리의 사진을 해시값에 따라 대상 디렉토리로 정리하는 핵심 로직입니다.
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

    image_files, visual_width = get_display_width(
        source_dir = source_dir,
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"},
        buffer_ratio = 0.25,
        min_width = 20,
        max_width = 50
    )

    image_number = len(image_files)

    status["images_scanned"]["value"] = image_number
    logger.info(f"{source_dir} 에서 {image_number}개의 이미지 파일을 찾았습니다.")

    if not image_files:
        logger.info("처리할 이미지가 없습니다.")
        return status

    created_dirs_in_this_run = set()
    digit_width = calc_digit_number(image_number)
    
    # tqdm 진행률 바와 로거 출력이 겹치지 않도록 처리
    if hasattr(logger, 'set_tqdm_aware'):
        logger.set_tqdm_aware(True)

    try: # 디스크 공간 부족 예외를 처리하기 위한 외부 try 블록
        # tqdm을 사용하여 진행률 바 추가
        with tqdm(total=image_number, desc="사진 정리 중", unit="파일", file=sys.stdout, dynamic_ncols=True) as pbar:
            for idx, img_path in enumerate(image_files):
                # 1. 파일 정보 처리 (유효성 검사, 해시, EXIF)
                process_result = _process_single_file(img_path, status)
                if not process_result:
                    pbar.update(1)
                    continue
                
                file_hash, date_str, original_name = process_result
                
                # 2. 대상 경로 결정
                dest_file_path = destination_dir / file_hash / date_str / original_name

                # 3. 파일 작업 실행
                try:
                    _execute_file_action(img_path, dest_file_path, action, dry_run, status, created_dirs_in_this_run)
                except Exception as e:
                    logger.error(f"  파일 처리 오류 ('{img_path}'): {e}", exc_info=True)
                    status["move_errors"]["value"] += 1
                
                # 진행률 바 업데이트 및 현재 처리 중인 파일 정보 표시
                safe_img_name = img_path.name.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8') # 안전한 문자열로 변환
                display_name = truncate_string(safe_img_name, visual_width) # 시각적 길이에 맞춰 자르기
                pbar.set_description_str(f"처리 중: {display_name.strip():<{visual_width}}")
                pbar.update(1)
    except DiskFullError as e:
        logger.critical(f"디스크 공간 부족으로 작업을 중단합니다. 오류: {e}")
    finally:
        # 작업 완료 후 로거를 원래 모드로 복원
        if hasattr(logger, 'set_tqdm_aware'):
            logger.set_tqdm_aware(False)

    status["subdirectories_created"]["value"] = len(created_dirs_in_this_run)
    return status

def run_main():
    """
    스크립트의 메인 실행 함수입니다.
    """
    parsed_args = get_argument()
    
    script_name = Path(__file__).stem
    
    if hasattr(logger, "setup"):
        date_str = datetime.now().strftime("%y%m%d_%H%M")
        log_file_name = f"{script_name}_{date_str}.log"
        full_log_path = Path(parsed_args.log_dir) / log_file_name
        logger.setup(
            logger_path=full_log_path,
            console_min_level = "warning",
            file_min_level=parsed_args.log_level.upper(),
            include_function_name=True,
            pretty_print=True
        )
    
    logger.info(f"애플리케이션 ({script_name}) 시작")
    logger.info(f"명령줄 인자: {vars(parsed_args)}")

    # --source-dir 또는 --target-dir을 소스 디렉토리로 사용
    input_dir_path = parsed_args.source_dir or parsed_args.target_dir

    if not input_dir_path:
        logger.error("정리할 사진이 있는 소스 디렉토리(--source-dir 또는 --target-dir)가 제공되지 않았습니다. 스크립트를 종료합니다.")
        sys.exit(1)
    if not parsed_args.destination_dir:
        logger.error("대상 디렉토리 (--destination-dir or -dst)가 제공되지 않았습니다. 스크립트를 종료합니다.")
        sys.exit(1)

    source_dir = Path(input_dir_path).expanduser().resolve()
    destination_dir = Path(parsed_args.destination_dir).expanduser().resolve()
    dry_run_mode = getattr(parsed_args, 'dry_run', False)
    action_mode = getattr(parsed_args, 'action', 'move') # 새로운 action 인자 가져오기

    if source_dir == destination_dir:
        logger.error("소스 디렉토리와 대상 디렉토리는 같을 수 없습니다. 스크립트를 종료합니다.")
        sys.exit(1)

    try:
        # organize_photos_by_hash_logic 함수에 action_mode 전달
        final_status = organize_photos_by_hash_logic(
            source_dir=source_dir,
            destination_dir=destination_dir,
            dry_run=dry_run_mode,
            action=action_mode
        )

        logger.warning("--- 사진 해시 기반 정리 처리 통계 ---")
        max_visual_msg_len = max(visual_length(v["msg"]) for v in DEFAULT_STATUS_TEMPLATE.values()) if DEFAULT_STATUS_TEMPLATE else 20
        max_val_for_width = max(s_item["value"] for s_item in final_status.values()) if final_status and any(final_status.values()) else 0
        digit_width_stats = calc_digit_number(max_val_for_width)
        
        for key, data in final_status.items():
            msg = DEFAULT_STATUS_TEMPLATE.get(key, {}).get("msg", key.replace("_", " ").capitalize())
            value = data["value"]
            padding_spaces = max(0, int(max_visual_msg_len - visual_length(msg)))
            logger.warning(f"{msg}{'-' * padding_spaces} : {value:>{digit_width_stats}}")
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

if __name__ == "__main__":
    run_main()
