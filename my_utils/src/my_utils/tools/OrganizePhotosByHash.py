# OrganizePhotosByHash.py
"""
이 스크립트는 지정된 소스 디렉토리(source_dir)의 모든 이미지 파일을 스캔하고,
각 파일의 내용에 기반한 SHA256 해시 값을 계산합니다.

계산된 해시 값을 이름으로 하는 하위 디렉토리를 대상 디렉토리(destination_dir)에 생성한 후,
원본 이미지 파일을 해당 하위 디렉토리로 이동시켜 정리합니다.

주요 동작:
1. 명령줄 인자로 소스 디렉토리와 대상 디렉토리 경로를 입력받습니다.
2. 소스 디렉토리 내의 모든 이미지 파일에 대해 SHA256 해시를 계산합니다.
3. 대상 디렉토리 아래에 '해시값' 이름의 하위 디렉토리를 생성합니다.
4. 원본 이미지 파일을 새로 생성된 해시 디렉토리로 이동시킵니다.
5. --dry-run 옵션을 통해 실제 파일 이동 없이 실행 결과를 미리 확인할 수 있습니다.
"""

import sys
from pathlib import Path
import shutil
import argparse
from datetime import datetime
import uuid
import copy
import hashlib
from typing import Optional, Dict, List, Any

try:
    from my_utils.config_utils.SimpleLogger import logger, calc_digit_number, get_argument, visual_length
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

# 지원하는 이미지 확장자 목록 (소문자로 통일)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

# 처리 상태를 기록하기 위한 기본 템플릿
DEFAULT_STATUS_TEMPLATE = {
    "images_scanned": {"value": 0, "msg": "소스 디렉토리에서 스캔된 이미지 수"},
    "hashes_calculated": {"value": 0, "msg": "계산된 해시 수"},
    "images_moved": {"value": 0, "msg": "대상 디렉토리로 이동된 이미지 수"},
    "images_deleted_as_duplicate": {"value": 0, "msg": "중복으로 인해 소스에서 삭제된 이미지 수"},
    "move_errors": {"value": 0, "msg": "파일 이동 중 발생한 오류 수"},
    "subdirectories_created": {"value": 0, "msg": "생성된 해시 하위 디렉토리 수"},
}

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

def organize_photos_by_hash_logic(source_dir: Path, destination_dir: Path, dry_run: bool = False):
    """
    소스 디렉토리의 사진을 해시값에 따라 대상 디렉토리로 정리하는 핵심 로직입니다.
    """
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

    image_files = [p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
    status["images_scanned"]["value"] = len(image_files)
    logger.info(f"{source_dir} 에서 {len(image_files)}개의 이미지 파일을 찾았습니다.")

    if not image_files:
        logger.info("처리할 이미지가 없습니다.")
        return status

    created_dirs_in_this_run = set()
    digit_width = calc_digit_number(len(image_files))

    for idx, img_path in enumerate(image_files):
        logger.debug(f"[{idx+1:{digit_width}}/{len(image_files)}] 처리 중: {img_path}")

        file_hash = calculate_sha256(img_path)
        if not file_hash:
            logger.warning(f"  해시 계산 실패: {img_path}. 건너뜁니다.")
            status["move_errors"]["value"] += 1
            continue

        status["hashes_calculated"]["value"] += 1

        hash_subdir = destination_dir / file_hash

        dest_file_path = hash_subdir / img_path.name

        try:
            # 대상 경로에 동일한 해시와 파일명을 가진 파일이 이미 있는지 확인합니다.
            if dest_file_path.exists():
                # 정확한 중복 파일이므로, 소스 파일을 삭제합니다.
                if dry_run:
                    logger.info(f"(Dry Run) 중복 파일 삭제 예정 (소스): '{img_path}' (대상에 동일 파일 존재)")
                else:
                    img_path.unlink() # 파일 삭제
                    logger.info(f"  중복 파일 삭제 (소스): '{img_path}' (대상에 동일 파일 존재)")
                status["images_deleted_as_duplicate"]["value"] += 1
            else:
                # 중복이 아니므로 파일을 이동합니다.
                # 이동 전, 대상 해시 디렉토리가 없으면 생성합니다.
                if not dry_run and not hash_subdir.exists():
                    hash_subdir.mkdir(parents=True, exist_ok=True)
                    created_dirs_in_this_run.add(hash_subdir)

                if dry_run:
                    log_msg = f"(Dry Run) 이동 예정: '{img_path}' -> '{dest_file_path}'"
                    if not hash_subdir.exists() and hash_subdir not in created_dirs_in_this_run:
                        log_msg += " (새 디렉토리 생성 예정)"
                        created_dirs_in_this_run.add(hash_subdir) # Dry run 시에도 통계 예측을 위해 추가
                    logger.info(log_msg)
                else:
                    shutil.move(str(img_path), str(dest_file_path))
                    logger.info(f"  이동 완료: '{img_path.name}' -> '{dest_file_path}'")

                status["images_moved"]["value"] += 1

        except Exception as e:
            logger.error(f"  파일 처리 오류 ('{img_path}'): {e}", exc_info=True)
            status["move_errors"]["value"] += 1

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
            min_level=parsed_args.log_level.upper(),
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

    if source_dir == destination_dir:
        logger.error("소스 디렉토리와 대상 디렉토리는 같을 수 없습니다. 스크립트를 종료합니다.")
        sys.exit(1)

    try:
        final_status = organize_photos_by_hash_logic(
            source_dir=source_dir,
            destination_dir=destination_dir,
            dry_run=dry_run_mode
        )

        logger.warning("--- 사진 해시 기반 정리 처리 통계 ---")
        max_visual_msg_len = max(visual_length(v["msg"]) for v in DEFAULT_STATUS_TEMPLATE.values()) if DEFAULT_STATUS_TEMPLATE else 20
        max_val_for_width = max(s_item["value"] for s_item in final_status.values()) if final_status else 0
        digit_width_stats = calc_digit_number(max_val_for_width) if "calc_digit_number" in globals() and callable(calc_digit_number) else 5
        
        for key, data in final_status.items():
            msg = DEFAULT_STATUS_TEMPLATE.get(key, {}).get("msg", key.replace("_", " ").capitalize())
            value = data["value"]
            padding_spaces = max(0, max_visual_msg_len - visual_length(msg))
            logger.warning(f"{msg}{'-' * padding_spaces} : {value:>{digit_width_stats}}")
        logger.warning("------------------------------------")

    except Exception as e:
        logger.error(f"애플리케이션 실행 중 최상위 오류 발생: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"애플리케이션 ({script_name}) 종료{ ' (Dry Run 모드)' if dry_run_mode else ''}")
        if hasattr(logger, "shutdown"):
            logger.shutdown()

if __name__ == "__main__":
    run_main()