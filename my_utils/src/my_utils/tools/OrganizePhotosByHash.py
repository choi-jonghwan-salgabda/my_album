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
from tqdm import tqdm # tqdm 임포트
from typing import Optional, Dict, List, Any

try:
    from my_utils.config_utils.display_utils import calc_digit_number, get_display_width, truncate_string, visual_length
    from my_utils.config_utils.arg_utils import get_argument
    from my_utils.config_utils.SimpleLogger import logger
    from my_utils.config_utils.file_utils import safe_move, safe_copy, DiskFullError
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

# 지원하는 이미지 확장자 목록 (소문자로 통일)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

# 처리 상태를 기록하기 위한 기본 템플릿
DEFAULT_STATUS_TEMPLATE = {
    "images_scanned":               {"value": 0, "msg": "소스 디렉토리에서 스캔된 이미지 수"},
    "hashes_calculated":            {"value": 0, "msg": "계산된 해시 수"},
    "images_processed":             {"value": 0, "msg": "이동 또는 복사된 이미지 수"},
    "images_deleted_as_duplicate":  {"value": 0, "msg": "중복으로 인해 삭제된 이미지 수 (이동 모드)"},
    "images_skipped_as_duplicate":  {"value": 0, "msg": "중복으로 인해 건너뛴 이미지 수 (복사 모드)"},
    "move_errors":                  {"value": 0, "msg": "파일 처리 중 발생한 오류 수"},
    "subdirectories_created":       {"value": 0, "msg": "생성된 해시 하위 디렉토리 수"},
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

def organize_photos_by_hash_logic(source_dir: Path, destination_dir: Path, dry_run: bool = False, action: str = 'move'):
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
                        # 정확한 중복 파일입니다. 'action'에 따라 처리합니다.
                        if action == 'move':
                            # '이동' 모드에서는 소스 파일이 불필요하므로 삭제합니다.
                            if dry_run:
                                logger.info(f"(Dry Run) 중복 파일 삭제 예정 (소스): '{img_path}' (대상에 동일 파일 존재)")
                            else:
                                img_path.unlink() # 파일 삭제
                                logger.info(f"  중복 파일 삭제 (소스): '{img_path}' (대상에 동일 파일 존재)")
                            status["images_deleted_as_duplicate"]["value"] += 1
                        elif action == 'copy':
                            # '복사' 모드에서는 소스 파일을 그대로 두고 작업을 건너뜁니다.
                            logger.info(f"  중복 파일 건너뛰기 (소스): '{img_path}' (대상에 동일 파일 존재)")
                            status["images_skipped_as_duplicate"]["value"] += 1
                    else:
                        # 중복이 아니므로 파일을 이동 또는 복사합니다.
                        # 이동/복사 전, 대상 해시 디렉토리가 없으면 생성합니다.
                        if not dry_run and not hash_subdir.exists():
                            hash_subdir.mkdir(parents=True, exist_ok=True)
                            created_dirs_in_this_run.add(hash_subdir)

                        if action == 'move':
                            if dry_run:
                                log_msg = f"(Dry Run) 이동 예정: '{img_path}' -> '{dest_file_path}'"
                                if not hash_subdir.exists() and hash_subdir not in created_dirs_in_this_run:
                                    log_msg += " (새 디렉토리 생성 예정)"
                                    created_dirs_in_this_run.add(hash_subdir) # Dry run 시에도 통계 예측을 위해 추가
                                logger.info(log_msg)
                            else:
                                safe_move(str(img_path), str(dest_file_path))
                                logger.info(f"  이동 완료: '{img_path.name}' -> '{dest_file_path}'")
                        elif action == 'copy':
                            if dry_run:
                                log_msg = f"(Dry Run) 복사 예정: '{img_path}' -> '{dest_file_path}'"
                                if not hash_subdir.exists() and hash_subdir not in created_dirs_in_this_run:
                                    log_msg += " (새 디렉토리 생성 예정)"
                                    created_dirs_in_this_run.add(hash_subdir) # Dry run 시에도 통계 예측을 위해 추가
                                logger.info(log_msg)
                            else:
                                safe_copy(str(img_path), str(dest_file_path)) # copy2는 메타데이터를 보존합니다.
                                logger.info(f"  복사 완료: '{img_path.name}' -> '{dest_file_path}'")
                        else:
                            logger.error(f"알 수 없는 작업: '{action}'. 파일 '{img_path}'를 처리하지 못했습니다.")
                            status["move_errors"]["value"] += 1

                        status["images_processed"]["value"] += 1

                except Exception as e:
                    logger.error(f"  파일 처리 오류 ('{img_path}'): {e}", exc_info=True)
                    status["move_errors"]["value"] += 1
                
                # 진행률 바 업데이트 및 현재 처리 중인 파일 정보 표시
                # 파일 이름이 너무 길 경우 잘라서 표시하여 줄이 깨지지 않도록 함
                # 유니코드 인코딩 오류 방지를 위해 파일 이름을 안전하게 처리
                safe_img_name = img_path.name.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8') # 안전한 문자열로 변환
                display_name = truncate_string(safe_img_name, visual_width) # 시각적 길이에 맞춰 자르기
                pbar.set_description_str(f"처리 중: {display_name:<{visual_width}}")
            #    pbar.set_description_str(f"{display_name}")
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