# MoveOriginalPhotos.py
"""
이 스크립트는 해시값에 따라 정리된 디렉토리 구조에서 각 중복 그룹별로
'가장 원본에 가까운' 대표 파일 하나를 선정하여 지정된 목적지 디렉토리로
이동하거나 복사하는 도구입니다.

'OrganizePhotosByHashExif.py'와 같은 스크립트로 1차 정리된 디렉토리를
후처리하는 데 사용될 수 있습니다.

주요 기능:
1.  대표 파일 선정:
    각 해시(중복 그룹) 디렉토리 내의 모든 이미지 파일들을 대상으로,
    다음과 같은 기준으로 가장 원본일 가능성이 높은 파일을 지능적으로 선정합니다.
    - 먼저, 'photo (1).jpg' -> 'photo.jpg'와 같이 파일명을 정제합니다.
    - 1순위: 파일 생성 시간 (오래될수록 좋음)
    - 2순위: 정제된 파일명 내 특수문자('_', '-') 개수 (적을수록 좋음)
    - 3순위: 정제된 파일명 내 숫자 개수 (적을수록 좋음)
    - 4순위: 정제된 파일명 길이 (짧을수록 좋음)

2.  체계적인 정리:
    선정된 대표 파일은 `[대상 디렉토리]/[해시값]/[원본 파일명]` 구조로 이동/복사됩니다.
    이를 통해 해시값으로 그룹화된 상태를 유지하면서 각 그룹의 대표 파일만 깔끔하게
    모을 수 있습니다.

3.  유연한 파일 처리 (`--action`):
    - `move` (기본값): 대표 파일을 대상 디렉토리로 이동시킵니다.
    - `copy`: 원본 파일을 유지한 채 대상 디렉토리로 복사합니다.

4.  안전한 테스트 실행 (`--dry-run`):
    이 옵션을 사용하면 실제 파일 이동이나 복사 없이 어떤 작업이 수행될지
    미리 로그로 확인할 수 있어 안전합니다.

5.  상세한 통계 및 로그:
    작업 완료 후, 총 몇 개의 해시 그룹을 처리했고, 몇 개의 파일을 이동/복사했으며,
    몇 개를 건너뛰었는지 상세한 통계를 제공합니다. 모든 과정은 로그 파일에 기록됩니다.

사용법 예시:
    # 해시로 정리된 사진들 중에서 원본만 골라 다른 폴더로 이동
    python MoveOriginalPhotos.py --source-dir /path/to/hashed_photos --destination-dir /path/to/original_only --action move

    # 원본은 유지한 채, 대표 파일만 복사 (테스트 실행)
    python MoveOriginalPhotos.py --source-dir /path/to/hashed_photos --destination-dir /path/to/original_copies --action copy --dry-run
"""

import sys
import copy
from pathlib import Path
from datetime import datetime
from typing import Optional

from tqdm import tqdm

# 프로젝트 공용 유틸리티 임포트
try:
    from my_utils.config_utils.arg_utils import get_argument
    from my_utils.config_utils.SimpleLogger import logger
    from my_utils.config_utils.configger import configger
    from my_utils.config_utils.file_utils import safe_move, safe_copy, DiskFullError, get_original_filename, get_unique_path
    from my_utils.config_utils.display_utils import calc_digit_number, visual_length, truncate_string
except ImportError as e:
    # logger가 설정되기 전이므로 print 사용
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

# 처리 상태를 기록하기 위한 기본 템플릿
DEFAULT_STATUS_TEMPLATE = {
    "hash_dirs_scanned":      {"value": 0, "msg": "검사한 해시 디렉토리 수"},
    "files_processed":        {"value": 0, "msg": "선별되어 이동/복사된 대표 파일 수"},
    "files_archived":         {"value": 0, "msg": "보관 처리된 나머지 중복 파일 수"},
    "empty_hash_dirs_skipped":{"value": 0, "msg": "이미지 파일이 없어 건너뛴 해시 디렉토리 수"},
    "file_op_errors":         {"value": 0, "msg": "파일 처리 중 발생한 오류 수"},
}

def select_and_process_originals_logic(source_dir: Path, destination_dir: Path, archive_dir: Optional[Path], allowed_extensions: set[str], action: str = 'move', dry_run: bool = False):
    """
    소스 디렉토리의 각 해시 하위 디렉토리에서 대표 원본 파일을 선택하여
    대상 디렉토리로 이동/복사하는 핵심 로직.

    Args:
        source_dir (Path): 해시값으로 정리된 사진들이 있는 소스 디렉토리.
        destination_dir (Path): 대표 파일들이 저장될 대상 디렉토리.
        archive_dir (Optional[Path]): 대표 파일이 아닌 나머지 파일들을 이동시킬 보관(archive) 디렉토리.
        allowed_extensions (set[str]): 처리할 이미지 파일 확장자 집합.
        action (str): 'move' 또는 'copy'.
        dry_run (bool): True이면 실제 파일 작업을 수행하지 않음.

    Returns:
        dict: 처리 통계 딕셔너리.
    """
    status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)
    hash_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    status["hash_dirs_scanned"]["value"] = len(hash_dirs)

    if not hash_dirs:
        logger.info("처리할 해시 디렉토리가 없습니다.")
        return status

    if hasattr(logger, 'set_tqdm_aware'):
        logger.set_tqdm_aware(True)

    # 진행률 표시줄의 디렉토리명 길이를 동적으로 계산합니다.
    # 평균 이름 길이를 계산하여 너무 길거나 짧지 않게 조절합니다.
    dir_name_lengths = [visual_length(d.name) for d in hash_dirs]
    avg_name_len = sum(dir_name_lengths) / len(dir_name_lengths) if dir_name_lengths else 30
    desc_width = max(20, min(int(avg_name_len * 1.2), 40))

    # 파일 처리 함수와 로그용 동사를 루프 전에 한 번만 결정
    action_func = safe_move if action == 'move' else safe_copy
    action_verb = "이동" if action == 'move' else "복사"
    log_prefix = "(Dry Run) " if dry_run else ""
    try:
        # file=sys.stdout 제거, 터미널 크기 변경에 대응하는 dynamic_ncols=True 추가
        with tqdm(total=len(hash_dirs), desc="대표 원본 정리 중", unit="폴더", dynamic_ncols=True) as pbar:
            for hash_dir in hash_dirs:
                pbar.set_description_str(f"처리 중: {truncate_string(hash_dir.name, desc_width):<{desc_width}}")

                # 1. 각 해시 디렉토리 내의 모든 이미지 파일 수집
                image_files = [
                    f for f in hash_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in allowed_extensions
                ]

                if not image_files:
                    logger.debug(f"'{hash_dir.name}' 디렉토리에 이미지 파일이 없어 건너뜁니다.")
                    status["empty_hash_dirs_skipped"]["value"] += 1
                    pbar.update(1)
                    continue

                # 2. '가장 원본'일 가능성이 높은 파일 선택
                def get_file_score(p: Path):
                    # 점수가 낮을수록 좋음 (낮은 점수가 높은 우선순위)

                    # 0순위: 파일명이 '$'로 시작하는지 여부. 시작하면 가장 낮은 우선순위 (높은 점수).
                    is_priority_file = 1 if p.name.startswith('$') else 0

                    # 1. get_original_filename을 적용하여 정리된 파일명 생성
                    cleaned_name = get_original_filename(p)
                    cleaned_stem = Path(cleaned_name).stem
                    cleaned_name_lower = cleaned_name.lower()

                    # 2. 우선순위에 따라 점수 튜플 반환
                    return (
                        is_priority_file,  # 0순위: '$'로 시작하지 않는 파일이 우선
                        p.stat().st_ctime,  # 1순위: 생성 시간이 오래될수록 좋음 (오름차순 정렬)
                        sum(c in ('_', '-') for c in cleaned_name_lower), # 2순위: '_' 또는 '-' 문자가 적을수록 좋음
                        sum(c.isdigit() for c in cleaned_stem),  # 3순위: 숫자가 적을수록 좋음
                        len(cleaned_name),  # 4순위: 정리된 이름이 짧을수록 좋음
                    )

                best_file = min(image_files, key=get_file_score)
                logger.debug(f"'{hash_dir.name}' 그룹의 대표 파일로 '{best_file.name}' 선택됨.")

                # 3. 대상 경로 생성 및 파일 작업 수행
                dest_subdir = destination_dir / hash_dir.name
                dest_file_path = dest_subdir / best_file.name

                if not dry_run:
                    dest_subdir.mkdir(parents=True, exist_ok=True)

                try:
                    if not dest_file_path.exists():
                        logger.info(f"{log_prefix}{action_verb} 예정: '{best_file}' -> '{dest_file_path}'")
                        if not dry_run:
                            action_func(str(best_file), str(dest_file_path))
                        status["files_processed"]["value"] += 1
                    else:
                        logger.info(f"건너뛰기: 대상 파일이 이미 존재합니다. '{dest_file_path}'")
                        # 이미 존재하는 경우도 처리된 것으로 간주할 수 있으나, 여기서는 명확히 구분
                except (DiskFullError, OSError, Exception) as e:
                    logger.error(f"파일 {action_verb} 실패: '{best_file}' -> '{dest_file_path}'. 오류: {e}")
                    status["file_op_errors"]["value"] += 1

                # 4. 나머지 파일들을 보관 디렉토리로 이동/복사 (archive_dir가 지정된 경우에만)
                if archive_dir:
                    remaining_files = [f for f in image_files if f != best_file]
                    for file_to_quarantine in remaining_files:
                        try:
                            # 보관 폴더에 동일한 이름의 파일이 있을 경우 덮어쓰지 않고 버전 꼬리표를 답니다.
                            archive_path = get_unique_path(archive_dir, file_to_quarantine.name)
                            
                            logger.info(f"{log_prefix}나머지 파일 보관({action_verb}) 예정: '{file_to_quarantine}' -> '{archive_path}'")
                            
                            if not dry_run:
                                action_func(str(file_to_quarantine), str(archive_path))
                            status["files_archived"]["value"] += 1
                        except (DiskFullError, OSError, Exception) as e:
                            logger.error(f"파일 보관 실패: '{file_to_quarantine}' -> '{archive_dir}'. 오류: {e}")
                            status["file_op_errors"]["value"] += 1
                pbar.update(1)

    except (KeyboardInterrupt, Exception) as e:
        # tqdm 루프 밖에서 발생하는 예외 처리
        if isinstance(e, KeyboardInterrupt):
            logger.warning("\n사용자에 의해 작업이 중단되었습니다.")
        else:
            logger.error(f"처리 중 예기치 않은 오류 발생: {e}", exc_info=True)
    finally:
        if hasattr(logger, 'set_tqdm_aware'):
            logger.set_tqdm_aware(False)

    return status

if __name__ == "__main__":
# def run_main():
    """스크립트의 메인 실행 함수."""
    # 이 스크립트는 source-dir과 destination-dir을 필수로 요구합니다.
    parsed_args = get_argument(required_args=['-src', '-dst'])

    script_name = Path(__file__).stem
    if hasattr(logger, 'setup'):
        date_str = datetime.now().strftime("%y%m%d_%H%M")
        log_file_name = f"{script_name}_{date_str}.log"
        full_log_path = Path(parsed_args.log_dir) / log_file_name
        logger.setup(
            logger_path=full_log_path,
            console_min_level="warning", # 콘솔에는 경고 이상만 표시하여 출력 줄임
            file_min_level=parsed_args.log_level.upper(),
            include_function_name=True,
            pretty_print=True
        )

    logger.info(f"애플리케이션 ({script_name}) 시작")
    logger.info(f"명령줄 인자: {vars(parsed_args)}")

    # 설정 파일에서 허용 확장자 로드
    allowed_extensions = set()
    try:
        config_manager = configger(root_dir=parsed_args.root_dir, config_path=parsed_args.config_path)
        logger.info("Configger 초기화 완료.")
        extensions_from_config = config_manager.get_value("processing.supported_image_extensions", default=[])
        if extensions_from_config:
            allowed_extensions = {ext.lower() for ext in extensions_from_config}
    except Exception as e:
        logger.error(f"Configger 초기화 또는 설정 로드 중 오류 발생: {e}", exc_info=True)

    if not allowed_extensions:
        logger.warning("설정 파일에서 허용 확장자를 찾지 못했거나 로드 중 오류가 발생하여, 기본 이미지 확장자를 사용합니다.")
        allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic"}

    source_dir = Path(parsed_args.source_dir).expanduser().resolve()
    destination_dir = Path(parsed_args.destination_dir).expanduser().resolve()
    action_mode = getattr(parsed_args, 'action', 'move')
    dry_run_mode = getattr(parsed_args, 'dry_run', False)
    
    # 나머지 중복 파일을 보관할 디렉토리 경로 결정 (--quarantine-dir 인자 사용)
    quarantine_dir = None
    if parsed_args.quarantine_dir:
        # 사용자가 명시적으로 경로를 지정한 경우
        quarantine_dir = Path(parsed_args.quarantine_dir).expanduser().resolve()
        logger.info(f"나머지 중복 파일은 다음 보관 디렉토리로 처리됩니다: '{quarantine_dir}'")

    # 입력 경로 유효성 검사
    if not source_dir.is_dir():
        logger.error(f"소스 디렉토리를 찾을 수 없습니다: '{source_dir}'")
        sys.exit(1)
    if source_dir == destination_dir:
        logger.error("소스 디렉토리와 대상 디렉토리는 같을 수 없습니다.")
        sys.exit(1)

    try:
        # 대상 및 보관 디렉토리 생성 (dry_run이 아닐 때만)
        if not dry_run_mode:
            destination_dir.mkdir(parents=True, exist_ok=True)
            if quarantine_dir:
                quarantine_dir.mkdir(parents=True, exist_ok=True)

        final_status = select_and_process_originals_logic(
            source_dir=source_dir,
            destination_dir=destination_dir,
            archive_dir=quarantine_dir, # 'archive_dir' 인자로 전달
            allowed_extensions=allowed_extensions,
            action=action_mode,
            dry_run=dry_run_mode
        )

        # 최종 통계 출력
        logger.warning("--- 대표 원본 사진 정리 통계 ---")
        max_visual_msg_len = max(visual_length(v["msg"]) for v in DEFAULT_STATUS_TEMPLATE.values()) if DEFAULT_STATUS_TEMPLATE else 20
        max_val_for_width = max((s_item["value"] for s_item in final_status.values()), default=0)
        digit_width_stats = calc_digit_number(max_val_for_width)

        for key, data in final_status.items():
            msg = DEFAULT_STATUS_TEMPLATE.get(key, {}).get("msg", key.replace("_", " ").capitalize())
            value = data["value"]
            padding = max(0, int(max_visual_msg_len - visual_length(msg)))
            logger.warning(f"{msg}{'-' * padding} : {value:>{digit_width_stats}}")
        logger.warning("------------------------------------")

    except KeyboardInterrupt:
        # select_and_process_originals_logic 내부에서 처리되지만, 만약을 위해 여기에도 둠
        logger.warning("\n사용자에 의해 작업이 중단되었습니다.")
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 최상위 오류 발생: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"애플리케이션 ({script_name}) 종료{ ' (Dry Run 모드)' if dry_run_mode else ''}")
        if hasattr(logger, "shutdown"):
            logger.shutdown()

# if __name__ == "__main__":
#     run_main()
