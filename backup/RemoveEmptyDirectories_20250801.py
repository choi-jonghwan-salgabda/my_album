"""
이 스크립트는 지정된 디렉토리와 그 안의 모든 하위 디렉토리를 검사하여
내용이 비어있는 폴더들을 모두 찾아 삭제하는 정리 도구입니다.

주요 기능 및 특징:
1.  안전한 상향식 삭제 (Bottom-up Deletion):
    가장 깊은 하위 폴더부터 검사를 시작하여 위로 올라오는 방식으로 작동합니다.
    폴더가 비어있으면 삭제하고, 그로 인해 상위 폴더가 비게 되면 상위 폴더도
    삭제 대상이 됩니다. 이 방식은 재귀 호출의 깊이 제한 없이 안정적으로
    동작하며, 여러 단계로 중첩된 빈 폴더들을 한 번에 정리할 수 있습니다.

2.  안전한 테스트 실행 (`--dry-run`):
    이 옵션을 사용하면 스크립트는 실제로는 아무것도 삭제하지 않고, 어떤
    폴더가 삭제될 예정인지만 로그로 보여줍니다. 중요한 데이터를 실수로
    삭제하는 것을 방지하기 위한 매우 유용한 기능입니다.

3.  최상위 폴더 삭제 옵션 (`--delete-top-if-empty`):
    기본적으로 스크립트는 사용자가 지정한 최상위 폴더는 비어있게 되더라도
    삭제하지 않습니다. 하지만 이 옵션을 주면, 정리 작업 후 최상위 폴더까지
    비게 될 경우 함께 삭제합니다.

4.  상세한 로그 및 통계:
    모든 작업 과정은 로그 파일에 기록됩니다. 실행이 끝나면, 총 몇 개의
    디렉토리를 검사했고, 몇 개를 삭제했으며, 오류는 몇 건이었는지 요약된 통계를
    보여주어 작업 결과를 쉽게 파악할 수 있습니다.

5.  무시 파일 처리:
    `.DS_Store`나 `Thumbs.db`와 같이 시스템이 생성하는 숨김 파일만 들어있는
    폴더도 '빈 폴더'로 간주하여 정리 대상으로 포함합니다.
"""
import sys
from pathlib import Path
import shutil
from datetime import datetime
import copy

from tqdm import tqdm

try:
    from my_utils.config_utils.SimpleLogger import logger
    from my_utils.config_utils.arg_utils import get_argument
    from my_utils.config_utils.file_utils import get_all_dirs
    from my_utils.config_utils.display_utils import calc_digit_number, visual_length, truncate_string
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

# 처리 상태를 기록하기 위한 기본 템플릿
DEFAULT_STATUS_TEMPLATE = {
    "directories_scanned": {"value": 0, "msg": "검사한 총 디렉토리 수"},
    "directories_deleted": {"value": 0, "msg": "삭제된 빈 디렉토리 수"},
    "deletion_errors": {"value": 0, "msg": "삭제 중 발생한 오류 수"},
}

def _remove_if_empty(
    dir_path: Path,
    status: dict,
    dry_run: bool,
    ignore_files: set
):
    """
    단일 디렉토리가 실질적으로 비어 있다면 삭제합니다.
    '실질적으로 비어있다'는 것은 폴더 안에 아무것도 없거나,
    무시 목록(ignore_files)에 포함된 파일만 있는 경우를 의미합니다.

    Args:
        dir_path (Path): 검사할 디렉토리 경로.
        status (dict): 통계 기록용 딕셔너리.
        dry_run (bool): True이면 실제 삭제를 수행하지 않습니다.
        ignore_files (set): 내용물 검사 시 무시할 파일명 집합.
    """
    if not dir_path.is_dir():
        return

    # 1. 디렉토리 내용물이 모두 무시 파일 목록에 있는지 확인하여 '실질적'으로 비었는지 판단
    is_effectively_empty = all(
        item.name in ignore_files for item in dir_path.iterdir()
    )

    if is_effectively_empty:
        # 무시 파일 삭제 시도
        if not dry_run:
            # 2. 실제 삭제 모드일 경우, 디렉토리 삭제(rmdir)가 가능하도록 무시 파일을 먼저 삭제합니다.
            for item in dir_path.iterdir():
                if item.name in ignore_files:
                    try:
                        item.unlink()
                        logger.debug(f"무시 목록 파일 삭제됨: '{item}'")
                    except OSError as e:
                        logger.error(f"무시 파일 삭제 실패: {item} → {e}")
                        status["deletion_errors"]["value"] += 1
                        return

        # 3. 무시 파일 삭제 후, 또는 dry_run 모드에서 디렉토리가 정말 비었는지 최종 확인
        is_now_empty = not any(dir_path.iterdir()) if not dry_run else True
        if not is_now_empty:
            return

        try:
            # 4. 디렉토리 삭제 실행 및 로그 기록
            if dry_run:
                logger.info(f"(Dry Run) 삭제 예정: '{dir_path}'")
            else:
                dir_path.rmdir()
                logger.info(f"삭제됨: '{dir_path}'")
            status["directories_deleted"]["value"] += 1
        except OSError as e:
            logger.error(f"디렉토리 삭제 실패: {dir_path} → {e}")
            status["deletion_errors"]["value"] += 1

def remove_empty_directories_logic(
    directory_path: Path,
    status: dict,
    dry_run: bool,
    ignore_files: set,
    delete_top_if_empty: bool = False
):
    """
    지정된 디렉토리 내의 모든 빈 하위 디렉토리를 삭제하는 핵심 로직입니다.
    재귀 호출 대신, 모든 하위 디렉토리 목록을 미리 만든 후 깊은 것부터 순서대로 처리하여
    안정성과 성능을 확보합니다.

    Args:
        directory_path (Path): 검사 및 삭제를 시작할 최상위 디렉토리 경로.
        status (dict): 처리 통계를 기록할 딕셔너리.
        dry_run (bool): True이면 실제 삭제 없이 로그만 남깁니다.
        ignore_files (set): 삭제 여부 판단 시 무시할 파일 이름의 집합.
        delete_top_if_empty (bool): True이고 최상위 directory_path도 비게 되면 삭제합니다.

    Returns:
        dict: 업데이트된 최종 통계 딕셔너리.
    """
    if not directory_path.is_dir():
        logger.error(f"'{directory_path}'는 유효한 디렉토리 아님")
        return status

    # 1. 모든 하위 디렉토리를 가져옵니다. get_all_dirs는 가장 깊은 디렉토리부터 순서대로 정렬하여 반환합니다.
    all_dirs = get_all_dirs(directory_path)
    status["directories_scanned"]["value"] = len(all_dirs)

    # 2. tqdm을 사용하여 진행 상황을 시각적으로 표시하며, 각 디렉토리를 처리합니다.
    with tqdm(total=len(all_dirs), desc="빈 디렉토리 검사 중", unit="폴더", dynamic_ncols=True, file=sys.stdout) as pbar:
        desc_width = 40 # 진행률 표시줄에 표시될 디렉토리 이름의 너비
        for sub_dir in all_dirs:
            pbar.set_description_str(f"검사 중: {truncate_string(sub_dir.name, desc_width):<{desc_width}}")
            _remove_if_empty(
                dir_path=sub_dir,
                status=status,
                dry_run=dry_run,
                ignore_files=ignore_files
            )
            pbar.update(1)

    # 3. 모든 하위 디렉토리 정리 후, 최상위 디렉토리도 비었는지 확인하고 옵션에 따라 삭제합니다.
    if delete_top_if_empty:
        status["directories_scanned"]["value"] += 1 # 최상위 디렉토리도 검사했으므로 카운트 추가
        _remove_if_empty(
            dir_path=directory_path,
            status=status,
            dry_run=dry_run,
            ignore_files=ignore_files
        )
    return status

# def remove_empty_directories_logic(
#     directory_path: Path,
#     status: dict,
#     dry_run: bool,
#     ignore_files: set,
#     delete_top_if_empty: bool = False
#     ):
#     """
#     지정된 디렉토리 내의 모든 빈 하위 디렉토리를 반복적으로 삭제하는 핵심 로직입니다.
#     가장 깊은 디렉토리부터 처리하기 위해 모든 하위 디렉토리를 스캔하고 정렬한 후 작업을 수행합니다.

#     Args:
#         directory_path (Path): 검사 및 삭제를 시작할 디렉토리 경로.
#         status (dict): 처리 통계를 기록할 딕셔너리.
#         dry_run (bool): True이면 실제 삭제 없이 로그만 남깁니다.
#         ignore_files (set): 삭제 여부 판단 시 무시할 파일 이름의 집합.
#         delete_top_if_empty (bool): True이고 최상위 directory_path도 비게 되면 삭제합니다.

#     Returns:
#         dict: 업데이트된 최종 통계 딕셔너리.
#     """

#     def _recursive_remover(current_path: Path, is_top_level: bool, ignore_files: set):
#         """재귀적으로 디렉토리를 탐색하고 실질적으로 비어있으면 삭제하는 헬퍼 함수."""
#         if not current_path.is_dir():
#             return

#         status["directories_scanned"]["value"] += 1
        
#         # 하위 디렉토리부터 재귀적으로 처리 (깊이 우선 탐색의 후위 순회 방식)
#         for item in list(current_path.iterdir()):
#             if item.is_dir():
#                 _recursive_remover(item, is_top_level=False, ignore_files=ignore_files)

#         # 현재 디렉토리가 실질적으로 비었는지 확인 (무시할 파일을 제외하고 내용이 없는지)
#         is_effectively_empty = all(item.name in ignore_files for item in current_path.iterdir())

#         if is_effectively_empty:
#             # 무시 목록에 있는 파일들을 먼저 삭제 시도
#             if not dry_run:
#                 for item in list(current_path.iterdir()):
#                     if item.name in ignore_files:
#                         try:
#                             item.unlink()
#                             logger.debug(f"무시 목록 파일 삭제됨: '{item}'")
#                         except OSError as e:
#                             logger.error(f"무시 목록 파일 '{item}' 삭제 중 오류 발생: {e}. 이 디렉토리는 삭제할 수 없습니다.")
#                             status["deletion_errors"]["value"] += 1
#                             return # 파일 삭제 실패 시 디렉토리 삭제를 진행하지 않음

#             # 이제 디렉토리가 완전히 비었는지 최종 확인
#             is_truly_empty = not any(current_path.iterdir()) if not dry_run else True
#             if not is_truly_empty: return

#             # 최상위 디렉토리인 경우, delete_top_if_empty 플래그에 따라 삭제 여부 결정
#             if is_top_level and not delete_top_if_empty:
#                 logger.info(f"최상위 디렉토리 '{current_path}'는 비어있지만, --delete-top-if-empty 옵션이 없어 삭제하지 않습니다.")
#                 return

#             try:
#                 if dry_run:
#                     logger.info(f"(Dry Run) 실질적으로 빈 디렉토리 삭제 예정: '{current_path}'")
#                 else:
#                     current_path.rmdir()
#                     logger.info(f"실질적으로 빈 디렉토리 삭제됨: '{current_path}'")
#                 status["directories_deleted"]["value"] += 1
#             except OSError as e:
#                 logger.error(f"디렉토리 '{current_path}' 삭제 중 오류 발생: {e}")
#                 status["deletion_errors"]["value"] += 1

#     if not directory_path.is_dir():
#         logger.error(f"오류: '{directory_path}'는 유효한 디렉토리가 아닙니다.")
#         return status

#     # 진행 표시줄 추가
#     all_dirs = get_all_dirs(directory_path)

#     for sub_dir in tqdm(all_dirs, desc="빈 디렉토리 검사/삭제 중", unit="dir"):
#         _recursive_remover(sub_dir, is_top_level=False, ignore_files=ignore_files)

#     # 마지막으로 최상위 디렉토리 처리 (최상위는 tqdm에 포함하지 않음)
#     _recursive_remover(directory_path, is_top_level=True, ignore_files=ignore_files)
#     return status

if __name__ == '__main__':
# def run_main():
    """
    스크립트의 메인 실행 함수입니다.
    명령줄 인자 파싱, 로깅 설정, 핵심 로직 호출 및 최종 통계 출력을 담당합니다.
    """
    parsed_args = get_argument(required_args=['-tgt'])
    script_name = Path(__file__).stem

    if hasattr(logger, "setup"):
        date_str = datetime.now().strftime("%y%m%d_%H%M")
        log_file_name = f"{script_name}_{date_str}.log"
        full_log_path = Path(parsed_args.log_dir) / log_file_name
        logger.setup(
            logger_path=full_log_path,
            file_min_level=parsed_args.log_level.upper(),
            include_function_name=True,
            pretty_print=True
        )
    
    logger.info(f"애플리케이션 ({script_name}) 시작")
    logger.info(f"명령줄 인자: {vars(parsed_args)}")

    # 명령줄 인자로부터 경로 및 옵션 설정
    target_dir_str = parsed_args.target_dir # arg_utils에서 필수로 검증되었으므로 None이 아님
    target_dir = Path(target_dir_str).expanduser().resolve()
    dry_run_mode = getattr(parsed_args, 'dry_run', False)
    delete_top_mode = getattr(parsed_args, 'delete_top_if_empty', False)

    # 향후 --ignore-files 인자로 받아오도록 확장 가능
    ignore_files_set = {'.DS_Store', 'Thumbs.db'}

    try:
        # 핵심 로직 함수 호출
        status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)
        final_status = remove_empty_directories_logic(
            directory_path=target_dir,
            status=status,
            dry_run=dry_run_mode,
            ignore_files=ignore_files_set,
            delete_top_if_empty=delete_top_mode
        )

        logger.warning("--- 빈 디렉토리 정리 통계 ---")
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

# if __name__ == '__main__':
#     run_main()
