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
    from my_utils.config_utils.configger import configger
    from my_utils.config_utils.file_utils import get_all_dirs
    from my_utils.config_utils.display_utils import calc_digit_number, visual_length, truncate_string, with_progress_bar
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
    I/O 호출을 최소화하기 위해 디렉토리 내용물은 한 번만 읽습니다.

    Args:
        dir_path (Path): 검사할 디렉토리 경로.
        status (dict): 통계 기록용 딕셔너리.
        dry_run (bool): True이면 실제 삭제를 수행하지 않습니다.
        ignore_files (set): 내용물 검사 시 무시할 파일명 집합.
    """
    if not dir_path.is_dir():
        return

    # 1. 디렉토리 내용물을 한 번만 읽어와 리스트에 저장 (I/O 최소화)
    try:
        items = list(dir_path.iterdir())
    except OSError as e:
        logger.error(f"디렉토리 '{dir_path}'의 내용을 읽는 중 오류 발생: {e}")
        status["deletion_errors"]["value"] += 1
        return

    # 2. 디렉토리가 비어있거나(not items), 내용물이 모두 무시 대상 파일인 경우 '실질적으로 비어있다'고 판단합니다.
    is_effectively_empty = not items or all(
        item.is_file() and item.name in ignore_files for item in items
    )

    if is_effectively_empty:
        # 3. 무시 파일 삭제 (dry_run이 아닐 경우)
        if not dry_run:
            for item in items:
                # is_effectively_empty가 True이므로, 모든 item은 무시 대상 파일임이 보장됨
                try:
                    item.unlink()
                    logger.debug(f"무시 목록 파일 삭제됨: '{item}'")
                except OSError as e:
                    logger.error(f"무시 파일 삭제 실패: {item} → {e}")
                    status["deletion_errors"]["value"] += 1
                    # 파일 하나라도 삭제 실패 시, 디렉토리 삭제를 진행하면 안 되므로 여기서 중단
                    return

        try:
            # 4. 디렉토리 삭제 실행 및 로그 기록
            if dry_run:
                # Dry run 모드에서는 실제 파일이 삭제되지 않았으므로,
                # is_effectively_empty가 True라는 사실만으로 삭제 예정 로그를 남길 수 있음
                logger.info(f"(Dry Run) 삭제 예정: '{dir_path}'")
            else:
                # 실제 삭제 모드에서는 무시 파일들이 모두 삭제되었으므로,
                # 이제 디렉토리가 비어있음을 확신하고 삭제할 수 있습니다.
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

    # 2. 각 디렉토리를 처리합니다. with_progress_bar를 사용하여 진행 상황을 표시합니다.
    dir_name_lengths = [visual_length(d.name) for d in all_dirs]
    avg_name_len = sum(dir_name_lengths) / len(dir_name_lengths) if dir_name_lengths else 30
    desc_width = max(20, min(int(avg_name_len * 1.2), 40))

    def task_function(sub_dir: Path, idx: int):
        _remove_if_empty(
            dir_path=sub_dir,
            status=status,
            dry_run=dry_run,
            ignore_files=ignore_files
        )

    def description_function(sub_dir: Path) -> str:
        return f"검사 중: {truncate_string(sub_dir.name, desc_width):<{desc_width}}"

    with_progress_bar(
        items=all_dirs,
        task_func=task_function,
        desc="빈 디렉토리 검사 중",
        unit="폴더",
        description_func=description_function
    )

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

if __name__ == "__main__":
    """
    스크립트의 메인 실행 함수입니다.
    명령줄 인자 파싱, 로깅 설정, 핵심 로직 호출 및 최종 통계 출력을 담당합니다.
    """
    script_name = Path(__file__).stem

    # 1단계: 콘솔 전용 임시 로거 설정 (설정 파일/인자 로드 전)
    logger.setup(
        logger_path=None,
        console_min_level="CONSOLE",
        include_function_name=False
    )
    logger.console(f"애플리케이션 ({script_name}) 시작. (로거 1/2 단계 초기화)")

    # 이 스크립트가 지원하는 인자 목록을 명시적으로 정의합니다.
    supported_args_for_script = [
        'target_dir',
        'log_mode',
        'dry_run',
        'delete_top_if_empty'
    ]
    parsed_args = get_argument(
        required_args=['-tgt'],
        supported_args=supported_args_for_script
    )

    # --- 설정 파일 경로 결정 ---
    if parsed_args.config_path is None:
        parsed_args.config_path = '../config/photo_album.yaml'
        logger.debug(f"--config-path가 지정되지 않아 기본 상대 경로를 사용합니다: '{parsed_args.config_path}'")

    # --- 로깅 설정 결정 (YAML vs. 명령줄) ---
    root_dir_from_yaml = None
    try:
        config_manager = configger(config_path=parsed_args.config_path, root_dir=parsed_args.root_dir)
        logger.debug("Configger 초기화 완료.")
        
        project_cfg = config_manager.get_config("project") or {}
        root_dir_from_yaml = project_cfg.get("root_dir")

        logging_cfg = config_manager.get_config("project.logging") or {}
        log_dir_from_yaml = logging_cfg.get("log_dir")
        log_level_from_yaml = logging_cfg.get("file_level", "INFO")
        console_level_from_yaml = logging_cfg.get("console_level", "CONSOLE")
        
        if parsed_args.log_dir:
            final_log_dir = parsed_args.log_dir
        elif log_dir_from_yaml:
            final_log_dir = log_dir_from_yaml
        elif root_dir_from_yaml:
            final_log_dir = Path(root_dir_from_yaml) / 'logs'
        else:
            final_log_dir = None

        final_log_level = parsed_args.log_level or log_level_from_yaml
        final_console_level = console_level_from_yaml

    except Exception as e:
        logger.error(f"Configger 초기화 또는 설정 로드 중 오류 발생: {e}", exc_info=True)
        final_log_dir = parsed_args.log_dir
        final_log_level = parsed_args.log_level or "INFO"
        final_console_level = "CONSOLE"
        config_manager = None

    if final_log_dir is None:
        final_root_dir = parsed_args.root_dir or root_dir_from_yaml or Path.cwd()
        final_log_dir = Path(final_root_dir) / 'logs'
        logger.console(f"로그 디렉토리가 지정되지 않아 기본 경로를 사용합니다: '{final_log_dir}'")

    try:
        final_log_dir_path = Path(final_log_dir).expanduser().resolve()
        final_log_dir_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"로그 디렉토리 '{final_log_dir}'를 생성할 수 없습니다: {e}", exc_info=True)
        sys.exit(1)

    # 2단계: 파일 로깅을 포함한 전체 로거 설정
    date_str = datetime.now().strftime("%y%m%d_%H%M")
    log_file_name = f"{script_name}_{date_str}.log"
    full_log_path = final_log_dir_path / log_file_name

    logger.setup(
        logger_path=full_log_path,
        console_min_level=final_console_level.upper(),
        file_min_level=final_log_level.upper(),
        include_function_name=True,
        pretty_print=True
    )
    logger.console(f"애플리케이션 ({script_name}) 시작. (로거 2/2 단계 초기화 완료)")
    logger.console(f"로그 파일 위치: {full_log_path}")
    logger.console(f"명령줄 인자: {vars(parsed_args)}")
    logger.console(f"최종 로그 레벨: 파일='{final_log_level.upper()}', 콘솔='{final_console_level.upper()}'")

    # 설정 파일에서 무시할 파일 목록 로드
    ignore_files_set = {'.DS_Store', 'Thumbs.db'} # 기본값
    try:
        if config_manager is None:
            raise RuntimeError("설정 관리자(configger)를 초기화하지 못했습니다.")
        processing_cfg = config_manager.get_config("processing") or {}
        # YAML 설정 파일에서 'ignore_files_in_empty_dirs' 키를 찾아 무시 목록을 확장합니다.
        # 이 키가 없으면 빈 리스트를 반환하여 기본값만 사용됩니다.
        files_from_config = processing_cfg.get("ignore_files_in_empty_dirs", [])
        if files_from_config:
            ignore_files_set.update(files_from_config)
            logger.console(f"설정 파일에서 무시할 파일 목록을 로드하여 확장했습니다: {ignore_files_set}")
    except Exception as e:
        logger.error(f"Configger에서 무시 파일 목록 로드 중 오류: {e}", exc_info=True)
        logger.console(f"기본 무시 파일 목록을 사용합니다: {ignore_files_set}")

    # 명령줄 인자로부터 경로 및 옵션 설정
    target_dir_str = parsed_args.target_dir # arg_utils에서 필수로 검증되었으므로 None이 아님
    target_dir = Path(target_dir_str).expanduser().resolve()
    dry_run_mode = getattr(parsed_args, 'dry_run', False)
    delete_top_mode = getattr(parsed_args, 'delete_top_if_empty', False)

    # 향후 --ignore-files 인자로 받아오도록 확장 가능
    # ignore_files_set는 위에서 이미 초기화 및 확장되었습니다.

    # 메인 로직 함수 호출
    logger.console("+++++++++++++++++++++++++++++++++++++")
    logger.console("--- 빈 디렉토리 정리 시작 ---")
    logger.console("-------------------------------------")
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

        # 최종 통계 출력
        logger.console("+++++++++++++++++++++++++++++++++++++")
        logger.console("--- 빈 디렉토리 정리 마침 ---")
        logger.console("-------------------------------------")
        logger.console("--- 빈 디렉토리 정리 통계 ---")
        max_visual_msg_len = max(visual_length(v["msg"]) for v in DEFAULT_STATUS_TEMPLATE.values()) if DEFAULT_STATUS_TEMPLATE else 20
        max_val_for_width = max((s_item["value"] for s_item in final_status.values()), default=0)
        digit_width_stats = calc_digit_number(max_val_for_width)
        
        for key, data in final_status.items():
            msg = DEFAULT_STATUS_TEMPLATE.get(key, {}).get("msg", key.replace("_", " ").capitalize())
            value = data["value"]
            padding = ' ' * (max_visual_msg_len - visual_length(msg))
            logger.console(f"  {msg}{padding} : {value:>{digit_width_stats}}")
        logger.console("------------------------------------")

    except Exception as e:
        logger.error(f"애플리케이션 실행 중 최상위 오류 발생: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.console(f"애플리케이션 ({script_name}) 종료{ ' (Dry Run 모드)' if dry_run_mode else ''}")
        if hasattr(logger, "shutdown"):
            logger.shutdown()
