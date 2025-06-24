import sys
from pathlib import Path
import shutil
from datetime import datetime
import copy

try:
    from my_utils.config_utils.SimpleLogger import logger, calc_digit_number, get_argument, visual_length
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

# 처리 상태를 기록하기 위한 기본 템플릿
DEFAULT_STATUS_TEMPLATE = {
    "directories_scanned": {"value": 0, "msg": "검사한 총 디렉토리 수"},
    "directories_deleted": {"value": 0, "msg": "삭제된 빈 디렉토리 수"},
    "deletion_errors": {"value": 0, "msg": "삭제 중 발생한 오류 수"},
}

def remove_empty_directories_logic(
    directory_path: Path,
    status: dict,
    dry_run: bool = False,
    delete_top_if_empty: bool = False
):
    """
    지정된 디렉토리 내의 모든 빈 하위 디렉토리를 재귀적으로 삭제하는 핵심 로직입니다.
    내부적으로 재귀 헬퍼 함수를 호출하여 작업을 수행합니다.

    Args:
        directory_path (Path): 검사 및 삭제를 시작할 디렉토리 경로.
        status (dict): 처리 통계를 기록할 딕셔너리.
        dry_run (bool): True이면 실제 삭제 없이 로그만 남깁니다.
        delete_top_if_empty (bool): True이고 최상위 directory_path도 비게 되면 삭제합니다.

    Returns:
        dict: 업데이트된 최종 통계 딕셔너리.
    """
    def _recursive_remover(current_path: Path, is_top_level: bool):
        """재귀적으로 디렉토리를 탐색하고 비어있으면 삭제하는 헬퍼 함수."""
        if not current_path.is_dir():
            return

        status["directories_scanned"]["value"] += 1
        
        # 하위 디렉토리부터 재귀적으로 처리 (깊이 우선 탐색의 후위 순회 방식)
        for item in list(current_path.iterdir()):
            if item.is_dir():
                _recursive_remover(item, is_top_level=False)

        # 현재 디렉토리가 비었는지 다시 확인 (하위 빈 디렉토리들이 삭제된 후)
        if not any(current_path.iterdir()):
            # 최상위 디렉토리인 경우, delete_top_if_empty 플래그에 따라 삭제 여부 결정
            if is_top_level and not delete_top_if_empty:
                logger.info(f"최상위 디렉토리 '{current_path}'는 비어있지만, --delete-top-if-empty 옵션이 없어 삭제하지 않습니다.")
                return

            try:
                if dry_run:
                    logger.info(f"(Dry Run) 빈 디렉토리 삭제 예정: '{current_path}'")
                else:
                    current_path.rmdir()
                    logger.info(f"빈 디렉토리 삭제됨: '{current_path}'")
                status["directories_deleted"]["value"] += 1
            except OSError as e:
                logger.error(f"디렉토리 '{current_path}' 삭제 중 오류 발생: {e}")
                status["deletion_errors"]["value"] += 1

    if not directory_path.is_dir():
        logger.error(f"오류: '{directory_path}'는 유효한 디렉토리가 아닙니다.")
        return status

    _recursive_remover(directory_path, is_top_level=True)
    return status

def run_main():
    """
    스크립트의 메인 실행 함수입니다.
    명령줄 인자 파싱, 로깅 설정, 핵심 로직 호출 및 최종 통계 출력을 담당합니다.
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

    # 필수 디렉토리 인자 확인
    target_dir_path = parsed_args.target_dir
    if not target_dir_path:
        logger.error("정리할 대상 디렉토리(--target-dir)가 제공되지 않았습니다. 스크립트를 종료합니다.")
        sys.exit(1)

    # 명령줄 인자로부터 경로 및 옵션 설정
    target_dir = Path(target_dir_path).expanduser().resolve()
    dry_run_mode = getattr(parsed_args, 'dry_run', False)
    delete_top_mode = getattr(parsed_args, 'delete_top_if_empty', False)

    try:
        # 핵심 로직 함수 호출
        status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)
        final_status = remove_empty_directories_logic(
            directory_path=target_dir,
            status=status,
            dry_run=dry_run_mode,
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

if __name__ == '__main__':
    run_main()
