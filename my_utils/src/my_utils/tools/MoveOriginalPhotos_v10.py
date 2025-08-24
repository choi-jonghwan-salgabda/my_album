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
import io
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import concurrent.futures
import multiprocessing
import threading
import functools
from tqdm import tqdm

# 프로젝트 공용 유틸리티 임포트
try:
    from my_utils.config_utils.arg_utils import get_argument
    from my_utils.config_utils.SimpleLogger import logger
    from my_utils.config_utils.configger import configger
    from my_utils.config_utils.file_utils import safe_move, safe_copy, DiskFullError, get_original_filename, get_unique_path
    from my_utils.config_utils.display_utils import calc_digit_number, visual_length, truncate_string, with_progress_bar
    from my_utils.object_utils.photo_utils import get_exif_date_taken
except ImportError as e:
    # logger가 설정되기 전이므로 print 사용
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

# EXIF 날짜가 없는 파일을 정렬 시 가장 뒤로 보내기 위한 값
MAX_DATETIME = datetime.max

def get_file_score(file_path: Path) -> tuple:
    """
    파일의 '원본성'을 평가하여 점수(튜플)를 반환합니다. 점수가 낮을수록 원본에 가깝습니다.

    평가 기준 (우선순위 순):
    1. EXIF 촬영 날짜: 오래된 날짜일수록 점수가 낮습니다. (가장 중요)
       - EXIF 날짜가 없는 파일은 있는 파일보다 항상 높은 점수를 받습니다.
    2. 파일 생성 시간(ctime): 오래된 시간일수록 점수가 낮습니다.
    3. 파일명 특수문자 수: '_', '-'의 개수가 적을수록 점수가 낮습니다.
    4. 파일명 숫자 수: 숫자가 적을수록 점수가 낮습니다.
    5. 파일명 길이: 짧을수록 점수가 낮습니다.
    """
    try:
        # 파일을 한 번만 읽어 메모리 내에서 EXIF 처리
        with open(file_path, 'rb') as f:
            file_content = io.BytesIO(f.read())
        date_str, _ = get_exif_date_taken(file_content)
        # 날짜 문자열이 있으면 datetime 객체로, 없으면 MAX_DATETIME으로 설정
        exif_date = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S") if date_str else MAX_DATETIME
    except (IOError, ValueError, TypeError):
        # 파일 읽기/EXIF 파싱 오류 시
        exif_date = MAX_DATETIME

    try:
        ctime = file_path.stat().st_ctime
    except FileNotFoundError:
        ctime = float('inf') # 파일이 없으면 가장 높은 점수

    clean_name = get_original_filename(file_path)
    stem = Path(clean_name).stem
    
    special_chars = stem.count('_') + stem.count('-')
    num_digits = sum(c.isdigit() for c in stem)
    name_len = len(stem)

    # 점수를 튜플로 반환하여 다중 기준으로 정렬
    return (exif_date, ctime, special_chars, num_digits, name_len)

# 처리 상태를 기록하기 위한 기본 템플릿
DEFAULT_STATUS_TEMPLATE = {
    "hash_dirs_scanned":      {"value": 0, "msg": "검사한 해시 디렉토리 수"},
    "files_processed":        {"value": 0, "msg": "선별되어 이동/복사된 대표 파일 수"},
    "files_archived":         {"value": 0, "msg": "보관 처리된 나머지 중복 파일 수"},
    "empty_hash_dirs_skipped":{"value": 0, "msg": "이미지 파일이 없어 건너뛴 해시 디렉토리 수"},
    "selection_ties":         {"value": 0, "msg": "대표 파일 선정 시 동점이 발생한 그룹 수"},
    "file_op_errors":         {"value": 0, "msg": "파일 처리 중 발생한 오류 수"},
}

def log_listener_process(queue: multiprocessing.Queue):
    """
    [병렬 처리] 멀티프로세싱 로그 리스너.

    여러 워커 프로세스에서 발생하는 로그 메시지를 중앙에서 처리하기 위한 전담 프로세스(또는 스레드)입니다.
    공유된 `log_queue`에서 로그 레코드(레벨, 메시지)를 계속 가져와, 메인 프로세스의 `logger` 인스턴스를
    사용하여 파일 및 콘솔에 순차적으로 기록합니다.

    이유: 각 워커 프로세스가 직접 하나의 로그 파일에 쓰려고 하면 파일 잠금 문제(Race Condition)가
    발생할 수 있습니다. 하나의 전담 리스너가 모든 로그를 처리함으로써 이 문제를 해결합니다.
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
    [병렬 처리] 각 워커 프로세스를 초기화합니다.

    `ProcessPoolExecutor`가 새로운 워커 프로세스를 생성할 때마다 호출되는 함수입니다.
    각 워커 프로세스 내의 전역 `logger` 인스턴스를 설정하여, 모든 로그 메시지(예: logger.info)가
    파일이나 콘솔에 직접 기록되지 않고, 대신 공유 `log_queue`로 보내지도록 재구성합니다.
    """
    logger.setup(mp_log_queue=queue)

def _process_one_hash_dir_parallel(args: Tuple[Path, Path, Optional[Path], set, str, bool]) -> Dict[str, int]:
    """
    [병렬 워커] 단일 해시 디렉토리를 처리하고 결과를 딕셔너리로 반환합니다.
    이 함수는 ProcessPoolExecutor의 워커 프로세스에서 실행됩니다.
    """
    hash_dir, destination_dir, archive_dir, allowed_extensions, action, dry_run = args
    
    worker_status = {
        "files_processed": 0,
        "files_archived": 0,
        "empty_hash_dirs_skipped": 0,
        "selection_ties": 0,
        "file_op_errors": 0,
    }
    
    action_func = safe_move if action == 'move' else safe_copy
    action_verb = "이동" if action == 'move' else "복사"
    log_prefix = "(Dry Run) " if dry_run else ""
    
    # 경로 생성 시 'embedded null byte' 오류를 방지하기 위해 디렉토리 이름에서 NULL 바이트를 제거합니다.
    safe_hash_dir_name = hash_dir.name.replace('\0', '')

    try:
        image_files = [f for f in hash_dir.rglob("*") if f.is_file() and f.suffix.lower() in allowed_extensions]
        if not image_files:
            logger.debug(f"'{safe_hash_dir_name}' 디렉토리에 이미지 파일이 없어 건너뜁니다.")
            worker_status["empty_hash_dirs_skipped"] = 1
            return worker_status
        
        best_file = min(image_files, key=get_file_score)
        # 파일명에 포함될 수 있는 NULL 바이트를 제거합니다.
        safe_best_file_name = best_file.name.replace('\0', '')
        logger.debug(f"'{safe_hash_dir_name}' 그룹의 대표 파일로 '{safe_best_file_name}' 선택됨.")
        
        dest_subdir = destination_dir / safe_hash_dir_name
        dest_file_path = dest_subdir / safe_best_file_name
        if not dry_run:
            dest_subdir.mkdir(parents=True, exist_ok=True)
        
        if not dest_file_path.exists():
            logger.info(f"{log_prefix}{action_verb} 예정: '{best_file}' → '{dest_file_path}'")
            if not dry_run:
                # safe_move/copy 함수 내부에서도 NULL 바이트를 처리하지만, 경로 생성 단계에서부터 방지하는 것이 안전합니다.
                action_func(str(best_file), str(dest_file_path))
            worker_status["files_processed"] = 1
        else:
            logger.info(f"건너뜀: 대상 파일 이미 존재 → '{dest_file_path}'")
        
        if archive_dir:
            for other_file in [f for f in image_files if f != best_file]:
                # 파일명에 포함될 수 있는 NULL 바이트를 제거합니다.
                safe_other_file_name = other_file.name.replace('\0', '')
                archive_path = get_unique_path(archive_dir, safe_other_file_name)
                logger.info(f"{log_prefix}보관({action_verb}) 예정: '{other_file}' → '{archive_path}'")
                if not dry_run:
                    action_func(str(other_file), str(archive_path))
                worker_status["files_archived"] += 1
    except Exception as e:
        worker_status["file_op_errors"] += 1
    return worker_status


def select_and_process_originals_logic(
    source_dir: Path, 
    destination_dir: Path, 
    archive_dir: Optional[Path], 
    allowed_extensions: set[str], 
    action: str = 'move', 
    dry_run: bool = False,
    parallel: bool = False,
    max_workers: Optional[int] = None
):
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
        parallel (bool): 병렬 처리 활성화 여부.
        max_workers (int, optional): 병렬 처리 시 사용할 최대 워커 수.

    Returns:
        dict: 처리 통계 딕셔너리.
    """
    status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)

    # parallel과 max_workers 인자를 함수 내부에서 사용하기 위해 로직을 수정합니다.
    # 이 부분은 이미 올바르게 구현되어 있으나, 함수 시그니처가 맞지 않아 오류가 발생했습니다.
    # 이제 시그니처가 수정되었으므로 아래 코드는 정상 동작합니다.
    manager = None
    log_queue = None
    listener_thread = None
    initializer = None
    if parallel:
        manager = multiprocessing.Manager()
        log_queue = manager.Queue()
        listener_thread = threading.Thread(target=log_listener_process, args=(log_queue,))
        # --- 데몬 스레드 설정 ---
        # 메인 프로그램이 종료될 때 이 스레드가 실행 중이더라도 강제 종료되도록 설정합니다.
        # 이를 통해 '로그 리스너 스레드가 시간 내에 종료되지 않았습니다' 경고 후 프로그램이 멈추는 현상을 방지합니다.
        listener_thread.daemon = True
        listener_thread.start()
        initializer = functools.partial(init_worker, log_queue)

    try:
        hash_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
        status["hash_dirs_scanned"]["value"] = len(hash_dirs)
        
        if not hash_dirs:
            logger.info("처리할 해시 디렉토리가 없습니다.")
            return status

        processing_args = [
            (d, destination_dir, archive_dir, allowed_extensions, action, dry_run)
            for d in hash_dirs
        ]

        group_results = []
        if parallel:
            logger.info(f"병렬 모드로 작업을 시작합니다 (최대 워커 수: {max_workers or '기본값'}).")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=initializer) as executor:
                results_iterator = executor.map(_process_one_hash_dir_parallel, processing_args)
                group_results = list(tqdm(results_iterator, total=len(hash_dirs), desc="대표 원본 정리 중 (병렬)", unit="폴더"))
        else:
            logger.info("순차 모드로 작업을 시작합니다.")
            for args_tuple in tqdm(processing_args, desc="대표 원본 정리 중 (순차)", unit="폴더"):
                group_results.append(_process_one_hash_dir_parallel(args_tuple))

        for worker_status in group_results:
            for key, value in worker_status.items():
                if key in status:
                    status[key]["value"] += value
        
        return status

    finally:
        if parallel and listener_thread:
            logger.info("로그 리스너 스레드 종료 신호 전송...")
            if log_queue:
                log_queue.put(None)
            listener_thread.join(timeout=5)
            if listener_thread.is_alive():
                logger.warning("로그 리스너 스레드가 시간 내에 종료되지 않았습니다.")
            else:
                logger.info("로그 리스너 스레드가 성공적으로 종료되었습니다.")

if __name__ == "__main__":
    """스크립트의 메인 실행 함수."""
    script_name = Path(__file__).stem

    # 1단계: 콘솔 전용 임시 로거 설정 (설정 파일/인자 로드 전)
    # 이 스크립트는 사용자에게 중요한 정보를 'CONSOLE' 레벨로 전달하므로, 해당 레벨을 사용합니다.
    logger.setup(
        logger_path=None,  # 파일 로깅은 아직 비활성화
        console_min_level="CONSOLE",
        include_function_name=False
    )
    logger.console(f"애플리케이션 ({script_name}) 시작. (로거 1/2 단계 초기화)")

    # 이 스크립트는 source-dir과 destination-dir을 필수로 요구합니다.
    supported_args_for_script = [
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

    # --- 설정 파일 경로 결정 ---
    if parsed_args.config_path is None:
        # --config-path가 지정되지 않았을 때 기본 경로를 생성합니다.
        # root_dir은 arg_utils에서 현재 작업 디렉토리로 기본값이 설정되어 있습니다.
        # configger가 상대 경로를 root_dir 기준으로 처리하므로, 여기서는 상대 경로를 전달합니다.
        parsed_args.config_path = '../config/photo_album.yaml'
        logger.debug(f"--config-path가 지정되지 않아 기본 상대 경로를 사용합니다: '{parsed_args.config_path}'")

    # --- 로깅 설정 결정 (YAML vs. 명령줄) ---
    # 1. YAML 설정 로드
    root_dir_from_yaml = None # NameError 방지를 위해 None으로 초기화
    try:
        # configger가 root_dir=None일 경우 자동으로 현재 작업 디렉토리를 사용하도록 수정되었습니다.
        config_manager = configger(config_path=parsed_args.config_path, root_dir=parsed_args.root_dir)
        logger.debug("Configger 초기화 완료.")
        
        # 2. YAML에서 로그 및 경로 설정 가져오기
        project_cfg = config_manager.get_config("project") or {}
        root_dir_from_yaml = project_cfg.get("root_dir")

        logging_cfg = config_manager.get_config("project.logging") or {}
        log_dir_from_yaml = logging_cfg.get("log_dir")
        log_level_from_yaml = logging_cfg.get("file_level", "INFO")
        console_level_from_yaml = logging_cfg.get("console_level", "CONSOLE")
        
        # 3. 최종 로그 설정 결정 (우선순위: 명령줄 > YAML > 기본값)
        # 로그 디렉토리 결정
        if parsed_args.log_dir:
            final_log_dir = parsed_args.log_dir
        elif log_dir_from_yaml:
            final_log_dir = log_dir_from_yaml
        elif root_dir_from_yaml:
            final_log_dir = Path(root_dir_from_yaml) / 'logs'
        else:
            final_log_dir = None # 아래에서 최종 fallback 처리

        # 로그 레벨 결정
        final_log_level = parsed_args.log_level or log_level_from_yaml
        final_console_level = console_level_from_yaml

    except Exception as e:
        logger.error(f"Configger 초기화 또는 설정 로드 중 오류 발생: {e}", exc_info=True)
        # 설정 파일 로드 실패 시 안전한 기본값 사용
        final_log_dir = parsed_args.log_dir # YAML이 없으므로 명령줄 인자만 고려
        final_log_level = parsed_args.log_level or "INFO"
        final_console_level = "CONSOLE"
        config_manager = None # config_manager 사용 불가 표시

    # 최종 로그 디렉토리 경로 결정 및 생성
    if final_log_dir is None:
        final_root_dir = parsed_args.root_dir or root_dir_from_yaml or Path.cwd()
        final_log_dir = Path(final_root_dir) / 'logs'
        logger.info(f"로그 디렉토리가 지정되지 않아 기본 경로를 사용합니다: '{final_log_dir}'")

    try:
        final_log_dir_path = Path(final_log_dir).expanduser().resolve()
        final_log_dir_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"로그 디렉토리 '{final_log_dir}'를 생성할 수 없습니다: {e}", exc_info=True)
        sys.exit(1)

    # 2단계: 파일 로깅을 포함한 전체 로거 설정 (최종 설정값 사용)
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
    logger.info(f"로그 파일 위치: {full_log_path}")
    logger.info(f"명령줄 인자: {vars(parsed_args)}")
    logger.info(f"최종 로그 레벨: 파일='{final_log_level.upper()}', 콘솔='{final_console_level.upper()}'")

    # 설정 파일에서 허용 확장자 로드
    allowed_extensions = set()
    try:
        if config_manager is None:
            raise RuntimeError("설정 관리자(configger)를 초기화하지 못했습니다.")
        processing_cfg = config_manager.get_config("processing") or {}
        extensions_from_config = processing_cfg.get("supported_image_extensions", [])
        if extensions_from_config:
            allowed_extensions = {ext.lower() for ext in extensions_from_config}
    except Exception as e:
        logger.error(f"Configger 설정 로드 중 오류 발생: {e}", exc_info=True)

    if not allowed_extensions:
        logger.warning("설정 파일에서 허용 확장자를 찾지 못했거나 로드 중 오류가 발생하여, 기본 이미지 확장자를 사용합니다.")
        allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic"}

    source_dir = Path(parsed_args.source_dir).expanduser().resolve()
    destination_dir = Path(parsed_args.destination_dir).expanduser().resolve()
    action_mode = parsed_args.action
    dry_run_mode = parsed_args.dry_run
    parallel_mode = parsed_args.parallel
    
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

        # 메인 로직 함수 호출
        logger.console("+++++++++++++++++++++++++++++++++++++")
        logger.console("--- 대표 원본 사진 정리 시작 ---")
        logger.console("-------------------------------------")
        final_status = select_and_process_originals_logic(
            source_dir=source_dir,
            destination_dir=destination_dir,
            archive_dir=quarantine_dir, # 'archive_dir' 인자로 전달
            allowed_extensions=allowed_extensions,
            action=action_mode,
            dry_run=dry_run_mode,
            parallel=parallel_mode,
            max_workers=parsed_args.max_workers
        )

        # 최종 통계 출력
        logger.console("-------------------------------------")
        logger.console("--- 대표 원본 사진 정리 마침 ---")
        logger.console("+++++++++++++++++++++++++++++++++++++")
        logger.console("--- 대표 원본 사진 정리 통계 ---")
        max_visual_msg_len = max(visual_length(v["msg"]) for v in DEFAULT_STATUS_TEMPLATE.values()) if DEFAULT_STATUS_TEMPLATE else 20
        max_val_for_width = max((s_item["value"] for s_item in final_status.values()), default=0)
        digit_width_stats = calc_digit_number(max_val_for_width)

        for key, data in final_status.items():
            msg = DEFAULT_STATUS_TEMPLATE.get(key, {}).get("msg", key.replace("_", " ").capitalize())
            value = data["value"]
            padding = max(0, int(max_visual_msg_len - visual_length(msg)))
            logger.console(f"{msg}{'-' * padding} : {value:>{digit_width_stats}}")
        logger.console("------------------------------------")

    except KeyboardInterrupt:
        # select_and_process_originals_logic 내부에서 처리되지만, 만약을 위해 여기에도 둠
        logger.warning("\n사용자에 의해 작업이 중단되었습니다.")
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 최상위 오류 발생: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.console(f"애플리케이션 ({script_name}) 종료{ ' (Dry Run 모드)' if dry_run_mode else ''}")
        if hasattr(logger, "shutdown"):
            logger.shutdown()
