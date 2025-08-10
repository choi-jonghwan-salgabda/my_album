# display_utils.py
"""
시각적 문자열 길이 계산, 문자열 자르기 등 표시 형식에 대한 유틸리티 함수를 제공합니다.
"""

import math
import unicodedata
import os
from pathlib import Path
from typing import List, Tuple, Callable, Iterable, Set
from typing import Any, Optional, TypeVar, Union, Dict
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# 프로젝트 공용 로거 임포트
try:
    from my_utils.config_utils.arg_utils import get_argument, visual_length
    from my_utils.config_utils.SimpleLogger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 제네릭 타입을 위한 TypeVar 정의
T = TypeVar('T')
R = TypeVar('R')

def calc_digit_number(in_number: int) -> int:
    """주어진 정수의 자릿수를 계산합니다."""
    if in_number == 0:
        return 1
    in_number = abs(in_number)
    return math.floor(math.log10(in_number)) + 1

def truncate_string(text: str, max_visual_width: int, ellipsis: str = '...') -> str:
    """문자열을 최대 시각적 길이에 맞춰 자르고, 필요한 경우 말줄임표를 추가합니다."""
    if visual_length(text) <= max_visual_width:
        return text

    ellipsis_visual_len = visual_length(ellipsis)
    current_visual_len = 0
    truncated_text_chars = []
    for char in text:
        char_visual_len = visual_length(char)
        if current_visual_len + char_visual_len + ellipsis_visual_len <= max_visual_width:
            truncated_text_chars.append(char)
            current_visual_len += char_visual_len
        else:
            break
    return "".join(truncated_text_chars) + ellipsis

def _scan_worker(dir_to_scan: Path, extensions: Set[str]) -> Tuple[List[Path], List[int], Optional[str]]:
    """
    [Worker] 지정된 단일 디렉토리와 그 하위 디렉토리를 재귀적으로 스캔하여
    조건에 맞는 파일 목록과 파일 이름의 시각적 길이 목록을 반환합니다.
    오류 발생 시 오류 메시지를 반환합니다.
    이 함수는 병렬 처리를 위해 별도의 프로세스에서 실행됩니다.

    Args:
        dir_to_scan (Path): 스캔할 디렉토리 경로.
        extensions (Set[str]): 허용되는 파일 확장자 집합 (소문자).

    Returns:
        Tuple[List[Path], List[int], Optional[str]]:
            - 첫 번째 요소: 찾은 파일 Path 객체 리스트.
            - 두 번째 요소: 찾은 파일들의 이름 시각적 길이 리스트.
            - 세 번째 요소: 오류 메시지 (성공 시 None).
    """
    local_files: List[Path] = []
    local_name_lengths: List[int] = []
    err_msg: Optional[str] = None # 오류 메시지를 저장할 변수 초기화

    try:
        if not dir_to_scan.is_dir():
            err_msg = f"스캔 대상이 디렉토리가 아니거나 존재하지 않음: '{dir_to_scan}'"
            return [], [], err_msg # 디렉토리 오류 시 빈 리스트와 메시지 반환

        # rglob은 비교적 작은 하위 디렉토리 내에서는 충분히 효율적입니다.
        for p in dir_to_scan.rglob("*"):
            if p.suffix.lower() in extensions and p.is_file():
                local_files.append(p)
                local_name_lengths.append(visual_length(p.name))
    except Exception as e:
        # 오류 메시지를 err_msg 변수에 저장합니다.
        err_msg = f"'{dir_to_scan}' 디렉토리 스캔 중 오류 발생: {e}"
        # 오류 발생 시, 부분적으로 채워진 리스트 대신 명확하게 빈 리스트를 반환합니다.
        return [], [], err_msg

    # 오류가 없었으면 err_msg는 None 상태로 반환됩니다.
    return local_files, local_name_lengths, err_msg


def scan_files_and_get_display_width(
    source_dir: Path,
    extensions: set,
    buffer_ratio: float = 0.15,
    min_width: int = 20,
    max_width: int = 50,
) -> Tuple[Optional[List[Path]], int, str]:
    """
    디렉토리에서 지정한 확장자를 가진 파일들을 검색하고,
    해당 파일 이름의 평균 시각적 길이에 따라 권장 표시 너비를 계산합니다.
    하위 디렉토리가 많을 경우, 스캔 작업을 병렬로 처리하여 성능을 향상시킵니다.

    Parameters:
        source_dir (Path): 검색할 기준 디렉토리 경로.
        extensions (set): 대상 확장자들 (예: {'.py', '.txt'}).
        buffer_ratio (float): 평균 길이에 곱해지는 여유 비율 (기본값: 0.15).
        min_width (int): 표시 너비의 최소값 (기본값: 20).
        max_width (int): 표시 너비의 최대값 (기본값: 50).

    Returns:
        Tuple[Optional[List[Path]], int, str]:
            - 조건에 맞는 파일 목록 (Path 객체 리스트) 또는 오류 시 None
            - 권장 표시 너비 (int)
            - 오류 또는 정보 메시지 (str)
    """
    all_files: List[Path] = []
    all_name_lengths: List[int] = []
    error_messages: List[str] = []

    # 1. 최상위 디렉토리의 파일들을 먼저 스캔합니다.
    try:
        if not source_dir.is_dir():
            err_msg = f"스캔 대상이 디렉토리가 아니거나 존재하지 않음: '{source_dir}'"
            logger.error(err_msg)
            return None, min_width, err_msg
        
        for p in source_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in extensions:
                all_files.append(p)
                all_name_lengths.append(visual_length(p.name))
    except OSError as e:
        err_msg = f"'{source_dir}'의 파일 스캔 중 오류 발생: {e}"
        logger.error(err_msg)
        error_messages.append(err_msg)
        # 오류가 발생했더라도 스캔이 불가능하지 않다면 계속 진행합니다.

    # 2. os.scandir를 사용하여 1단계 하위 디렉토리 목록을 효율적으로 가져옵니다.
    sub_dirs: List[Path] = []
    try:
        sub_dirs = [Path(entry.path) for entry in os.scandir(source_dir) if entry.is_dir()]
    except OSError as e:
        err_msg = f"'{source_dir}'의 하위 디렉토리 목록 스캔 중 오류 발생: {e}"
        logger.error(err_msg)
        error_messages.append(err_msg)

    if sub_dirs:
        logger.info(f"{len(sub_dirs)}개의 하위 디렉토리에 대해 병렬 스캔을 시작합니다...")
        
        # 3. ProcessPoolExecutor를 사용하여 각 하위 디렉토리를 병렬로 스캔합니다.
        with ProcessPoolExecutor() as executor:
            future_to_dir = {executor.submit(_scan_worker, sub_dir, extensions): sub_dir for sub_dir in sub_dirs}
            
            # tqdm으로 진행 상황을 표시하며 완료되는 작업부터 결과를 집계합니다.
            with tqdm(total=len(sub_dirs), desc="하위 폴더 스캔 중", unit="폴더", file=sys.stdout) as pbar:
                for future in as_completed(future_to_dir):
                    sub_dir_path = future_to_dir[future]
                    try:
                        files, name_lengths, err_msg = future.result()
                        if err_msg:
                            logger.warning(f"워커 오류: {err_msg}")
                            error_messages.append(err_msg)
                        
                        # 워커가 오류를 반환했더라도 빈 리스트를 반환하므로, extend는 안전합니다.
                        all_files.extend(files)
                        all_name_lengths.extend(name_lengths)
                    except Exception as exc:
                        err_msg = f"'{sub_dir_path}' 디렉토리 처리 중 예외 발생: {exc}"
                        logger.error(err_msg, exc_info=True)
                        error_messages.append(err_msg)
                    
                    pbar.update(1)

    # 4. 최종 표시 너비 계산
    if all_name_lengths:
        avg_length = sum(all_name_lengths) / len(all_name_lengths)
        display_width = int(avg_length * (1 + buffer_ratio))
        display_width = max(min_width, min(display_width, max_width))
    else:
        # 파일이 하나도 없을 경우 최소 너비로 설정
        display_width = min_width

    # 5. 최종 결과 반환
    final_message = "; ".join(error_messages) if error_messages else "스캔 완료"
    return all_files, display_width, final_message


def create_dynamic_description_func(
    prefix: str,
    width: int,
    item_to_name_func: Callable[[Any], str] = lambda item: getattr(item, 'name', str(item))
    ) -> Callable[[Any], str]:
    """
    tqdm 진행 표시줄의 설명을 동적으로 생성하는 함수를 반환하는 팩토리 함수입니다.

    반환된 함수는 각 항목을 처리할 때마다 호출되어 진행 표시줄의 설명을 업데이트합니다.

    Args:
        prefix (str): 설명 앞에 붙일 고정 접두사 (예: "처리 중: ").
        width (int): 이름이 표시될 최대 시각적 너비.
        item_to_name_func (Callable[[Any], str], optional):
            처리 항목에서 표시할 이름을 추출하는 함수.
            기본값은 항목의 'name' 속성을 사용하거나, 없으면 문자열로 변환합니다.

    Returns:
        Callable[[Any], str]: tqdm의 `description_func` 인자로 사용할 수 있는 함수.
    """
    def description_function(item: Any) -> str:
        """tqdm 설명을 동적으로 생성하는 실제 함수."""
        name_str = item_to_name_func(item)
        # 터미널에 안전하게 표시할 수 있도록 인코딩/디코딩
        safe_name = name_str.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
        # 계산된 너비에 맞게 이름 자르기
        display_name = truncate_string(safe_name, width)
        # 왼쪽 정렬하여 일관된 너비 유지
        return f"{prefix}{display_name:<{width}}"
    return description_function


def with_progress_bar(
    items: Iterable,
    task_func: Callable[[Any, int], None],
    desc: str = "처리 중",
    unit: str = "항목",
    total: Optional[int] = None,
    dynamic_ncols: bool = True,
    description_func: Optional[Callable[[Any], str]] = None
):
    """
    tqdm 진행 표시줄과 함께 항목들을 처리하는 범용 유틸리티 함수.
    로거의 tqdm 호환 모드를 자동으로 관리하며, 예외 발생 시에도 로거 상태를 복원합니다.

    Args:
        items (Iterable): 처리할 항목 리스트 또는 제너레이터.
                          total이 None이면, 메모리 사용량에 주의해야 합니다 (전체 항목이 리스트로 변환됨).
        task_func (Callable): 각 항목에 대해 실행할 함수 (인자: item, index).
        desc (str): tqdm 진행 표시줄의 기본 설명.
        unit (str): 진행 단위 (예: "파일", "디렉토리").
        total (int, optional): 전체 항목 수. 제공되지 않으면 items로부터 계산됩니다.
        dynamic_ncols (bool): 터미널 폭 자동 조절 여부.
        description_func (Callable, optional): 각 항목 처리 후 pbar의 설명을 동적으로 설정하는 함수.
                                                이 함수는 현재 처리 중인 item을 인자로 받습니다.
    """
    # total이 제공되지 않은 경우, iterable의 길이를 계산합니다.
    # len()이 불가능한 제너레이터의 경우, 리스트로 변환하며 이는 메모리 사용량에 영향을 줄 수 있습니다.
    if total is None:
        try:
            total = len(items)
        except TypeError:
            logger.debug("Iterable에 len()을 사용할 수 없어 리스트로 변환합니다. 메모리 사용량에 주의하세요.")
            items = list(items)
            total = len(items)

    if hasattr(logger, 'set_tqdm_aware'):
        logger.set_tqdm_aware(True)
    try:
        with tqdm(total=total, desc=desc, unit=unit, file=sys.stdout, dynamic_ncols=dynamic_ncols) as pbar:
            for idx, item in enumerate(items):
                task_func(item, idx)
                # description_func가 제공되면, 진행 표시줄의 설명을 동적으로 업데이트합니다.
                if description_func:
                    pbar.set_description(description_func(item))
                pbar.update(1)
    finally:
        # 작업이 성공하든 실패하든, 로거 설정을 원래대로 복원합니다.
        if hasattr(logger, 'set_tqdm_aware'):
            logger.set_tqdm_aware(False)

def _worker_init_tqdm():
    """Initializer for ProcessPoolExecutor workers to make the global logger tqdm-aware."""
    # 각 워커 프로세스는 자신만의 전역 로거 인스턴스를 가집니다.
    # 여기서 해당 로거를 임포트하고 설정해야 합니다.
    if hasattr(logger, 'set_tqdm_aware'):
        logger.set_tqdm_aware(True)

def with_parallel_progress_bar(
    items: Iterable[T],
    task_func: Callable[[T], R],
    desc: str = "처리 중",
    unit: str = "항목",
    postfix_func: Optional[Callable[[Any], str]] = None,
    preserve_order: bool = False,
    max_workers: Optional[int] = None
) -> List[Union[R, Exception]]:
    """
    ProcessPoolExecutor와 tqdm을 사용해 항목들을 병렬로 처리하고 결과를 반환합니다.
    워커 프로세스의 로거를 자동으로 tqdm 호환 모드로 설정합니다.

    Args:
        items (Iterable[T]): 처리할 항목 리스트 또는 제너레이터.
        task_func (Callable[[T], R]): 각 항목에 대해 실행할 함수. 결과를 반환해야 합니다.
        desc (str): tqdm 진행 표시줄의 기본 설명.
        unit (str): 진행 단위.
        postfix_func (Callable, optional): 각 항목 처리 완료 후 pbar의 후행 텍스트(postfix)를 동적으로 설정하는 함수.
                                            이 함수는 현재 처리 완료된 item을 인자로 받습니다.
        preserve_order (bool): True이면 입력 순서와 동일한 순서로 결과를 반환합니다. 
                               이 경우, 실패한 작업은 결과 리스트에 Exception 객체로 포함됩니다.
                               False(기본값)이면 완료되는 순서대로 결과를 반환하며, 실패한 작업은 결과에서 제외됩니다.
        max_workers (int, optional): 사용할 최대 워커 프로세스 수. None이면 기본값을 사용합니다.

    Returns:
    """
    results = []
    # len()을 사용할 수 없는 제너레이터 등을 위해 리스트로 변환
    if not hasattr(items, '__len__'):
        items = list(items)

    with ProcessPoolExecutor() as executor:
        future_to_item = {executor.submit(task_func, item): item for item in items}
        with tqdm(total=len(items), desc=desc, unit=unit, dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                    if description_func:
                        pbar.set_postfix_str(description_func(item), refresh=True)
                except Exception as e:
                    logger.error(f"❗ 워커 오류 발생: {item} → {e}")
                pbar.update(1)
    return results
