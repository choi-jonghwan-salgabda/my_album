# display_utils.py
"""
시각적 문자열 길이 계산, 문자열 자르기 등 표시 형식에 대한 유틸리티 함수를 제공합니다.
"""

import math
import unicodedata
from pathlib import Path
from typing import List, Tuple, Callable, Iterable, Any, Optional, TypeVar, Union
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

def get_display_width(
    source_dir: Path,
    extensions: set,
    buffer_ratio: float = 0.15,
    min_width: int = 20,
    max_width: int = 50,
) -> Tuple[List[Path], int]:
    """
    디렉토리에서 지정한 확장자를 가진 파일들을 검색하고,
    해당 파일 이름의 평균 시각적 길이에 따라 권장 표시 너비를 계산합니다.

    Parameters:
        source_dir (Path): 검색할 기준 디렉토리 경로.
        extensions (set): 대상 확장자들 (예: {'.py', '.txt'}).
        buffer_ratio (float): 평균 길이에 곱해지는 여유 비율 (기본값: 0.15).
        min_width (int): 표시 너비의 최소값 (기본값: 20).
        max_width (int): 표시 너비의 최대값 (기본값: 50).

    Returns:
        Tuple[List[Path], int]: 
            - 조건에 맞는 파일 목록 (Path 객체 리스트)
            - 권장 표시 너비 (int)
    """
    files = []
    name_lengths = []

    for p in source_dir.rglob("*"):
        if p.suffix.lower() in extensions and p.is_file():
            files.append(p)
            name_lengths.append(visual_length(p.name))

    if name_lengths:
        avg_length = sum(name_lengths) / len(name_lengths)
        display_width = int(avg_length * (1 + buffer_ratio))
        display_width = max(min_width, min(display_width, max_width))
    else:
        display_width = min_width

    return files, display_width


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
    from my_utils.config_utils.SimpleLogger import logger
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
