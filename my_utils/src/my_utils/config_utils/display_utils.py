# display_utils.py
"""
시각적 문자열 길이 계산, 문자열 자르기 등 표시 형식에 대한 유틸리티 함수를 제공합니다.
"""

import math
import unicodedata
import statistics
from pathlib import Path
from typing import List, Tuple

def calc_digit_number(in_number: int) -> int:
    """주어진 정수의 자릿수를 계산합니다."""
    if in_number == 0:
        return 1
    in_number = abs(in_number)
    return math.floor(math.log10(in_number)) + 1

def visual_length(text, space_width=1):
    """전각 문자를 고려하여 문자열의 시각적 길이를 계산합니다."""
    length = 0
    for ch in text:
        if ch == ' ':
            length += space_width
        elif unicodedata.east_asian_width(ch) in ('W', 'F'):
            length += 2
        else:
            length += 1
    return length

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

def calculate_median(data_list):
    """숫자 리스트의 중앙값을 계산합니다."""
    if not data_list:
        return None
    return statistics.median(data_list)

def get_display_width(
    source_dir: Path,
    extensions: set,
    buffer_ratio: float = 0.15,
    min_width: int = 20,
    max_width: int = 50,
) -> Tuple[List[Path], int]:
    """디렉토리에서 파일을 스캔하고 파일 이름에 대한 권장 표시 너비를 계산합니다."""
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