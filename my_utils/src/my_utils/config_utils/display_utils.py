# display_utils.py
"""
Provides utility functions for display formatting, such as calculating
visual string length and truncating strings.
"""

import math
import unicodedata
import statistics
from pathlib import Path
from typing import List, Tuple

def calc_digit_number(in_number: int) -> int:
    """Calculates the number of digits in an integer."""
    if in_number == 0:
        return 1
    in_number = abs(in_number)
    return math.floor(math.log10(in_number)) + 1

def visual_length(text, space_width=1):
    """Calculates the visual length of a string, considering full-width characters."""
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
    """Truncates a string to a maximum visual width, adding an ellipsis if needed."""
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
    """Calculates the median of a list of numbers."""
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
    """
    Scans a directory for files and calculates a recommended display width for their names.
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