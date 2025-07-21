# arg_utils.py
"""
Provides a centralized function for parsing command-line arguments
used by various tool scripts in the project.
"""

import os
import argparse
from pathlib import Path

try:
    from my_utils.config_utils.display_utils import visual_length
except ImportError:
    # Fallback for standalone execution or if display_utils is not found
    def visual_length(text, space_width=1):
        """A simple fallback for visual_length."""
        return len(text)

def get_argument() -> argparse.Namespace:
    """
    Parses command-line arguments for the tool scripts.
    """
    curr_dir = os.getcwd()

    # 사용법 예시를 포함하는 도움말ท้าย(epilog)을 정의합니다.
    epilog_text = """
사용 예시:

  - 사진을 해시값에 따라 정리하기 (OrganizePhotosByHash.py):
    python OrganizePhotosByHash.py --source_dir /path/to/photos --destination_dir /path/to/organized --action copy

  - 두 디렉토리(A, B)를 비교하여 중복 사진을 다른 곳(C)으로 옮기기 (MoveSearchedDupPhoto.py):
    python MoveSearchedDupPhoto.py --source_dir /path/to/dir_A --target_dir /path/to/dir_B --destination_dir /path/to/duplicates

  - 한 디렉토리(B) 내에서 중복 사진을 찾아 다른 곳(C)으로 옮기기 (MoveSearchedDupPhoto.py):
    python MoveSearchedDupPhoto.py --target_dir /path/to/dir_B --destination_dir /path/to/duplicates

  - 빈 하위 디렉토리 모두 삭제하기 (RemoveEmptyDirectories.py):
    python RemoveEmptyDirectories.py --target_dir /path/to/cleanup --dry_run
"""

    # RawTextHelpFormatter를 사용하여 epilog의 줄바꿈을 유지합니다.
    parser = argparse.ArgumentParser(description="스크립트 실행 경로 및 로깅 레벨 설정", epilog=epilog_text, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--root-dir', '--root_dir', '-root',
        type=str,
        default=curr_dir,
        help='프로젝트의 루트 디렉토리. (기본값: 현재 작업 디렉토리)'
    )
    parser.add_argument(
        '--log-dir', '--log_dir', '-log',
        type=str,
        default=(Path(curr_dir) / 'logs').expanduser().resolve(),
        help='로그 파일을 저장할 디렉토리.'
    )
    parser.add_argument(
        '--log-level', '--log_level', '-lvl',
        type=str,
        default='warning',
        choices=["debug", "info", "warning", "error", "critical"],
        help='로깅 레벨. (기본값: warning)'
    )
    parser.add_argument(
        '--config-path', '--config_path', '-cfg',
        type=str,
        default=(Path(curr_dir) / '../config' / 'photo_album.yaml').expanduser().resolve(),
        help='설정 파일(YAML)의 경로.'
    )
    parser.add_argument(
        '--source-dir', '--source_dir', '-src',
        type=str,
        required=False,
        help='소스(source) 디렉토리. (기본값: 제공되지 않음)'
    )
    parser.add_argument(
        '--destination-dir', '--destination_dir', '-dst',
        type=str,
        required=False,
        help='대상(destination) 디렉토리. (기본값: 제공되지 않음)'
    )
    parser.add_argument(
        '--target-dir', '--target_dir', '-tgt',
        type=str,
        required=False,
        help='특정 작업의 대상(target) 디렉토리. (기본값: 제공되지 않음)'
    )
    parser.add_argument(
        '--action', '-act',
        type=str,
        default='move',
        choices=['move', 'copy'],
        help='파일 처리 방식: "move"(이동) 또는 "copy"(복사). (기본값: move)'
    )
    parser.add_argument(
        '--dry-run', '--dry_run', '-dry',
        action="store_true",
        help="실제 파일 작업 없이 실행 과정만 보여주는 테스트 실행을 수행합니다."
    )
    parser.add_argument(
        '--delete-top-if-empty', '--delete_top_if_empty',
        action="store_true",
        help="최상위 디렉토리도 비어있을 경우 함께 삭제합니다 (remove_empty_directories.py에서 사용)."
    )

    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # --- Print parsed arguments for user confirmation ---
    print("--- 파싱된 인자 ---")
    arg_print_definitions = [
        ("루트 디렉토리 (--root-dir)", lambda: args.root_dir, lambda: True),
        ("로그 디렉토리 (--log-dir)", lambda: args.log_dir, lambda: True),
        ("로그 레벨 (--log-level)", lambda: args.log_level, lambda: True),
        ("설정 파일 경로 (--config-path)", lambda: args.config_path, lambda: True),
        ("파일 처리 방식 (--action)", lambda: args.action, lambda: True),
    ]
    if args.source_dir is not None:
        arg_print_definitions.append(("소스 디렉토리 (--source-dir)", lambda: args.source_dir, lambda: True))
    if args.destination_dir is not None:
        arg_print_definitions.append(("대상 디렉토리 (--destination-dir)", lambda: args.destination_dir, lambda: True))
    if args.target_dir is not None:
        arg_print_definitions.append(("타겟 디렉토리 (--target-dir)", lambda: args.target_dir, lambda: True))

    # 불리언(Boolean) 플래그들은 True일 때만 표시합니다.
    arg_print_definitions.append(("테스트 실행 모드 (Dry Run)", lambda: "활성화", lambda: args.dry_run))
    arg_print_definitions.append(("최상위 빈 디렉토리 삭제", lambda: "활성화", lambda: args.delete_top_if_empty))

    items_to_print = [(label, value_func()) for label, value_func, cond_func in arg_print_definitions if cond_func()]

    if not items_to_print:
        max_label_vl = 0
    else:
        max_label_vl = max(visual_length(label) for label, _ in items_to_print)

    target_value_start_column_vl = visual_length("  ") + max_label_vl + 3

    for label_text, value in items_to_print:
        prefix_with_spaces = f"  {label_text}"
        prefix_vl = visual_length(prefix_with_spaces)
        num_hyphens_vl_needed = target_value_start_column_vl - prefix_vl
        num_hyphens = max(1, int(num_hyphens_vl_needed))
        print(f"{prefix_with_spaces}{'-' * num_hyphens}{value}")
    print(f"--------------------------------------------------\n")

    return args