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
    parser = argparse.ArgumentParser(description="Script execution path and logging level settings")
    parser.add_argument(
        '--root-dir', '-root',
        type=str,
        default=curr_dir,
        help='The root directory of the project. (Default: current working directory)'
    )
    parser.add_argument(
        '--log-dir', '-log',
        type=str,
        default=(Path(curr_dir) / 'logs').expanduser().resolve(),
        help='Directory to save log files.'
    )
    parser.add_argument(
        '--log-level', '-lvl',
        type=str,
        default='warning',
        choices=["debug", "info", "warning", "error", "critical"],
        help='Logging level. (Default: warning)'
    )
    parser.add_argument(
        '--config-path', '-cfg',
        type=str,
        default=(Path(curr_dir) / '../config' / 'photo_album.yaml').expanduser().resolve(),
        help='Path to the configuration (YAML) file.'
    )
    parser.add_argument(
        '--source-dir', '-src',
        type=str,
        required=False,
        help='Source directory. (Default: not provided)'
    )
    parser.add_argument(
        '--destination-dir', '-dst',
        type=str,
        required=False,
        help='Destination directory. (Default: not provided)'
    )
    parser.add_argument(
        '--target-dir', '-tgt',
        type=str,
        required=False,
        help='Target directory for a specific operation. (Default: not provided)'
    )
    parser.add_argument(
        '--action', '-act',
        type=str,
        default='move',
        choices=['move', 'copy'],
        help='File operation: "move" or "copy". (Default: move)'
    )
    parser.add_argument(
        '--dry-run', '--dry_run', '-dry',
        dest='dry_run',
        action="store_true",
        help="Perform a dry run without actual file operations."
    )
    parser.add_argument(
        '--delete-top-if-empty',
        dest='delete_top_if_empty',
        action="store_true",
        help="Also delete the top-level directory if it becomes empty (used by remove_empty_directories.py)."
    )

    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # --- Print parsed arguments for user confirmation ---
    print("--- Parsed Arguments ---")
    arg_print_definitions = [
        ("Root Directory (--root-dir)", lambda: args.root_dir, lambda: True),
        ("Log Directory (--log-dir)", lambda: args.log_dir, lambda: True),
        ("Log Level (--log-level)", lambda: args.log_level, lambda: True),
        ("Config Path (--config-path)", lambda: args.config_path, lambda: True),
        ("Action (--action)", lambda: args.action, lambda: True),
    ]
    if args.source_dir is not None:
        arg_print_definitions.append(("Source Directory (--source-dir)", lambda: args.source_dir, lambda: True))
    if args.destination_dir is not None:
        arg_print_definitions.append(("Destination Directory (--destination-dir)", lambda: args.destination_dir, lambda: True))
    if args.target_dir is not None:
        arg_print_definitions.append(("Target Directory (--target-dir)", lambda: args.target_dir, lambda: True))

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