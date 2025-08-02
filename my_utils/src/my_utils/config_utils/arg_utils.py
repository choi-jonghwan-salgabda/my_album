# arg_utils.py
"""
Provides a centralized function for parsing command-line arguments
used by various tool scripts in the project.
"""

import argparse
import sys
from pathlib import Path
import unicodedata

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

def get_argument(required_args: list[str] = None) -> argparse.Namespace:
    """
    인자를 파싱하며, required_args로 지정된 필수 인자가 명시되지 않은 경우,
    위치 인자(positional arguments)를 순서대로 매핑하여 자동 보완합니다.

    Args:
        required_args (list[str], optional): '--source-dir'와 같이 필수로 지정할 인자 목록.
                                             이 목록의 순서에 따라 위치 인자가 매핑됩니다.
    """
    if required_args is None:
        required_args = []

    curr_dir = Path.cwd()

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
    # 스크립트 설명 및 사용 예시(epilog 포함)를 표시하는 argparse 객체 생성
    # RawTextHelpFormatter를 사용하여 epilog에 줄바꿈을 유지함
    parser = argparse.ArgumentParser(
        description="스크립트 실행 경로 및 로깅 레벨 설정",
        epilog=epilog_text,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 모든 잠재적 인자들을 정의합니다.

    parser.add_argument(
        # --root-dir, --root_dir, -root → 하나의 인자 root_dir로 매핑됨
        # default: 사용자 입력이 없을 때 사용할 기본값 지정
        '--root-dir', '--root_dir', '-root',
        type=str,
        default=curr_dir,
        help='프로젝트의 루트 디렉토리. (기본값: 현재 작업 디렉토리)'
    )
    parser.add_argument(
        # --log-dir, --rlog_dir, -log → 하나의 인자 log_dir로 매핑됨
        # default: 사용자 입력이 없을 때 사용할 기본값 지정
        '--log-dir', '--log_dir', '-log',
        type=str,
        default=(Path(curr_dir) / 'logs').expanduser().resolve(),
        help='로그 파일을 저장할 디렉토리.'
    )
    parser.add_argument(
        # --log-level, --log_level, -lvl → 하나의 인자 log_level로 매핑됨
        # default: 사용자 입력이 없을 때 사용할 기본값 지정
        '--log-level', '--log_level', '-lvl',
        type=str,
        default='warning',
        choices=["debug", "info", "warning", "error", "critical"],
        help='로깅 레벨. (기본값: warning)'
    )
    parser.add_argument(
        # --config-path, --config_path, -cfg → 하나의 인자 config_path로 매핑됨
        # default: 사용자 입력이 없을 때 사용할 기본값 지정
        '--config-path', '--config_path', '-cfg',
        type=str,
        default=(Path(curr_dir) / '../config' / 'photo_album.yaml').expanduser().resolve(),
        help='설정 파일(YAML)의 경로.'
    )
    parser.add_argument(
        # --action, -act → 하나의 인자 action로 매핑됨
        # default: 사용자 입력이 없을 때 사용할 기본값 지정
        '--action', '-act',
        type=str,
        default='copy',
        choices=['move', 'copy'],
        help='파일 처리 방식: "move"(이동) 또는 "copy"(복사). (기본값: copy)'
    )
    parser.add_argument(
        # --source-dir', '--source_dir', '-src', → 하나의 인자 source_dir로 매핑됨
        # default: 사용자 입력이 없을 때 사용할 기본값 없음
        '--source-dir', '--source_dir', '-src',
        type=str,
        default=None,
        help='소스(source) 디렉토리.'
    )
    parser.add_argument(
        # --destination-dir', '--destination_dir', '-dst', → 하나의 인자 destination_dir로 매핑됨
        # default: 사용자 입력이 없을 때 사용할 기본값 없음
        '--destination-dir', '--destination_dir', '-dst',
        type=str,
        default=None,
        help='대상(destination) 디렉토리.'
    )
    parser.add_argument(
        # --target-dir', '--target_dir', '-tgt', → 하나의 인자 target_dir로 매핑됨
        # default: 사용자 입력이 없을 때 사용할 기본값 없음
        '--target-dir', '--target_dir', '-tgt',
        type=str,
        default=None,
        help='특정 작업의 대상(target) 디렉토리.'
    )
    parser.add_argument(
        # --quarantine-dir', '--quarantine_dir', '-q', → 하나의 인자 quarantine_dir로 매핑됨
        '--quarantine-dir', '--quarantine_dir', '-qrt',
        type=str,
        default=None,
        help='오류 파일을 이동할 격리 디렉토리. (기본값: 소스 디렉토리 내 "quarantine" 폴더)'
    )
    parser.add_argument(
        # --execute, -exec 플래그가 있으면 dry_run은 False가 됩니다.
        # 플래그가 없으면 기본적으로 dry_run은 True입니다.
        '--execute', '-exec',
        dest='dry_run',
        action='store_false', # 이 action은 기본값을 True로 만듭니다.
        help="실제 파일 작업을 수행합니다. 이 플래그가 없으면 기본적으로 테스트 실행(Dry Run) 모드로 동작합니다."
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=False,
        help="병렬 처리를 사용하여 작업을 수행합니다 (지원하는 스크립트만 해당)."
    )
    parser.add_argument(
        # --delete-top-if-empty', '--delete_top_if_empty', → 하나의 인자 delete_top_if_empty로 매핑됨
        '--delete-top-if-empty', '--delete_top_if_empty',
        action="store_true",
        help="최상위 디렉토리도 비어있을 경우 함께 삭제합니다 (remove_empty_directories.py에서 사용)."
    )
    parser.add_argument(
        'pos_args',
        nargs='*',  # 0개 이상의 위치 인자를 리스트로 받습니다.
        help='위치 기반 인자. required_args에 지정된 순서대로 매핑됩니다.'
    )

    # 1. 파서에 정의된 모든 인자의 기본값을 미리 추출합니다.
    # 이후 사용자 입력값과 비교하여 실제 지정된 인자만 출력하기 위해 사용됨
    default_values = {}
    option_string_to_dest = {} # '--source-dir' -> 'source_dir' 매핑용
    for action in parser._actions:
        if action.dest and action.dest != argparse.SUPPRESS:
            default_values[action.dest] = action.default
            # 옵션 문자열(예: '--source-dir')과 dest(예: 'source_dir')를 매핑합니다.
            for option in action.option_strings:
                option_string_to_dest[option] = action.dest

    # 실제 명령줄 인자를 파싱합니다.
    args = parser.parse_args()

    # --- 위치 인자(Positional Arguments) 처리 ---
    if args.pos_args:
        # 위치 인자가 required_args보다 많으면 오류
        if len(args.pos_args) > len(required_args):
            parser.error(f"위치 인자가 너무 많습니다. 필수 인자({len(required_args)}개)보다 많은 {len(args.pos_args)}개가 제공되었습니다.")

        # 위치 인자를 required_args 순서대로 매핑
        for i, pos_arg_value in enumerate(args.pos_args):
            required_arg_option = required_args[i]  # 예: '--source-dir'
            dest_name = option_string_to_dest.get(required_arg_option)

            if dest_name:
                # 플래그와 위치 인자가 동시에 사용되었는지 확인 (충돌 방지)
                if getattr(args, dest_name) is not None:
                    parser.error(f"인자 '{required_arg_option}'가 위치 기반('{pos_arg_value}')과 플래그 기반으로 동시에 지정되었습니다. 하나만 사용해주세요.")
                
                # 위치 인자 값을 해당 속성에 할당
                setattr(args, dest_name, pos_arg_value)
            else:
                # 이 경우는 required_args에 잘못된 값이 들어왔을 때 발생
                print(f"경고: 'required_args'에 지정된 '{required_arg_option}'는 유효한 인자가 아닙니다.")

    # --- 최종 필수 인자 검증 ---
    missing_args = []
    for req_arg_option in required_args:
        dest_name = option_string_to_dest.get(req_arg_option)
        if dest_name and getattr(args, dest_name) is None:
            missing_args.append(req_arg_option)

    if missing_args:
        parser.error(f"다음 필수 인자가 누락되었습니다: {', '.join(missing_args)}")

    # 로그 디렉토리 생성은 계속 유지합니다.
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # --- 파싱된 인자 출력 ---
    print("+++++++++ 파싱된 인자 +++++++++++++++++++++++++++++")

    # 출력할 인자와 레이블 매핑
    arg_display_map = {
        'root_dir': "루트 디렉토리 (--root-dir)",
        'log_dir': "로그 디렉토리 (--log-dir)",
        'log_level': "로그 레벨 (--log-level)",
        'config_path': "설정 파일경로 (--config-path)",
        'source_dir': "소스 디렉토리 (--source-dir)",
        'destination_dir': "대상 디렉토리 (--destination-dir)",
        'target_dir': "타겟 디렉토리 (--target-dir)",
        'quarantine_dir': "격리 디렉토리 (--quarantine-dir)",
        'action': "파일 처리방식 (--action)",
        # 'dry_run'은 특별 처리되므로 여기서 제외합니다.
        'parallel': "병렬 처리 모드 (--parallel)",
        'delete_top_if_empty': "최상위 빈 디렉토리 삭제",
    }

    # 사용자가 명시적으로 지정했거나, 필수 인자로 지정된 값만 출력하기 위해 필터링
    items_to_print = []

    # --execute 플래그가 사용되었는지 (dry_run == False) 먼저 확인합니다.
    # 기본값(dry_run=True)일 때는 아무것도 출력하지 않아 혼동을 줄입니다.
    if not args.dry_run: # --execute가 사용된 경우
        items_to_print.append(("실행 모드", "실제 파일 작업 수행 (--execute 사용됨)"))
    else: # 기본값인 Dry Run 모드인 경우
        items_to_print.append(("실행 모드", "테스트 실행 (Dry Run)"))

    for arg_name, current_value in vars(args).items():
        # arg_display_map에 있는 모든 유효한 인자를 출력하도록 수정합니다.
        # 이전 로직은 사용자가 명시적으로 값을 변경했거나 필수 인자인 경우에만 출력했습니다.
        if arg_name in arg_display_map:
            # '--action' 인자는 사용자가 명시적으로 입력했을 때만 표시합니다.
            if arg_name == 'action':
                # sys.argv를 직접 확인하여 사용자가 '--action' 또는 '-act'를 사용했는지 검사합니다.
                action_arg_provided = any(arg == '--action' or arg.startswith('--action=') or arg == '-act' for arg in sys.argv)
                if not action_arg_provided:
                    continue # 사용자가 입력하지 않았으면 건너뜁니다.

            label = arg_display_map[arg_name]
            # 부울 값이고 True일 때만 표시 (예: --delete-top-if-empty)
            if isinstance(current_value, bool) and current_value:
                items_to_print.append((label, "활성화"))
            # 부울이 아니고 값이 None이 아닐 때 표시
            elif not isinstance(current_value, bool) and current_value is not None:
                items_to_print.append((label, current_value))

    if not items_to_print:
        max_label_vl = 0
    else:
        max_label_vl = max(visual_length(label) for label, _ in items_to_print)

    target_value_start_column_vl = visual_length("  ") + max_label_vl + 3

    # 특수 병합 대상: log_dir + log_level
    merged = False

    i = 0
    while i < len(items_to_print):
        label_text, value = items_to_print[i]

        if label_text == arg_display_map['log_dir']:
            # log_dir 라인일 경우 → log_level과 함께 출력할지 확인
            line = ""
            prefix1 = f"  {label_text}"
            hyphens1 = "-" * (target_value_start_column_vl - visual_length(prefix1))
            text1 = f"{prefix1}{hyphens1}{value}"

            # 다음 항목이 log_level이면 병합 출력
            if i + 1 < len(items_to_print):
                next_label, next_value = items_to_print[i + 1]
                if next_label == arg_display_map['log_level']:
                    prefix2 = f"{next_label}"
                    hyphens2 = "-" * (target_value_start_column_vl - visual_length(prefix2))
                    text2 = f"{prefix2}{hyphens2}{next_value}"
                    line = f"{text1}   {text2}"
                    merged = True
                    i += 2
                    print(line)
                    continue

            # 병합 안 된 경우 단독 출력
            print(text1)
            i += 1
        elif merged and label_text == arg_display_map['log_level']:
            # 이미 위에서 log_level 출력했으면 생략
            i += 1
        else:
            prefix = f"  {label_text}"
            hyphens = "-" * (target_value_start_column_vl - visual_length(prefix))
            print(f"{prefix}{hyphens}{value}")
            i += 1
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")

    return args
