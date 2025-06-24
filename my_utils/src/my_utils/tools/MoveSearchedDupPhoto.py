# MoveDupPhto.py (Move Duplicate Photos)
"""
이 스크립트는 두 개의 지정된 소스 디렉토리(A와 B)에서 이미지 파일을 스캔하여,
내용이 동일한 중복 이미지들을 찾아냅니다. 발견된 중복 이미지들은 지정된
대상 디렉토리(C)로 이동시켜 정리합니다.

중복 판단 기준:
- 각 이미지 파일의 SHA256 해시 값을 계산하여 비교합니다.
- 해시 값이 동일한 파일들은 내용이 동일한 중복 파일로 간주됩니다.

주요 동작:
1. 명령줄 인자로 두 개의 소스 디렉토리 경로와 한 개의 대상 디렉토리 경로를 입력받습니다.
2. 각 소스 디렉토리 내의 모든 이미지 파일에 대해 SHA256 해시를 계산합니다.
3. 두 소스 디렉토리 간에 공통으로 존재하는 해시(즉, 중복된 이미지)를 식별합니다.
4. 식별된 중복 이미지들을 대상 디렉토리로 이동시킵니다. 이때, 각 중복 그룹(동일 해시를 가진 파일들)은
   대상 디렉토리 내에 별도의 하위 디렉토리로 그룹화되어 저장됩니다.
5. 처리 과정 및 결과를 로그 파일에 기록합니다.
"""

import sys
from pathlib import Path
import shutil
import argparse
from datetime import datetime
import uuid # 고유 이름 생성을 위해 uuid 모듈 임포트
import copy
import hashlib
from typing import Optional, Dict, List, Any # 타입 힌트 호환성을 위해 typing 모듈 임포트

# shared_utils 패키지에서 configger 클래스 가져오기
# shared_utils 프로젝트의 src/utility/configger.py에 configger 클래스가 있다고 가정
# object_detector.py 파일 내 임포트 구문
# 사용자 정의 유틸리티 모듈 임포트
try:
    from my_utils.config_utils.SimpleLogger import logger, calc_digit_number, get_argument, visual_length # type: ignore
    # from my_utils.object_utils.photo_utils import compute_sha256_from_file
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)
# 이 파일 내에서 직접적인 로깅은 최소화하고, 호출하는 쪽에서 로깅을 처리한다고 가정합니다.
# 필요시 logger 객체를 함수 인자로 받거나 전역 로거를 사용할 수 있습니다.


# 지원하는 이미지 확장자 목록 (소문자로 통일)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

# 처리 상태를 기록하기 위한 기본 템플릿
DEFAULT_STATUS_TEMPLATE = {
    "images_scanned_sorc_dir": {"value": 0, "msg": "기준 디렉토리 A에서 스캔된 이미지 수"},
    "images_scanned_trgt_dir": {"value": 0, "msg": "중복을 찾을 디렉토리 B에서 스캔된 이미지 수"},
    "hashes_calculated_sorc_dir": {"value": 0, "msg": "기준 디렉토리 A에서 계산된 해시 수"},
    "hashes_calculated_trgt_dir": {"value": 0, "msg": "중복을 찾을 디렉토리 B에서 계산된 해시 수"},
    "unique_hashes_sorc_dir": {"value": 0, "msg": "기준 디렉토리 A의 고유 이미지 해시 수"},
    "unique_hashes_trgt_dir": {"value": 0, "msg": "중복을 찾을 디렉토리 B의 고유 이미지 해시 수"},
    "duplicate_groups_found": {"value": 0, "msg": "발견된 중복 이미지 그룹 수 (A와 B 사이)"},
    "total_duplicate_images_processed": {"value": 0, "msg": "처리된 총 중복 이미지 파일 수 (A+B)"},
    "images_moved_to_c": {"value": 0, "msg": "중복된 사진을 모으는곳 C 디렉토리로 성공적으로 이동된 이미지 수"},
    "source_files_removed_as_redundant": {"value": 0, "msg": "중복으로 인해 원본 위치에서 삭제된 파일 수 (이동되지 않은 경우)"},
    "move_errors": {"value": 0, "msg": "파일 이동/삭제 중 오류 수"},
    "subdirectories_created_in_c": {"value": 0, "msg": "C 디렉토리에 생성된 하위 디렉토리 수"},
}

def calculate_sha256(file_path: Path) -> Optional[str]:
    """
    주어진 파일 경로에 대해 SHA256 해시 값을 계산하여 문자열로 반환합니다.

    Args:
        file_path (Path): 해시를 계산할 파일의 경로.

    Returns:
        str | None: 계산된 SHA256 해시 값 (16진수 문자열). 파일 읽기 오류 시 None을 반환합니다.
    """
    sha256_hash = hashlib.sha256()
    try:
        # 파일을 바이너리 읽기 모드('rb')로 열어서 처리
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block) # 파일 내용을 조금씩 읽어 해시 업데이트
        return sha256_hash.hexdigest()
    except IOError as e:
        logger.error(f"파일 읽기 오류 {file_path}: {e}")
        return None

def collect_image_hashes(
        directory_path: Path,
        status: dict, 
        status_key_scanned: str, 
        status_key_hashed: str, 
        status_key_unique: str
    ) -> Dict[str, List[Path]]: # 타입 힌트 수정
    """
    지정된 디렉토리 및 하위 디렉토리에서 이미지 파일을 찾아 각 파일의 SHA256 해시를 계산합니다.
    결과로 해시 값을 키로, 해당 해시를 가진 파일 경로 리스트를 값으로 하는 딕셔너리를 반환합니다.
    처리 과정에 대한 통계는 입력된 status 딕셔너리에 업데이트됩니다.

    Args:
        directory_path (Path): 이미지 파일을 스캔할 디렉토리 경로.
        status (dict): 처리 통계를 기록할 딕셔너리.
        status_key_scanned (str): 스캔된 총 이미지 파일 수를 기록할 status 딕셔너리의 키.
        status_key_hashed (str): 해시가 성공적으로 계산된 파일 수를 기록할 status 딕셔너리의 키.
        status_key_unique (str): 발견된 고유 해시의 수를 기록할 status 딕셔너리의 키.

    Returns:
        dict[str, list[Path]]: 해시 값을 키로, 해당 해시를 가진 파일 경로(Path 객체) 리스트를 값으로 하는 딕셔너리.
                                디렉토리가 유효하지 않으면 빈 딕셔너리를 반환합니다.
    """
    hashes: Dict[str, List[Path]] = {}
    if not directory_path.is_dir():
        logger.error(f"디렉토리를 찾을 수 없습니다: {directory_path}")
        return hashes

    logger.info(f"{directory_path} 에서 이미지 스캔 중...")
    # 디렉토리 내 모든 파일을 재귀적으로 탐색하여 지원하는 확장자를 가진 파일만 필터링
    image_files = [p for p in directory_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
    status[status_key_scanned]["value"] = len(image_files) # 스캔된 이미지 파일 수 업데이트
    
    digit_width = calc_digit_number(len(image_files)) # 로그 출력 시 정렬을 위한 자릿수 계산

    for idx, img_path in enumerate(image_files):
        logger.debug(f"[{idx+1:{digit_width}}/{len(image_files)}] 해시 계산 중: {img_path.name}")
        file_hash = calculate_sha256(img_path)
        if file_hash:
            status[status_key_hashed]["value"] += 1 # 해시 계산 성공 수 업데이트
            if file_hash not in hashes:
                hashes[file_hash] = [] # 새 해시 값이면 리스트 초기화 (defaultdict 사용 시 불필요)
            hashes[file_hash].append(img_path)
    
    status[status_key_unique]["value"] = len(hashes)
    logger.info(f"{directory_path} 에서 {len(image_files)}개 이미지 스캔, {len(hashes)}개의 고유 해시 발견.")
    return hashes

def move_internal_duplicates_logic(
        trgt_dir: Path, 
        dest_dir: Path, 
        dry_run=False
        ):
    """
    지정된 단일 소스 디렉토리(trgt_dir) 내에서 중복된 사진을 찾아
    대상 디렉토리(dest_dir)로 이동시키는 로직을 수행합니다.

    Args:
        trgt_dir (Path): 중복을 검사할 소스 이미지 디렉토리.
        dest_dir (Path): 중복 이미지를 이동시킬 대상 디렉토리.
        dry_run (bool): True이면 실제 파일 이동 없이 로그만 남깁니다.

    Returns:
        dict: 처리 과정에 대한 통계 정보를 담은 딕셔너리.
    """
    status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)

    # 1. 입력 디렉토리 유효성 검사
    if not trgt_dir.is_dir():
        logger.error(f"중복된 사진이 있는지 찾을곳(trgt_dir)을 찾을 수 없습니다: {trgt_dir}")
        return status

    # 2. 출력 디렉토리 C 생성
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"중복된 사진을 옮겨 둘곳(dest_dir): {dest_dir}")
    except OSError as e:
        logger.error(f"대상 디렉토리(C)를 생성할 수 없습니다 ({dest_dir}): {e}")
        return status

    # 3. 소스 디렉토리에서 이미지 해시 수집
    hashes_trgt = collect_image_hashes(
        trgt_dir, 
        status,
        "images_scanned_trgt_dir", 
        "hashes_calculated_trgt_dir", 
        "unique_hashes_trgt_dir"
    )

    # 4. 디렉토리 내의 중복 해시(중복된 이미지) 찾기
    # 해시에 대해 2개 이상의 파일 경로가 있는 그룹만 필터링합니다.
    duplicate_groups = {h: paths for h, paths in hashes_trgt.items() if len(paths) > 1}
    status["duplicate_groups_found"]["value"] = len(duplicate_groups)

    if not duplicate_groups:
        logger.info(f"{trgt_dir} 내에서 중복된 이미지를 찾지 못했습니다.")
        return status
    
    logger.info(f"{len(duplicate_groups)}개의 중복 이미지 그룹(해시 기준)을 찾았습니다. 파일 이동을 시작합니다...")

    # 5. 중복 그룹별로 파일 이동 처리
    group_digit_width = calc_digit_number(len(duplicate_groups))
    for group_idx, (h_val, files_in_group) in enumerate(duplicate_groups.items()):
        status["total_duplicate_images_processed"]["value"] += len(files_in_group)

        # 해시값 자체를 하위 디렉토리 이름으로 사용
        dest_group_dir = dest_dir / h_val
        
        try:
            if not dry_run:
                dest_group_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"하위 디렉토리 생성 실패 {dest_group_dir}: {e}")
            status["move_errors"]["value"] += len(files_in_group)
            continue
        
        logger.info(f"[{group_idx+1:{group_digit_width}}/{len(duplicate_groups)}] 그룹 '{h_val}' 처리 중 ({len(files_in_group)}개 파일 처리 예정)")

        # 그룹 내 모든 중복 파일을 대상 디렉토리로 이동
        for file_to_move in files_in_group:
            dest_file_path = dest_group_dir / file_to_move.name
            
            # 파일명 충돌 처리: 대상 경로에 파일이 이미 존재하면 고유한 이름으로 변경
            if not dry_run and dest_file_path.exists():
                new_name = f"{file_to_move.stem}_{uuid.uuid4().hex[:8]}{file_to_move.suffix}"
                dest_file_path = dest_group_dir / new_name
                logger.warning(f"  파일명 충돌: '{file_to_move.name}'이(가) 대상 폴더에 이미 존재합니다. 새 이름 '{new_name}'(으)로 저장합니다.")

            try:
                if dry_run:
                    logger.info(f"(Dry Run) 이동 예정: '{file_to_move}' -> '{dest_file_path}'")
                else:
                    shutil.move(str(file_to_move), str(dest_file_path))
                logger.info(f"  이동: '{file_to_move}' -> '{dest_file_path}'")
                status["images_moved_to_c"]["value"] += 1
            except Exception as e_move:
                logger.error(f"  파일 이동 오류 ('{file_to_move}'): {e_move}")
                status["move_errors"]["value"] += 1
    
    status["subdirectories_created_in_c"]["value"] = len(duplicate_groups)
    return status

def move_duplicate_photos_logic(
        sorc_dir: Path, 
        trgt_dir: Path, 
        dest_dir: Path, 
        dry_run=False
        ):
    """
    두 개의 소스 디렉토리(src_dir_path, dst_dir_path)에서 중복된 사진을 찾아
    대상 디렉토리(tst_dir_path)로 이동시키는 핵심 로직을 수행합니다.

    Args:
        src_dir_path (Path): 첫 번째 소스 이미지 디렉토리 (A).
        tgt_dir_path (Path): 두 번째 소스 이미지 디렉토리 (B).
        dst_dir_path (Path): 중복 이미지를 이동시킬 대상 디렉토리 (C).

    Returns:
        dict: 처리 과정에 대한 통계 정보를 담은 딕셔너리.
    """
    status = copy.deepcopy(DEFAULT_STATUS_TEMPLATE)

    # 1. 입력 디렉토리 유효성 검사
    if not sorc_dir.is_dir():
        logger.error(f"기준사진이 있는 곳(srrc_dir)을 찾을 수 없습니다: {sorc_dir}")
        return status
    if not trgt_dir.is_dir():
        logger.error(f"중복된 사진이 있는지 찾을곳(trgt_dir)를 찾을 수 없습니다: {trgt_dir}")
        return status

    # 2. 출력 디렉토리 C 생성
    try:
        dest_dir.mkdir(parents=True, exist_ok=True) # 대상 디렉토리 생성 (이미 존재하면 무시)
        logger.info(f"중복된 사진을 옮겨 둘곳(dest_dir): {dest_dir}")
    except OSError as e:
        logger.error(f"대상 디렉토리(C)를 생성할 수 없습니다 ({dest_dir}): {e}") # 오타 수정
        return status

    # 3. 첫 번째 소스 디렉토리(A)에서 이미지 해시 수집
    hashes_sorc = collect_image_hashes(
                sorc_dir, status,
                "images_scanned_sorc_dir", 
                "hashes_calculated_sorc_dir", 
                "unique_hashes_sorc_dir"
                )
    
    # 4. 두 번째 소스 디렉토리(B)에서 이미지 해시 수집
    hashes_trgt = collect_image_hashes(
        trgt_dir, 
        status,
        "images_scanned_trgt_dir", 
        "hashes_calculated_trgt_dir", 
        "unique_hashes_trgt_dir"
        ) # 오타 수정: status_key_scanned_b -> images_scanned_trgt_dir

    # 5. 디렉토리 A와 B 사이의 공통 해시(중복된 이미지) 찾기
    common_hashes = set(hashes_sorc.keys()) & set(hashes_trgt.keys())
    status["duplicate_groups_found"]["value"] = len(common_hashes)

    if not common_hashes:
        logger.info("Sorc와 target 사이에 중복된 이미지를 찾지 못했습니다.")
        return status
    
    logger.info(f"{len(common_hashes)}개의 중복 이미지 그룹(해시 기준)을 찾았습니다. 파일 이동을 시작합니다...")

    # 6. 공통 해시 그룹별로 파일 이동 처리
    group_digit_width = calc_digit_number(len(common_hashes)) # 로그 출력용 자릿수
    for group_idx, h_val in enumerate(common_hashes):
        paths_from_sorc = hashes_sorc.get(h_val, [])
        paths_from_trgt = hashes_trgt.get(h_val, [])
        # 해당 해시를 가진 A와 B의 모든 파일 (통계용)
        all_files_in_group_for_stats = paths_from_sorc + paths_from_trgt
        status["total_duplicate_images_processed"]["value"] += len(all_files_in_group_for_stats)

        if not paths_from_trgt: # 대상 디렉토리(B)에 해당 해시의 파일이 없으면 건너<0xEB><0><0x8F><0xBB>니다.
            logger.debug(f"  해시 {h_val}에 대해 대상 디렉토리(B)에 파일이 없습니다. 건너<0xEB><0><0x8F><0xBB>니다.")
            continue

        # 대표 파일명을 사용하여 하위 디렉토리 이름 결정 (첫 번째 파일의 이름 사용) + 고유 ID 추가
        # all_files_in_group_for_stats는 paths_from_trgt가 비어있지 않으므로 최소 1개 이상의 요소를 가집니다.
        representative_file_path = all_files_in_group_for_stats[0]
        #  subdir_name_stem = f"{representative_file_path.stem}_{uuid.uuid4().hex[:8]}"     # 확장자 제외한 파일명으로 디렉토리 만들기
        subdir_name_stem = h_val  # 해시값 자체를 디렉토리 이름으로 사용
        dest_group_dir = dest_dir / subdir_name_stem # 대상 디렉토리 C 아래에 하위 디렉토리 경로 생성
        
        # 실제 생성된 디렉토리 추적용 set 정의 및 카운팅

        try:
            dest_group_dir.mkdir(parents=True, exist_ok=True)
            # 이전에 생성되지 않았다면 카운트 (이미 존재할 수도 있음 - 다른 해시 그룹이 같은 stem을 가질 경우 드물게 발생 가능)
            # 좀 더 정확하려면 생성된 디렉토리 set을 관리해야 하지만, 여기서는 단순화.
            # status["subdirectories_created_in_c"]["value"] += 1 # 이 방식은 중복 카운트 가능
        except OSError as e:
            logger.error(f"하위 디렉토리 생성 실패 {dest_group_dir}: {e}")
            status["move_errors"]["value"] += len(paths_from_trgt) # 이 그룹의 대상 디렉토리(B) 파일 이동 실패로 간주
            continue
        
        logger.info(f"[{group_idx+1:{group_digit_width}}/{len(common_hashes)}] 그룹 '{subdir_name_stem}' 처리 중 (대상 디렉토리(B)에서 {len(paths_from_trgt)}개 파일 처리 예정)")

        moved_filenames_in_dest_subdir = set() # 현재 그룹의 대상 하위 디렉토리에 이미 이동된 파일명들을 추적 (파일명 충돌 방지)

        # 대상 디렉토리(B)의 파일만 처리합니다.
        for file_from_trgt_dir in paths_from_trgt:
            dest_file_path = dest_group_dir / file_from_trgt_dir.name # 최종 이동될 경로
            
            try:
                if file_from_trgt_dir.name not in moved_filenames_in_dest_subdir:
                    # 대상 하위 디렉토리에 아직 동일한 이름의 파일이 이동되지 않았다면 이동
                    if dry_run:
                        logger.info(f"(Dry Run) 이동 예정 (B에서 C로): '{file_from_trgt_dir}' -> '{dest_file_path}'")
                    else:
                        shutil.move(str(file_from_trgt_dir), str(dest_file_path))
                    logger.info(f"  이동 (B에서 C로): '{file_from_trgt_dir}' -> '{dest_file_path}'")
                    moved_filenames_in_dest_subdir.add(file_from_trgt_dir.name)
                    status["images_moved_to_c"]["value"] += 1
                else:
                    # 대상 디렉토리(B) 내에서 같은 해시 그룹에 속하지만 파일명이 동일하여,
                    # 하나는 C로 이동되고 나머지는 B에서 삭제되는 경우입니다.
                    logger.info(f"  중복 파일명 (B 내): '{file_from_trgt_dir.name}' (대상 폴더 '{dest_group_dir.name}'에 이미 동일 해시의 다른 B파일 존재). 원본 '{file_from_trgt_dir}'(B에 위치) 삭제 중...")
                    if dry_run:
                        logger.info(f"(Dry Run) 삭제 예정 (B에서): '{file_from_trgt_dir}'")
                    else:
                        file_from_trgt_dir.unlink() # 원본 파일(B에 위치) 삭제
                    status["source_files_removed_as_redundant"]["value"] += 1
            except Exception as e_move:
                logger.error(f"  파일 처리 오류 (B의 파일 '{file_from_trgt_dir}'): {e_move}")
                status["move_errors"]["value"] += 1
    
    # 생성된 하위 디렉토리 수 업데이트 (실제 생성된 유니크한 디렉토리 수)
    # common_hashes의 각 항목이 고유한 subdir_name_stem을 만든다고 가정 (대부분의 경우)
    # 더 정확하게는 생성된 target_group_dir의 set 크기를 사용해야 함.
    # 여기서는 발견된 중복 그룹 수로 대략적인 값을 사용합니다.
    status["subdirectories_created_in_c"]["value"] = len(common_hashes) 

    return status # 최종 통계 반환

def run_main():
    """
    스크립트의 메인 실행 함수입니다.
    명령줄 인자 파싱, 로깅 설정, 핵심 로직 호출 및 최종 통계 출력을 담당합니다.
    """
    # get_argument()는 프로젝트 내 유틸리티 함수로 가정.
    # fallback은 my_utils가 없을 경우를 대비한 것입니다.
    parsed_args = get_argument()
    
    script_name = Path(__file__).stem # 현재 스크립트 파일 이름 (확장자 제외)
    
    # 로거 설정 (my_utils.SimpleLogger 사용 또는 표준 로깅 사용)
    # logger.setup은 프로젝트 내 유틸리티로 가정.
    if hasattr(logger, "setup"):
        date_str = datetime.now().strftime("%y%m%d_%H%M")
        log_file_name = f"{script_name}_{date_str}.log"
        # parsed_args.log_dir이 Path 객체가 아닐 수 있으므로 Path로 변환
        # 로그 파일 경로가 이미 핸들러에 추가되었는지 확인하여 중복 추가 방지
        full_log_path = Path(parsed_args.log_dir) / log_file_name
        logger.setup(
            logger_path=full_log_path, # 로거 이름 설정
            min_level=parsed_args.log_level.upper(), # 로그 레벨은 대문자로
            include_function_name=True,
            pretty_print=True # pretty_print 옵션이 있다면 사용
        )
    else:
        # my_utils.SimpleLogger가 없을 경우 표준 로깅으로 파일 핸들러 설정
        log_dir_path = Path(parsed_args.log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True) # 로그 디렉토리 생성
        date_str = datetime.now().strftime("%y%m%d")
        log_file_name = f"{script_name}_{date_str}.log"
        full_log_path = log_dir_path / log_file_name

        file_handler = logging.FileHandler(str(full_log_path), encoding='utf-8')
        log_level_std = getattr(logging, parsed_args.log_level.upper(), logging.INFO)
        file_handler.setLevel(log_level_std)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # 동일 경로의 FileHandler가 이미 있는지 확인하여 중복 추가 방지 (basicConfig에 의해 이미 추가되었을 수 있음)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(full_log_path) for h in logger.handlers):
             logger.addHandler(file_handler)
             logger.setLevel(log_level_std) # logger 인스턴스의 레벨도 설정
    
    logger.info(f"애플리케이션 ({script_name}) 시작")
    logger.info(f"명령줄 인자: {vars(parsed_args)}")

    # 필수 디렉토리 인자 확인
    if parsed_args.target_dir is None:
        logger.error("중복을 찾을 디렉토리 (--target-dir or -tgt)가 제공되지 않았습니다. 스크립트를 종료합니다.")
        sys.exit(1)
    if parsed_args.destination_dir is None:
        logger.error("결과물 디렉토리 (--destination-dir or -dst)가 제공되지 않았습니다. 스크립트를 종료합니다.")
        sys.exit(1)

    # 명령줄 인자로부터 디렉토리 경로를 Path 객체로 변환 및 절대 경로로 해석
    tst_dir = Path(parsed_args.target_dir).expanduser().resolve() # 중복을 찾을 디렉토리 (B)
    dst_dir = Path(parsed_args.destination_dir).expanduser().resolve() # 결과물 디렉토리 (C)

    # dry_run 인자 값 가져오기
    dry_run_mode = getattr(parsed_args, 'dry_run', False) # get_argument_fallback에 dry_run이 없으면 기본값 False

    try:
        if parsed_args.source_dir:
            # 모드 1: 두 디렉토리(A, B) 간의 중복 비교
            logger.info("모드 1: 두 디렉토리 간의 중복 파일을 검색합니다 (A vs B).")
            src_dir = Path(parsed_args.source_dir).expanduser().resolve()
            final_status = move_duplicate_photos_logic(
                    sorc_dir=src_dir, 
                    trgt_dir=tst_dir, 
                    dest_dir=dst_dir, 
                    dry_run=dry_run_mode
            )
        else:
            # 모드 2: 단일 디렉토리(B) 내의 중복 비교
            logger.info("모드 2: 단일 디렉토리 내의 중복 파일을 검색합니다 (B vs B).")
            logger.warning("기준 디렉토리(--source-dir)가 지정되지 않았습니다. --target-dir 내에서 중복을 찾습니다.")
            final_status = move_internal_duplicates_logic(
                    trgt_dir=tst_dir,
                    dest_dir=dst_dir,
                    dry_run=dry_run_mode
            )

        logger.warning("--- 중복 사진 이동 처리 통계 ---")
        # 통계 메시지 중 가장 긴 것을 기준으로 출력 너비 조절
        # max_msg_len = max(len(v["msg"]) for v in DEFAULT_STATUS_TEMPLATE.values()) if DEFAULT_STATUS_TEMPLATE else 20
        # visual_length를 사용하여 시각적 최대 길이 계산
        max_visual_msg_len = 0
        if DEFAULT_STATUS_TEMPLATE:
            max_visual_msg_len = max(visual_length(v["msg"]) for v in DEFAULT_STATUS_TEMPLATE.values())
        
        # 통계 출력 시 사용할 숫자 너비 계산 (가장 큰 값 기준)
        max_val_for_width = 0
        if final_status and "images_scanned_sorc_dir" in final_status and final_status["images_scanned_sorc_dir"]["value"] > 0 : # final_status가 비어있지 않고 값이 있다면
             max_val_for_width = max(s_item["value"] for s_item in final_status.values())
        
        digit_width_stats = calc_digit_number(max_val_for_width) if "calc_digit_number" in globals() and callable(calc_digit_number) else 5
        
        for key, data in final_status.items():
            # DEFAULT_STATUS_TEMPLATE에 해당 키가 있는지 확인
            msg = DEFAULT_STATUS_TEMPLATE.get(key, {}).get("msg", key.replace("_", " ").capitalize())
            value = data["value"]
            # f-string의 기본 정렬은 문자 개수 기준이므로, visual_length에 맞춰 수동으로 패딩 추가
            padding_spaces = max(0, max_visual_msg_len - visual_length(msg))
            logger.warning(f"{msg}{'-' * padding_spaces} : {value:>{digit_width_stats}}")
        logger.warning("-----------------------------")

    except Exception as e:
        logger.error(f"애플리케이션 실행 중 최상위 오류 발생: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"애플리케이션 ({script_name}) 종료{ ' (Dry Run 모드)' if dry_run_mode else ''}") # Dry Run 모드 표시
        if hasattr(logger, "shutdown"): # logger.shutdown 함수가 있다면 호출
            pass # 표준 로깅에는 shutdown이 없음. SimpleLogger에만 있다면 호출.
        
if __name__ == "__main__":
    # 스크립트가 직접 실행될 때 run_main() 함수 호출
    run_main()
