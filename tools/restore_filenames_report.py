"""
주요 기능 및 특징:

맵 파일 로드 및 분석: organize_photos_map.json 파일을 읽어, 원본 파일명과 이동된 파일명이 다른 경우를 찾습니다.
이름 변경 패턴 확인: 정규식(re.fullmatch)을 사용하여 이동된 파일명이 원본이름_숫자.확장자 형태인지 정확하게 확인합니다.
복구 경로 충돌 확인: 원래 이름으로 되돌릴 경로에 이미 다른 파일이 있는지 os.path.exists()로 확인합니다.
충돌 시: 사용자에게 경고하고 해당 파일은 건너<0xEB><0><0x8F><0xBC>니다. (덮어쓰기 방지)
사용자 확인 (기본): 각 복구 대상 파일에 대해 사용자에게 복구 여부를 묻습니다 (y/n).
자동 확인 옵션 (--yes 또는 -y): 명령줄에서 이 옵션을 주면 확인 질문 없이, 복구 경로에 충돌이 없는 경우 자동으로 파일명을 변경합니다. 주의해서 사용해야 합니다.
파일명 변경: os.rename()을 사용하여 파일명을 원래대로 변경합니다.
맵 데이터 업데이트: 파일명 복구에 성공하면, 메모리에 로드된 map_data의 해당 항목에서 '이동된 경로' 정보를 업데이트합니다.
맵 파일 저장: 모든 파일 처리가 끝난 후, 맵 데이터에 변경 사항이 있었으면 save_map_file 함수를 호출하여 업데이트된 내용을 JSON 파일에 저장합니다.
사용 방법:

위 코드를 restore_filenames.py와 같은 이름으로 저장합니다.

터미널에서 스크립트를 실행합니다. --map_file 인자는 필수입니다.

각 파일마다 확인하며 복구:
bash
python restore_filenames.py --map_file organize_photos_map.json
확인 없이 자동으로 복구 (충돌 없는 경우만):
bash
python restore_filenames.py --map_file organize_photos_map.json --yes
(또는 python restore_filenames.py --map_file organize_photos_map.json -y)
이 스크립트를 사용하면 맵 파일 정보를 기반으로 안전하게 파일명을 원래대로 복구할 수 있습니다.
"""
# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import argparse
import logging
import shutil # shutil 추가 (맵 파일 백업용)
from typing import Optional, Dict, List, Any # 타입 힌트

# --- 로거 설정 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ==============================================================================
# 맵 파일 로드/저장 함수 (review_photos.py 와 동일)
# ==============================================================================
def load_map_file(map_file_path: str) -> Dict[str, Dict[str, Any]]:
    """JSON 맵 파일을 로드합니다. 파일이 없거나 손상되었으면 빈 딕셔너리를 반환합니다."""
    if os.path.exists(map_file_path):
        try:
            with open(map_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    logger.info(f"맵 파일 로드 완료: {map_file_path}")
                    return data
                else:
                    logger.error(f"맵 파일 내용이 유효한 JSON 객체가 아닙니다: {map_file_path}")
                    return {}
        except json.JSONDecodeError:
            logger.error(f"맵 파일 형식이 잘못되었습니다: {map_file_path}")
            try:
                shutil.copyfile(map_file_path, map_file_path + ".corrupted")
                logger.info(f"손상된 맵 파일을 백업했습니다: {map_file_path}.corrupted")
            except Exception as backup_e:
                logger.error(f"손상된 맵 파일 백업 실패: {backup_e}")
            return {}
        except IOError as e:
            logger.error(f"맵 파일 읽기 오류 '{map_file_path}': {e}")
            return {}
        except Exception as e:
            logger.error(f"맵 파일 로드 중 예상치 못한 오류 발생 '{map_file_path}': {e}")
            return {}
    else:
        logger.error(f"맵 파일을 찾을 수 없습니다: {map_file_path}")
        return {}

def save_map_file(map_data: Dict[str, Dict[str, Any]], map_file_path: str):
    """맵 데이터를 JSON 파일로 안전하게 저장합니다."""
    try:
        temp_map_file_path = map_file_path + ".tmp"
        with open(temp_map_file_path, 'w', encoding='utf-8') as f:
            json.dump(map_data, f, ensure_ascii=False, indent=2)
        os.replace(temp_map_file_path, map_file_path)
        logger.debug(f"맵 파일 저장 완료: {map_file_path}")
    except IOError as e:
        logger.error(f"맵 파일 쓰기 오류 '{map_file_path}': {e}")
    except Exception as e:
        logger.error(f"맵 파일 저장 중 예상치 못한 오류 발생 '{map_file_path}': {e}")
        if os.path.exists(temp_map_file_path):
            try:
                os.remove(temp_map_file_path)
            except OSError:
                logger.warning(f"임시 맵 파일 삭제 실패: {temp_map_file_path}")

# ==============================================================================
# 파일명 복구 함수
# ==============================================================================
def restore_renamed_files(map_file_path: str, auto_confirm: bool = False):
    """
    맵 파일을 분석하여 이름 충돌로 변경된 파일(_숫자 추가)을 찾아
    원래 이름으로 복구를 시도합니다.
    """
    map_data = load_map_file(map_file_path)
    if not map_data:
        logger.info("맵 데이터가 비어 있어 복구를 진행할 수 없습니다.")
        return

    files_to_restore = [] # 복구 대상 파일 정보를 저장할 리스트

    logger.info("맵 파일을 분석하여 이름이 변경된 파일(_숫자 추가)을 찾습니다...")

    for file_hash, hash_data in map_data.items():
        originals_map = hash_data.get("originals", {})
        for original_path, moved_path in originals_map.items():
            if not moved_path or not os.path.exists(moved_path): # 이동된 경로가 없거나 파일이 없으면 건너<0xEB><0><0x8F><0xBC>기
                continue

            original_filename = os.path.basename(original_path)
            moved_filename = os.path.basename(moved_path)

            # 원본 파일명과 이동된 파일명이 다른 경우 검사
            if original_filename != moved_filename:
                original_name_part, original_ext = os.path.splitext(original_filename)
                moved_name_part, moved_ext = os.path.splitext(moved_filename)

                # 확장자가 동일하고, 이동된 이름이 "원본이름_숫자" 형태인지 확인
                if original_ext.lower() == moved_ext.lower():
                    match = re.fullmatch(rf"{re.escape(original_name_part)}_(\d+)", moved_name_part)
                    if match:
                        potential_original_dest_path = os.path.join(os.path.dirname(moved_path), original_filename)
                        files_to_restore.append({
                            "hash": file_hash,
                            "original_path": original_path,
                            "moved_path": moved_path,
                            "original_filename": original_filename,
                            "potential_original_dest_path": potential_original_dest_path
                        })

    # --- 복구 대상 보고 및 확인 ---
    if not files_to_restore:
        logger.info("파일 이름 충돌로 인해 이름이 변경된 파일(_숫자 추가)을 찾지 못했습니다.")
        return

    logger.info(f"총 {len(files_to_restore)}개의 파일 이름 복구를 시도할 수 있습니다.")
    logger.warning("주의: 복구하려는 원래 이름으로 다른 파일이 이미 존재하면 복구되지 않습니다.")

    map_changed = False # 맵 데이터 변경 여부 플래그
    restored_count = 0
    skipped_count = 0
    error_count = 0

    for item in files_to_restore:
        moved_path = item['moved_path']
        potential_original_dest_path = item['potential_original_dest_path']
        original_path = item['original_path']
        file_hash = item['hash']

        print("-" * 60)
        print(f"  현재 경로: {moved_path}")
        print(f"  복구 경로: {potential_original_dest_path}")
        print(f"  (원본 경로: {original_path})")

        # 복구 경로에 파일이 이미 존재하는지 확인
        if os.path.exists(potential_original_dest_path):
            logger.warning(f"  [건너<0xEB><0><0x8F><0xBC>] 복구 경로에 이미 다른 파일이 존재합니다: {potential_original_dest_path}")
            skipped_count += 1
            continue

        # 사용자 확인 (auto_confirm이 False일 경우)
        confirm = 'y' # 기본값을 'y'로 설정 (auto_confirm=True일 때 사용)
        if not auto_confirm:
            confirm = input("  이 파일의 이름을 원래대로 복구하시겠습니까? (y/n): ").strip().lower()

        if confirm == 'y':
            try:
                os.rename(moved_path, potential_original_dest_path)
                logger.info(f"  [성공] 파일 이름 복구: '{os.path.basename(moved_path)}' -> '{os.path.basename(potential_original_dest_path)}'")
                restored_count += 1

                # 맵 데이터 업데이트
                if file_hash in map_data and "originals" in map_data[file_hash] and original_path in map_data[file_hash]["originals"]:
                    map_data[file_hash]["originals"][original_path] = potential_original_dest_path
                    map_changed = True
                    logger.debug(f"  맵 데이터 업데이트: {original_path} -> {potential_original_dest_path}")

            except OSError as e:
                logger.error(f"  [오류] 파일 이름 변경 실패: {e}")
                error_count += 1
            except Exception as e:
                 logger.error(f"  [오류] 파일 이름 변경 중 예상치 못한 오류 발생: {e}")
                 error_count += 1
        else:
            logger.info("  [취소] 사용자가 복구를 취소했습니다.")
            skipped_count += 1

    print("-" * 60)
    logger.info("파일명 복구 시도 결과:")
    logger.info(f"  - 성공: {restored_count}개")
    logger.info(f"  - 건너<0xEB><0><0x8F><0xBC> (이름 충돌 또는 사용자 취소): {skipped_count}개")
    logger.info(f"  - 오류: {error_count}개")

    # 맵 데이터가 변경되었으면 저장
    if map_changed:
        logger.info("변경된 맵 데이터를 저장합니다...")
        save_map_file(map_data, map_file_path)
        logger.info("맵 파일 업데이트 완료.")
    else:
        logger.info("맵 데이터 변경 사항이 없습니다.")

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="사진 정리 맵 파일을 분석하여 이름 충돌로 변경된 파일(_숫자 추가)을 원래 이름으로 복구합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--map_file", type=str, required=True,
                        help="분석 및 업데이트할 맵 파일 경로 (_map.json 파일) (필수)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="각 파일 복구 시 확인 질문을 생략하고 자동으로 진행합니다. (주의해서 사용!)")
    parser.add_argument("--debug", action="store_true",
                        help="디버그 레벨 로그를 활성화합니다.")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)

    if args.debug:
        logger.info("디버그 로깅 활성화됨.")

    try:
        map_file_abs = os.path.abspath(os.path.expanduser(args.map_file))
    except Exception as e:
        logger.error(f"맵 파일 경로 처리 중 오류 발생: {e}")
        sys.exit(1)

    logger.info("="*20 + " 파일명 복구 시작 " + "="*20)
    logger.info(f"맵 파일: {map_file_abs}")
    if args.yes:
        logger.warning("자동 확인(--yes) 모드로 실행됩니다. 복구 경로에 파일이 없으면 자동으로 이름이 변경됩니다.")
    logger.info("=" * 50 )

    restore_renamed_files(map_file_abs, auto_confirm=args.yes)

    logger.info("파일명 복구 작업이 완료되었습니다.")
