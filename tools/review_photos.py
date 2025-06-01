"""
# interactive_review 함수 로직 변경:
# 스크립트 시작 시 map_data를 순회하여 len(originals_map) > 1인 항목(중복 세트)만 찾습니다.
# 각 중복 세트에 대해 (대표 원본 경로, 해시, 중복 파일 수) 튜플을 만들어 duplicate_sets 리스트에 저장합니다.
# duplicate_sets 리스트를 대표 원본 경로 기준으로 정렬합니다.
# 사용자에게 duplicate_sets 목록을 번호와 함께 보여줍니다.
# 사용자가 번호를 입력하면, 해당 번호에 맞는 중복 세트의 해시(selected_hash)를 가져옵니다.
# map_data에서 selected_hash를 사용하여 해당 세트의 모든 파일 정보(originals_map)를 가져옵니다.
# 세트에 포함된 모든 파일의 원본 경로와 이동된 경로를 출력합니다.
# 세트에 포함된 모든 파일 중 실제로 존재하는 이동된 파일들의 경로를 image_paths_to_display 리스트에 추가합니다.
# display_images 함수를 호출하여 이미지들을 표시합니다.
"""

# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
import json
from typing import Optional, Dict, List, Any # 타입 힌트

# --- 로거 설정 ---
# 기본 로거 설정 (파일 핸들러는 main 블록에서 추가)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # 로그를 화면에도 출력
logger = logging.getLogger(__name__) # 로거 객체 생성

# --- 이미지 표시 위한 라이브러리 (필수) ---
try:
    # try:
    #     import matplotlib
    #     matplotlib.use('TkAgg') # 또는 'Qt5Agg' 등
    #     import matplotlib.pyplot as plt
    #     print("Matplotlib 백엔드:", plt.get_backend()) # 현재 백엔드 확인
    #     plt.plot([1, 2, 3])
    #     plt.title("Matplotlib Test Plot")
    #     print("테스트 플롯 표시 시도...")
    #     plt.show()
    #     print("테스트 플롯 창이 닫혔습니다.")
    # except Exception as e:
    #     print(f"Matplotlib 테스트 플롯 표시 중 오류 발생: {e}")

    import matplotlib.pyplot as plt
    from PIL import Image, UnidentifiedImageError
    # --- 한글 폰트 설정 ---
    from matplotlib import font_manager, rc
    import platform

    font_path = None # 폰트 경로 초기화
    font_name = None # 폰트 이름 초기화

    # 운영체제별 한글 폰트 경로 설정 (예시)
    if platform.system() == 'Windows':
        font_path = "c:/Windows/Fonts/malgun.ttf" # 윈도우: 맑은 고딕
    elif platform.system() == 'Darwin': # macOS
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc" # macOS: Apple SD Gothic Neo
    else: # Linux 등 기타
        # 시스템에 설치된 나눔고딕 폰트 경로를 찾거나 지정해야 함
        try:
            # fallback_to_default=True로 변경하여 나눔고딕 없으면 기본 폰트 시도
            font_path = font_manager.findfont('NanumGothic', fallback_to_default=True)
        except ValueError: # findfont가 ValueError를 발생시킬 수 있음
            logger.warning("NanumGothic 폰트를 찾지 못했습니다. 시스템 기본 폰트를 찾습니다.")
            try:
                # 시스템 폰트 목록 가져오기
                system_fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
                if system_fonts: # 폰트 목록이 비어있지 않으면
                    font_path = system_fonts[0] # 첫 번째 폰트 사용
                    logger.info(f"시스템 기본 폰트 사용: {font_path}")
                else:
                    # 시스템 폰트도 못 찾으면 경고만 하고 진행 (오류 방지)
                    logger.error("시스템에서 사용 가능한 TTF 폰트를 찾을 수 없습니다. 이미지 제목에 문자가 표시되지 않을 수 있습니다.")
                    font_path = None # 폰트 경로 없음
            except Exception as e_fallback:
                logger.error(f"시스템 기본 폰트 검색 중 오류 발생: {e_fallback}")
                font_path = None

    # 폰트 경로가 유효하면 matplotlib 설정 적용
    if font_path and os.path.exists(font_path):
        try:
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font_name) # matplotlib의 기본 폰트 설정 변경
            plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
            logger.info(f"Matplotlib 폰트를 '{font_name}'({font_path})으로 설정했습니다.")
            MATPLOTLIB_AVAILABLE = True
        except Exception as e:
            logger.error(f"Matplotlib 폰트 설정 중 오류 발생: {e}")
            logger.error("기본 폰트를 사용합니다. 한글이 깨질 수 있습니다.")
            # 폰트 설정 실패 시 MATPLOTLIB_AVAILABLE을 False로 할지 결정 필요
            # 여기서는 일단 True로 두어 이미지 자체는 표시되도록 함
            MATPLOTLIB_AVAILABLE = True
    elif font_path: # 경로는 찾았으나 파일이 없는 경우
         logger.error(f"지정된 폰트 파일을 찾을 수 없습니다: {font_path}")
         MATPLOTLIB_AVAILABLE = True # 이미지 표시는 가능하도록 유지
    else: # font_path가 None인 경우 (폰트를 전혀 못 찾음)
        logger.error("사용 가능한 폰트가 없어 Matplotlib 폰트 설정을 건너<0xEB><0><0x8F><0xBC>니다.")
        MATPLOTLIB_AVAILABLE = True # 이미지 표시는 가능하도록 유지

except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("오류: 이 스크립트를 실행하려면 'matplotlib'와 'Pillow' 라이브러리가 필요합니다.")
    print("      pip install matplotlib Pillow")
    sys.exit(1) # 라이브러리 없으면 종료


# --- 로거 설정 ---
# 기본 로거 설정 (파일 핸들러는 필요 없음, 화면 출력만)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__) # 로거 객체 생성

# ==============================================================================
# 맵 파일 로드 함수 (organize_photos.py와 동일)
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
            return {}
        except IOError as e:
            logger.error(f"맵 파일 읽기 오류 '{map_file_path}': {e}")
            return {}
        except Exception as e:
            logger.error(f"맵 파일 로드 중 예상치 못한 오류 발생 '{map_file_path}': {e}")
            return {}
    else:
        logger.error(f"맵 파일을 찾을 수 없습니다: {map_file_path}")
        return {} # 파일 없으면 빈 딕셔너리 반환

# ==============================================================================
# json파일 만드는 함수
# ==============================================================================

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
# 이미지 표시 및 대화형 검토 함수
# ==============================================================================
def display_images(image_paths: List[str], reference_path: Optional[str] = None):
    """
    주어진 경로의 이미지들을 matplotlib 윈도우에 나란히 표시합니다.
    reference_path가 주어지면 해당 이미지 제목에 '(대표)'를 추가합니다.
    ESC 키를 누르면 창을 닫습니다.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("이미지를 표시할 수 없습니다. 'matplotlib'와 'Pillow' 라이브러리 및 폰트 설정이 필요합니다.")
        return

    num_images = len(image_paths)
    if num_images == 0:
        print("표시할 이미지가 없습니다.")
        return

    valid_images = []
    valid_paths = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            valid_images.append(img.copy())
            valid_paths.append(img_path)
            img.close()
        except FileNotFoundError:
            logger.warning(f"이미지 파일을 찾을 수 없습니다: {img_path}")
        except UnidentifiedImageError:
            logger.warning(f"이미지 파일을 열 수 없습니다 (손상되었거나 지원하지 않는 형식): {img_path}")
        except Exception as e:
            logger.error(f"이미지 로드 중 오류 발생 ({img_path}): {e}")

    num_valid_images = len(valid_images)
    if num_valid_images == 0:
        print("표시할 유효한 이미지가 없습니다.")
        return

    display_count = min(num_valid_images, 5)
    fig, axes = plt.subplots(1, display_count, figsize=(4 * display_count, 4))

    if display_count == 1:
        axes = [axes]

    fig.suptitle("이미지 비교 (ESC 키로 닫기)", fontsize=14) # 제목에 안내 추가

    for i in range(display_count):
        ax = axes[i]
        img = valid_images[i]
        path = valid_paths[i]

        ax.imshow(img)
        title = os.path.basename(path) + "\n" + os.path.dirname(path)
        if path == reference_path:
            title = "(대표)\n" + title
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    # --- 키 이벤트 핸들러 정의 및 연결 ---
    def on_key(event):
        # print(f"Key pressed: {event.key}") # 어떤 키가 눌렸는지 확인 (디버깅용)
        if event.key == 'escape': # ESC 키가 눌렸는지 확인
            plt.close(fig) # 현재 창(figure) 닫기

    # 'key_press_event'에 on_key 함수 연결
    fig.canvas.mpl_connect('key_press_event', on_key)
    # ------------------------------------

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # 창 표시 및 이벤트 루프 시작

def interactive_review(map_file_path: str):
    """ 저장된 맵 파일을 기반으로 중복 세트 목록을 보여주고, 선택된 세트의 이미지를 표시하며, 중복 파일 삭제 기능을 제공합니다. """
    map_data = load_map_file(map_file_path)

    if not map_data:
        logger.info("처리된 파일 정보가 없습니다. 검토를 종료합니다.")
        return

    # --- 중복 세트 목록 생성 함수 (내부 함수로 정의하여 재사용) ---
    def generate_duplicate_list(current_map_data):
        dup_sets = []
        for file_hash, hash_data in current_map_data.items():
            originals_map = hash_data.get("originals", {})
            if len(originals_map) > 1:
                representative_original = next(iter(originals_map.keys()), None)
                if representative_original:
                    dup_sets.append((representative_original, file_hash, len(originals_map)))
        dup_sets.sort(key=lambda x: x[0])
        return dup_sets

    # --- 초기 중복 세트 목록 생성 및 표시 ---
    duplicate_sets = generate_duplicate_list(map_data)

    if not duplicate_sets:
        logger.info("맵 데이터에 중복된 이미지 세트가 없습니다.")
        return

    def print_duplicate_list(dup_sets):
        print("\n--- 중복 이미지 세트 검토 ---")
        print("발견된 중복 세트 목록 (대표 원본 파일 기준):")
        if not dup_sets:
            print("  (더 이상 중복 세트가 없습니다)")
            return
        for i, (rep_orig, _, count) in enumerate(dup_sets):
            print(f"  {i+1}: {rep_orig} (총 {count}개)")

    print_duplicate_list(duplicate_sets)

    # --- 사용자 입력 루프 ---
    while True:
        try:
            # 목록이 비었으면 종료
            if not duplicate_sets:
                print("\n모든 중복 세트 처리가 완료되었거나 목록이 비었습니다.")
                break

            user_input = input("\n확인/삭제할 중복 세트 번호를 입력하세요 (목록 새로고침 'r', 종료 'q'): ").strip().lower() # 옵션 추가
            if user_input == 'q':
                break
            if user_input == 'r': # 목록 새로고침 옵션
                map_data = load_map_file(map_file_path) # 맵 파일 다시 로드
                duplicate_sets = generate_duplicate_list(map_data)
                print_duplicate_list(duplicate_sets)
                continue # 다시 입력 받기

            selected_index = int(user_input) - 1
            if 0 <= selected_index < len(duplicate_sets):
                selected_rep_original, selected_hash, selected_count = duplicate_sets[selected_index]
                print("-" * 50)
                print(f"선택된 중복 세트 (대표 원본: {selected_rep_original})")
                print(f"  - 해시: {selected_hash}")
                print(f"  - 총 파일 수: {selected_count}")

                hash_data = map_data.get(selected_hash)
                if hash_data:
                    originals_map = hash_data.get("originals", {})
                    reference_moved_path = originals_map.get(selected_rep_original)

                    image_paths_to_display = []
                    duplicates_to_process = []
                    if reference_moved_path and os.path.exists(reference_moved_path):
                        image_paths_to_display.append(reference_moved_path)
                    else:
                         logger.warning(f"대표 이미지의 이동된 위치를 찾을 수 없거나 파일이 없습니다: {reference_moved_path}")

                    print("\n  --- 이 세트에 포함된 파일들 ---")
                    for orig, dest in sorted(originals_map.items()):
                        is_reference = (orig == selected_rep_original)
                        ref_marker = "(대표)" if is_reference else ""
                        print(f"    - 원본: {orig} {ref_marker}")
                        print(f"      이동된 위치: {dest if dest else '알 수 없음'}")

                        if not is_reference:
                            duplicates_to_process.append((orig, dest))
                            if dest and os.path.exists(dest):
                                image_paths_to_display.append(dest)
                            elif dest:
                                logger.warning(f"파일의 이동된 위치를 찾을 수 없거나 파일이 없습니다: {dest}")

                    if image_paths_to_display:
                        print(f"\n이미지 표시 시도: {len(image_paths_to_display)}개")
                        display_images(image_paths_to_display, reference_path=reference_moved_path)
                    else:
                        print("표시할 수 있는 이미지 파일을 찾지 못했습니다.")

                    # --- 삭제 여부 확인 및 실행 ---
                    if duplicates_to_process:
                        paths_to_delete = [(orig, dest) for orig, dest in duplicates_to_process if dest and os.path.exists(dest)]

                        if paths_to_delete:
                            delete_confirm = input(f"\n대표 이미지 외 {len(paths_to_delete)}개의 중복 이미지를 삭제하시겠습니까? (y/n): ").strip().lower()
                            if delete_confirm == 'y':
                                logger.info(f"{len(paths_to_delete)}개 중복 이미지 삭제 시작...")
                                deleted_count = 0
                                failed_count = 0
                                originals_to_remove_from_map = []

                                for orig_path, dest_path in paths_to_delete:
                                    try:
                                        os.remove(dest_path)
                                        logger.info(f"삭제 완료: {dest_path}")
                                        deleted_count += 1
                                        originals_to_remove_from_map.append(orig_path)
                                    except OSError as e:
                                        logger.error(f"파일 삭제 오류 '{dest_path}': {e}")
                                        failed_count += 1
                                    except Exception as e:
                                         logger.error(f"파일 삭제 중 예상치 못한 오류 발생 '{dest_path}': {e}")
                                         failed_count += 1

                                logger.info(f"삭제 완료: {deleted_count}개 성공, {failed_count}개 실패.")

                                if originals_to_remove_from_map:
                                    logger.info("맵 데이터 업데이트 중...")
                                    map_changed = False
                                    if selected_hash in map_data and "originals" in map_data[selected_hash]:
                                        current_originals = map_data[selected_hash]["originals"]
                                        for orig_to_remove in originals_to_remove_from_map:
                                            if orig_to_remove in current_originals:
                                                current_originals.pop(orig_to_remove)
                                                map_changed = True

                                        if len(current_originals) <= 1:
                                            map_data.pop(selected_hash, None)
                                            logger.info(f"해시 {selected_hash}는 더 이상 중복이 아니므로 맵에서 제거합니다.")
                                            map_changed = True

                                    if map_changed:
                                        save_map_file(map_data, map_file_path)
                                        logger.info("맵 파일 업데이트 완료.")
                                        # --- 목록 갱신 및 계속 진행 ---
                                        print("\n맵 데이터가 변경되었습니다. 중복 세트 목록을 갱신합니다.")
                                        duplicate_sets = generate_duplicate_list(map_data) # 갱신된 map_data로 목록 재생성
                                        print_duplicate_list(duplicate_sets) # 갱신된 목록 출력
                                        continue # 다음 입력 받기 (while 루프의 시작으로)
                                        # ---------------------------
                            else:
                                print("삭제하지 않았습니다.")
                        else:
                            print("실제로 삭제할 수 있는 파일이 없습니다.")
                    else:
                        print("\n삭제할 중복 이미지가 없습니다 (대표 이미지 외 파일 없음).")

                else:
                    print("  - 해당 해시에 대한 상세 정보를 찾을 수 없습니다.")
                print("-" * 50)

            else:
                print(f"잘못된 번호입니다. 1부터 {len(duplicate_sets)} 사이의 숫자를 입력하세요.")

        except ValueError:
            print("잘못된 입력입니다. 숫자를 입력하거나 'q', 'r'을 입력하세요.") # 안내 메시지 수정
        except Exception as e:
            logger.error(f"검토 중 오류 발생: {e}", exc_info=True)

    print("검토 세션을 종료합니다.")

# ==============================================================================
# 메인 실행 블록 (`if __name__ == "__main__":`)
# ==============================================================================

if __name__ == "__main__":
    # --- 명령줄 인자 파서 설정 ---
    parser = argparse.ArgumentParser(
        description="사진 정리 결과(맵 파일)를 기반으로 대화형으로 검토하고 이미지를 확인합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- 필수 인자 ---
    parser.add_argument("--map_file", type=str, required=True,
                        help="검토할 맵 파일 경로 (organize_photos.py가 생성한 _map.json 파일) (필수)")
    # --- 선택적 옵션 ---
    parser.add_argument("--debug", action="store_true",
                        help="디버그 레벨 로그를 활성화합니다.")

    # 명령줄 인자 파싱
    args = parser.parse_args()

    # --- 로그 레벨 설정 ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level)
    # 화면 출력 핸들러의 레벨도 설정
    for handler in logger.handlers:
        handler.setLevel(log_level)

    if args.debug:
        logger.info("디버그 로깅 활성화됨.")

    # --- 맵 파일 경로 처리 ---
    try:
        map_file_abs = os.path.abspath(os.path.expanduser(args.map_file))
    except Exception as e:
        logger.error(f"맵 파일 경로 처리 중 오류 발생: {e}")
        sys.exit(1)

    # --- 스크립트 시작 정보 로깅 ---
    logger.info("="*20 + " 사진 검토 시작 " + "="*20)
    logger.info(f"맵 파일: {map_file_abs}")
    logger.info("=" * 50 )

    # --- 대화형 검토 함수 호출 ---
    if os.path.exists(map_file_abs):
        # 맵 파일이 존재하면 검토 함수 실행
        interactive_review(map_file_abs)
    else:
        # 맵 파일이 없으면 오류 메시지 출력 후 종료
        logger.error(f"맵 파일을 찾을 수 없습니다: {map_file_abs}")
        logger.error("먼저 organize_photos.py를 실행하여 맵 파일을 생성해야 합니다.")
        sys.exit(1)

