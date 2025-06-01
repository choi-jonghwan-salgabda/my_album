
#
# 내가 만든 공용 함수를 쓰기위해 환경을 만든다.
# (디렉토리 구조 주석은 이전과 동일하게 유지)
# ...

import os
import sys
import json
import shutil
import argparse
import logging # logging 모듈 직접 임포트

# --- 초기 설정 ---
worker_name = os.path.splitext(os.path.basename(__file__))[0] # 현재 스크립트 파일 이름 (확장자 제외)
worker_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 스크립트가 있는 디렉토리
project_root = os.path.abspath(os.path.join(worker_dir, '..')) # 프로젝트 루트 디렉토리 계산

# 프로젝트 루트를 sys.path에 추가하여 유틸리티 임포트
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 사용자 정의 유틸리티 임포트 ---
try:
    from my_utility import setup_logger, get_logger # combine_paths는 여기서는 필요 없을 수 있음
    # 로거 초기화 (파일은 INFO, 콘솔은 INFO 레벨 기본값 사용)
    logger = setup_logger(log_file=f"{worker_name}.log", console=True,
                          file_level=logging.INFO, console_level=logging.INFO)
    logger.info(f"{worker_name}.py 용 로거 초기화 완료.")
    logger.info(f"프로젝트 루트: {project_root}")
except ImportError as e:
    print(f"Import 오류: {e}")
    print("프로젝트 구조가 올바른지, 'my_utility' 접근 가능한지 확인하세요.")
    print(f"  - 예상 프로젝트 루트: {project_root}")
    print(f"  - '{os.path.join(project_root, 'my_utility', '__init__.py')}' 파일 존재 여부 확인.")
    sys.exit(1) # 스크립트 비정상 종료
except Exception as e:
    print(f"설정 중 예상치 못한 오류 발생: {e}")
    sys.exit(1) # 스크립트 비정상 종료


def move_images_from_json(src_dir, dest_dir, json_path):
    """
    JSON 파일에서 이미지 파일명 목록을 읽어 src_dir에서 dest_dir로 이동합니다.
    파일명의 대소문자를 구분하지 않고 처리합니다.
    src_dir에 없는 파일은 JSON 파일과 같은 디렉토리에 목록으로 저장합니다.

    Args:
        src_dir (str): 원본 이미지가 있는 절대 경로.
        dest_dir (str): 이미지를 이동할 대상 디렉토리의 절대 경로.
        json_path (str): 이동할 이미지 파일명 목록이 포함된 JSON 파일의 절대 경로.
                         ('images' 키 아래에 파일명을 키로 갖는 딕셔너리가 있다고 가정)
    """
    logger.info("="*50)
    logger.info("JSON 목록 기반 이미지 이동 작업을 시작합니다 (대소문자 무시)...") # 메시지 수정
    logger.info(f"원본 이미지 디렉토리      : {src_dir}")
    logger.info(f"대상 이미지 디렉토리      : {dest_dir}")
    logger.info(f"JSON 파일 경로              : {json_path}")
    logger.info("="*50)

    # --- 입력 경로 유효성 검사 ---
    if not os.path.isdir(src_dir):
        logger.error(f"원본 이미지 디렉토리를 찾을 수 없습니다: {src_dir}")
        return # 함수 종료
    if not os.path.isfile(json_path):
        logger.error(f"JSON 파일을 찾을 수 없습니다: {json_path}")
        return # 함수 종료
    # 대상 디렉토리는 없으면 생성
    try:
        os.makedirs(dest_dir, exist_ok=True)
        logger.info(f"대상 디렉토리 확인/생성 완료: {dest_dir}")
    except OSError as e:
        logger.error(f"대상 디렉토리 생성 실패: {e}")
        return # 함수 종료

    # --- JSON 데이터 로드 ---
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"JSON 파일 로드 성공: {json_path}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파일 디코딩 오류: {json_path} - {e}")
        return # 함수 종료
    except Exception as e:
        logger.error(f"JSON 파일 읽기 중 예상치 못한 오류 발생: {json_path} - {e}")
        return # 함수 종료

    # --- 이미지 파일명 추출 ('images' 키 확인) ---
    if 'images' not in data or not isinstance(data.get('images'), dict):
        logger.error(f"JSON 파일에 'images' 키가 없거나 값이 딕셔너리 형태가 아닙니다. 경로: {json_path}")
        return # 함수 종료

#    json_image_filenames = list(data['images'].keys()) # JSON에 있는 파일명 목록
    json_image_filenames = [key.lower() for key in data['images'].keys()]
    total_files_in_json = len(json_image_filenames)
    logger.info(f"JSON 파일에서 {total_files_in_json}개의 이미지 파일명을 찾았습니다.")

    if total_files_in_json == 0:
        logger.warning("JSON 파일에 이미지 목록이 없습니다. 이동할 파일이 없습니다.")
        return # 함수 종료

    # --- 원본 디렉토리 파일 목록 읽고 소문자 맵 생성 (대소문자 무시 처리 위함) ---
    logger.info(f"원본 디렉토리({src_dir}) 파일 목록 읽는 중...")
    try:
        actual_files_in_src = os.listdir(src_dir)
        # 소문자 파일명을 키로, 원본 파일명을 값으로 하는 딕셔너리 생성
        src_file_map_lower = {f.lower(): f for f in actual_files_in_src}
        logger.info(f"원본 디렉토리에서 {len(actual_files_in_src)}개의 항목을 찾았고, 대소문자 무시 맵 생성 완료.")
    except OSError as e:
        logger.error(f"원본 디렉토리({src_dir})를 읽는 중 오류 발생: {e}")
        return

    # --- 이미지 처리 및 이동 ---
    missing_files = [] # 누락된 파일 목록을 저장할 리스트
    moved_count = 0    # 성공적으로 이동된 파일 수
    error_count = 0    # 이동 중 오류 발생 수
    remove_count = 0   # 원본과 대상이 같아 원본에서 제거된 파일 수

    logger.info(f"{total_files_in_json}개 파일에 대한 처리 시작...")
    for index, filename_from_json in enumerate(json_image_filenames):
        filename_lower = filename_from_json.lower() # JSON 파일명을 소문자로 변환

        # 진행 상황 로깅 (예: 1000개마다)
        if (index + 1) % 1000 == 0:
            logger.info(f"처리 중... ({index + 1}/{total_files_in_json})")

        # 3. 소문자 맵을 사용하여 원본 디렉토리에서 파일 존재 확인 (대소문자 무시)
        if filename_lower in src_file_map_lower:
            # 실제 원본 파일명 (대소문자 유지)
            actual_source_filename = src_file_map_lower[filename_lower]
            source_file_path = os.path.join(src_dir, actual_source_filename)
            # 대상 파일 경로는 JSON의 파일명 기준 (대소문자 유지)
            dest_file_path = os.path.join(dest_dir, filename_from_json)

            # 소스와 대상 경로가 동일한 경우 원본에서 제거 (수정된 로직)
            # os.path.abspath로 절대 경로 비교
            if os.path.abspath(source_file_path) == os.path.abspath(dest_file_path):
                try:
                    os.remove(source_file_path)
                    logger.warning(f"원본과 대상 경로가 동일하여 원본에서 제거합니다: {actual_source_filename}")
                    remove_count += 1
                except OSError as e:
                    logger.error(f"원본 파일 제거 오류 '{actual_source_filename}': {e}")
                    error_count += 1 # 제거 오류도 오류 카운트에 포함
                continue # 다음 파일로 넘어감
            # 소스와 대상 경로가 다르면 이동 시도
            try:
                shutil.move(source_file_path, dest_file_path)
                # logger.debug(f"이동 완료: {actual_source_filename} -> {dest_dir}") # 상세 로깅 필요시 debug 사용
                moved_count += 1
            except (IOError, OSError) as e:
                logger.error(f"파일 이동 오류 '{actual_source_filename}': {e}")
                error_count += 1
            except Exception as e:
                logger.error(f"파일 이동 중 예상치 못한 오류 발생 '{actual_source_filename}': {e}")
                error_count += 1
        else:
            # 4. src_dir에 파일이 없으면 missing_files 목록에 추가 (JSON 파일명 기준)
            logger.warning(f"원본 파일을 찾을 수 없음 ('{src_dir}', 대소문자 무시): {filename_from_json}")
            missing_files.append(filename_from_json)

    logger.info(f"파일 처리 완료.")
    # 5. 각 항목 처리 결과 로깅
    logger.info(f"이미지 이동 작업 요약:")
    logger.info(f"  - 성공적으로 이동된 파일 수 : {moved_count}")
    logger.info(f"  - 원본 경로에 없던 파일 수   : {len(missing_files)}")
    logger.info(f"  - 중복으로 제거된 파일 수   : {remove_count}") # 수정된 부분 반영
    logger.info(f"  - 처리 중 오류 발생 수      : {error_count}")

    # --- 존재하지 않는 파일 목록 저장 ---
    if missing_files:
        # 누락 목록 파일 이름 생성 (JSON 파일 이름 기반)
        json_basename = os.path.splitext(os.path.basename(json_path))[0]
        missing_list_filename = f"{json_basename}_nonexist_list.lst" # 파일 이름 형식
        json_directory = os.path.dirname(json_path) # JSON 파일이 있는 디렉토리 (수정된 로직)
        missing_list_path = os.path.join(json_directory, missing_list_filename) # 저장 경로
        logger.info(f"미존재 파일 목록 저장 경로: {missing_list_path}")

        try:
            with open(missing_list_path, 'w', encoding='utf-8') as f:
                for filename in missing_files:
                    f.write(filename + '\n')
            logger.info(f"{len(missing_files)}개의 미존재 파일 목록 저장 완료: {missing_list_path}")
        except IOError as e:
            logger.error(f"미존재 파일 목록 저장 실패: {missing_list_path} - {e}")
        except Exception as e:
            logger.error(f"미존재 파일 목록 저장 중 예상치 못한 오류 발생: {missing_list_path} - {e}")
    else:
        logger.info("원본 경로에 존재하지 않는 파일이 없습니다.")

    logger.info("이미지 이동 작업이 종료되었습니다.")
    logger.info("="*50 + "\n")



# --- 메인 실행 블록 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JSON 파일에 명시된 이미지 목록을 기반으로 원본 디렉토리에서 대상 디렉토리로 파일을 이동시키고, 누락된 파일 목록을 JSON 파일 옆에 기록합니다.", # 설명 수정
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 1. 각 정보를 인자로 받는다 (절대 경로 권장)
    parser.add_argument('--path', type=str, required=True,
                        help='읽고자 하는 파일의 경로와 이름 (예: images)')

    args = parser.parse_args()

    # 입력값 기본 검증 (파일/디렉토리 존재 여부)
    valid_inputs = True
    # 입력 경로에 ~가 있을 경우 확장하고 절대 경로로 변환
    abs_src_dir = os.path.abspath(os.path.expanduser(args.src_dir))
    abs_dest_dir = os.path.abspath(os.path.expanduser(args.dest_dir))
    abs_json_path = os.path.abspath(os.path.expanduser(args.json_path))

    if not os.path.isdir(abs_src_dir):
        logger.error(f"원본 디렉토리가 존재하지 않습니다: {abs_src_dir}")
        valid_inputs = False
    if not os.path.isfile(abs_json_path):
        logger.error(f"JSON 파일이 존재하지 않습니다: {abs_json_path}")
        valid_inputs = False
    # dest_dir는 함수 내에서 생성하므로 여기서는 검사하지 않음

    if valid_inputs:
        logger.info("입력 경로 검증 완료. 이미지 이동 함수를 호출합니다.")
        # 함수 호출 시 절대 경로로 변환된 값 사용
        move_images_from_json(
            src_dir=abs_src_dir,
            dest_dir=abs_dest_dir,
            json_path=abs_json_path
        )
    else:
        logger.error("입력 경로 오류로 스크립트 실행을 중단합니다.")
        sys.exit(1) # 스크립트 비정상 종료
