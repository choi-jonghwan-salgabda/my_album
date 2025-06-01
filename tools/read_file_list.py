"""
파일의 목록을 담은 파일을 일러준다
입력 : file_list_path : 파일목록이 있는 경로과 파일이름
돌려주는 갑 : 읽은 목록 -> List
"""


import os
from typing import List
from logger import setup_logger  # my_utility 로거 임포트

# 로거 설정 (my_utility의 setup_logger 사용)
log = setup_logger()


def read_json_list(file_list_path: str) -> List:
    """
    주어진 경로의 파일에서 JSON 파일 이름 또는 상대 경로 목록을 읽어 리스트로 반환합니다.

    Args:
        file_list_path (str): JSON 파일 이름/경로 목록이 담긴 텍스트 파일의 경로.

    Returns:
        list: JSON 파일 이름/경로 문자열의 리스트. 파일 읽기 실패 시 빈 리스트 반환.
    """
    log.info(f"JSON 목록 파일을 읽습니다: {file_list_path}")
    json_file_name = []
    try:
        with open(file_list_path, 'r', encoding='utf-8') as f_list:
            # 모든 경로를 읽고, 양 끝 공백 제거 및 빈 줄 무시
            json_file_name = [line.strip() for line in f_list if line.strip()]
        log.info(f"총 {len(json_file_name)}개의 JSON 경로(상대)를 찾았습니다.")
    except FileNotFoundError:
        log.error(f"오류: 입력 목록 파일을 찾을 수 없습니다: {file_list_path}")
    except Exception as e:
        log.error(f"오류: 입력 목록 파일 읽기 중 오류 발생 {file_list_path}: {e}")
    return json_file_name



if __name__ == "__main__":
    direction_dir = os.getcwd()
    print(f"os.getcwd() = {direction_dir}")
    run_file_name = os.path.basename(__file__).replace(".py", "")
    print(f'os.path.basename(__file__).replace(".py", "") = {run_file_name}')
    log = setup_logger(direction_dir, run_file_name, console=True)

    #log.info(f"경로만들기 프로그램 처음:{os.path.basename(__file__)}")
    #make_file_list('/data/ephemeral/home/Upstage/industry-partnership-project-brainventures/data/aihub_60m')
    #log.info(f"경로만들기 프로그램 끝:{os.path.basename(__file__)}")
    base_dir = '/data/ephemeral/home/Upstage/industry-partnership-project-brainventures/data/aihub_60m/'
    file_list_path = os.path.join(base_dir, 'lists/val_file_list.lst')
    
    file_list = read_json_list(file_list_path)
    print(file_list)