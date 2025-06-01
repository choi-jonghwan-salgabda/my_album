import os
import sys
import shutil
import argparse
import logging

# --- 초기 설정 및 로거 설정 ---
worker_name = os.path.splitext(os.path.basename(__file__))[0]
worker_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(worker_dir, '..'))

# 프로젝트 루트를 sys.path에 추가 (my_utility 사용 위함)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from my_utility import setup_logger
    logger = setup_logger(log_file=f"{worker_name}.log", console=True,
                          file_level=logging.INFO, console_level=logging.INFO)
    logger.info(f"{worker_name}.py 용 로거 초기화 완료.")
    logger.info(f"프로젝트 루트: {project_root}")
except ImportError as e:
    # my_utility 로드 실패 시 기본 로거 사용
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(worker_name)
    logger.warning(f"my_utility 로드 실패 ({e}). 기본 로거를 사용합니다.")
except Exception as e:
    print(f"로거 설정 중 예상치 못한 오류 발생: {e}")
    sys.exit(1)

# --- 고정 경로 정의 ---
# 원본 이미지들이 있는 디렉토리
SRC_IMAGE_BASE_DIR = "/data/ephemeral/home/data/aihub/Fastcampus_project/images"
# 파일 목록 및 대상 디렉토리의 기본 경로
UPSTAGE_JSONS_BASE_DIR = "/data/ephemeral/home/Upstage/brainventures/data/jsons"

def read_lst_file(file_path: str) -> List[str]:
    """ 파일이름 목록을 읽어 json파일 이름 목록을 만들어 반환한다. """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        log.error(f"file_path 읽기 실패: => read_lst_file {e}")
        return []

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="지정된 파일을 읽어, 읽은 내용을 돌려줍니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 외부 인자로 'test', 'train', 'val' 중 하나를 받도록 설정
    parser.add_argument('set_name', type=str, choices=['test', 'train', 'val'],
                        help="처리할 데이터셋 이름 (예: test, train, val)")

    args = parser.parse_args()

    # 함수 호출
    read_lst_file(args.set_name)
