# ⚙️1. 프로그램 환경설정
## 1) 필요한 라이브러리 설치
import os
import sys


### 2) 전역변수의 정의
if len(sys.argv) != 2:
    print("❌ 사용법: python script.py <project_root_directory>")
    sys.exit(1)

PRJT_DIRS = os.path.abspath(sys.argv[1])
print(f"이 프로젝트 디렉토리(PRJT_DIRS) : {PRJT_DIRS}")

UTIL_PATH = os.path.abspath('./my_utility')
print(f"도움의 구성정보 위치(UTIL_PATH) : {UTIL_PATH}")

# 현재 파일(main.py)이 위치한 디렉토리 기준으로 utility 디렉토리 추가
# 각 디렉토리 확인 후 없으면 종료
for directory in [PRJT_DIRS, UTIL_PATH]:
    if not os.path.exists(directory):
        print(f"Error: {directory} 가 존재하지 않습니다.")
        sys.exit(1)
    sys.path.append(directory)

from my_utility import setup_logger, get_logger, combine_paths, ConfigDirectoryManager

manager = ConfigDirectoryManager(os.path.join(UTIL_PATH, 'my_config.yaml'))

if PRJT_DIRS != manager.getpath('PRJT_DIRS', '.'):
    print("❌ 사용법: python script.py <project_root_directory>")
    print(f"❌ 입력하신 프로젝트 위치 PRJT_DIRS : {PRJT_DIRS}")
    print(f"❌ 찾아낸   프로젝트 위치 PRJT_DIRS : {init_config.get('PRJT_DIRS', os.path.abspath('.'))}")
    sys.exit(1)

LOGS_DIRS = combine_paths(PRJT_DIRS, manager.getpath('LOGS_DIRS', './logs'))
print(f"프로젝트의 log 디렉토리(LOGS_DIRS) : {LOGS_DIRS}")

log = setup_logger(log_dir=LOGS_DIRS, console=True)

FileName = os.path.basename(__file__)
log.info(f"지금 일하는 파일이름은 : {FileName}")

OUTPUT_DIRS = combine_paths(PRJT_DIRS, manager.getpath('OUTPUT_DIRS', './logs'))
log.info(f"프로젝트의 OUTPUT_DIRS 디렉토리(LOGS_DIRS) : {OUTPUT_DIRS}")
DATA_DIRS = combine_paths(PRJT_DIRS, manager.getpath('DATA_DIRS', './data'))
log.info(f"프로젝트의 DATA_DIRS 디렉토리(LOGS_DIRS) : {DATA_DIRS}")

#===================== 여기까지가 기본설정임 ============================


# 간판 카테고리
Sign_dict = {
    '가로형간판': 'Horizontal_Sign',
    '돌출간판': 'Projecting_Sign',
    '세로형간판': 'Vertical_Sign',
    '실내간판': 'Indoor_Sign_Board',
    '실내안내판': 'Indoor_Guide_Sign',
    '지주이용간판': 'Pillar_Sign',
    '창문이용광고물': 'Window_Advertisement',
    '현수막': 'Banner',
    '기타': 'Other',
    '간판': 'Sign'  # '간판' 변환은 마지막에 수행되어야 함
}

# 책표지 카테고리
bookcover_dict = {
    '종교': 'Religion',
    '총류': 'General_Category',
    '역사': 'History',
    '언어': 'Language',
    '철학': 'Philosophy',
    '자연과학': 'Natural_Science',
    '기술과학': 'Technology_Science',
    '사회과학': 'Social_Science',
    '예술': 'Art',
    '문학': 'Literature',
    '책표지': 'Bookcover'  # '책표지' 변환은 마지막에 수행
}

# 변환 딕셔너리 (긴 문자열부터 변환하도록 정렬)
conversion_dict = {
    **dict(sorted(Sign_dict.items(), key=lambda x: -len(x[0]))),
    **dict(sorted(bookcover_dict.items(), key=lambda x: -len(x[0])))
}

def rename_files_in_directory(directory, conversion_dict):
    """디렉터리 내 모든 파일명을 변환"""
    log.info(f"convert_filenames: {directory}")
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)

        # 파일이 아닌 경우 건너뛰기
        if not os.path.isfile(old_path):
            continue

        # 파일명 변경
        new_filename = filename
        for kor, eng in conversion_dict.items():
            new_filename = new_filename.replace(kor, eng)
            log.info(f"new_filename: {new_filename}, new_filename: {new_filename}")

        # 변경이 필요한 경우 이름 변경
        if new_filename != filename:
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            log.info(f"파일 변경: {filename} → {new_filename}")

def rename_directories(root_directory, conversion_dict):
    """루트 디렉토리 이하의 모든 디렉토리 및 파일명을 변환"""

    # 1️⃣ 하위 파일명을 먼저 변환
    for root, dirs, files in os.walk(root_directory, topdown=False):  
        rename_files_in_directory(root, conversion_dict)

        # 2️⃣ 디렉토리 이름 변경 (topdown=False로 하위 폴더부터 변경)
        for dir_name in dirs:
            old_dir_path = os.path.join(root, dir_name)
            new_dir_name = dir_name
            for kor, eng in conversion_dict.items():
                new_dir_name = new_dir_name.replace(kor, eng)

            if new_dir_name != dir_name:
                new_dir_path = os.path.join(root, new_dir_name)
                os.rename(old_dir_path, new_dir_path)
                log.info(f"디렉토리 변경: {dir_name} → {new_dir_name}")


# 실행
#===================== 여기부터가 기본 main 설정임 =======================
if __name__ == "__main__":
# 경로 설정
    # 사용 예제 (변경할 디렉터리 지정)
    root_directory = "/data/ephemeral/home/industry-partnership-project-brainventures/data/"  # 원하는 경로로 변경
    DATA_DIR = os.path.join(PRJT_DIR, 'data', 'datasets')
    log.info(f"{DATA_DIR} : 프로젝트의 data 디렉토리")
    
    if len(sys.argv) != 2:
        print("❌ 사용법: python script.py <root_directory>")
        sys.exit(1)

    root_directory = sys.argv[1]

    if not os.path.exists(root_directory):
        log.ERROR(f"❌ 오류: 입력한 디렉토리 '{root_directory}'가 존재하지 않습니다.")
        sys.exit(1)

    log.info(f"디렉토리 변경: {root_directory}")
    rename_directories(root_directory, conversion_dict)
    log.info("✅ 변환 완료!")