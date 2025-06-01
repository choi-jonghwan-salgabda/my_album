# ⚙️1. 프로그램 환경설정
## 1) 필요한 라이브러리 설치
import os
import sys


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

def convert_text(text):
    """문자열 내 한국어 단어를 영어로 변환"""
    for kor, eng in conversion_dict.items():
        text = text.replace(kor, eng)
    return text

def convert_file_content(file_path):
    """파일 내용을 변환하여 저장"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        converted_content = convert_text(content)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(converted_content)
        
        print(f"파일 내용 변환 완료: {file_path}")
    except Exception as e:
        print(f"❌ 파일 변환 실패 ({file_path}): {e}")


def rename_files_in_directory(directory):
    """디렉터리 내 모든 파일명을 변환"""
    print(f"convert_filenames: {directory}")
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)

        # 파일이 아닌 경우 건너뛰기
        if not os.path.isfile(old_path):
            continue

        # 파일명 변경
        new_filename = filename
        for kor, eng in conversion_dict.items():
            new_filename = new_filename.replace(kor, eng)
            print(f"new_filename: {new_filename}, new_filename: {new_filename}")

        # 변경이 필요한 경우 이름 변경
        if new_filename != filename:
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"파일 변경: {filename} → {new_filename}")

def rename_directories(root_directory):
    """루트 디렉토리 이하의 모든 디렉토리 및 파일명을 변환"""

    # 1️⃣ 하위 파일명을 먼저 변환
    for root, dirs, files in os.walk(root_directory, topdown=False):  
        rename_files_in_directory(root)

        # 2️⃣ 디렉토리 이름 변경 (topdown=False로 하위 폴더부터 변경)
        for dir_name in dirs:
            old_dir_path = os.path.join(root, dir_name)
            new_dir_name = dir_name
            for kor, eng in conversion_dict.items():
                new_dir_name = new_dir_name.replace(kor, eng)

            if new_dir_name != dir_name:
                new_dir_path = os.path.join(root, new_dir_name)
                os.rename(old_dir_path, new_dir_path)
                print(f"디렉토리 변경: {dir_name} → {new_dir_name}")


# 실행
#===================== 여기부터가 기본 main 설정임 =======================
if __name__ == "__main__":
# 경로 설정
    # 사용 예제 (변경할 디렉터리 지정)
    if len(sys.argv) != 3:
        print("❌ 사용법: python script.py <option -f(File Contents) / -d(Directory) > <content>")
        sys.exit(1)

    option = sys.argv[1]
    content = sys.argv[2]

    if not os.path.exists(content):
        print(f"❌ 오류: 입력한 디렉토리 '{content}'가 존재하지 않습니다.")
        sys.exit(1)

    print(f"디렉토리 변경: {content}")
    if option == "-f":
        convert_file_content(content)
    elif option == "-d":
        rename_directories(content)
    else:
        print(f"❌ 오류: 입력한 디렉토리 '{content}'가 존재하지 않습니다.")
    
    print("✅ 변환 완료!")