import os
import sys

def remove_last_json_from_lines(file_path):
    """파일 내용에서 각 줄의 마지막 '.json' 제거"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        new_lines = [line.rstrip('\n').rstrip('.json') + '\n' for line in lines]  # 줄 끝 .json 제거

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(new_lines)

        print(f"변환 완료: {file_path}")
    
    except Exception as e:
        print(f"❌ 파일 처리 중 오류 발생 ({file_path}): {e}")

def process_all_files(directory):
    """디렉토리 내 모든 파일을 처리"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        print(f"file_path : {file_path}")

        if os.path.isfile(file_path):  # 파일만 처리
            remove_last_json_from_lines(file_path)
    


# 실행 예제
if __name__ == "__main__":
    # 사용 예제 (변경할 디렉터리 지정)
    if len(sys.argv) != 2:
        print("❌ 사용법: python script.py <option -f(File Contents) / -d(Directory) > <content>")
        sys.exit(1)

    directory_path = sys.argv[1]
    process_all_files(directory_path)
