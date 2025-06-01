import json

# JSON 파일 경로
file_path = '/data/ephemeral/home/outputs/ocr_training/submissions/20250411_134058.json'

try:
    # JSON 파일 열기 및 읽기 (UTF-8 인코딩 명시)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 'images' 키가 있는지 확인
    if 'images' in data:
        # 'images' 딕셔너리의 키 개수(이미지 개수) 세기
        num_images = len(data['images'])
        print(f"'{file_path}' 파일의 'images' 키 안에는 총 {num_images}개의 이미지가 있습니다.")
    else:
        print(f"'{file_path}' 파일 안에 'images' 키를 찾을 수 없습니다.")

except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
except json.JSONDecodeError:
    print(f"오류: '{file_path}' 파일이 유효한 JSON 형식이 아닙니다.")
except Exception as e:
    print(f"알 수 없는 오류 발생: {e}")
