project_name = "my_photo_album"
"""
my_photo_album/
├── config/.my_config.yaml              # 구성 파일
├── data/raw_photos/                    # 원본 이미지
├── data/indexed_faces/                # 얼굴 crop
├── data/face_index.pkl                 # 인덱스 저장 결과
├── sorc/
│   ├── face_indexer.py                # 인덱싱
│   ├── face_search_web.py            # 웹 서버
│   ├── face_labeler.py              # 라벨링 UI
│   └── config_loader.py              # 설정 로더
├── templates/search.html              # 검색 UI
├── static/results/                   # 결과 이미지 (복사 비사용 시 미활용)
├── README.md
"""
import os
from pathlib import Path

BASE_DIR = Path(project_name)
folders = [
    "config", "data/raw_photos", "data/indexed_faces",
    "src", "templates", "static/results"
]

files = {
    "./config/.${project_name}_config.yaml": """# 프로젝트 설정 파일
# 얼굴 인덱싱 및 검색 시스템용

data_path: data/raw_photos  # 원본 이미지 폴더 (label 없음)
cropped_faces_dir: data/indexed_faces  # 얼굴만 잘라 저장될 폴더
index_output: data/face_index.pkl  # 인덱싱 결과 저장 경로
face_model: cnn  # 'hog' 또는 'cnn' (정확도 우선이면 cnn 권장)
tolerance: 0.6  # 유사도 판단 기준 거리 임계값
top_k: null  # top_k 제한 없음 (전체 보여주기 위함)
use_copy: false  # 이미지 복사하지 않고 원본 사용
""",

    "templates/search.html": """<!DOCTYPE html>
<html>
<head><title>Face Search</title></head>
<body>
<h2>Upload a face to search</h2>
<form action=\"/search\" enctype=\"multipart/form-data\" method=\"post\">
  <input name=\"file\" type=\"file\" accept=\"image/*\">
  <input type=\"submit\">
</form>
<hr>
{% if results %}
<h3>Results:</h3>
<div style=\"display: flex; flex-wrap: wrap;\">
  {% for r in results %}
    <div style=\"margin: 5px; text-align: center;\">
      <img src=\"{{ r.path }}\" width=\"120\"><br>
      <small>{{ r.distance | round(3) }}</small>
    </div>
  {% endfor %}
</div>
{% endif %}
</body>
</html>
""",

    "src/config_loader.py": "import yaml\n\ndef load_config(path):\n    with open(path, 'r') as f:\n        return yaml.safe_load(f)\n",

    "README.md": """# 나의 사진첩
얼굴 인덱싱 + 검색 + 라벨링을 지원하는 Python 기반 시스템입니다.
- `data/raw_photos/`: 원본 사진
- `data/indexed_faces/`: 얼굴만 추출해 저장
- `face_index.pkl`: 인덱스 결과 파일
- 웹 기반 유사 얼굴 검색 (`face_search_web.py`)
"""
}


def create_structure():
    # folders 리스트에 있는 각 폴더를 생성합니다.
    for folder in folders:
        # BASE_DIR 경로 아래에 folder 이름으로 폴더를 생성합니다.
        # parents=True는 상위 디렉토리가 없으면 생성하도록 하고,
        # exist_ok=True는 이미 폴더가 존재해도 오류를 발생시키지 않습니다.
        (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)
    
    # files 딕셔너리에 있는 각 파일 경로와 내용을 사용하여 파일을 생성합니다.
    for file_path, content in files.items():
        # BASE_DIR 경로와 file_path를 결합하여 전체 파일 경로를 생성합니다.
        full_path = BASE_DIR / file_path
        
        # 지정된 경로에 파일을 쓰기 모드로 열고, UTF-8 인코딩으로 설정합니다.
        with open(full_path, "w", encoding="utf-8") as f:
            # 파일에 content 내용을 씁니다.
            f.write(content)
    
    # 모든 폴더와 파일 생성이 완료되었음을 알리는 메시지를 출력합니다.
    print("✅ 'my_photo_album' 프로젝트 템플릿이 생성되었습니다!")

if __name__ == "__main__":
    create_structure()
