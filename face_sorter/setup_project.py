"""
face_sorter/
├── pyproject.toml             ✅ 생성됨 (내용 자동 작성 가능)
├── .gitignore                 ✅ 기본 설정 포함
├── README.md                  ✅ 템플릿 포함
├── src/
│   └── face_sorter/
│       ├── __init__.py
│       ├── data/              ✅ 폴더만 생성
│       ├── models/            ✅ 폴더만 생성
│       ├── clustering.py      ✅ 빈 파이썬 파일
│       ├── face_extractor.py  ✅ 여기에 첫 기능 작성할 예정
│       └── main.py            ✅ 실행 진입점
"""


import os
from pathlib import Path

def create_file(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# 프로젝트 기본 경로 설정
project_root = Path("face_sorter")
src_dir = project_root / "src" / "face_sorter"
folders = [
    project_root,
    project_root / "src",
    src_dir / "data",
    src_dir / "models",
]

# 파일들 정의
files = {
    project_root / "README.md":
        "# Face Sorter\n\nDesktop face clustering and labeling system.\n",

    project_root / ".gitignore":
        "*.pyc\n__pycache__/\n.env\npoetry.lock\n.idea/\n.vscode/\n",

    project_root / "pyproject.toml":
        """[tool.poetry]
name = "face_sorter"
version = "0.1.0"
description = "Desktop face clustering and labeling system"
authors = ["Your Name <your@email.com>"]
packages = [{ include = "face_sorter" }]

[tool.poetry.dependencies]
python = "^3.10"
insightface = "^0.7.3"
opencv-python = "^4.9"
scikit-learn = "^1.4"
matplotlib = "^3.8"
numpy = "^1.26"
tqdm = "^4.66"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
""",

    src_dir / "__init__.py": "",

    src_dir / "face_extractor.py":
        "# 얼굴 임베딩 추출 모듈\n\n"
        "def extract_face_embeddings():\n"
        "    pass\n",

    src_dir / "clustering.py":
        "# 얼굴 임베딩 클러스터링\n\n"
        "def cluster_embeddings():\n"
        "    pass\n",

    src_dir / "main.py":
        "# 메인 실행 스크립트\n\n"
        "if __name__ == '__main__':\n"
        "    print('📸 Face Sorter 실행 준비 완료')\n",
}

# 디렉토리 생성
for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)

# 파일 생성
for path, content in files.items():
    create_file(path, content)

print("✅ 프로젝트 템플릿 생성 완료!")
print("➡  'cd face_sorter' 후 'poetry install' 로 환경을 설정하세요.")
