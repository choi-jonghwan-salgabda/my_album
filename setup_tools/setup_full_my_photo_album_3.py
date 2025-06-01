import os
import sys
import subprocess
import yaml
import requests
from pathlib import Path

# === 설정 ===
PROJECT_CONFIG_FILE = "./config/my_photo_album_3.yaml"
HAAR_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
HAAR_CASCADE_URL = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{HAAR_CASCADE_FILENAME}"

# === 기본 함수 ===

def load_project_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_directories(config):
    root_dir = Path(config["project"]["root_dir"])
    dirs_to_create = [
        root_dir / "config",
        root_dir / "dataset",
        root_dir / "output",
        root_dir / "src",
        root_dir / "templates"
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        print(f"📁 디렉토리 생성: {d}")

def create_pyproject_toml(root_dir):
    pyproject_content = """\
[tool.poetry]
name = "my-photo-album-3"
version = "0.1.0"
description = "Face Detection and Indexing Project"
authors = ["Your Name <your@email.com>"]

[tool.poetry.dependencies]
python = "^3.8"
opencv-python = "^4.8.0"
PyYAML = "^6.0"
fastapi = "^0.95.0"
uvicorn = "^0.22.0"
pillow = "^10.0.0"
numpy = "^1.24.0"
mediapipe = "^0.10.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""
    pyproject_path = root_dir / "pyproject.toml"
    pyproject_path.write_text(pyproject_content, encoding="utf-8")
    print(f"✅ pyproject.toml 생성: {pyproject_path}")

def create_requirements_txt(root_dir):
    requirements_content = """\
face_recognition
git+https://github.com/ageitgey/face_recognition_models.git
setuptools
"""
    requirements_path = root_dir / "requirements.txt"
    requirements_path.write_text(requirements_content, encoding="utf-8")
    print(f"✅ requirements.txt 생성: {requirements_path}")

def download_haar_cascade(root_dir):
    config_dir = root_dir / "config"
    haar_path = config_dir / HAAR_CASCADE_FILENAME
    if not haar_path.exists():
        response = requests.get(HAAR_CASCADE_URL)
        haar_path.write_bytes(response.content)
        print(f"✅ Haar Cascade 다운로드 완료: {haar_path}")
    else:
        print(f"ℹ️ Haar Cascade 파일이 이미 존재합니다: {haar_path}")

def setup_poetry_environment(root_dir):
    # 1. 가상환경 초기화
    subprocess.run(["poetry", "env", "remove", "python"], cwd=root_dir)
    # 2. pyproject.toml 기반 설치
    subprocess.run(["poetry", "install"], cwd=root_dir)
    print("✅ Poetry 환경 구축 완료")

def install_requirements_with_pip(root_dir):
    subprocess.run(["poetry", "run", "pip", "install", "-r", "requirements.txt"], cwd=root_dir)
    print("✅ 추가 requirements.txt 패키지 설치 완료")

# === 메인 실행 ===
if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "./config/my_config.yaml"

    # 0. 기즘 내가 일하는 곳은"
    direction_dir = os.getcwd()
    print(f"지금 쥔계서 계신곳(direction_dir) : {direction_dir}")
    worker_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"지금 일꾼이 일하는곳(worker_dir)  : {worker_dir}")
    
    # 1. 프로젝트 설정 로드
    config = load_project_config(config_path)
    root_dir = Path(config["project"]["root_dir"])

    # 2. 디렉토리 생성
    create_directories(config)

    # 3. pyproject.toml, requirements.txt 생성
    create_pyproject_toml(root_dir)
    create_requirements_txt(root_dir)

    # 4. Haar Cascade 다운로드
    download_haar_cascade(root_dir)

    # 5. Poetry 환경 구축
    setup_poetry_environment(root_dir)

    # 6. pip으로 추가 패키지 설치
    install_requirements_with_pip(root_dir)

    print("🎉 프로젝트 초기화가 완료되었습니다!")

