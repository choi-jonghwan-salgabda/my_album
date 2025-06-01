import os
from pathlib import Path
import yaml
import textwrap

# 프로젝트 디렉토리 이름
PROJECT_NAME = "my_photo_album_2"
PROJECT_ROOT = Path.cwd() / PROJECT_NAME

def create_dirs():
    dirs = [
        "config",
        "data/raw_photos",
        "data/indexed_faces",
        "static/results",
        "templates",
        "src"
    ]
    for d in dirs:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {path}")

def create_config():
    config = {
        "data_path": "data/raw_photos",
        "cropped_faces_dir": "data/indexed_faces",
        "index_output": "data/face_index.pkl",
        "face_model": "cnn",
        "use_mediapipe": True,
        "tolerance": 0.6,
        "top_k": None,
        "use_copy": False,
        "visualization_output": "static/results/index_distribution.png"
    }
    config_path = PROJECT_ROOT / "config/.my_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"⚙️  Configuration written: {config_path}")

def create_pyproject_toml():
    toml_content = textwrap.dedent(f"""\
    [tool.poetry]
    name = "{PROJECT_NAME}"
    version = "0.1.0"
    description = "Personal face-based photo classification and search system."
    authors = ["you@example.com"]
    package-mode = false

    [tool.poetry.dependencies]
    python = "^3.10"
    opencv-python-headless = ">=4.8.0"
    numpy = "*"
    matplotlib = "*"
    fastapi = ">=0.100.0"
    uvicorn = {{ extras = ["standard"], version = ">=0.24.0" }}
    python-multipart = ">=0.0.6"
    Pillow = ">=9.0.0"
    requests = "*"
    PyYAML = "*"
    scikit-learn = "*"
    mediapipe = "*"

    [tool.poetry.group.dev.dependencies]
    ruff = ">=0.4.0"
    pytest = ">=7.4.0"
    mypy = ">=1.8.0"
    pre-commit = ">=3.5.0"

    [build-system]
    requires = ["poetry-core>=1.0.0"]
    build-backend = "poetry.core.masonry.api"
    """)
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    pyproject_path.write_text(toml_content)
    print(f"📦 pyproject.toml created: {pyproject_path}")

def create_readme():
    readme = PROJECT_ROOT / "README.md"
    readme.write_text("# My Photo Album 2\n\n얼굴 기반 사진 자동 분류 및 검색 시스템입니다.")
    print(f"📝 README.md created.")

def create_template():
    html = """<html><body><h2>🔎 얼굴 검색 UI (준비 중)</h2></body></html>"""
    (PROJECT_ROOT / "templates/search.html").write_text(html)
    print("🌐 search.html created.")

if __name__ == "__main__":
    print(f"🚀 Setting up project: {PROJECT_NAME}")
    PROJECT_ROOT.mkdir(exist_ok=True)
    create_dirs()
    create_config()
    create_pyproject_toml()
    create_readme()
    create_template()
    print(f"✅ 프로젝트 초기화 완료: {PROJECT_ROOT}")
