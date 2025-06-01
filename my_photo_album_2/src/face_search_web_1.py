import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 현재 디렉토리
import io
import shutil
import pickle
import uuid
import logging
from pathlib import Path

import numpy as np
import face_recognition
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config_loader import load_config  # 환경설정 로더
import re

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-6s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI 앱 생성 및 정적 파일 서빙
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# 설정 로딩 및 인덱스 파일 준비
config = load_config("config/.my_config.yaml")
INDEX_FILE = Path(config["index_output"])
TOP_K = config.get("top_k", None)
TOLERANCE = config.get("tolerance", 0.6)

if INDEX_FILE.exists():
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
        INDEX_ENCODINGS = np.array(data["encodings"])
        INDEX_PATHS = data["paths"]
else:
    INDEX_ENCODINGS = np.array([])
    INDEX_PATHS = []

STATIC_RESULT_DIR = os.path.join("static", "results")
os.makedirs(STATIC_RESULT_DIR, exist_ok=True)


def remove_face_number(filename: str) -> str:
    return re.sub(r'_face\d+', '', filename)

def recover_original_path(face_path: str) -> str:
    return re.sub(r'_face\d*', '', face_path)

def find_similar_faces(upload_image: Image.Image):
    image_np = np.array(upload_image)
    face_locations = face_recognition.face_locations(image_np)
    if not face_locations:
        return []

    face_encodings = face_recognition.face_encodings(image_np, face_locations)
    if not face_encodings:
        return []

    query_enc = face_encodings[0]
    distances = face_recognition.face_distance(INDEX_ENCODINGS, query_enc)
    match_indices = np.where(distances <= TOLERANCE)[0]

    results = []
    for i in match_indices:
        results.append((float(distances[i]), INDEX_PATHS[i]))
    results = sorted(results, key=lambda x: x[0])
    if TOP_K:
        results = results[:TOP_K]
    return results

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def main_page():
    return HTML_PAGE

@app.post("/find_faces/", response_class=JSONResponse)
async def find_faces(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        uploaded_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logger.error(f"이미지 읽기 실패: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.")

    results = find_similar_faces(uploaded_image)
    served_result_images = []
    for dist, img_path in results:
        original_path = remove_face_number(str(img_path))

        if os.path.isfile(original_path):
            filename = os.path.basename(original_path)
            dest_path = os.path.join(STATIC_RESULT_DIR, filename)
            try:
                shutil.copy(original_path, dest_path)
                served_result_images.append({
                    "thumbnail": f"/static/results/{filename}",
                    "full": f"/static/results/{filename}"
                })
            except Exception as e:
                logger.warning(f"결과 파일 복사 실패 ({original_path}): {e}")
        else:
            logger.warning(f"원본 이미지가 존재하지 않음: {original_path}")

    return {
        "message": "유사 얼굴 검색 성공",
        "uploaded_filename": image.filename,
        "found_similar_images": served_result_images
    }

# 인라인 HTML
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>유사 얼굴 찾기</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: sans-serif; margin: 20px; line-height: 1.6; }
        .container { display: flex; gap: 20px; margin-top: 20px; }
        .left-panel, .right-panel { flex: 1; }
        .right-panel { border-left: 1px solid #ccc; padding-left: 20px; }
        #result-list { list-style: none; padding: 0; border: 1px solid #eee; max-height: 60vh; overflow-y: auto; }
        #result-list li { padding: 8px; display: flex; align-items: center; cursor: pointer; border-bottom: 1px solid #eee; }
        #result-list li:hover { background: #f0f0f0; }
        #result-list li.selected { background: #d0e0ff; font-weight: bold; }
        img.thumbnail { width: 40px; height: 40px; object-fit: cover; margin-right: 10px; border: 1px solid #ccc; }
        #image-preview img { max-width: 100%; max-height: 70vh; margin-top: 10px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>유사 얼굴 찾기</h1>
    <form id="uploadForm" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">찾기</button>
    </form>
    <div id="error-message" style="color: red; margin-top: 10px;"></div>
    <div class="container">
        <div class="left-panel">
            <h2>결과 목록</h2>
            <ul id="result-list"></ul>
        </div>
        <div class="right-panel">
            <h2>미리보기</h2>
            <div id="image-preview"><p>목록에서 이미지를 선택하세요.</p></div>
        </div>
    </div>
    <script>
        const form = document.getElementById('uploadForm');
        const resultList = document.getElementById('result-list');
        const previewDiv = document.getElementById('image-preview');
        const errorDiv = document.getElementById('error-message');

        function showPreview(fullUrl, filename) {
            previewDiv.innerHTML = '';
            const img = document.createElement('img');
            img.src = fullUrl;
            img.alt = filename;
            previewDiv.appendChild(img);
        }

        form.onsubmit = async (e) => {
            e.preventDefault();
            resultList.innerHTML = '';
            previewDiv.innerHTML = '<p>목록에서 이미지를 선택하세요.</p>';
            errorDiv.textContent = '';

            const formData = new FormData(form);
            const res = await fetch('/find_faces/', { method: 'POST', body: formData });

            if (!res.ok) {
                const err = await res.json();
                errorDiv.textContent = `오류: ${err.detail}`;
                return;
            }

            const data = await res.json();
            data.found_similar_images.forEach(item => {
                const li = document.createElement('li');
                const img = document.createElement('img');
                img.src = item.thumbnail;
                img.className = 'thumbnail';
                const span = document.createElement('span');
                span.textContent = item.thumbnail.split('/').pop();

                li.appendChild(img);
                li.appendChild(span);
                li.onclick = () => {
                    document.querySelectorAll('#result-list li').forEach(li => li.classList.remove('selected'));
                    li.classList.add('selected');
                    showPreview(item.full, span.textContent);
                };

                resultList.appendChild(li);
            });
        };
    </script>
</body>
</html>
"""
