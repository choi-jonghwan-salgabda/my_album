import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # ✅ 먼저 sys.path 설정

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import shutil
import numpy as np
import face_recognition
from PIL import Image
import pickle
import io
from pathlib import Path
import logging
from config_loader import load_config  # ✅ 이제 올바르게 불러올 수 있음

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load config
config = load_config("config/.my_config.yaml")
INDEX_FILE = Path(config["index_output"])
TOP_K = config.get("top_k", None)
TOLERANCE = config.get("tolerance", 0.6)

# Load face index
if INDEX_FILE.exists():
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
        INDEX_ENCODINGS = np.array(data["encodings"])
        INDEX_PATHS = data["paths"]
else:
    INDEX_ENCODINGS = np.array([])
    INDEX_PATHS = []

def find_similar_faces(upload_image):
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

    # 정렬 + 제한 없음 (top_k=None이면 전부 보여줌)
    results = sorted(results, key=lambda x: x[0])
    if TOP_K:
        results = results[:TOP_K]
    return results

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    results = find_similar_faces(image)

    return templates.TemplateResponse("search.html", {
        "request": request,
        "results": results,
        "uploaded": True
    })
