from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import face_recognition
import numpy as np
import pickle
import shutil
import os
import uvicorn
from config_loader import load_config

app = FastAPI()

# Static and templates setup
app.mount("/indexed_faces", StaticFiles(directory="data/indexed_faces"), name="indexed_faces")
templates = Jinja2Templates(directory="templates")

# Load index
CONFIG_PATH = "config/.my_config.yaml"
config = load_config(CONFIG_PATH)
INDEX_FILE = Path(config["index_output"])

with open(INDEX_FILE, "rb") as f:
    index_data = pickle.load(f)
    face_encodings = np.array(index_data["encodings"])
    face_paths = index_data["paths"]

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search_faces(request: Request, file: UploadFile = File(...)):
    # Save uploaded image
    uploaded_path = Path("static") / "query.jpg"
    with uploaded_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = face_recognition.load_image_file(str(uploaded_path))
    locations = face_recognition.face_locations(image, model=config.get("face_model", "cnn"))
    if not locations:
        return templates.TemplateResponse("search.html", {"request": request, "results": [], "error": "얼굴이 감지되지 않았습니다."})

    query_encoding = face_recognition.face_encodings(image, known_face_locations=locations)[0]
    distances = face_recognition.face_distance(face_encodings, query_encoding)
    sorted_matches = sorted(zip(distances, face_paths), key=lambda x: x[0])

    results = [
        {"path": f"/indexed_faces/{Path(p).name}", "distance": float(d)}
        for d, p in sorted_matches
    ]

    return templates.TemplateResponse("search.html", {"request": request, "results": results})

if __name__ == "__main__":
    uvicorn.run("face_search_web:app", host="0.0.0.0", port=8000, reload=True)
