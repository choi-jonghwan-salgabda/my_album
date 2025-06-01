"""
함수는 detect_faces_with_polygon()의 결과를 받아서
JSON 파일로 저장해주는 함수입니다.

매개변수	설명
image_path	원본 이미지 경로 (Path)
image_hash	SHA-256 해시 (중복 제거 및 식별자 용도)
faces	detect_faces_with_polygon() 함수의 결과
output_path	저장할 JSON 파일 경로
"""
import json
from pathlib import Path
from typing import List, Dict

def save_face_json_with_polygon(image_path: Path, image_hash: str, faces: List[Dict], output_path: Path) -> None:
    """
    얼굴 윤곽 정보와 이미지 해시를 포함한 JSON 파일 저장
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_data = {
        "image_path": str(image_path),
        "image_hash": image_hash,
        "num_faces": len(faces),
        "faces": faces
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
        
"""
✅ 함께 사용할 흐름 예시

from pathlib import Path
from PIL import Image
import hashlib
from src.face_cropper_polygon import detect_faces_with_polygon, save_face_json_with_polygon

img_path = Path("data/raw_photos/images/group1.jpg")
img = Image.open(img_path)

# SHA-256 해시 생성
img_hash = hashlib.sha256(img.tobytes()).hexdigest()

# 얼굴 윤곽 추출
faces = detect_faces_with_polygon(img)

# JSON 저장
json_out = Path("data/raw_photos/jsons") / f"{img_path.stem}.json"
save_face_json_with_polygon(img_path, img_hash, faces, json_out)

"""