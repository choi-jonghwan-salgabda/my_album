import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from typing import Union, List, Dict
import hashlib
import json
from pathlib import Path

mp_face_detection = mp.solutions.face_detection

def compute_sha256(image: Union[np.ndarray, Image.Image]) -> str:
    """이미지의 SHA-256 해시를 반환합니다."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    _, buffer = cv2.imencode('.png', image)
    return hashlib.sha256(buffer.tobytes()).hexdigest()

def detect_faces_with_hash(
    image: Union[np.ndarray, Image.Image],
    image_path: Union[str, Path] = None,
    min_confidence: float = 0.5,
    target_size: tuple = (224, 224)
) -> Dict:
    """
    얼굴 검출 및 해시 포함 결과 생성

    Returns:
        Dict: {
            image_hash: str,
            image_path: str,
            faces: List[Dict]
        }
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    height, width, _ = image.shape
    image_hash = compute_sha256(image)

    faces = []

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_confidence) as detector:
        results = detector.process(image)

        if results.detections:
            for i, det in enumerate(results.detections):
                box = det.location_data.relative_bounding_box
                x = int(box.xmin * width)
                y = int(box.ymin * height)
                w = int(box.width * width)
                h = int(box.height * height)

                face_info = {
                    "face_id": i,
                    "box": {"x": x, "y": y, "width": w, "height": h},
                    "score": float(det.score[0])
                }
                faces.append(face_info)

    return {
        "image_hash": image_hash,
        "image_path": str(image_path) if image_path else None,
        "faces": faces
    }

def save_face_json(json_data: Dict, output_path: Union[str, Path]):
    """검출된 얼굴 정보를 JSON으로 저장합니다."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
