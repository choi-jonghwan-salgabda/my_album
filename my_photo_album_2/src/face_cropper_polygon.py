from pathlib import Path
from typing import List
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def get_face_landmarks(image: np.ndarray) -> List[np.ndarray]:
    """MediaPipe Face Mesh를 사용하여 얼굴 랜드마크 좌표를 반환"""
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return []
        return [
            np.array([(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                      for landmark in face_landmarks.landmark], dtype=np.int32)
            for face_landmarks in results.multi_face_landmarks
        ]

def crop_polygon_face(image: np.ndarray, landmarks: np.ndarray, output_size=(224, 224)) -> np.ndarray:
    """랜드마크를 기반으로 다각형 마스크를 생성하고 얼굴 영역만 자름"""
    contour_idx = list(range(0, 17)) + list(range(68, 83))  # 턱선 + 눈썹 아래쪽
    polygon = landmarks[contour_idx]

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    face_only = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(polygon)
    cropped = face_only[y:y+h, x:x+w]

    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
    return resized

def crop_face_from_file(img_path: Path, output_dir: Path) -> bool:
    image = np.array(Image.open(img_path).convert("RGB"))
    landmarks_list = get_face_landmarks(image)
    if not landmarks_list:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, landmarks in enumerate(landmarks_list):
        cropped = crop_polygon_face(image, landmarks)
        out_path = output_dir / f"{img_path.stem}_face{i}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    return True


"""
✅ 사용 예시
poetry add mediapipe opencv-python-headless pillow numpy


from pathlib import Path
from src.face_cropper_polygon import crop_face_from_file

img_path = Path("data/raw_photos/images/group1.jpg")
output_dir = Path("data/indexed_faces_polygon")

crop_face_from_file(img_path, output_dir)

"""