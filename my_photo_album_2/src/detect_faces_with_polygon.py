"""
MediaPipe Face Mesh를 이용해 이미지에서 얼굴의 윤곽선 다각형 추출

score, polygon 좌표, bounding box 등의 정보를 포함하는 JSON-like 결과 반환

return
[
  {
    "score": 1.0,
    "polygon": [
      {"x": 120.5, "y": 210.3},
      {"x": 122.2, "y": 208.9},
      ...
    ]
  }
]
✅ 연계 활용
create_face_jsons_polygon.py에서 사용해 JSON 생성

face_cropper_polygon.py와 함께 써서 얼굴 이미지를 자르면서 JSON도 생성

face_search_web.py에서 업로드된 이미지의 윤곽 추출에도 사용 가능
"""
from typing import List, Dict
from PIL import Image
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def detect_faces_with_polygon(image: Image.Image, min_confidence: float = 0.5) -> List[Dict]:
    """
    이미지에서 얼굴을 다각형 윤곽 기반으로 검출
    반환: 각 얼굴에 대해 polygon 좌표 및 score 포함된 dict 리스트
    """
    image_np = np.array(image.convert("RGB"))
    height, width = image_np.shape[:2]

    faces = []

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, refine_landmarks=True,
                                min_detection_confidence=min_confidence) as face_mesh:
        results = face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            return []

        for face_landmarks in results.multi_face_landmarks:
            # 턱선 (0~16), 눈썹 하단 (68~83) 중심으로 polygon 구성
            indices = list(range(0, 17)) + list(range(68, 83))
            polygon = [
                {
                    "x": round(face_landmarks.landmark[i].x * width, 2),
                    "y": round(face_landmarks.landmark[i].y * height, 2)
                } for i in indices
            ]

            faces.append({
                "score": 1.0,  # MediaPipe Face Mesh는 별도 score 없음
                "polygon": polygon
            })

    return faces
