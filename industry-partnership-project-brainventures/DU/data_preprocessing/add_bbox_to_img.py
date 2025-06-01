import cv2
import json
import numpy as np
import os
import glob

# JSON 파일 로드 (구조: { "images": { "파일명.jpg": { "words": { ... } } } })
json_path = "/data/ephemeral/home/datasets/jsons/train.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# JSON의 "images" 부분 추출
images_data = data["images"]

# 이미지 폴더 경로 (jpg 파일 기준)
img_folder = "/data/ephemeral/home/datasets/images/train"
img_paths = glob.glob(os.path.join(img_folder, "*.jpg"))

# 결과 이미지를 저장할 폴더 생성
output_folder = "/data/ephemeral/home/datasets/images/train_output"
os.makedirs(output_folder, exist_ok=True)

for img_path in img_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue

    filename = os.path.basename(img_path)
    # JSON에 해당 이미지가 있을 경우 bounding box 추가
    if filename in images_data:
        words = images_data[filename].get("words", {})
        for word_id, word_info in words.items():
            points = np.array(word_info["points"], dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            # 빨간색 선으로 bounding box 그리기
            cv2.polylines(img, [points], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # 결과 이미지 저장
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")