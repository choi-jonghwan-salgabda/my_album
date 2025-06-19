import cv2
import json
import numpy as np

# 이미지 읽기
img_path = "/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/img/책표지_총류_000001.jpg"
json_path = "/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/label/000001.json"

img = cv2.imread(img_path)

# JSON 파일 읽기
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 각 annotation에 대해 bbox와 텍스트 그리기
for ann in data['annotations']:
    # bbox 좌표 (x,y,w,h -> x1,y1,x2,y2로 변환)
    x, y, w, h = ann['bbox']
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    
    # bbox 그리기 (녹색)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 텍스트 그리기 (흰색)
    text = ann['text']
    cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# 결과 이미지 저장
output_path = "output_visualized.jpg"
cv2.imwrite(output_path, img)

print(f"시각화된 이미지가 {output_path}에 저장되었습니다.")