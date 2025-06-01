import json
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
DATA_DIR = '/data/ephemeral/home/'
"""
annotation정보가 있는 json파일을 받아 
그안의 정보로 이미지를 읽고 그 이미지에 
검출된 이미지의 특징표시인 
bbox를 그려넣어 저장한다.
"""

def draw_annotations(annotation_data):
    for image_info in annotation_data["images"]:
        image_path = image_info["file_name"]
        
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found.")
            continue
        
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        for annotation in annotation_data["annotations"]:
            if annotation["image_id"] == image_info["id"]:
                bbox = annotation["bbox"]
                text = annotation["text"]
                
                # bbox 유효성 검사 및 수정
                if len(bbox) == 4:
                    x, y, width, height = bbox
                    if width <= 0 or height <= 0:
                        print(f"Invalid bbox for annotation id {annotation['id']}: {bbox}")
                        continue
                    # bbox의 정보가 시작정돠 가로 세로 거리일경우 가로점, 아래 왼쪽점, 아애 오른쏙점을 계산한다.
                    # 오른쪽 하단 모서리 계산
                    x1 = x + width
                    y1 = y + height
                    
                    # 사각형 그리기
                    draw.rectangle([x, y, x1, y1], outline="red", width=2)
                    
                    # 텍스트 위치
                    text_position = (x, y)
                    draw.text(text_position, text, fill="white")
                else:
                    print(f"Invalid bbox length for annotation id {annotation['id']}: {bbox}")

        # 이미지 파일로 저장
        output_path = f"annotated_{image_info['file_name']}"
        image.save(output_path)
        print(f"Annotated image saved as {output_path}")

# JSON 파일 경로
json_file_path = os.path.join(DATA_DIR, '간판_가로형간판_037597.json')

# JSON 파일 읽기
with open(json_file_path, 'r', encoding='utf-8') as f:
    annotation_data = json.load(f)

# 어노테이션 그리기
draw_annotations(annotation_data)
