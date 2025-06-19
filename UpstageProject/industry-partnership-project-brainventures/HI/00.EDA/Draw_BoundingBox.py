import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import matplotlib.font_manager as fm

# 사용 가능한 폰트 목록 출력
available_fonts = [f.name for f in fm.fontManager.ttflist]
print("사용 가능한 폰트:", available_fonts)

# 한글 지원 폰트 찾기
korean_fonts = [f for f in available_fonts if any(keyword in f.lower() for keyword in ['nanum', 'malgun', 'gothic', '고딕', '돋움'])]
print("한글 지원 폰트:", korean_fonts)

# 발견된 한글 폰트 중 첫 번째 폰트 사용
if korean_fonts:
    plt.rcParams['font.family'] = korean_fonts[0]
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'  # 기본 대체 폰트

# OpenCV 이미지를 PIL 이미지로 변환하는 함수
def cv2_to_pil(cv_img):
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv_img_rgb)

# PIL 이미지를 OpenCV 이미지로 변환하는 함수
def pil_to_cv2(pil_img):
    return np.array(pil_img)

# 한글 텍스트를 이미지에 넣는 함수
def put_korean_text(img, text, position, font_size=24, font_color=(255, 0, 0)):
    # OpenCV 이미지를 PIL 이미지로 변환
    pil_img = cv2_to_pil(img)
    
    # 이미지 드로우 객체 생성
    draw = ImageDraw.Draw(pil_img)
    
    # 폰트 설정 (먼저 시스템에 설치된 폰트를 사용)
    try:
        # Ubuntu/Debian 계열에 기본 설치된 한글 폰트 경로
        font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size)
    except:
        try:
            # 다른 경로에서 폰트 시도
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", font_size)
        except:
            try:
                # 시스템 기본 폰트 사용
                font = ImageFont.load_default()
                print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
            except:
                # 마지막 방법: PIL 기본 폰트
                print("폰트를 불러올 수 없습니다.")
                return img
    
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=font_color)
    
    # PIL 이미지를 OpenCV 이미지로 다시 변환
    return pil_to_cv2(pil_img)

# 파일 경로
sample_dir = "/data/ephemeral/home/industry-partnership-project-brainventures/data/02.raw_data/1.Training/sample"
label_dir = "/data/ephemeral/home/industry-partnership-project-brainventures/data/02.raw_data/1.Training/label"
output_dir = "./bbox_visualizations"

os.makedirs(output_dir, exist_ok=True)

# 이미지 파일 찾기
image_files = []
for root, dirs, files in os.walk(sample_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.append(os.path.join(root, file))

print(f"총 {len(image_files)}개 이미지 파일 발견")
def process_image(image_path):
    # 이미지 로드
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
        else:
            # BGR을 RGB로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 이미지 파일명 추출

            img_filename = os.path.basename(image_path)
            img_name_without_ext = os.path.splitext(img_filename)[0]
            
            # 상대경로 추출 (sample 디렉토리 기준)
            rel_path = os.path.relpath(os.path.dirname(image_path), sample_dir)
            
            # 매칭되는 JSON 파일 찾기
            json_file = None
            
            # 경로 패턴 처리 (간판_가로형간판_000013.jpg -> 간판_가로형간판_000013.json)
            search_pattern = img_name_without_ext + ".json"
            
            # 가능한 JSON 경로 탐색
            for json_root, _, json_files in os.walk(label_dir):
                for jf in json_files:
                    if jf.lower() == search_pattern.lower():
                        json_file = os.path.join(json_root, jf)
                        break
                if json_file:
                    break
            
            if not json_file:
                print(f"일치하는 JSON 파일을 찾을 수 없습니다: {img_filename}")
                return False

            # JSON 파일 로드
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 원본 이미지 복사
            image_with_bbox = image.copy()
            
            # annotations 데이터 가져오기
            if 'data' in json_data:
                annotations = json_data['data'].get('annotations', [])
            else:
                annotations = json_data.get('annotations', [])
            
            seen = set() 
            # 바운딩 박스 그리기
            for anno in annotations:
                if isinstance(anno, dict) and 'bbox' in anno:
                    bbox = anno.get('bbox', [])
                    text = anno.get('text', '')
                    
                    # if text == 'xxx': 
                    #     continue
                    
                    # if bbox is None or not bbox:
                    #     continue

                    # # null 값이 포함된 bbox 필터링  
                    # if any(val is None for val in bbox):
                    #     continue


                    key = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{text}"

                    if len(bbox) == 4 and key not in seen:
                        try:
                            # bbox 좌표 추출
                            x, y, w, h = [int(val) for val in bbox]
                            
                            # 박스 그리기
                            cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            
                            # 텍스트 그리기 (텍스트가 있는 경우)
                            if text:
                                # 한글 텍스트 추가 (PIL 사용)
                                label_text = f"#{text}"
                                image_with_bbox = put_korean_text(
                                    image_with_bbox, 
                                    label_text, 
                                    (x, y - 30),  # 텍스트 위치
                                    font_size=50,  # 폰트 크기
                                    font_color=(255, 0, 0)  # 빨간색
                                )
                            
                                # seen.add(key)
                        except (ValueError, TypeError):
                            print(f"바운딩 박스 좌표 오류: {bbox}")
            
            # 결과 이미지 저장
            plt.figure(figsize=(12, 10))
            plt.imshow(image_with_bbox)
            plt.axis('off')
            plt.title("바운딩 박스가 표시된 이미지", fontsize=16)
            
            # 저장 경로 설정
            output_name = f"{os.path.splitext(img_filename)[0]}_bbox.jpg"
            output_path = os.path.join(output_dir, output_name)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
            
            print(f"바운딩 박스가 표시된 이미지가 저장되었습니다: {output_path}")
            
            # 시각화
            plt.show()
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


# 모든 이미지 처리
success_count = 0
for idx, img_path in enumerate(image_files):
    print(f"\n[{idx+1}/{len(image_files)}] 처리 중: {img_path}")
    if process_image(img_path):
        success_count += 1

print(f"\n처리 완료: 총 {len(image_files)}개 중 {success_count}개 성공")
print(f"결과 이미지는 {output_dir} 디렉토리에 저장되었습니다.")