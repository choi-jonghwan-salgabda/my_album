import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import random
import matplotlib as mpl
import matplotlib.font_manager as fm
# 한글 폰트 설정
# 리눅스에서 사용 가능한 한글 폰트 경로
font_paths = [
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # 우분투 나눔고딕
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',  # Noto Sans CJK
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',  # 다른 경로
]

# 폰트 경로 확인 및 설정
font_found = False
for font_path in font_paths:
    if os.path.exists(font_path):
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
        mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 표시 문제 해결
        print(f"한글 폰트 설정 완료: {font_path}")
        font_found = True
        break

# 폰트가 없는 경우 폰트 다운로드
if not font_found:
    try:
        # 나눔고딕 폰트 다운로드 및 설정
        import matplotlib.font_manager as fm
        from matplotlib import rc
        
        # 시스템에 있는 모든 폰트 확인
        font_list = [f.name for f in fm.fontManager.ttflist]
        print("사용 가능한 폰트:", font_list)
        
        # 한글 글꼴 선택 시도
        for font in font_list:
            if any(keyword in font.lower() for keyword in ['gothic', 'gulim', 'malgun', 'nanum', 'noto', 'batang']):
                plt.rcParams['font.family'] = font
                print(f"한글 폰트 찾음: {font}")
                font_found = True
                break
    except:
        print("폰트 설정 실패")

# 한글 텍스트를 이미지로 표시하는 함수
def draw_text_on_image(img, text, position):
    """텍스트를 OpenCV로 이미지에 직접 그리기"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position
    cv2.putText(img, text, (int(x), int(y)), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img

# 경로 설정
json_dir = '/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/json'
image_dir = '/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/images'
output_dir = '/data/ephemeral/home/industry-partnership-project-brainventures/HI/bbox_visualization'

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 파일 목록
image_list = [
    '간판_실내안내판_036594', '책표지_역사_001313', '간판_가로형간판_015330',
    '간판_가로형간판_015343', '간판_돌출간판_003642', '간판_돌출간판_023259',
    '간판_세로형간판_008997', '간판_실내간판_003639', '책표지_기타_001387',
    '책표지_사회과학_002812', '책표지_역사_001243', '간판_가로형간판_015314',
    '간판_실내간판_003022', '간판_실내안내판_000671', '간판_창문이용광고물_000557',
    '간판_현수막_007910', '책표지_기타_000322', '책표지_기타_000611',
    '책표지_사회과학_002796', '간판_가로형간판_015767', '간판_돌출간판_023138',
    '간판_실내간판_009669', '책표지_문학_002328', '책표지_문학_002332',
    '책표지_문학_002369', '책표지_문학_002370', '책표지_문학_002371'
]

# 각 이미지 처리
for image_name in image_list:
    # 파일 경로 설정
    json_path = os.path.join(json_dir, f"{image_name}.json")
    image_path = os.path.join(image_dir, f"{image_name}.jpg")
    
    # 파일 존재 확인
    if not os.path.exists(json_path):
        print(f"JSON 파일이 없음: {json_path}")
        continue
    
    image_found = False
    for root, dirs, files in os.walk(image_dir):
        possible_path = os.path.join(root, f"{image_name}.jpg")
        if os.path.exists(possible_path):
            image_path = possible_path
            image_found = True
            print(f"이미지 찾음: {image_path}")
            break
        if image_found:
            break
    
    # JSON 파일 로드
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"JSON 파일 로드 오류: {json_path} - {e}")
        continue
    
    # 이미지 로드
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"이미지 로드 오류: {image_path} - {e}")
        continue
    
    # matplotlib으로 시각화
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    ax = plt.gca()
    
    # 바운딩 박스 그리기
    try:
        if 'annotations' in data:
            for annotation in data['annotations']:
                # bbox 좌표 추출 (bbox 키가 있는 경우)
                if 'bbox' in annotation:
                    # bbox는 [x, y, width, height] 형식일 가능성이 높음
                    bbox = annotation['bbox']
                    
                    # bbox 포맷에 따라 처리 (x, y, w, h 또는 x1, y1, x2, y2)
                    if len(bbox) == 4:
                        # 일반적인 bbox 포맷: [x, y, width, height]
                        x, y, w, h = bbox
                        
                        # 사각형 그리기
                        rect = Rectangle((x, y), w, h, fill=False, edgecolor=np.random.rand(3,), linewidth=2)
                        ax.add_patch(rect)
                        
                       # 텍스트 추출 및 표시
                        if 'text' in annotation and annotation['text']:
                            text = annotation['text']
                            
                            # 방법 1: Matplotlib 텍스트 사용
                            if font_found:
                                plt.text(x, y-5, text, color='white', fontsize=8,
                                        bbox=dict(facecolor='black', alpha=0.5))
                        # 방법 2: 대안으로 이미지에 직접 텍스트 그리기
                        else:
                            img = draw_text_on_image(img, text, (x, y-5))
                
                # polygon 좌표도 체크 (일부 데이터는 polygon 형식일 수 있음)
                elif 'polygon' in annotation:
                    polygon = annotation['polygon']
                    polygon_coords = np.array(polygon).reshape(-1, 2)
                    
                    # 다각형 그리기
                    polygon_patch = Polygon(polygon_coords, fill=False, edgecolor=np.random.rand(3,), linewidth=2)
                    ax.add_patch(polygon_patch)
                    
                    # 텍스트 추출 및 표시
                    if 'text' in annotation and annotation['text']:
                        text = annotation['text']
                        x_min = min(p[0] for p in polygon)
                        y_min = min(p[1] for p in polygon)
                        plt.text(x_min, y_min-5, text, color='white', fontsize=8,
                                 bbox=dict(facecolor='black', alpha=0.5))
    except Exception as e:
        print(f"바운딩 박스 처리 오류: {image_name} - {e}")
        print(f"오류 세부 정보: {annotation}")
    
    # 축 제거 및 타이틀 설정
    plt.axis('off')
    plt.title(f"Image: {image_name}")
    
    # 이미지 저장
    output_path = os.path.join(output_dir, f"{image_name}_bbox.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"처리 완료: {image_name}")

print("모든 이미지 처리 완료")