import easyocr,os,json,cv2;from PIL import Image,ImageDraw,ImageFont;import numpy as np

img_path='/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/img/책표지_총류_000002.jpg'
output_folder="/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/output/easyocr/"
annotated_img_file=os.path.join(output_folder, "annotated_" + os.path.basename(img_path))
annotation_file=os.path.join(output_folder, "annotated_" + os.path.splitext(os.path.basename(img_path))[0] + ".json")
# PIL로 이미지 열고 numpy 배열로 변환 (RGB)
image=Image.open(img_path).convert("RGB")
image_np=np.array(image)

result1=easyocr.Reader(['ko']).readtext(image_np)#1.raw output
print(f"bbox 갯수: {len(result1)}")
# for tu_ple in result1:print(tu_ple)
# result2=easyocr.Reader(['ko']).readtext(img_path,detail=0);print(result2)#2.text만 list로
# result3=easyocr.Reader(['ko']).readtext(img_path,detail=0,paragraph=True);print(result3)#3.text만 자연스러운 문장으로

# JSON 직렬화(튜플 등을 리스트로)를 위해 numpy 자료형을 기본 자료형으로 변환
def convert_to_native(obj):
    if isinstance(obj, (tuple, list)):  # tuple 또는 list 처리
        return [convert_to_native(x) for x in obj]
    elif isinstance(obj, dict):  # dict 처리
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif hasattr(obj, 'item'):  # numpy 자료형 처리
        return obj.item()
    else:  # 기본 자료형 그대로 반환
        return obj
# EasyOCR 결과를 JSON 직렬화 가능한 형태로 변환
results_native = convert_to_native(result1)

with open(annotation_file, 'w', encoding='utf-8') as f:
    json.dump(results_native, f, ensure_ascii=False, indent=4)
print("OCR 결과가 JSON 파일로 저장되었습니다:", annotation_file)

# 저장된 JSON 파일을 불러와 이미지에 주석 추가
with open(annotation_file, 'r', encoding='utf-8') as f:
    loaded_results = json.load(f)

# 이미지에 bounding box와 텍스트 그리기
draw=ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 35)
except Exception:
    font = ImageFont.load_default()

for item in loaded_results:
    bbox, text, conf = item
    points = [tuple(point) for point in bbox]  # 각 좌표를 튜플로 변환
    draw.line(points + [points[0]], fill=(255, 0, 0), width=4)  # bbox 그리기
    text_position = (points[0][0], max(points[0][1] - 25, 0))
    draw.text(text_position, text, fill=(255, 0, 0), stroke_width=1, font=font)  # 텍스트 추가
    
    # 텍스트 크기 구하기
    bbox_text = draw.textbbox((0, 0), text, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    # 신뢰도 텍스트: 소수점 둘째자리, bbox의 첫번째 점 위쪽에 추가
    conf_text = f"{conf:.2f}"
    conf_position = (text_position[0] + text_width + 5, text_position[1])
    draw.text(conf_position, conf_text, fill=(0, 0, 255), stroke_width=1, font=font)  # 신뢰도 추가 (파란색)

# 주석이 추가된 이미지 저장
image.save(annotated_img_file)
print("Annotated 이미지가 저장되었습니다:", annotated_img_file)