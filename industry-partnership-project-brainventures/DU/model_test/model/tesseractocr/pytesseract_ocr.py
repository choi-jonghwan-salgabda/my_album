import pytesseract,cv2,base64,numpy as np

image_path = "/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/img/책표지_총류_000001.jpg"
with open(image_path, "rb") as image_file:
    base64_image=base64.b64encode(image_file.read()).decode("utf-8")
    
# Base64 이미지를 OpenCV 형식으로 변환
image_data = base64.b64decode(base64_image)
np_arr = np.frombuffer(image_data, np.uint8)
image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# OCR 수행 (텍스트 + 바운딩 박스 반환)
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

results = []
for i in range(len(data["text"])):
    if data["text"][i].strip():
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        results.append({"text": data["text"][i], "bbox": [x, y, x + w, y + h]})

# 결과 출력
print(results)