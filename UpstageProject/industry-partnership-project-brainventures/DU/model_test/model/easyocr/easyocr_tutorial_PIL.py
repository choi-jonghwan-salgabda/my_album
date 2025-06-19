import easyocr,os;from PIL import Image;import numpy as np

img_path='/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/img/책표지_총류_000002.jpg'

# PIL로 이미지 열고 numpy 배열로 변환 (RGB)
image=Image.open(img_path).convert("RGB")
image_np=np.array(image)

result=easyocr.Reader(['ko']).readtext(image_np)#1.raw output
for tuple in result:print(tuple)
result=easyocr.Reader(['ko']).readtext(img_path,detail=0);print(result)#2.text만 list로
result=easyocr.Reader(['ko']).readtext(img_path,detail=0,paragraph=True);print(result)#3.text만 자연스러운 문장으로