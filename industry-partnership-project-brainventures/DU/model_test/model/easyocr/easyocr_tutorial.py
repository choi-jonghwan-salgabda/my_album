import easyocr

img_path='/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/img/책표지_총류_000002.jpg'
output_path="/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/output/easyocr"

result=easyocr.Reader(['ko']).readtext(img_path)#1.raw output
for tuple in result:print(tuple)
result=easyocr.Reader(['ko']).readtext(img_path,detail=0);print(result)#2.text만 list로
result=easyocr.Reader(['ko']).readtext(img_path,detail=0,paragraph=True);print(result)#3.text만 자연스러운 문장으로