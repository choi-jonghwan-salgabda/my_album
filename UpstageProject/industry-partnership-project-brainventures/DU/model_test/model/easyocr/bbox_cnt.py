import json # bbox 숫자 비교

with open ("/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/label/4.json","r") as f:
    data = json.load(f)
print(f"길이: {len(data['annotations'])}")

with open ("/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/output/easyocr/annotated_4.json","r") as f:
    data = json.load(f)
print(f"길이: {len(data)}")


with open ("/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/label/000001.json","r") as f:
    data = json.load(f)
print(f"길이: {len(data['annotations'])}")

with open ("/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/output/easyocr/annotated_책표지_총류_000001.json","r") as f:
    data = json.load(f)
print(f"길이: {len(data)}")