# data_path 경로 수정해주시면 되세요 
import os
import json
from glob import glob

def merge_vl1_jsons(json_root, output_path):
    merged = {}
    
    # 모든 JSON 파일 찾기
    json_files = glob(os.path.join(json_root, "**", "*.json"), recursive=True)
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    # 파일 경로가 없는 경우 확인
    if len(json_files) == 0:
        print(f"경고: '{json_root}' 경로에서 JSON 파일을 찾을 수 없습니다.")
        print(f"현재 작업 디렉토리: {os.getcwd()}")
        print(f"디렉토리 존재 여부: {os.path.exists(json_root)}")
        
        # 디렉토리가 존재하면 내용 확인
        if os.path.exists(json_root):
            print(f"'{json_root}' 디렉토리 내용:")
            for item in os.listdir(json_root):
                print(f" - {item}")
    
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            print(f"파일 처리 중: {json_path}")
            print(f"키: {list(data.keys())}")
            
            # 데이터 구조 확인
            if "images" not in data or not data["images"]:
                print(f"경고: {json_path}에 'images' 키가 없거나 비어 있습니다.")
                continue
                
            if "annotations" not in data or not data["annotations"]:
                print(f"경고: {json_path}에 'annotations' 키가 없거나 비어 있습니다.")
                continue
            
            file_name = data["images"][0]["file_name"]
            text_boxes = []
            
            for ann in data["annotations"]:
                if "bbox" not in ann:
                    print(f"경고: {json_path}의 어노테이션에 'bbox' 키가 없습니다.")
                    continue
                
                if "text" not in ann:
                    print(f"경고: {json_path}의 어노테이션에 'text' 키가 없습니다.")
                    continue
                    
                bbox = ann["bbox"]
                text = ann["text"]
                
                if isinstance(bbox[0], list):
                    # 여러 bbox가 있는 경우
                    for i, box in enumerate(bbox):
                        text_boxes.append({
                            "bbox": box,
                            "text": text[i] if isinstance(text, list) and i < len(text) else text
                        })
                else:
                    # 단일 bbox인 경우
                    text_boxes.append({
                        "bbox": bbox,
                        "text": text
                    })
            
            merged[file_name] = text_boxes
            
        except Exception as e:
            print(f"오류 발생: {json_path} 처리 중 - {str(e)}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    print(f"[✓] 병합 완료: {output_path} (총 {len(merged)}개 이미지 데이터)")
    if len(merged) == 0:
        print("경고: 병합된 데이터가 없습니다. 입력 경로와 JSON 파일 형식을 확인하세요.")

# 경로 확인 및 수정   -> 경로 수정해주시면 되세요 
data_path = "/data/VL1"
if not os.path.exists(data_path):
    print(f"'{data_path}' 경로가 존재하지 않습니다. 가능한 경로를 찾는 중...")
    possible_paths = [
        "/data/VL1", 
        "data/VL1", 
        "./data/VL1", 
        "../data/VL1",
        "/content/data/VL1",
        "VL1"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            print(f"대체 경로를 찾았습니다: {data_path}")
            break

# VL1 데이터셋의 JSON 파일들을 병합하여 val_labels.json 파일로 저장
merge_vl1_jsons(data_path, "val_labels.json")
