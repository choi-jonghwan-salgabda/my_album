import os
import json
import random
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict

def load_json_file(directory_path):
    """
    지정된 디렉토리에서 모든 JSON 파일을 로드하여 
    {subclass: {filename: data}} 형식의 데이터로 반환
    """
    json_data = defaultdict(dict)
    file_count = 0
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 메타데이터에서 subclass 정보 추출
                if 'metadata' in data and len(data['metadata']) > 0:
                    subclass = data['metadata'][0].get('subclass')
                    if subclass:
                        json_data[subclass][filename] = data
                        file_count += 1
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"총 {file_count}개 파일 로드 완료")
    
    # 각 subclass별 파일 개수 계산
    distribution = {}
    for subclass, files in json_data.items():
        distribution[subclass] = len(files)
    
    print("\n각 subclass별 파일 개수:")
    for subclass, count in distribution.items():
        print(f"{subclass}: {count}개")
    
    # 내보낼 파일 수 계산
    export_count_list = {}
    test_ratio = 0.1
    val_ratio = 0.1
    
    for subclass, count in distribution.items():
        # 테스트 및 검증 세트에 들어갈 파일 수 계산
        export_count = int(count * 1/6)
        # 최소한 1개 이상의 파일을 내보내도록 함
        export_count_list[subclass] = max(1, export_count) if count > 0 else 0
    
    return json_data, export_count_list

def check_bbox(annotation):
  # bbox가 없거나 None이거나 리스트가 아닌 경우 건너뛰기
    if 'bbox' not in annotation or annotation['bbox'] is None or len(annotation['bbox']) == 0:
        return False

    if ('bbox' not in annotation or 
        annotation['bbox'] is None or 
        not isinstance(annotation['bbox'], list) or 
        len(annotation['bbox']) != 4 or 
        'text' not in annotation):
            return False
                
    bbox = annotation['bbox']
            
    # bbox 값이 숫자인지 확인 
    if not all(isinstance(coord, (int, float)) for coord in bbox):
        return False
        
    return True    


def remove_duplicates_in_annotations(data):
    """
    Remove annotations with duplicate bbox and text values.
    Returns the data with duplicates removed.
    """
    if 'annotations' not in data:
        return data
    
    # Create a set to track unique combinations of bbox and text
    unique_entries = set()
    filtered_annotations = []
    
    # Filter out duplicates
    for annotation in data['annotations']:
        # bbox가 없거나 null인 경우 건너뛰기
        if not check_bbox(annotation):
            continue

        bbox_tuple = tuple(annotation['bbox'])
        text = annotation['text']
        
        # Create a unique identifier from both bbox and text
        unique_id = (bbox_tuple, text)
        
        if unique_id not in unique_entries:
            unique_entries.add(unique_id)
            filtered_annotations.append(annotation)
    
    data['annotations'] = filtered_annotations
    return data

def find_and_move_image_as_jpg(base_name, src_dir, dst_dir):
    """
    여러 확장자(.jpg, .jpeg, .JPG 등)로 이미지 파일을 찾아서
    .jpg 확장자로 변환하여 대상 디렉토리로 이동합니다.
    """
    # 확인할 이미지 확장자 리스트
    extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
    
    for ext in extensions:
        src_path = os.path.join(src_dir, base_name + ext)
        if os.path.exists(src_path):
            # 항상 .jpg 확장자로 통일
            dst_path = os.path.join(dst_dir, base_name + '.jpg')
            shutil.move(src_path, dst_path)
            return True, src_path, dst_path  # 성공 시 경로 반환
    
    # 모든 확장자로 시도했지만 파일을 찾지 못한 경우
    return False, None, None

def convert_data_format(input_data, is_test = False):
    # 입력 JSON 파일 로드
    output_data = {"images": {}}

    # 파일 이름별로 데이터 처리
    for file_name, file_data in input_data.items():
        # JSON 파일 이름에서 이미지 파일 이름으로 변환
        img_file_name = file_name.replace('.json', '.jpg')
        
        if 'annotations' not in file_data or 'images' not in file_data:
            continue
            
        # 이미지 크기 정보 가져오기
        img_width = 1000  # 기본값
        img_height = 1000  # 기본값
        
        if len(file_data['images']) > 0:
            img_info = file_data['images'][0]
            img_width = img_info.get('width', 1000)
            img_height = img_info.get('height', 1000)
        
        # 이미지 데이터 초기화
        output_data["images"][img_file_name] = {
            "words": {},
            "img_w": img_width,
            "img_h": img_height
        }
        
        if is_test == True:
            continue
        
        subclass = "Horizontal"
        parts = file_name.split('_')
        if len(parts) >= 2:
            subclass = parts[1]
        
        for i, annotation in enumerate(file_data['annotations']):
            # bbox가 없거나 None이거나 리스트가 아닌 경우 건너뛰기
            if ('bbox' not in annotation or 
                annotation['bbox'] is None or 
                not isinstance(annotation['bbox'], list) or 
                len(annotation['bbox']) != 4 or 
                'text' not in annotation):
                continue
                
            bbox = annotation['bbox']
            
            # bbox 값이 숫자인지 확인 
            if not all(isinstance(coord, (int, float)) for coord in bbox):
                continue
                
            text = annotation['text']
            
            # word_id 생성 (4자리 숫자로 패딩)
            word_id = f"{i+1:04d}"
            
            try:
                # 좌표 포인트 생성
                points = [
                    [bbox[0], bbox[1]],                    # 좌상단
                    [bbox[0] + bbox[2], bbox[1]],          # 우상단
                    [bbox[0] + bbox[2], bbox[1] + bbox[3]], # 우하단
                    [bbox[0], bbox[1] + bbox[3]]           # 좌하단
                ]
                
                # words 객체에 데이터 추가
                output_data["images"][img_file_name]["words"][word_id] = {
                    "text": text,
                    "points": points,
                    "orientation": subclass,  # 파일 이름에서 추출한 subclass 사용
                    "language": ["ko"]
                }
            except Exception as e:
                print(f"좌표 처리 중 오류 발생 (파일: {file_name}, annotations[{i}]): {e}")
                print(f"문제의 bbox: {bbox}")
                continue
    
    print(f"변환 완료: {len(output_data['images'])}개 이미지")
    return output_data

def split_train_test_val(json_data, export_count_list, src_image_dir, json_dir, image_dir):
    np.random.seed(42)  # numpy 랜덤 시드 설정
    random.seed(42)     # 기존 random 시드도 유지

    stats = {'train': 0, 'test': 0, 'val': 0}
    subclass_stats = {}

    train_data = {}
    test_data = {}
    val_data = {}

   
    # 출력 이미지 디렉토리 생성
    train_img_dir = os.path.join(image_dir, 'train')
    test_img_dir = os.path.join(image_dir, 'test')
    val_img_dir = os.path.join(image_dir, 'val')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)


    for subclass, count in export_count_list.items():
        if subclass not in subclass_stats:
            subclass_stats[subclass] = {'train': 0, 'test': 0, 'val': 0}
            
        if subclass in json_data and count > 0:
            all_files = list(json_data[subclass].keys())
            
            selected_files, remaining_files = train_test_split(
                all_files, 
                train_size=count,
                random_state=42
            )
                
            test_files, val_files = train_test_split(
                selected_files,
                test_size=0.5,
                random_state=42
            )
                
            train_files = remaining_files

            # 각 파일의 데이터를 해당 분할 데이터 컬렉션에 추가
            for file_name in train_files:
                if file_name in json_data[subclass]:
                    train_data[file_name] = json_data[subclass][file_name]
                    stats['train'] += 1
                    subclass_stats[subclass]['train'] += 1
                    
                    # 이미지 파일 이동 (파일 이름에서 .json 확장자 제거)
                    base_name = file_name.replace('.json', '')
                    found, src_path, dst_path = find_and_move_image_as_jpg(base_name, src_image_dir, train_img_dir)
                    
                    if not found:
                        print(f"경고: 이미지 파일을 찾을 수 없음: {base_name}.*")
            
            for file_name in test_files:
                if file_name in json_data[subclass]:
                    test_data[file_name] = json_data[subclass][file_name]
                    stats['test'] += 1
                    subclass_stats[subclass]['test'] += 1
                    
                    # 이미지 파일 이동
                    base_name = file_name.replace('.json', '')
                    found, src_path, dst_path = find_and_move_image_as_jpg(base_name, src_image_dir, test_img_dir)
                    
                    if not found:
                        print(f"경고: 이미지 파일을 찾을 수 없음: {base_name}.*")
            
            for file_name in val_files:
                if file_name in json_data[subclass]:
                    val_data[file_name] = json_data[subclass][file_name]
                    stats['val'] += 1
                    subclass_stats[subclass]['val'] += 1
                    
                    # 이미지 파일 이동
                    base_name = file_name.replace('.json', '')
                    found, src_path, dst_path = find_and_move_image_as_jpg(base_name, src_image_dir, val_img_dir)
                    
                    if not found:
                        print(f"경고: 이미지 파일을 찾을 수 없음: {base_name}.*")

            print(f"{subclass}: train {len(train_files)}개, test {len(test_files)}개, val {len(val_files)}개")
    

    train_output_data = convert_data_format(train_data)
    val_output_data = convert_data_format(val_data)
    test_output_data = convert_data_format(test_data, True)

    # 각 분할별 데이터를 JSON 파일로 저장
    with open(os.path.join(json_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_output_data, f, ensure_ascii=False, indent=2)
   
    with open(os.path.join(json_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_output_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(json_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_output_data, f, ensure_ascii=False, indent=2)
   
    # 카테고리별 통계 출력
    print("\n각 카테고리별 분할 결과:")
    for subclass, counts in subclass_stats.items():
        print(f"{subclass}: train {counts['train']}개, test {counts['test']}개, val {counts['val']}개")
    
    print(f"\n총 분할 결과: train {stats['train']}개, test {stats['test']}개, val {stats['val']}개")
    print(f"이미지 파일 이동 완료: {stats['train']}개 (train), {stats['test']}개 (test), {stats['val']}개 (val)")
    
    return stats

if __name__ == "__main__":
    load_json_path = '/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/json'
    load_image_path = '/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/image'


    output_json_dir = '/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/jsons'
    output_image_dir = '/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/images'
    
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    json_data, export_count_list = load_json_file(load_json_path)
    split_train_test_val(json_data, export_count_list, load_image_path, output_json_dir, output_image_dir)
    
    print("모든 작업이 완료되었습니다.")