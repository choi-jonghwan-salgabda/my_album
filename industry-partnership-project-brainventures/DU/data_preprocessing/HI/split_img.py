import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def load_image_names_from_json(json_path):
    """JSON 파일에서 이미지 이름 목록을 로드합니다."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # JSON 구조가 {'images': {'image_name': {...}}} 형태라고 가정
            if 'images' in data:
                return list(data['images'].keys())
            else:
                return []
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []

def create_directories(base_path):
    """필요한 디렉토리를 생성합니다."""
    for subdir in ['train', 'test', 'val']:
        path = os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

def move_images(src_dir, dest_base_dir, test_images, train_images, val_images):
    """이미지를 해당하는 train/test/val 폴더로 이동하거나 복사합니다."""
    # 모든 이미지 파일 찾기
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        image_files.extend(list(Path(src_dir).glob(f'**/*{ext}')))
        image_files.extend(list(Path(src_dir).glob(f'**/*{ext.upper()}')))
    
    print(f"Found {len(image_files)} image files in {src_dir}")
    
    # 이미지 분류 결과 카운트
    counts = {'test': 0, 'train': 0, 'val': 0, 'unmatched': 0}
    
    # 이미지 이동 또는 복사
    for img_path in tqdm(image_files, desc="Processing images"):
        img_name = img_path.name
        
        # 어떤 세트에 속하는지 결정
        if img_name in test_images:
            target_dir = os.path.join(dest_base_dir, 'test')
            counts['test'] += 1
        elif img_name in train_images:
            target_dir = os.path.join(dest_base_dir, 'train')
            counts['train'] += 1
        elif img_name in val_images:
            target_dir = os.path.join(dest_base_dir, 'val')
            counts['val'] += 1
        else:
            # 어떤 세트에도 속하지 않는 경우 (선택적으로 처리)
            counts['unmatched'] += 1
            continue
        
        # 이미지 복사 또는 이동
        target_path = os.path.join(target_dir, img_name)
        shutil.move(img_path, target_path)
    
    return counts

def main():
    # 경로 설정
    base_path = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project"
    src_image_dir = os.path.join(base_path, "image")
    dest_base_dir = os.path.join(base_path,"images")  # 상위 폴더에 train/test/val 생성
    json_dir = os.path.join(base_path, "jsons")
    
    # JSON 파일 경로
    test_json = os.path.join(json_dir, "test.json")
    train_json = os.path.join(json_dir, "train.json")
    val_json = os.path.join(json_dir, "val.json")
    
    # 각 JSON 파일에서 이미지 이름 로드
    print("Loading image names from JSON files...")
    test_images = set(load_image_names_from_json(test_json))
    train_images = set(load_image_names_from_json(train_json))
    val_images = set(load_image_names_from_json(val_json))
    
    print(f"Found {len(test_images)} test images, {len(train_images)} train images, {len(val_images)} validation images in JSON files")
    
    # 디렉토리 생성
    create_directories(dest_base_dir)
    
    # 이미지 복사 (이동하려면 copy_only=False로 설정)
    counts = move_images(src_image_dir, dest_base_dir, test_images, train_images, val_images)
    
    # 결과 보고
    print("\nProcess completed!")
    print(f"Test images: {counts['test']}")
    print(f"Train images: {counts['train']}")
    print(f"Validation images: {counts['val']}")
    print(f"Unmatched images: {counts['unmatched']}")

if __name__ == "__main__":
    main()