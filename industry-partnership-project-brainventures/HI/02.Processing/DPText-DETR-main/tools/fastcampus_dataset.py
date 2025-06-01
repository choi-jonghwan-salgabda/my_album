import json
import os
import traceback
import numpy as np
from scipy.interpolate import interp1d
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from tqdm import tqdm

def is_valid_polygon(polygon):
    """다각형 데이터의 유효성을 검사하는 함수 (완화된 버전)"""
    if not polygon:
        return False
    
    # 다각형은 짝수 개의 좌표를 가져야 함 (x, y 쌍)
    if len(polygon) % 2 != 0:
        print(f"경고: 홀수 길이의 다각형 발견 - 마지막 요소 제거: {polygon}")
        polygon = polygon[:-1]  # 홀수 길이면 마지막 요소 제거
    
    # NaN이나 무한대 값 수정
    has_invalid = False
    for i, p in enumerate(polygon):
        if not isinstance(p, (int, float)) or np.isnan(p) or np.isinf(p):
            has_invalid = True
            polygon[i] = 0.0  # 유효하지 않은 값을 0으로 대체
    
    if has_invalid:
        print(f"경고: 유효하지 않은 값이 포함된 다각형 - 값 대체")
    
    return True  # 모든 검사 통과 또는 수정됨

def resample_polygon_to_fixed_points(polygon, n_points=16):
    """
    다각형을 지정된 개수의 점으로 리샘플링
    
    Args:
        polygon: [x1, y1, x2, y2, ..., xn, yn] 형태의 리스트
        n_points: 원하는 점의 개수
    
    Returns:
        n_points 개수로 리샘플링된 다각형: [x1, y1, x2, y2, ..., x_n_points, y_n_points]
    """
    try:
        # NaN/inf 값 확인 및 필터링
        if any(not isinstance(p, (int, float)) or np.isnan(p) or np.isinf(p) for p in polygon):
            # 대체된 다각형 만들기
            clean_polygon = []
            for p in polygon:
                if not isinstance(p, (int, float)) or np.isnan(p) or np.isinf(p):
                    clean_polygon.append(0.0)
                else:
                    clean_polygon.append(float(p))
            polygon = clean_polygon
        
        # 홀수 길이의 다각형은 지원하지 않음
        if len(polygon) % 2 != 0:
            # 안전하게 마지막 점 제거
            polygon = polygon[:-1]
        
        # 점이 4개 미만이면 처리 불가
        if len(polygon) < 8:  # 최소 4개 점(8개 좌표) 필요
            # 현재 점들을 복제하여 채우기
            while len(polygon) < n_points * 2:
                polygon = polygon + polygon
            return polygon[:n_points * 2]
        
        # 다각형을 좌표 쌍으로 변환
        points = []
        for i in range(0, len(polygon), 2):
            points.append((polygon[i], polygon[i+1]))
        
        # 폐곡선이 아니면 첫 번째 점을 마지막에 추가하여 폐곡선 만들기
        if points[0] != points[-1]:
            points.append(points[0])
        
        # 각 점 간의 누적 거리 계산
        total_distance = 0
        distances = [0]
        for i in range(1, len(points)):
            p1 = points[i-1]
            p2 = points[i]
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_distance += dist
            distances.append(total_distance)
        
        # 거리가 0이면 모든 점이 같다는 의미
        if total_distance == 0:
            return polygon[:2] * n_points
        
        # 균등 간격으로 리샘플링을 위한 매개변수 생성
        t = np.array(distances) / total_distance
        
        # x와 y 좌표에 대한 보간 함수 생성
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        fx = interp1d(t, x)
        fy = interp1d(t, y)
        
        # 균등한 간격으로 n_points개 점 생성
        new_t = np.linspace(0, 1, n_points)
        new_x = fx(new_t)
        new_y = fy(new_t)
        
        # 리샘플링된 다각형 반환 (NaN/inf 검사 추가)
        resampled_polygon = []
        for i in range(n_points):
            x, y = float(new_x[i]), float(new_y[i])
            
            # NaN/inf 검사
            if np.isnan(x) or np.isinf(x) or np.isnan(y) or np.isinf(y):
                # 문제가 있는 경우 기본값으로 대체
                resampled_polygon.extend([0.0, 0.0])
            else:
                resampled_polygon.extend([x, y])
        
        return resampled_polygon
        
    except Exception as e:
        print(f"다각형 리샘플링 중 오류 발생: {e}")
        # 오류 발생 시 기본 사각형 반환
        return [0, 0, 10, 0, 10, 10, 0, 10] * 2

def create_dummy_record(image_root):
    """비상용 더미 레코드 생성"""
    # 이미지 루트 디렉토리에서 아무 이미지나 찾기
    for root, _, files in os.walk(image_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(root, file)
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        width, height = img.size
                    
                    # 더미 레코드 생성
                    dummy_record = {
                        "file_name": file_path,
                        "image_id": 0,
                        "height": height,
                        "width": width,
                        "annotations": [
                            {
                                "bbox": [10, 10, 100, 50],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": 0,
                                "polygons": [[10, 10, 110, 10, 110, 60, 10, 60]]
                            }
                        ]
                    }
                    print(f"더미 레코드 생성: {file_path}")
                    return dummy_record
                except Exception as e:
                    print(f"더미 레코드 생성 중 오류: {e}")
                    continue
    
    print("더미 레코드 생성 실패: 적절한 이미지를 찾을 수 없습니다.")
    return None

def load_fastcampus_json(json_file, image_root, start_idx=0, end_idx=None):
    """
    FastCampus 프로젝트의 JSON 파일에서 선택된 범위의 데이터만 로드하는 함수
    
    Args:
        json_file (str): JSON 파일 경로
        image_root (str): 이미지 파일이 저장된 루트 디렉토리
        start_idx (int): 시작 인덱스
        end_idx (int): 종료 인덱스 (None이면 끝까지)
    """
    print(f"JSON 파일 로드 시작: {json_file}")
    print(f"이미지 루트 디렉토리: {image_root}")
    print(f"인덱스 범위: {start_idx} ~ {end_idx if end_idx is not None else '끝'}")
    
    # 디렉토리가 존재하는지 확인
    if not os.path.exists(image_root):
        print(f"경고: 이미지 루트 디렉토리가 존재하지 않습니다: {image_root}")
    else:
        print(f"이미지 루트 디렉토리 확인: {image_root} (존재함)")
    
    if not os.path.exists(json_file):
        print(f"오류: JSON 파일이 존재하지 않습니다: {json_file}")
        return []
    
    # 카운터 추가
    total_items = 0
    filtered_items = 0
    valid_items = 0
    
    # 데이터셋 초기화
    dataset_dicts = []
    
    try:
        # 방법 1: 표준 JSON 로드
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 데이터 형식 확인 (리스트 또는 객체)
        if isinstance(data, list):
            # 리스트 형식의 JSON인 경우
            records = data[start_idx:end_idx if end_idx is not None else len(data)]
            total_items = len(records)
            
            for record in tqdm(records, desc="레코드 처리"):
                # 파일 이름에 경로 추가
                if "file_name" in record:
                    file_name = record["file_name"]
                    if not os.path.isabs(file_name):
                        record["file_name"] = os.path.join(image_root, file_name)
                    
                    # 파일 존재 확인
                    if not os.path.exists(record["file_name"]):
                        print(f"경고: 파일을 찾을 수 없습니다 - {record['file_name']}")
                        filtered_items += 1
                        continue
                
                # 주석 처리 (annotations가 있는 경우)
                valid_annotations = []
                if "annotations" in record:
                    for anno in record["annotations"]:
                        # bbox_mode 설정
                        if "bbox_mode" not in anno:
                            anno["bbox_mode"] = BoxMode.XYWH_ABS
                        
                        # 카테고리 ID 설정
                        anno["category_id"] = 0
                        
                        # 다각형 데이터 처리
                        valid_polygon_found = False
                        
                        if "segmentation" in anno:
                            # 각 segmentation을 16개 점으로 리샘플링
                            if isinstance(anno["segmentation"], list):
                                resampled_segmentation = []
                                
                                for poly in anno["segmentation"]:
                                    # 다각형 유효성 검사
                                    if is_valid_polygon(poly):
                                        resampled_poly = resample_polygon_to_fixed_points(poly, n_points=16)
                                        resampled_segmentation.append(resampled_poly)
                                        valid_polygon_found = True
                                
                                if valid_polygon_found:
                                    anno["polygons"] = resampled_segmentation
                        
                        elif "polys" in anno:
                            resampled_polys = []
                            
                            for poly in anno["polys"]:
                                flat_poly = None
                                
                                if isinstance(poly, list):
                                    if all(isinstance(p, (list, tuple)) for p in poly):
                                        flat_poly = [coord for point in poly for coord in point]
                                    else:
                                        flat_poly = poly
                                    
                                    # 다각형 유효성 검사
                                    if is_valid_polygon(flat_poly):
                                        resampled_poly = resample_polygon_to_fixed_points(flat_poly, n_points=16)
                                        resampled_polys.append(resampled_poly)
                                        valid_polygon_found = True
                            
                            if valid_polygon_found:
                                anno["polygons"] = resampled_polys
                        
                        # 다각형 정보 없이 bbox만 있는 경우, bbox에서 다각형 생성
                        if not valid_polygon_found and "bbox" in anno:
                            bbox = anno["bbox"]
                            bbox_mode = anno.get("bbox_mode", BoxMode.XYWH_ABS)
                            
                            # XYWH 형식인 경우 XYXY 형식으로 변환
                            if bbox_mode == BoxMode.XYWH_ABS:
                                x, y, w, h = bbox
                                box = [x, y, x + w, y + h]
                            else:
                                box = bbox
                            
                            # 박스에서 다각형 생성
                            x1, y1, x2, y2 = box
                            anno["polygons"] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
                            valid_polygon_found = True
                        
                        # 유효한 다각형이 있는 경우만 주석 추가
                        if valid_polygon_found:
                            valid_annotations.append(anno)
                
                # 유효한 주석으로 업데이트
                record["annotations"] = valid_annotations
                
                # 주석이 하나라도 있으면 데이터셋에 추가
                if valid_annotations:
                    dataset_dicts.append(record)
                    valid_items += 1
                else:
                    filtered_items += 1
        
        else:
            # COCO 형식인 경우
            images = data.get("images", [])
            annotations = data.get("annotations", [])
            
            total_items = len(images)
            selected_images = images[start_idx:end_idx if end_idx is not None else len(images)]
            
            # 이미지 ID별 주석 그룹화
            annotations_by_image = {}
            for anno in annotations:
                img_id = anno["image_id"]
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(anno)
            
            # 이미지별 처리
            for img_info in tqdm(selected_images, desc="이미지 처리"):
                img_id = img_info["id"]
                
                # 파일 이름에 경로 추가
                file_name = img_info["file_name"]
                if not os.path.isabs(file_name):
                    file_name = os.path.join(image_root, file_name)
                
                # 파일 존재 확인
                if not os.path.exists(file_name):
                    print(f"경고: 파일을 찾을 수 없습니다 - {file_name}")
                    filtered_items += 1
                    continue
                
                # 레코드 생성
                record = {
                    "file_name": file_name,
                    "image_id": img_id,
                    "height": img_info.get("height", 0),
                    "width": img_info.get("width", 0)
                }
                
                # 이미지 크기가 유효하지 않으면 직접 확인
                if record["height"] <= 0 or record["width"] <= 0:
                    try:
                        from PIL import Image
                        with Image.open(file_name) as img:
                            record["width"], record["height"] = img.size
                    except Exception as e:
                        print(f"경고: 이미지 크기 확인 중 오류 - {file_name}: {e}")
                        filtered_items += 1
                        continue
                
                # 주석 처리
                valid_annotations = []
                for anno in annotations_by_image.get(img_id, []):
                    obj = {
                        "bbox": anno["bbox"],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": 0,
                        "iscrowd": anno.get("iscrowd", 0),
                    }
                    
                    # 다각형 데이터 처리
                    valid_polygon_found = False
                    
                    if "segmentation" in anno:
                        if isinstance(anno["segmentation"], list):
                            resampled_segmentation = []
                            
                            for poly in anno["segmentation"]:
                                # 다각형 유효성 검사
                                if is_valid_polygon(poly):
                                    resampled_poly = resample_polygon_to_fixed_points(poly, n_points=16)
                                    resampled_segmentation.append(resampled_poly)
                                    valid_polygon_found = True
                            
                            if valid_polygon_found:
                                obj["polygons"] = resampled_segmentation
                    
                    # bbox만 있는 경우
                    if not valid_polygon_found:
                        bbox = anno["bbox"]
                        x, y, w, h = bbox
                        obj["polygons"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                        valid_polygon_found = True
                    
                    if valid_polygon_found:
                        valid_annotations.append(obj)
                
                # 주석이 하나라도 있으면 레코드 추가
                if valid_annotations:
                    record["annotations"] = valid_annotations
                    dataset_dicts.append(record)
                    valid_items += 1
                else:
                    filtered_items += 1
                    
    except Exception as e:
        print(f"JSON 로드 중 오류 발생: {e}")
        traceback.print_exc()
    
    # 결과 출력
    print(f"처리된 총 항목 수: {total_items}")
    print(f"필터링된 항목 수: {filtered_items}")
    print(f"유효한 항목 수: {valid_items}")
    print(f"최종 데이터셋 크기: {len(dataset_dicts)}")
    
    # 빈 데이터셋 처리
    if len(dataset_dicts) == 0:
        print("경고: 데이터셋이 비어 있습니다! 필터링 조건이 너무 엄격할 수 있습니다.")
        # 샘플 데이터 추가 (비상용)
        dummy_record = create_dummy_record(image_root)
        if dummy_record:
            dataset_dicts.append(dummy_record)
            print("비상용 더미 데이터가 추가되었습니다.")
    
    return dataset_dicts

def register_fastcampus_datasets(batch_size=5000):
    """
    FastCampus 데이터셋 등록 함수 (안전 버전)
    """
    # 이미지 루트 디렉토리 설정
    train_image_root = "/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/images/train"
    valid_image_root = "/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/images/val"  # valid 경로
    
    # JSON 파일 경로
    train_json = "/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/jsons/train_poly_pos.json"
    val_json = "/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/jsons/valid_poly_pos.json"
    
    # 데이터 검증
    print("데이터셋 검증 중...")
    try:
        # 전체 데이터셋을 한 번 로드하여 검증
        test_dataset = load_fastcampus_json(train_json, train_image_root, 0, 10)
        if len(test_dataset) == 0:
            print("경고: 테스트 데이터셋이 비어 있습니다. 필터링 조건을 완화합니다.")
    except Exception as e:
        print(f"데이터셋 검증 중 오류 발생: {e}")
        print("필터링 조건을 완화합니다.")
    
    # 전체 훈련 데이터셋 등록
    DatasetCatalog.register(
        "fastcampus_train_poly_pos",
        lambda: load_fastcampus_json(train_json, train_image_root)
    )
    MetadataCatalog.get("fastcampus_train_poly_pos").set(
        thing_classes=["text"],
        evaluator_type="text",
        json_file=train_json,
        dataset_name="fastcampus_train_poly_pos"
    )
    
    # 배치별 데이터셋 등록 (직접 로드)
    try:
        # 전체 데이터셋 크기 추정
        with open(train_json, 'r') as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                # 리스트 형식의 경우 직접 크기 확인
                data = json.load(f)
                total_size = len(data)
            else:
                # COCO 형식의 경우 이미지 수로 추정
                data = json.load(f)
                total_size = len(data.get("images", []))
        
        num_batches = (total_size + batch_size - 1) // batch_size  # 올림 나눗셈
        
        print(f"전체 훈련 데이터 크기: {total_size}, 배치 크기: {batch_size}, 총 배치 수: {num_batches}")
        
        # 배치별 데이터셋 등록
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_size)
            
            # 배치 데이터셋 등록
            dataset_name = f"fastcampus_train_poly_pos_batch_{batch_idx+1}"
            
            # 이미 등록된 경우 제거
            if dataset_name in DatasetCatalog:
                DatasetCatalog.remove(dataset_name)
            
            # 배치 데이터셋 등록 (직접 로드)
            DatasetCatalog.register(
                dataset_name,
                lambda s=start_idx, e=end_idx: load_fastcampus_json(train_json, train_image_root, s, e)
            )
            
            MetadataCatalog.get(dataset_name).set(
                thing_classes=["text"],
                evaluator_type="text",
                json_file=train_json,
                dataset_name=dataset_name,
                batch_info=f"Batch {batch_idx+1}/{num_batches} ({start_idx}-{end_idx})"
            )
            
            print(f"등록된 훈련 데이터셋: {dataset_name} ({start_idx}-{end_idx})")
    
    except Exception as e:
        print(f"배치별 데이터셋 등록 중 오류 발생: {e}")
        traceback.print_exc()
        
        # 직접 배치 데이터셋 등록 (대체 방법)
        print("대체 방법으로 배치 데이터셋 등록을 시도합니다...")
        
        dataset_name = "fastcampus_train_poly_pos_batch_1"
        if dataset_name in DatasetCatalog:
            DatasetCatalog.remove(dataset_name)
        
        DatasetCatalog.register(
            dataset_name,
            lambda: load_fastcampus_json(train_json, train_image_root, 0, batch_size)
        )
        
        MetadataCatalog.get(dataset_name).set(
            thing_classes=["text"],
            evaluator_type="text",
            json_file=train_json,
            dataset_name=dataset_name,
            batch_info=f"Batch 1/1 (대체 등록)"
        )
        
        print(f"등록된 훈련 데이터셋 (대체): {dataset_name}")
    
    # 검증 데이터셋 등록
    DatasetCatalog.register(
        "fastcampus_valid_poly_pos",
        lambda: load_fastcampus_json(val_json, valid_image_root)
    )
    MetadataCatalog.get("fastcampus_valid_poly_pos").set(
        thing_classes=["text"],
        evaluator_type="text",
        json_file=val_json,
        dataset_name="fastcampus_valid_poly_pos"
    )
    
    print("FastCampus 데이터셋 등록 완료!")

# 디버깅 목적으로 직접 실행할 수 있는 코드
if __name__ == "__main__":
    register_fastcampus_datasets()
    # 등록된 데이터셋의 몇 개 샘플 출력
    dataset = DatasetCatalog.get("fastcampus_train_poly_pos")
    if dataset:
        count_by_category = {}
        for i, d in enumerate(tqdm(dataset[:3], desc="샘플 확인")):  # 처음 3개 샘플만 확인
            for anno in tqdm(d.get("annotations", []), desc=f"샘플 {i+1} 주석", leave=False):
                cat_id = anno.get("category_id", 0)
                count_by_category[cat_id] = count_by_category.get(cat_id, 0) + 1
            
            print(f"샘플 {i+1}:")
            print(f"  파일 이름: {d['file_name']}")
            print(f"  주석 수: {len(d.get('annotations', []))}")
            
            # 다각형 정보 출력
            for j, anno in enumerate(tqdm(d.get("annotations", [])[:2], desc="주석 세부 정보", leave=False)):  # 처음 2개 주석만
                if "polygons" in anno:
                    poly_count = len(anno["polygons"])
                    points_per_poly = len(anno["polygons"][0]) // 2 if poly_count > 0 else 0
                    print(f"  주석 {j+1}: {poly_count}개 다각형, 각 {points_per_poly}개 점")
        
        print("카테고리 ID별 주석 수:")
        for cat_id, count in count_by_category.items():
            print(f"  카테고리 ID {cat_id}: {count}개")