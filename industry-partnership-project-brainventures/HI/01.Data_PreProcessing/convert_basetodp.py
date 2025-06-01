#!/usr/bin/env python3
"""
커스텀 JSON 형식을 DPText-DETR 형식으로 변환하는 스크립트
모든 텍스트 박스(xxx 포함)를 유지합니다.
"""

import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d  # 이 줄을 추가
from detectron2.structures.boxes import BoxMode

def parse_args():
    parser = argparse.ArgumentParser(description="커스텀 JSON을 DPText-DETR JSON 형식으로 변환")
    parser.add_argument("--input", required=True, help="입력 어노테이션 파일 경로")
    parser.add_argument("--output", required=True, help="출력 어노테이션 파일 경로")
    parser.add_argument("--image-dir", default="", help="이미지 디렉토리 경로")
    return parser.parse_args()

def convert_to_dptext_detr_format(input_json):
    """
    커스텀 JSON 형식을 DPText-DETR 형식으로 변환
    
    Args:
        input_json: 입력 어노테이션 JSON 데이터
        
    Returns:
        DPText-DETR 형식의 어노테이션 JSON 데이터
    """
    # 결과 데이터 초기화
    result = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 이미지와 어노테이션 ID 초기화
    image_id = 1
    ann_id = 1
    
    # 각 이미지에 대해 처리
    for filename, img_data in tqdm(input_json["images"].items(), desc="이미지 변환 중"):
        # 이미지 정보 추가
        result["images"].append({
            "id": image_id,
            "file_name": filename,
            "height": img_data["img_h"],
            "width": img_data["img_w"]
        })
        
        result["categories"].append({"id": image_id, "name": "text"})
        
        # 이미지 내 텍스트 어노테이션 처리
        for word_id, word_info in img_data["words"].items():
            # 모든 텍스트 처리 ("xxx" 포함)
            text = word_info["text"]
                
            # 다각형 좌표 추출 및 평탄화
            points = word_info["points"]
            flat_points = []
            for point in points:
                flat_points.extend(point)  # [x1, y1, x2, y2, ...] 형식으로 평탄화
            
            # 경계 상자 계산 (x, y, width, height)
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            width, height = x_max - x_min, y_max - y_min
            
            # 면적 계산
            area = width * height
            
            # 텍스트 인코딩 (ASCII 코드)
            rec = [ord(c) for c in text]
            
            # 어노테이션 추가
            annotation = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,  # text 카테고리
                "bbox": [x_min, y_min, width, height],
                "area": area,
                "iscrowd": 0,
                "polys": flat_points,
                "segmentation": [flat_points],
                "rec": rec,
                "text": text  # 원본 텍스트도 유지
            }
            
            # 추가 정보 유지
            if "orientation" in word_info:
                annotation["orientation"] = word_info["orientation"]
            if "language" in word_info:
                annotation["language"] = word_info["language"]
                
            result["annotations"].append(annotation)
            ann_id += 1
            
        image_id += 1
        
    return result


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

def modify_json_annotations(input_json_path, output_json_path, n_points=16):
    """
    JSON 파일의 구조를 유지하면서 annotation만 수정하여 저장
    
    Args:
        input_json_path (str): 입력 JSON 파일 경로
        output_json_path (str): 출력 JSON 파일 경로
        n_points (int): 폴리곤 리샘플링할 점 개수
    
    Returns:
        bool: 성공 여부
    """
    print(f"원본 JSON 파일 로드 중: {input_json_path}")
    
    # 원본 JSON 파일 로드
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"JSON 파일 로드 실패: {e}")
        return False
    
    # 데이터 형식 확인 (리스트 또는 객체)
    if isinstance(data, list):
        print("리스트 형식 JSON 처리 중...")
        # 수정된 데이터 초기화
        modified_data = []
        
        # 각 레코드 처리
        for record in tqdm(data, desc="레코드 처리 중"):
            modified_record = record.copy()  # 원본 레코드 복사
            
            # annotations 필드가 있는 경우만 처리
            if "annotations" in record:
                valid_annotations = []
                
                for anno in record["annotations"]:
                    # 기본 정보 복사
                    processed_anno = anno.copy()
                    valid_polygon_found = False
                    
                    # 다양한 다각형 데이터 형식 처리
                    if "polygons" in anno:
                        resampled_polygons = []
                        
                        for poly in anno["polygons"]:
                            # 다각형 유효성 검사 후 리샘플링
                            if is_valid_polygon(poly):
                                resampled_poly = resample_polygon_to_fixed_points(poly, n_points=n_points)
                                resampled_polygons.append(resampled_poly)
                                valid_polygon_found = True
                        
                        if valid_polygon_found:
                            processed_anno["polygons"] = resampled_polygons
                    
                    # segmentation 필드가 있는 경우
                    elif "segmentation" in anno:
                        resampled_segmentation = []
                        
                        for poly in anno["segmentation"]:
                            if is_valid_polygon(poly):
                                resampled_poly = resample_polygon_to_fixed_points(poly, n_points=n_points)
                                resampled_segmentation.append(resampled_poly)
                                valid_polygon_found = True
                        
                        if valid_polygon_found:
                            processed_anno["polygons"] = resampled_segmentation
                            # 원본 형식 유지를 위해 segmentation 필드도 업데이트
                            processed_anno["segmentation"] = resampled_segmentation
                    
                    # 다각형 정보 없이 bbox만 있는 경우
                    if (not valid_polygon_found or "polygons" not in processed_anno) and "bbox" in anno:
                        bbox = anno["bbox"]
                        bbox_mode = anno.get("bbox_mode", BoxMode.XYWH_ABS)
                        
                        # XYWH 형식을 XYXY 형식으로 변환
                        if bbox_mode == BoxMode.XYWH_ABS:
                            x, y, w, h = bbox
                            box_poly = [x, y, x + w, y, x + w, y + h, x, y + h]
                        else:
                            x1, y1, x2, y2 = bbox
                            box_poly = [x1, y1, x2, y1, x2, y2, x1, y2]
                        
                        # bbox에서 생성한 다각형을 리샘플링
                        resampled_poly = resample_polygon_to_fixed_points(box_poly, n_points=n_points)
                        processed_anno["polygons"] = [resampled_poly]
                        valid_polygon_found = True
                    
                    # 유효한 다각형이 있는 경우만 주석에 추가
                    if valid_polygon_found:
                        valid_annotations.append(processed_anno)
                
                # 수정된 주석을 레코드에 업데이트
                modified_record["annotations"] = valid_annotations
            
            # 수정된 레코드를 결과에 추가
            modified_data.append(modified_record)
        
        # 수정된 데이터 저장
        try:
            with open(output_json_path, 'w') as f:
                json.dump(modified_data, f, indent=2)
            print(f"수정된 JSON 파일 저장 완료: {output_json_path}")
            print(f"총 {len(modified_data)}개 레코드 처리됨")
            return True
        except Exception as e:
            print(f"JSON 파일 저장 중 오류: {e}")
            return False
    
    elif isinstance(data, dict) and "images" in data and "annotations" in data:
        print("COCO 형식 JSON 처리 중...")
        # 원본 구조 복사
        modified_data = {
            "images": data["images"],
            "annotations": []
        }
        
        # 카테고리 정보가 있으면 유지
        if "categories" in data:
            modified_data["categories"] = data["categories"]
        
        # 기타 메타데이터 필드 유지
        for key, value in data.items():
            if key not in ["images", "annotations", "categories"]:
                modified_data[key] = value
        
        # 이미지 ID별 주석 그룹화
        annotations_by_image = {}
        for anno in data["annotations"]:
            img_id = anno["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(anno)
        
        # 이미지별 주석 처리
        for img_id, annos in tqdm(annotations_by_image.items(), desc="이미지별 주석 처리 중"):
            for anno in annos:
                processed_anno = anno.copy()  # 원본 주석 복사
                valid_polygon_found = False
                
                # segmentation 필드가 있는 경우
                if "segmentation" in anno:
                    resampled_segmentation = []
                    
                    for poly in anno["segmentation"]:
                        if is_valid_polygon(poly):
                            resampled_poly = resample_polygon_to_fixed_points(poly, n_points=n_points)
                            resampled_segmentation.append(resampled_poly)
                            valid_polygon_found = True
                    
                    if valid_polygon_found:
                        processed_anno["segmentation"] = resampled_segmentation
                
                # 다각형 정보 없이 bbox만 있는 경우
                if not valid_polygon_found and "bbox" in anno:
                    bbox = anno["bbox"]
                    x, y, w, h = bbox
                    box_poly = [x, y, x + w, y, x + w, y + h, x, y + h]
                    
                    # bbox에서 생성한 다각형을 리샘플링
                    resampled_poly = resample_polygon_to_fixed_points(box_poly, n_points=n_points)
                    processed_anno["segmentation"] = [resampled_poly]
                    valid_polygon_found = True
                
                # 수정된 주석을 결과에 추가
                if valid_polygon_found or "segmentation" in processed_anno:
                    modified_data["annotations"].append(processed_anno)
        
        # 수정된 데이터 저장
        try:
            with open(output_json_path, 'w') as f:
                json.dump(modified_data, f, indent=2)
            print(f"수정된 JSON 파일 저장 완료: {output_json_path}")
            print(f"총 {len(modified_data['images'])}개 이미지, {len(modified_data['annotations'])}개 주석 처리됨")
            return True
        except Exception as e:
            print(f"JSON 파일 저장 중 오류: {e}")
            return False
    
    else:
        print(f"지원하지 않는 JSON 형식입니다.")
        return False

def main():
    args = parse_args()
    
    # 입력 JSON 파일 읽기
    print(f"입력 파일 '{args.input}' 읽는 중...")
    with open(args.input, 'r', encoding='utf-8') as f:
        try:
            input_json = json.load(f)
        except json.JSONDecodeError:
            print("JSON 파일을 파싱하는 중 오류가 발생했습니다. 파일 형식을 확인하세요.")
            return
    
    # DPText-DETR 형식으로 변환
    print("DPText-DETR 형식으로 변환 중...")
    result_json = convert_to_dptext_detr_format(input_json)
    
    # 변환 통계 출력
    print(f"변환 완료: {len(result_json['images'])} 이미지, {len(result_json['annotations'])} 어노테이션")
    
    # 출력 JSON 파일 저장
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"결과 저장 중: '{args.output}'")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    
    print("변환 완료!")

    modified_output_path = os.path.join(os.path.dirname(args.output), "modified.json")
    #modify_json_annotations(args.output, modified_output_path, n_points=16)

if __name__ == "__main__":
    main()

#conda activate DPText-DETR
#python /data/ephemeral/home/industry-partnership-project-brainventures/HI/01.Data_PreProcessing/convert_basetodp.py --input /data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/jsons/train.json --output /data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/jsons/train_poly_pos.json
#python /data/ephemeral/home/industry-partnership-project-brainventures/HI/01.Data_PreProcessing/convert_basetodp.py --input /data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/jsons/val.json --output /data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/jsons/valid_poly_pos.json