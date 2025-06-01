import os
import json
from collections import defaultdict, Counter
import pandas as pd
from scipy.spatial.distance import pdist
import numpy as np

training_distance = 382

def analyze_hierarchical_json_counts(root_path, output_file="json_counts_by_detail.csv", export_lists=True):
    """
    계층적 구조로 JSON 파일 개수를 분석하여 세부 카테고리별 CSV 파일로 저장합니다.
    유효한 어노테이션이 3개 이상 있고, 가장 작은 바운딩 박스가 가장 큰 바운딩 박스의 반 이하인 파일만 카운트합니다.
    export_lists=True인 경우, 각 카테고리별로 유효한 파일 목록을 별도 파일로 저장합니다.
    
    Args:
        root_path: 파일 탐색을 시작할 루트 디렉토리 경로
        output_file: 저장할 CSV 파일 이름
        export_lists: 세부 카테고리별 파일 목록을 저장할지 여부
    """
    # 세부 카테고리별 유효한 어노테이션이 있는 JSON 파일 카운터
    detail_counts = Counter()
    
    # 세부 카테고리별 유효한 파일 목록
    detail_files = defaultdict(list)
    
    # 루트 경로가 존재하는지 확인
    if not os.path.exists(root_path):
        print(f"Error: 디렉토리 '{root_path}'가 존재하지 않습니다.")
        return None
    
    # 통계 카운터
    total_files = 0
    valid_files = 0
    size_condition_met = 0
    
    # 바운딩 박스 개수별 파일 카운트
    bbox_count_stats = Counter()
    
    # 모든 하위 디렉토리 탐색
    for dirpath, dirnames, filenames in os.walk(root_path):
        # 현재 경로를 루트 경로에 대한 상대 경로로 변환
        rel_path = os.path.relpath(dirpath, root_path)
        path_parts = rel_path.split(os.path.sep)
        
        # 카테고리 경로 구성
        if len(path_parts) >= 3:
            category = path_parts[0]
            subcategory = path_parts[1]
            detail = path_parts[2]
            detail_path = f"{category}/{subcategory}/{detail}"
        elif len(path_parts) >= 2:
            category = path_parts[0]
            subcategory = path_parts[1]
            detail_path = f"{category}/{subcategory}"
        elif len(path_parts) >= 1 and path_parts[0] != '.':
            category = path_parts[0]
            detail_path = f"{category}"
        else:
            continue  # 루트 디렉토리는 건너뛰기
        
        # 현재 디렉토리의 JSON 파일 카운트
        for filename in filenames:
            if not filename.lower().endswith('.json'):
                continue
                
            total_files += 1
            
            # JSON 파일 읽기
            json_path = os.path.join(dirpath, filename)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # data 키가 있는지 확인
                if 'data' in json_data:
                    json_data = json_data['data']
                
                # annotations 확인
                annotations = json_data.get('annotations', [])
                
                valid_bbox_count = 0
                valid_bboxes = [] 

                # 중복된 annotation 제거
                seen = set()  # 이미 본 annotation의 특성을 추적
                
                for anno in annotations:
                    if not isinstance(anno, dict):
                        continue
                        
                    # annotation의 고유 특성을 식별하기 위한 키 생성
                    # bbox 좌표와 텍스트를 사용하여 고유 키 생성
                    bbox = anno.get('bbox', [])
                    text = anno.get('text', '')
                    
                    if text == 'xxx':
                        continue

                    if any(val is None for val in bbox):
                        continue

                    if bbox and len(bbox) == 4:
                        # bbox와 text를 문자열로 변환하여 고유 키 생성
                        key = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{text}"
                        
                        if key not in seen:
                            seen.add(key)
                            valid_bboxes.append((bbox[0], bbox[1], bbox[2], bbox[3]))
                            valid_bbox_count += 1

                # 유효한 바운딩 박스 개수 통계 추가
                bbox_count_stats[valid_bbox_count] += 1
                
                # 유효한 바운딩 박스가 3개 이상인 파일만 처리
                if valid_bbox_count >= 2:
                    # 박스 iou 계산 겹치는 박스는 제외하고 박스 크기 비교
                    valid_bbox_sizes = []
                    converted_bboxes = []
                    bbox_centers = []  # 바운딩 박스 중심점 저장 리스트

                    for x, y, w, h in valid_bboxes:
                        converted_bboxes.append([x, y, x + w, y + h])
                        valid_bbox_sizes.append(w * h)
                        # 바운딩 박스 중심점 계산
                        center_x = x + w/2
                        center_y = y + h/2
                        bbox_centers.append([center_x, center_y])
                    
                    if len(bbox_centers) >= 2:  # 최소 2개 이상의 중심점이 필요함
                        distances = pdist(np.array(bbox_centers))
                        min_distance = np.min(distances)
                        
                        # 거리가 500 미만인 경우만 계속 진행
                        if min_distance > training_distance:
                            # 거리 조건을 만족하지 않으면 건너뛰기
                            #print(f"거리 조건 불만족 (최소거리: {min_distance:.2f}): {json_path}")
                            continue

                    # # 겹치는 박스 필터링
                    # overlapping = False
                    # non_overlapping_bboxes = []
                    # for i, bbox1 in enumerate(converted_bboxes):    
                    #     if(overlapping == True):
                    #         break 

                    #     # 이미 선택된 박스들과 비교
                    #     for bbox2 in non_overlapping_bboxes:
                    #         # 두 박스의 겹치는 영역 계산
                    #         x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
                    #         y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
                            
                    #         # 겹치는 영역의 넓이
                    #         overlap_area = x_overlap * y_overlap
                            
                    #         # 각 박스의 넓이
                    #         bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                    #         bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                            
                    #         # IoU 계산 (Intersection over Union)
                    #         iou = overlap_area / (bbox1_area + bbox2_area - overlap_area)
                            
                    #         # IoU가 임계값보다 크면 겹친다고 판단
                    #         if iou > 0.5:  # 임계값 0.5 (조정 가능)
                    #             overlapping = True
                    #             print(" 겹치는 박스 발견:", bbox1, bbox2)
                    #             break

                    #         if not overlapping:
                    #             non_overlapping_bboxes.append(bbox1)

                    # 가장 작은 박스와 가장 큰 박스의 크기 비교
                    if valid_bbox_sizes:
                        min_size = min(valid_bbox_sizes)
                        max_size = max(valid_bbox_sizes)
                        
                        # 작은 박스가 큰 박스의 반 이하인 경우에만 카운트
                        if min_size <= max_size / 2:
                            detail_counts[detail_path] += 1
                            valid_files += 1
                            size_condition_met += 1
                            
                            # 유효한 파일 목록에 추가
                            if export_lists:
                                detail_files[detail_path].append(json_path)
                    
            except Exception as e:
                print(f"Error reading JSON file {json_path}: {e}")
                continue
    
    # 결과 출력
    print(f"총 JSON 파일 수: {total_files}")
    print(f"유효한 바운딩 박스가 3개 이상인 JSON 파일 수: {bbox_count_stats[3] + bbox_count_stats[4] + bbox_count_stats[5] + bbox_count_stats[6] + bbox_count_stats[7] + bbox_count_stats[8] + bbox_count_stats[9] + bbox_count_stats[10] + sum(bbox_count_stats[n] for n in range(11, 100))}")
    print(f"조건을 만족하는 파일 수 (바운딩 박스 3개 이상 + 크기 조건): {valid_files}")
    print(f"크기 조건을 만족하는 파일 비율: {(size_condition_met / (bbox_count_stats[3] + bbox_count_stats[4] + bbox_count_stats[5] + bbox_count_stats[6] + bbox_count_stats[7] + bbox_count_stats[8] + bbox_count_stats[9] + bbox_count_stats[10] + sum(bbox_count_stats[n] for n in range(11, 100))) * 100 if bbox_count_stats[3] + bbox_count_stats[4] + bbox_count_stats[5] + bbox_count_stats[6] + bbox_count_stats[7] + bbox_count_stats[8] + bbox_count_stats[9] + bbox_count_stats[10] + sum(bbox_count_stats[n] for n in range(11, 100)) else 0):.2f}%")
    
    # 세부 카테고리별 데이터프레임 생성 및 저장
    detail_df = pd.DataFrame(list(detail_counts.items()), columns=['Detail', 'Count'])
    detail_df = detail_df.sort_values('Detail', ascending=True)  # 이름 오름차순 정렬
    detail_df.to_csv(output_file, index=False)
    print(f"세부 카테고리별 유효한 바운딩 박스가 3개 이상이고 크기 조건을 만족하는 JSON 파일 개수가 '{output_file}' 파일로 저장되었습니다.")
    
    # 각 카테고리별 파일 목록 저장
    if export_lists:
         # 디렉토리 생성
        list_dir = "./data/valid_files_by_category"
        os.makedirs(list_dir, exist_ok=True)
        
        # 파일 수가 있는 카테고리마다 목록 파일 생성
        for detail_path, files in detail_files.items():
            # 파일명에 사용할 안전한 이름 생성 (/ 문자를 _로 변환)
            safe_name = detail_path.replace('/', '_')
            
            # 파일명만 있는 간결한 목록만 저장 (전체 경로 목록은 저장하지 않음)
            short_list_file = os.path.join(list_dir, f"{safe_name}_short.txt")
            with open(short_list_file, 'w', encoding='utf-8') as f:
                for file_path in sorted(files):
                    f.write(f"{os.path.basename(file_path)}\n")
                    
            print(f"카테고리 '{detail_path}'의 파일 목록이 '{short_list_file}'에 저장되었습니다.")
        
        print(f"모든 카테고리의 파일 목록이 '{list_dir}' 디렉토리에 저장되었습니다.")
    
    return detail_df

if __name__ == "__main__":
    # 데이터 경로 설정
    root_path = "/data/ephemeral/home/industry-partnership-project-brainventures/data/02.raw_data/1.Training/label"
    
    # 계층적 분석 및 세부 카테고리별 조건을 만족하는 파일 수 CSV 저장
    # 각 카테고리별 유효한 파일 목록도 함께 저장
    detail_df = analyze_hierarchical_json_counts(root_path, "json_counts_by_detail.csv", export_lists=True)
    
    # 최종 결과 출력
    print("\n=== 세부 카테고리별 조건을 만족하는 JSON 파일 개수 (크기 조건 포함) ===")
    print(detail_df.head(10))