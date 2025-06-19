#!/usr/bin/env python
# coding: utf-8

# 간판 데이터 EDA 스크립트
# 데이터셋 구조 분석 및 시각화

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from datetime import datetime
from collections import defaultdict
from scipy.spatial.distance import pdist
import re


#한국어-영어 매핑 사전
korean_to_english_dict = {
    #간판 카테고리
    '가로형간판': 'Horizontal_Sign',
    '돌출간판': 'Projecting_Sign',
    '세로형간판': 'Vertical_Sign',
    '실내': 'Indoor_Sign',
    '실내간판': 'Indoor_Sign_Board',
    '실내안내판': 'Indoor_Guide_Sign',
    '지주이용간판': 'Pillar_Sign',
    '창문이용광고물': 'Window_Advertisement',
    '현수막': 'Banner',
    '기타': 'Other',
    
    #책표지 카테고리
    '종교': 'Religion',
    '총류': 'General_Category',
    '역사': 'History',
    '언어': 'Language',
    '철학': 'Philosophy',
    '자연과학': 'Natural_Science',
    '기술과학': 'Technology_Science',
    '사회과학': 'Social_Science',
    '예술': 'Art',
    '문학': 'Literature'
}

def remove_file_extension(filename):
    """파일 확장자를 제거합니다."""
    return os.path.splitext(filename)[0]

def load_all_json_files(image_directory, check_origin = False):
    """JSON 파일을 로드하고 이미지 파일 존재 여부를 확인합니다."""
    label_directory = f"{image_directory}/json"
    origin_directory = f"{image_directory}/image"
    file_list = {}
    missing_images = []
    processed_count = 0
    
    # 디렉토리가 존재하는지 확인
    if not os.path.exists(label_directory):
        print(f"Label directory not found: {label_directory}")
        return file_list
    
    # JSON 파일 순회
    for root, _, files in os.walk(label_directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                processed_count += 1
                
                try:
                    # JSON 파일 로드
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 상대 경로 계산 (label 디렉토리 기준)
                    rel_path = os.path.relpath(root, label_directory)
                    file_name_without_extension = remove_file_extension(file)
                    
                    # 이미지 파일 경로 
                    img_path = os.path.join(origin_directory, rel_path, f"{file_name_without_extension}.jpg")
                    
                    # 이미지 파일 존재 여부 확인
                    img_exists = os.path.exists(img_path)
                    if not img_exists:
                        missing_images.append(f"{rel_path}/{file_name_without_extension}")
                    
                    # 데이터와 이미지 정보를 함께 저장
                    data['_image_path'] = img_path
                    data['_image_exists'] = img_exists
                    
                    # 파일 목록에 추가
                    file_list[file_name_without_extension] = data
                    
                    # 진행 상황 출력 (1000개마다)
                    if processed_count % 1000 == 0:
                        print(f"Processed {processed_count} JSON files...")
                        
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    # 이미지 누락 파일 수 출력
    print(f"\nProcessed total {processed_count} JSON files")
    
    if check_origin and missing_images:
        print(f"Missing images: {len(missing_images)} files")
        raise ValueError("일부 JSON 파일에 대응하는 이미지 파일이 없습니다. 프로세스를 중단합니다.")
    
    return file_list

def count_subclass_occurrences(json_data):
    """JSON 데이터에서 서브클래스 빈도수를 계산합니다."""
    subclass_counter = Counter()

    for file_name, content in json_data.items():
        try:
            # metadata가 리스트인 경우
            if isinstance(content.get('metadata', None), list):
                metadata_list = content.get('metadata', [])
                for metadata_item in metadata_list:
                    if isinstance(metadata_item, dict):  # 딕셔너리인지 확인
                        subclass = metadata_item.get('subclass', None)
                        if subclass:
                            subclass_counter[subclass] += 1
                            break  # 첫 번째 유효한 subclass를 찾으면 중단
            
            # metadata가 딕셔너리인 경우 (기존 로직)
            elif isinstance(content.get('metadata', None), dict):
                subclass = content.get('metadata', {}).get('subclass', None)
                if subclass:
                    subclass_counter[subclass] += 1
                    
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    return subclass_counter

def plot_subclass_counts(subclass_counter):
    """서브클래스별 빈도수를 막대 그래프로 시각화합니다."""
    # 데이터 변환 및 정렬
    sorted_data = sorted(subclass_counter.items(), key=lambda x: x[1], reverse=True)
    korean_names = [item[0] for item in sorted_data]
    english_names = [korean_to_english_dict.get(name, name) for name in korean_names]
    counts = [item[1] for item in sorted_data]
    
    # 그래프 생성
    plt.figure(figsize=(16, 8))
    bars = plt.bar(range(len(counts)), counts, color='skyblue')
    
    # 바 위에 값 표시
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                 str(counts[i]), ha='center', va='bottom')
    
    # 서브클래스명(영어)으로 x축 설정
    plt.xticks(range(len(english_names)), english_names, rotation=45, ha='right')
    
    # 그래프 제목 및 라벨 설정
    plt.xlabel('Subclass')
    plt.ylabel('Count')
    plt.title(f'Frequency of Subclasses ({title})')
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig(f'{title}.png')
    plt.close()
    
    print(f"\nSubclass distribution graph saved as {title}.png")

def create_augmentation_excel(subclass_counter):
    """증강이 필요한 서브클래스를 CSV 파일로 저장합니다."""
    data = []
    for korean_name, count in subclass_counter.items():
        english_name = korean_to_english_dict.get(korean_name, korean_name)
        
        # 500개 미만인 경우만 포함
        if count < 500:
            # 증강 필요 개수 계산 (목표: 500개)
            augment_needed = 500 - count
            
            # 증강 배수 계산 (500개 목표)
            augment_multiplier = round(500 / count, 2) if count > 0 else "N/A"
            
            # 데이터 리스트에 추가
            data.append({
                'Korean_Name': korean_name,
                'English_Name': english_name,
                'Current_Count': count,
                'Target_Count': 500,
                'Augmentation_Needed': augment_needed,
                'Augmentation_Multiplier': augment_multiplier
            })
    
    # 데이터 프레임 생성
    if data:
        df = pd.DataFrame(data)
        
        # 증강 필요량 기준으로 내림차순 정렬
        df = df.sort_values(by='Augmentation_Needed', ascending=False)
        
        # CSV 파일로 저장
        csv_path = f'{title}_augmentation_plan.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"\nAugmentation plan saved to: {csv_path}")
        print(f"Found {len(data)} subclasses with less than 500 images")
    else:
        print("\nNo subclasses with less than 500 images found.")

def plot_bbox_count_distribution(json_data):
    """서브클래스별 바운딩 박스 개수 분포를 히트맵으로 시각화합니다."""
    # 데이터 수집
    bbox_count_dict = {}
    
    for file_name, content in json_data.items():
        # 서브클래스 추출
        subclass = None
        if isinstance(content.get('metadata', None), list):
            for metadata_item in content.get('metadata', []):
                if isinstance(metadata_item, dict) and 'subclass' in metadata_item:
                    subclass = metadata_item['subclass']
                    break
        elif isinstance(content.get('metadata', None), dict):
            subclass = content.get('metadata', {}).get('subclass', None)
        
        if not subclass:
            continue
            
        # bbox 개수 세기 (유효한 bbox만 카운트)
        annotations = content.get('annotations', [])
        valid_bbox_count = 0
        
        for anno in annotations:
            if not isinstance(anno, dict):
                continue
                
            # 'xxx' 텍스트가 있는 경우 제외
            if anno.get('text', '') == 'xxx':
                continue
                
            # bbox 확인
            bbox = anno.get('bbox', [])
            
            # 유효하지 않은 bbox 필터링
            if bbox is None or not bbox:
                continue
            
            # null 값이 있는 bbox 필터링
            if any(val is None for val in bbox):
                continue
                
            # 유효한 bbox인 경우 카운트
            if 'bbox' in anno and len(bbox) == 4:
                valid_bbox_count += 1
        
        # 데이터 누적
        if subclass not in bbox_count_dict:
            bbox_count_dict[subclass] = Counter()
        bbox_count_dict[subclass][valid_bbox_count] += 1
    
    # 상위 15개 서브클래스만 선택 (이미지 수 기준)
    top_subclasses = sorted(
        [(k, sum(v.values())) for k, v in bbox_count_dict.items()], 
        key=lambda x: x[1], 
        reverse=True
    )[:20]
    
    # 데이터 프레임 준비
    rows = []
    for subclass, _ in top_subclasses:
        counter = bbox_count_dict[subclass]
        total = sum(counter.values())
        
        # 최대 8개 bbox까지만 표시 (필요에 따라 조정)
        for i in range(1, 20):
            count = counter.get(i, 0)
            try:
                # 타입 체크 및 안전한 계산
                total = float(total) if total else 0
                count = int(count) if count else 0
                
                if total > 0:
                    percentage = round((count / total) * 100, 2)
                else:
                    percentage = 0.0
                    
                # 안전한 포맷팅
                str_per = f'{percentage:.2f}%'
                
                # 디버깅 정보 출력
                print(f"DEBUG: subclass={subclass}, i={i}, count={count}, total={total}, percentage={percentage}")
                
                # 데이터 추가
                rows.append({
                    'Subclass': korean_to_english_dict.get(subclass, subclass),
                    'Bbox Count': str(i),
                    'Image Count': count,
                    'Percentage': percentage
                })
            except Exception as e:
                print(f"Error processing bbox count {i} for subclass {subclass}: {e}")
                continue
    
    df = pd.DataFrame(rows)
    

    # 피벗 테이블 생성
    pivot_df = df.pivot(index='Subclass', columns='Bbox Count', values='Image Count')
    
    # 히트맵 생성
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='YlGnBu', 
                linewidths=.5, cbar_kws={'label': 'Image Count of Images (%)'})
    
    plt.title('Distribution of Bounding Box Counts by Subclass (%)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{title}_bbox_count_heatmap.png')
    plt.close()
    
    print(f"Heatmap saved to '{title}_bbox_count_heatmap.png'")
    
    # 상세 데이터 CSV 저장
    csv_path = f'{title}_bbox_count_distribution.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Detailed distribution saved to '{csv_path}'")

def analyze_bbox_distances(json_data):
    output_dir = f'{title}_bbox_distances'
    os.makedirs(output_dir, exist_ok=True)

    distances_by_subclass = defaultdict(list)
    samples_by_subclass = defaultdict(int)

    for file_name, content in json_data.items():
        subclass = None
        if isinstance(content.get('metadata', None), list):
            for metadata_item in content.get('metadata', []):
                if isinstance(metadata_item, dict) and 'subclass' in metadata_item:
                    subclass = metadata_item['subclass']
                    break
        elif isinstance(content.get('metadata', None), dict):
            subclass = content.get('metadata', {}).get('subclass', None)
        
        if not subclass:
            continue

        bboxes = []
        for anno in content.get('annotations', []):
            if isinstance(anno, dict) and 'bbox' in anno:
                bbox = anno.get('bbox', [])

                # [null, null, null, null] 같은 잘못된 bbox 필터링
                if bbox is None or not bbox:
                    continue
                    
                # null 값이 포함된 bbox 필터링
                if any(val is None for val in bbox):
                    continue
                
                if len(bbox) == 4:
                    try:
                        x, y, w, h = [float(val) for val in bbox]
                        
                        # 유효한 값인지 확인
                        if w <= 0 or h <= 0:
                            continue
                            
                        # 중심점 계산
                        center_x = x + w/2
                        center_y = y + h/2

                        #바운딩 박스의 겹치는 지점 계산

                        bboxes.append([center_x, center_y])
                    except (ValueError, TypeError):
                        continue

        if len(bboxes) >= 2:
            try:
                centers = np.array(bboxes)
                distances = pdist(centers, 'euclidean')
                
                # 서브클래스별로 저장
                distances_by_subclass[subclass].extend(distances)
                samples_by_subclass[subclass] += 1
                
            except Exception as e:
                continue

    # 유효한 서브클래스 선택
    valid_subclasses = [s for s, count in samples_by_subclass.items() if count >= 5]

    if not valid_subclasses:
        print("충분한 데이터가 없습니다.")
        return

    summary_data = []
    
    # 1. 개별 히스토그램 대신 모든 서브클래스를 위한 하나의 통합 그리드 히스토그램 생성
    # 상위 20개 서브클래스 선택 (또는 유효한 서브클래스 전체가 20개 미만인 경우)
    top_subclasses = sorted(
        [(s, samples_by_subclass[s]) for s in valid_subclasses],
        key=lambda x: x[1],
        reverse=True
    )[:min(20, len(valid_subclasses))]
    
    # 그리드 크기 계산 (최대 20개 서브클래스를 4x5 또는 5x4 그리드로 배치)
    num_subclasses = len(top_subclasses)
    if num_subclasses <= 16:
        grid_rows = 4
        grid_cols = 4
    else:
        grid_rows = 4
        grid_cols = 5
    
    # 전체 그림 생성
    fig = plt.figure(figsize=(5*grid_cols, 4*grid_rows))
    fig.suptitle('Bounding Box Distance Distributions by Subclass', fontsize=20, y=0.98)
    
    # 각 서브클래스에 대한 서브플롯 생성
    for idx, (subclass, sample_count) in enumerate(top_subclasses, 1):
        distances = distances_by_subclass[subclass]
        if not distances:
            continue
            
        # 영어 이름 변환
        eng_name = korean_to_english_dict.get(subclass, subclass)
        
        # 통계 계산
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)

        filtered_distances = [d for d in distances]
        
        # 요약 정보 저장 (기존 코드 유지)
        summary_data.append({
            'Subclass_Korean': subclass,
            'Subclass_English': eng_name,
            'Sample_Count': sample_count,
            'Mean_Distance':round(mean_distance,2),
            'Median_Distance': round(median_distance,2),
            'Min_Distance': round(np.min(distances),2),
            'Max_Distance': round(np.max(distances),2),
            'Std_Distance': round(np.std(distances),2)
        })
        
        # 서브플롯 생성
        ax = fig.add_subplot(grid_rows, grid_cols, idx)
        
        # KDE 곡선만 그리기 (hist=False로 막대그래프 제거)
        sns.kdeplot(filtered_distances, ax=ax, color='skyblue')

        ax.axvline(mean_distance, color='red', linestyle='--', label=f'Mean :{mean_distance: .2f}')
        ax.axvline(median_distance, color='green', linestyle='-', label=f'Median :{median_distance: .2f}')
        
        # 타이틀과 레이블 설정
        ax.set_title(f'{eng_name} (n={sample_count})', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_xlabel('Distance (pixels)', fontsize=10)
        ax.legend(fontsize=8)
        
        # 숫자 크기 조정
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 전체 제목을 위한 여백 조정
    
    # 통합 그림 저장
    combined_output_path = f'{title}_bbox_distances.png'
    plt.savefig(combined_output_path, dpi=150)
    plt.close()
    
    print(f"모든 서브클래스의 히스토그램이 하나의 이미지로 저장되었습니다: {combined_output_path}")
    if summary_data:
        # 샘플 수로 내림차순 정렬
        summary_data.sort(key=lambda x: x['Sample_Count'], reverse=True)
        
        # 데이터프레임 생성 및 저장
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = f'{title}_bbox_distance_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        
        print(f"서브클래스별 거리 통계가 CSV 파일로 저장되었습니다: {summary_csv_path}")


if __name__ == "__main__": 
    # 데이터 디렉토리 경로 설정
    title = "file list"  # 파일 이름에 사용될 제목
    image_directory = "/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project"
    
    print(f"== 데이터셋 분석 시작: {title} ==")
    print(f"데이터 경로: {image_directory}")
    
    try:
        # 1. JSON 파일 로드
        print("\n1. JSON 파일 로드 중...")
        all_json_data = load_all_json_files(image_directory=image_directory)
        print(f"로드 완료: {len(all_json_data)}개 파일")
        
        # 2. 서브클래스별 데이터 개수 분석
        print("\n2. 서브클래스별 데이터 개수 분석 중...")
        subclass_counter = count_subclass_occurrences(all_json_data)
        print("서브클래스별 개수:")
        for subclass, count in sorted(subclass_counter.items(), key=lambda x: x[1], reverse=True):
            eng_name = korean_to_english_dict.get(subclass, subclass)
            print(f"- {subclass} ({eng_name}): {count}")
        
        # 3. 서브클래스 분포 그래프 생성
        print("\n3. 서브클래스 분포 그래프 생성 중...")
        plot_subclass_counts(subclass_counter)
        
        # # 4. 증강 계획 CSV 생성
        # print("\n4. 증강 계획 CSV 생성 중...")
        # create_augmentation_excel(subclass_counter)
        
        # # 5. 바운딩 박스 개수 분포 분석
        # print("\n5. 바운딩 박스 개수 분포 분석 중...")
        # plot_bbox_count_distribution(all_json_data)
        
        # # 6. 바운딩 박스 간 거리 분석
        # print("\n6. 바운딩 박스 간 거리 분석 중...")
        # analyze_bbox_distances(all_json_data)
        
        print(f"\n== 데이터셋 분석 완료: {title} ==")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
# import os
# import json
# import re

# def extract_subclass_from_path(file_path):
#     """파일 경로에서 서브클래스 정보를 추출합니다."""
#     # 경로에서 파일 이름만 추출
#     filename = os.path.basename(file_path)
    
#     # 파일 이름에서 패턴 매칭 (간판_가로형간판_000013.json)
#     match = re.match(r'간판_([^_]+)_', filename)
#     if match:
#         return match.group(1)  # 첫 번째 그룹 (가로형간판)
    
#     return None

# def update_json_subclass(json_directory):
#     """JSON 파일의 subclass 필드를 경로에서 추출한 값으로 업데이트합니다."""
#     processed_count = 0
#     updated_count = 0
    
#     # 재귀적으로 모든 JSON 파일 탐색
#     for root, _, files in os.walk(json_directory):
#         for file in files:
#             if not file.endswith(".json"):
#                 continue
                
#             file_path = os.path.join(root, file)
#             processed_count += 1
            
#             try:
#                 # JSON 파일 로드
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
                
#                 # 경로에서 서브클래스 추출
#                 subclass_from_path = extract_subclass_from_path(file_path)
#                 if not subclass_from_path:
#                     continue
                
#                 # subclass 필드 업데이트
#                 updated = False
                
#                 # metadata가 리스트인 경우
#                 if isinstance(data.get('metadata', None), list):
#                     for item in data['metadata']:
#                         if isinstance(item, dict):
#                             if 'subclass' not in item or item['subclass'] != subclass_from_path:
#                                 item['subclass'] = subclass_from_path
#                                 updated = True
                
#                 # metadata가 딕셔너리인 경우
#                 elif isinstance(data.get('metadata', None), dict):
#                     if 'subclass' not in data['metadata'] or data['metadata']['subclass'] != subclass_from_path:
#                         data['metadata']['subclass'] = subclass_from_path
#                         updated = True
                
#                 # metadata 필드가 없는 경우
#                 else:
#                     data['metadata'] = {'subclass': subclass_from_path}
#                     updated = True
                
#                 # 변경사항이 있으면 파일 저장
#                 if updated:
#                     with open(file_path, 'w', encoding='utf-8') as f:
#                         json.dump(data, f, ensure_ascii=False, indent=2)
#                     updated_count += 1
                
#                 # 진행상황 출력
#                 if processed_count % 1000 == 0:
#                     print(f"Processed {processed_count} files, updated {updated_count}")
                    
#             except Exception as e:
#                 print(f"Error processing {file_path}: {e}")
    
#     return processed_count, updated_count

# if __name__ == "__main__":
#     json_directory = "/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/json"
    
#     print(f"Starting to update subclass fields in JSON files...")
#     processed, updated = update_json_subclass(json_directory)
    
#     print(f"\nProcess completed!")
#     print(f"Processed {processed} JSON files")
#     print(f"Updated {updated} JSON files with new subclass values")