import os
import glob
import csv

def convert_txt_to_csv(input_dir, output_csv):
    """
    input_dir에 있는 모든 .txt 파일을 읽어
    filename,polygons 형식의 CSV 파일로 변환
    
    Args:
        input_dir: .txt 파일이 있는 디렉토리 경로
        output_csv: 결과 CSV 파일 경로
    """
    if not os.path.exists(input_dir):
        print(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
        return
    
    # 모든 .txt 파일 찾기
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    if not txt_files:
        print(f"입력 디렉토리에 .txt 파일이 없습니다: {input_dir}")
        return
    
    print(f"{len(txt_files)}개의 .txt 파일을 찾았습니다.")
    
    # CSV 파일 작성
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 헤더 작성
        csv_writer.writerow(['filename', 'polygons'])
        
        # 각 .txt 파일 처리
        for txt_file in sorted(txt_files):
            filename = os.path.basename(txt_file)
            # .txt 확장자 제거 후 원본 파일명(.jpg) 생성
            if filename.startswith('res_'):
                # ICDAR 형식인 경우 (res_ 제거)
                image_filename = filename[4:].replace('.txt', '.jpg')
            else:
                # 일반 형식
                image_filename = filename.replace('.txt', '.jpg')
            
            # .txt 파일에서 폴리곤 좌표 읽기
            polygons = []
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 각 폴리곤 좌표를 공백으로 구분된 형태로 저장
                        numbers = line.split(',')
                        # Total-Text 형식 (y,x 형식)을 x,y 형식으로 변환
                        coords = []
                        for i in range(0, len(numbers), 2):
                            if i+1 < len(numbers):
                                # y,x 형식이므로 x,y로 순서 변경
                                coords.append(numbers[i+1])  # x
                                coords.append(numbers[i])    # y
                        
                        polygons.append(' '.join(coords))
            
            # 모든 폴리곤을 '|'로 연결
            all_polygons = '|'.join(polygons)
            
            # CSV에 행 추가
            csv_writer.writerow([image_filename, all_polygons])
    
    print(f"CSV 파일이 성공적으로 생성되었습니다: {output_csv}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='텍스트 파일을 CSV로 변환')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='입력 txt 파일이 있는 디렉토리')
    parser.add_argument('--output_csv', type=str, required=True, 
                        help='출력 CSV 파일 경로')
    
    args = parser.parse_args()
    
    convert_txt_to_csv(args.input_dir, args.output_csv)