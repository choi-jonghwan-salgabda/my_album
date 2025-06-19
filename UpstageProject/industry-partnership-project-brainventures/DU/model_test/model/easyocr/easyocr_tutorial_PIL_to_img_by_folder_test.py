import easyocr, os, json, cv2, csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
# import pandas as pd # pandas is not used

img_input_folder="/data/ephemeral/home/Fastcampus_project/Fastcampus_project/images/test"
img_output_folder="/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/easyocr/img" # Not used in this script logic, but keep for context
csv_output_path="/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/easyocr/easyocr_test.csv"

# img_path="/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/img/책표지_총류_000001.jpg" # Example path, not used in loop
# img_name=os.path.basename(img_path) # Example name, not used in loop

def get_image_files(folder_path):
    # 이미지 확장자 리스트
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    return [f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in img_extensions]

def format_coordinates(bbox):
    """좌표를 공백으로 구분된 문자열로 변환"""
    coords = []
    # Ensure points are integers before converting to string
    for point in bbox:
        coords.extend([str(int(point[0])), str(int(point[1]))])
    return " ".join(coords)

print(f"Processing images from: {img_input_folder}")
print(f"Writing CSV output to: {csv_output_path}")

try: # Add try-except block for better error handling during file processing
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        # --- MODIFICATION: Change delimiter to comma ---
        writer = csv.writer(csvfile, delimiter=',') # 콤마로 구분

        # --- MODIFICATION: Add header row ---
        writer.writerow(['filename', 'polygons']) # 헤더 추가

        image_files = get_image_files(img_input_folder)
        total_files = len(image_files)
        print(f"Found {total_files} image files to process.")

        # Initialize EasyOCR reader outside the loop for efficiency
        try:
            reader = easyocr.Reader(['ko', 'en'])
            print("EasyOCR Reader initialized.")
        except Exception as e:
             print(f"Error initializing EasyOCR Reader: {e}")
             print("Please ensure EasyOCR is installed correctly and models are available.")
             exit() # Exit if reader cannot be initialized

        processed_count = 0
        for img_name in image_files:
            img_path = os.path.join(img_input_folder, img_name)

            try:
                image = Image.open(img_path).convert("RGB")
                image_np = np.array(image)
                # Use the reader initialized outside the loop
                result = reader.readtext(image_np)

                # 모든 검출 결과의 좌표 수집
                coords_list = []
                for r in result:
                    bbox = r[0]  # 바운딩 박스 좌표
                    # Add basic validation for bbox structure
                    if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(p, list) and len(p) == 2 for p in bbox):
                         coords = format_coordinates(bbox)
                         coords_list.append(coords)
                    else:
                         print(f"Warning: Skipping invalid bbox format in {img_name}: {bbox}")


                # 파일명과 좌표 리스트를 콤마로 구분하여 저장 (handled by csv.writer)
                # join()으로 coords_list를 '|'로 연결하여 하나의 문자열로 만듦
                writer.writerow([img_name, '|'.join(coords_list)])
                processed_count += 1
                if processed_count % 50 == 0: # Print progress
                    print(f"  Processed {processed_count}/{total_files} images...")

            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
                # Optionally write an error row or skip
                # writer.writerow([img_name, f"ERROR: {e}"])

    print(f"\nProcessing complete. Processed {processed_count} images.")
    print(f"CSV file saved to {csv_output_path}")

except IOError as e:
    print(f"Error opening or writing to CSV file {csv_output_path}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
