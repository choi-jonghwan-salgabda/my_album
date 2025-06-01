# Import necessary libraries
import os
import csv
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR, draw_ocr # Import PaddleOCR
import traceback # For detailed error printing

# --- Configuration ---
img_input_folder = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/images/test"
# Define output path for PaddleOCR results
csv_output_path = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/paddleocr/paddleocr_test.csv"
# Optional: Define a folder to save annotated images (like EasyOCR script)
# img_output_folder = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/paddleocr/img_test"

# --- Helper Functions ---
def get_image_files(folder_path):
    """Gets a list of image files from a folder."""
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff') # Added tiff
    return [f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(img_extensions)]

def format_coordinates(bbox):
    """
    Formats PaddleOCR bounding box coordinates (list of [x, y] points)
    into a space-separated string "x1 y1 x2 y2 x3 y3 x4 y4".
    """
    coords = []
    # Ensure points are integers before converting to string
    for point in bbox:
        # PaddleOCR might return floats, ensure conversion to int
        coords.extend([str(int(point[0])), str(int(point[1]))])
    return " ".join(coords)

# --- Main Processing Logic ---
print(f"Processing images from: {img_input_folder}")
print(f"Writing CSV output to: {csv_output_path}")
# if img_output_folder:
#     os.makedirs(img_output_folder, exist_ok=True)
#     print(f"Saving annotated images to: {img_output_folder}")

# Ensure the directory for the CSV file exists
try:
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
except OSError as e:
    print(f"Error creating directory {os.path.dirname(csv_output_path)}: {e}")
    exit()


try: # Add try-except block for better error handling during file processing
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Use comma as delimiter
        writer = csv.writer(csvfile, delimiter=',')

        # Add header row
        writer.writerow(['filename', 'polygons'])

        image_files = get_image_files(img_input_folder)
        total_files = len(image_files)
        print(f"Found {total_files} image files to process.")

        # Initialize PaddleOCR reader outside the loop for efficiency
        try:
            # Initialize PaddleOCR. Set use_gpu=False if you don't have GPU or paddlepaddle-gpu installed.
            # lang='korean' should handle both Korean and English if the model supports it.
            # Check PaddleOCR documentation for specific multilingual models if needed.
            ocr_engine = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=False) # Modify use_gpu if needed
            print("PaddleOCR Engine initialized (lang='korean', use_gpu=False).")
        except Exception as e:
             print(f"Error initializing PaddleOCR Engine: {e}")
             print(traceback.format_exc())
             print("Please ensure PaddleOCR and paddlepaddle are installed correctly.")
             exit() # Exit if reader cannot be initialized

        processed_count = 0
        for img_name in image_files:
            img_path = os.path.join(img_input_folder, img_name)

            try:
                # --- Image Loading ---
                # PaddleOCR can accept file paths directly, which might be slightly more efficient
                # Or load with PIL and convert to NumPy array like the EasyOCR script
                # Using file path method here for simplicity with PaddleOCR:
                # image = Image.open(img_path).convert("RGB")
                # image_np = np.array(image)

                # --- Run PaddleOCR ---
                # Pass the image path directly to PaddleOCR
                result = ocr_engine.ocr(img_path, cls=True)

                # --- Parse PaddleOCR Result ---
                coords_list = []
                # PaddleOCR returns a list where the first element contains the results for the image.
                # It can be [[detection1], [detection2], ...] or sometimes [[None]] if nothing is found.
                if result and result[0] is not None and isinstance(result[0], list):
                    detections = result[0]
                    for detection in detections:
                        # Each detection is [[points], (text, confidence)]
                        if isinstance(detection, list) and len(detection) == 2:
                            bbox = detection[0]  # The list of points [[x1,y1], [x2,y2], ...]

                            # Basic validation for bbox structure (4 points, each a list/tuple of 2 numbers)
                            if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in bbox):
                                coords = format_coordinates(bbox)
                                coords_list.append(coords)
                            else:
                                print(f"Warning: Skipping invalid bbox format in {img_name}: {bbox}")
                        else:
                            print(f"Warning: Skipping unexpected detection format in {img_name}: {detection}")
                elif result and result[0] is None:
                     print(f"Info: No text detected by PaddleOCR in {img_name}.")
                else:
                     # Handle cases where result might be empty or have an unexpected structure
                     print(f"Warning: Unexpected or empty result structure from PaddleOCR for {img_name}: {result}")


                # --- Write to CSV ---
                # Join the collected coordinates with '|'
                writer.writerow([img_name, '|'.join(coords_list)])
                processed_count += 1
                if processed_count % 50 == 0: # Print progress
                    print(f"  Processed {processed_count}/{total_files} images...")

                # --- Optional: Save annotated image ---
                # if img_output_folder and result and result[0] is not None:
                #     try:
                #         image = Image.open(img_path).convert('RGB')
                #         boxes = [line[0] for line in result[0]] # Extract boxes
                #         # txts = [line[1][0] for line in result[0]] # Extract text
                #         # scores = [line[1][1] for line in result[0]] # Extract scores
                #         # font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" # Adjust if needed
                #         # im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
                #         # Simplified drawing just boxes:
                #         draw = ImageDraw.Draw(image)
                #         for box in boxes:
                #              # Ensure box points are tuples for drawing
                #              draw_points = [tuple(map(int, p)) for p in box]
                #              draw.polygon(draw_points, outline=(255, 0, 0), width=2) # Draw red polygon
                #
                #         annotated_img_path = os.path.join(img_output_folder, f"annotated_{img_name}")
                #         image.save(annotated_img_path)
                #     except Exception as draw_e:
                #         print(f"Error drawing/saving annotated image for {img_name}: {draw_e}")


            except FileNotFoundError:
                 print(f"Error: Image file not found during processing: {img_path}")
                 writer.writerow([img_name, "ERROR: File not found"])
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
                print(traceback.format_exc())
                # Optionally write an error row or skip
                writer.writerow([img_name, f"ERROR: {e}"])

    print(f"\nProcessing complete. Processed {processed_count} images.")
    print(f"CSV file saved to {csv_output_path}")

except IOError as e:
    print(f"Error opening or writing to CSV file {csv_output_path}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print(traceback.format_exc())
