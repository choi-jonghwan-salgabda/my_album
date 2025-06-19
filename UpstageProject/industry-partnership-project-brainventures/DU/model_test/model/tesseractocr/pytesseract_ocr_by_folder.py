# Import necessary libraries
import os
import csv
import cv2 # Using OpenCV for image loading as in the single file script
import pytesseract # Import pytesseract
import traceback # For detailed error printing
import numpy as np # cv2 uses numpy arrays

# --- Configuration ---
# Use ORIGINAL images as input for Tesseract
img_input_folder = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/images/test"
# Define output path for Tesseract results
csv_output_path = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/tesseract/tesseract_test.csv"
# Optional: Define a folder to save annotated images (can be added later if needed)
# img_output_folder = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/tesseract/img_test"

# Tesseract configuration
tesseract_lang = 'kor' # Specify the language(s)
# Common PSM modes:
# 3: Fully automatic page segmentation, but no OSD. (Default)
# 6: Assume a single uniform block of text.
# 11: Sparse text. Find as much text as possible in no particular order.
# 12: Sparse text with OSD.
tesseract_config = r'--oem 3 --psm 11' # Example config: LSTM engine, sparse text mode
confidence_threshold = 0 # Tesseract confidence threshold (0-100). Set higher (e.g., 50) to filter weak detections.

# --- Helper Functions ---
def get_image_files(folder_path):
    """Gets a list of image files from a folder."""
    # Add more extensions if needed
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')
    return [f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(img_extensions)]

def format_coordinates_tesseract(x, y, w, h):
    """
    Formats Tesseract bounding box (x, y, width, height)
    into a space-separated string "x1 y1 x2 y2 x3 y3 x4 y4".
    """
    x1, y1 = x, y
    x2, y2 = x + w, y
    x3, y3 = x + w, y + h
    x4, y4 = x, y + h
    # Ensure points are integers before converting to string
    return f"{int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(x3)} {int(y3)} {int(x4)} {int(y4)}"

# --- Pre-check Tesseract Installation ---
try:
    tesseract_version = pytesseract.get_tesseract_version()
    print(f"Tesseract version {tesseract_version} found.")
except pytesseract.TesseractNotFoundError:
    print("Tesseract Error: tesseract is not installed or it's not in your PATH.")
    print("Please install Tesseract OCR engine and the required language packs (e.g., tesseract-ocr, tesseract-ocr-kor).")
    exit()
except Exception as e:
    print(f"An error occurred checking Tesseract version: {e}")
    # Decide if you want to exit or try to continue
    # exit()

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

        processed_count = 0
        for img_name in image_files:
            img_path = os.path.join(img_input_folder, img_name)

            try:
                # --- Image Loading using OpenCV ---
                # Tesseract generally works better with clean, preprocessed images
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load image {img_name}. Skipping.")
                    writer.writerow([img_name, "ERROR: Could not load image"])
                    continue

                # Optional Preprocessing (Example: Grayscale)
                # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # You might add more preprocessing like thresholding depending on image quality

                # --- Run Tesseract ---
                # Use image_to_data to get bounding boxes and confidence
                data = pytesseract.image_to_data(
                    image, # Use the loaded image (or preprocessed version like gray_image)
                    lang=tesseract_lang,
                    config=tesseract_config,
                    output_type=pytesseract.Output.DICT
                )

                # --- Parse Tesseract Result ---
                coords_list = []
                num_boxes = len(data['level']) # Use 'level' which should always exist

                for i in range(num_boxes):
                    # Check confidence level
                    conf = int(data['conf'][i])
                    text = data['text'][i].strip()

                    # Filter based on confidence and non-empty text
                    # Tesseract often returns boxes with -1 confidence for structure elements
                    if conf > confidence_threshold and text:
                        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])

                        # Ensure width and height are positive
                        if w <= 0 or h <= 0:
                            # print(f"Warning: Skipping box with non-positive dimensions in {img_name}: w={w}, h={h}")
                            continue

                        # Format coordinates to x1 y1 x2 y2 x3 y3 x4 y4
                        coords_str = format_coordinates_tesseract(x, y, w, h)
                        coords_list.append(coords_str)

                # --- Write to CSV ---
                # Join the collected coordinates with '|'
                writer.writerow([img_name, '|'.join(coords_list)])
                processed_count += 1
                if processed_count % 50 == 0: # Print progress
                    print(f"  Processed {processed_count}/{total_files} images...")

                # --- Optional: Save annotated image (Add drawing logic here if needed) ---
                # if img_output_folder:
                #    # Load image with PIL or use the cv2 image
                #    # Draw boxes based on the filtered data (x, y, w, h)
                #    # Save the annotated image

            except FileNotFoundError:
                 print(f"Error: Image file not found during processing: {img_path}")
                 writer.writerow([img_name, "ERROR: File not found"])
            except pytesseract.TesseractError as e:
                 print(f"Tesseract Error processing image {img_name}: {e}")
                 print(traceback.format_exc())
                 writer.writerow([img_name, f"ERROR: Tesseract failed - {e}"])
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
                print(traceback.format_exc())
                writer.writerow([img_name, f"ERROR: {e}"])

    print(f"\nProcessing complete. Processed {processed_count} images.")
    print(f"CSV file saved to {csv_output_path}")

except IOError as e:
    print(f"Error opening or writing to CSV file {csv_output_path}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print(traceback.format_exc())