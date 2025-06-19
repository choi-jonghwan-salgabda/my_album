import os
import csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- Configuration ---
csv_input_path = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/easyocr/easyocr.csv"
img_input_folder = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/images/val"
img_output_folder = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/easyocr/img"

# --- Ensure output directory exists ---
os.makedirs(img_output_folder, exist_ok=True)
print(f"Output directory created or already exists: {img_output_folder}")

# --- Font setup (Optional, if you want to add text later, but not needed for just boxes) ---
# try:
#     # Adjust font path and size if needed
#     font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 15)
# except Exception:
#     print("Warning: NanumGothic font not found. Using default font.")
#     font = ImageFont.load_default()

# --- Process the CSV file ---
print(f"Reading CSV file: {csv_input_path}")
processed_count = 0
skipped_count = 0

with open(csv_input_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')  # Use tab delimiter

    for i, row in enumerate(reader):
        if len(row) != 2:
            print(f"Warning: Skipping malformed row {i+1}: {row}")
            skipped_count += 1
            continue

        img_name, coords_str = row
        img_path = os.path.join(img_input_folder, img_name)
        annotated_img_path = os.path.join(img_output_folder, "annotated_" + img_name)

        # Check if image file exists
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found, skipping: {img_path}")
            skipped_count += 1
            continue

        try:
            # --- Load Image ---
            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            # --- Parse Coordinates ---
            # The coordinate string is like "x1 y1 x2 y2 x3 y3 x4 y4|x1 y1 x2 y2..."
            box_strings = coords_str.split('|')

            for box_str in box_strings:
                if not box_str:  # Skip empty strings if they occur (e.g., trailing '|')
                    continue

                # Split the string "x1 y1 x2 y2 x3 y3 x4 y4" into individual numbers
                coord_values_str = box_str.strip().split(' ')
                if len(coord_values_str) != 8:
                    print(f"Warning: Malformed coordinate string for {img_name}: '{box_str}'. Skipping this box.")
                    continue

                try:
                    # Convert string coordinates to integers
                    coords = [int(c) for c in coord_values_str]

                    # Group into points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    bbox_points = [
                        (coords[0], coords[1]),
                        (coords[2], coords[3]),
                        (coords[4], coords[5]),
                        (coords[6], coords[7])
                    ]

                    # --- Draw Bounding Box ---
                    # draw.line requires a list of tuples, close the polygon by adding the first point at the end
                    draw.line(bbox_points + [bbox_points[0]], fill=(255, 0, 0), width=3) # Red box, width 3

                    # --- Optional: Draw text/confidence (if you had that data) ---
                    # text_position = (bbox_points[0][0], max(bbox_points[0][1] - 20, 0)) # Position above top-left
                    # draw.text(text_position, "TEXT", fill=(0, 255, 0), font=font) # Example text

                except ValueError:
                    print(f"Warning: Non-integer coordinate found for {img_name} in '{box_str}'. Skipping this box.")
                    continue

            # --- Save Annotated Image ---
            image.save(annotated_img_path)
            processed_count += 1
            if processed_count % 50 == 0: # Print progress every 50 images
                 print(f"Processed {processed_count} images...")

        except FileNotFoundError:
            print(f"Error: Image file not found during processing: {img_path}")
            skipped_count += 1
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            skipped_count += 1

print(f"\nProcessing complete.")
print(f"Successfully annotated and saved: {processed_count} images.")
print(f"Skipped images/rows: {skipped_count}.")
print(f"Annotated images saved to: {img_output_folder}")
