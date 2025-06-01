import json
import csv
import os
from PIL import Image, ImageDraw
import traceback # Import traceback for detailed error printing

# --- Configuration ---
gt_json_path = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/jsons/val.json"
pred_csv_path = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/easyocr/easyocr.csv"
# IMPORTANT: Use the ORIGINAL image folder as the base
img_input_folder = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/images/val"
# Create a NEW output folder for comparison images
img_output_folder = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/easyocr/comparison_img"

gt_color = (0, 175, 0)  # Dark Green for Ground Truth
pred_color = (255, 0, 0) # Red for Predictions
line_width = 2          # Width of the bounding box lines

# --- Ensure output directory exists ---
os.makedirs(img_output_folder, exist_ok=True)
print(f"Comparison output directory created or exists: {img_output_folder}")

# --- 1. Load Predictions from CSV into a dictionary ---
predictions = {}
print(f"Loading predictions from CSV: {pred_csv_path}")
try:
    with open(pred_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for i, row in enumerate(reader):
            if len(row) == 2:
                img_name, coords_str = row
                predictions[img_name] = coords_str
            else:
                print(f"Warning: Skipping malformed row {i+1} in CSV: {row}")
    print(f"Loaded predictions for {len(predictions)} images from CSV.")
except FileNotFoundError:
    print(f"Error: Prediction CSV file not found at {pred_csv_path}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading prediction CSV: {e}")
    traceback.print_exc()
    exit()

# --- 2. Load Ground Truth JSON Data ---
print(f"Loading Ground Truth JSON data from: {gt_json_path}")
try:
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    print("GT JSON data loaded successfully.")
except FileNotFoundError:
    print(f"Error: GT JSON file not found at {gt_json_path}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {gt_json_path}. Check file format.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading GT JSON: {e}")
    traceback.print_exc()
    exit()

# --- 3. Process Images: Iterate through GT data ---
processed_count = 0
skipped_count = 0
gt_entries_count = 0

# --- MODIFICATION START: Handle the specific JSON structure ---
# Check if gt_data is a dictionary and has the 'images' key,
# and if the value associated with 'images' is also a dictionary.
if isinstance(gt_data, dict) and 'images' in gt_data and isinstance(gt_data['images'], dict):
    # Get the dictionary where keys are filenames
    images_dict = gt_data['images']
    gt_entries_count = len(images_dict)
    print(f"GT JSON structure: Dictionary with 'images' dictionary ({gt_entries_count} entries). Filenames are keys.")
    # Prepare items for iteration: (filename, details_dict)
    gt_items_to_iterate = images_dict.items()
else:
    # If the structure is different, print an error and exit
    print(f"Error: Unsupported GT JSON structure. Expected a dictionary with an 'images' key containing another dictionary (filename: details). Found structure type: {type(gt_data)}")
    if isinstance(gt_data, dict):
        print(f"Keys found: {list(gt_data.keys())}")
        if 'images' in gt_data:
            print(f"Type of gt_data['images']: {type(gt_data['images'])}")
    exit()
# --- MODIFICATION END ---

print(f"Starting image processing. Reading originals from: {img_input_folder}")

# Iterate through the items (filename, details_dict) from the images_dict
for gt_filename, image_details in gt_items_to_iterate:

    # gt_filename is now correctly assigned the image filename (the key)

    original_img_path = os.path.join(img_input_folder, gt_filename)
    output_img_path = os.path.join(img_output_folder, f"compare_{gt_filename}")

    # Check if the original image file exists
    if not os.path.exists(original_img_path):
        print(f"Warning: Original image file not found, skipping: {original_img_path}")
        skipped_count += 1
        continue

    try:
        # --- Load Original Image ---
        image = Image.open(original_img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        gt_boxes_drawn = 0
        pred_boxes_drawn = 0

        # --- 4a. Draw GT Bounding Boxes (Green) ---
        # --- MODIFICATION START: Get annotations from image_details ---
        # 'image_details' is the dictionary associated with the filename key
        # It should contain the 'words' dictionary
        gt_annotations_dict = image_details.get('words', {})
        if not isinstance(gt_annotations_dict, dict):
             print(f"Warning: Expected 'words' to be a dictionary for {gt_filename}, found {type(gt_annotations_dict)}. Skipping GT annotations.")
             gt_annotations_to_process = []
        else:
             # Get the list of word dictionaries (the values within the 'words' dict)
             gt_annotations_to_process = gt_annotations_dict.values()
        # --- MODIFICATION END ---

        for annotation in gt_annotations_to_process:
            # 'annotation' is now a dictionary like:
            # {"text": "...", "points": [[x1,y1],[x2,y2],...], ...}
            points = annotation.get('points') # 'points' seems to be the key based on your JSON snippet

            if points is None or not isinstance(points, list):
                print(f"Warning: 'points' key missing or not a list in GT annotation for {gt_filename}. Annotation: {annotation}")
                continue # Skip if no valid points found

            # Parse GT points (handles nested list format [[x1, y1], [x2, y2], ...])
            gt_bbox_points = None
            try:
                # Check for the nested list format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                if len(points) == 4 and all(isinstance(p, list) and len(p) == 2 for p in points):
                     gt_bbox_points = [(int(p[0]), int(p[1])) for p in points]
                # Add checks for other formats if necessary (e.g., flat list)
                # elif len(points) == 8 and all(isinstance(p, (int, float)) for p in points):
                #     coords = [int(p) for p in points]
                #     gt_bbox_points = [(coords[0], coords[1]), (coords[2], coords[3]), (coords[4], coords[5]), (coords[6], coords[7])]
                else:
                    print(f"Warning: Unexpected GT points format for {gt_filename}. Points: {points}")
                    continue

                if gt_bbox_points:
                    draw.line(gt_bbox_points + [gt_bbox_points[0]], fill=gt_color, width=line_width)
                    gt_boxes_drawn += 1
            except (ValueError, TypeError, IndexError) as parse_err:
                 print(f"Warning: Invalid coordinate format/value in GT points for {gt_filename}. Points: {points}. Error: {parse_err}")
                 continue

        # --- 4b. Draw Prediction Bounding Boxes (Red) ---
        # This part remains the same - it uses the predictions dict loaded from CSV
        pred_coords_str = predictions.get(gt_filename)

        if pred_coords_str:
            box_strings = pred_coords_str.split('|')
            for box_str in box_strings:
                if not box_str: continue

                coord_values_str = box_str.strip().split(' ')
                if len(coord_values_str) == 8:
                    try:
                        coords = [int(c) for c in coord_values_str]
                        pred_bbox_points = [
                            (coords[0], coords[1]), (coords[2], coords[3]),
                            (coords[4], coords[5]), (coords[6], coords[7])
                        ]
                        draw.line(pred_bbox_points + [pred_bbox_points[0]], fill=pred_color, width=line_width)
                        pred_boxes_drawn += 1
                    except ValueError:
                        print(f"Warning: Non-integer coordinate found in prediction CSV for {gt_filename}. Box string: '{box_str}'. Skipping this pred box.")
                        continue
                else:
                     print(f"Warning: Malformed coordinate string in prediction CSV for {gt_filename}: '{box_str}'. Expected 8 values, got {len(coord_values_str)}. Skipping this pred box.")

        # --- 5. Save Combined Image ---
        image.save(output_img_path)
        processed_count += 1
        # Only print if boxes were actually drawn or if prediction existed but had no boxes
        if gt_boxes_drawn > 0 or pred_boxes_drawn > 0 or (pred_coords_str is not None):
             print(f"Processed '{gt_filename}': Drew {gt_boxes_drawn} GT boxes (Green), {pred_boxes_drawn} Pred boxes (Red). Saved to {output_img_path}")
        # else: # Optional: print even if nothing was drawn
        #      print(f"Processed '{gt_filename}': No boxes drawn (GT: {gt_boxes_drawn}, Pred: {pred_boxes_drawn}). Saved to {output_img_path}")


    except FileNotFoundError:
        # This might happen if file disappears between os.path.exists and Image.open
        print(f"Error: Original image file disappeared unexpectedly during processing: {original_img_path}")
        skipped_count += 1
    except Exception as e:
        print(f"Error processing image {gt_filename} ({original_img_path}): {e}")
        traceback.print_exc() # Print full traceback for debugging
        skipped_count += 1

print("\n--- Comparison Processing Summary ---")
print(f"Total GT JSON entries considered: {gt_entries_count}")
print(f"Images successfully processed and saved: {processed_count}")
print(f"GT entries/Images skipped (not found or errors): {skipped_count}")
print(f"Comparison images saved to: {img_output_folder}")