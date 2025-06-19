# -*- coding: utf-8 -*-
import os
import json
import csv
import random
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import traceback
import re
# --- Removed deep_translator dependency ---

# --- Configuration ---

# 1. Prediction BBox Sources
PRED_CSV_DIR = "/data/ephemeral/home/industry-partnership-project-brainventures/output"

# 2. Ground Truth BBox Source
GT_JSON_PATH = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/jsons/test.json"

# 3. Category Sampling
NUM_SAMPLES_PER_CATEGORY = 5

# 4. Original Image Source
IMG_BASE_FOLDER = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/images/test/"

# 5. Output Configuration
OUTPUT_BASE_FOLDER = "/data/ephemeral/home/outputs/sample_vis/"
GT_COLOR = (0, 0, 255)    # Blue for Ground Truth
PRED_COLOR = (255, 0, 0)  # Red for Predictions
LINE_WIDTH = 8            # Width of the bounding box lines

# --- !!! Manually Curated Category Translation Dictionary !!! ---
# Please VERIFY and ADD any missing categories from your actual data.
CATEGORY_TRANSLATION = {
    "간판_가로형간판": "signboard_horizontal",
    "간판_돌출간판": "signboard_protruding", # Corrected based on common usage
    "간판_세로형간판": "signboard_vertical",
    "간판_실내간판": "signboard_indoor",
    "간판_실내안내판": "signboard_guide",
    "간판_지주이용간판": "signboard_pole", # Corrected based on common usage
    "간판_창문이용광고물": "signboard_window",
    "간판_현수막": "signboard_banner", # Corrected spelling
    "책표지_기술과학": "bookcover_science",
    "책표지_기타": "bookcover_etc",
    "책표지_문학": "bookcover_literature",
    "책표지_사회과학": "bookcover_social",
    "책표지_언어": "bookcover_language",
    "책표지_역사": "bookcover_history",
    "책표지_예술": "bookcover_art",
    "책표지_자연과학": "bookcover_natural",
    "책표지_종교": "bookcover_religion",
    "책표지_철학": "bookcover_philosophy",
    "책표지_총류": "bookcover_general",
    # --- Add other categories found in your data below ---
    # "실제_한글_카테고리": "desired_english_name",
    "상품_상품": "product_general", # Example from previous context
    "상품_패키지": "product_package", # Example from previous context
    "메뉴판": "menu",             # Example from previous context
    "문서": "document",           # Example from previous context
    "영수증": "receipt",           # Example from previous context
    "카드": "card",               # Example from previous context
    "기타": "etc",                # Example from previous context
}

# --- Helper Functions ---

def get_category_from_filename(filename):
    """Extracts the category part (e.g., '간판_가로형간판') from the filename."""
    match = re.match(r"^(.*?)_\d+\.jpg$", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        base, _ = os.path.splitext(filename)
        if '_' in base:
            category_part = base.rsplit('_', 1)[0]
            potential_number = base.rsplit('_', 1)[1]
            if potential_number.isdigit():
                 return category_part
            else:
                 print(f"Warning: Filename '{filename}' doesn't match 'category_number.jpg'. Using '{base}' as category.")
                 return base
        print(f"Warning: Could not determine category for filename: {filename}. Using 'unknown_category'.")
        return "unknown_category"

def load_gt_data(json_path):
    """Loads Ground Truth data from JSON file."""
    print(f"Loading Ground Truth data from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        print("GT JSON data loaded successfully.")
        if not isinstance(gt_data, dict) or 'images' not in gt_data or not isinstance(gt_data['images'], dict):
            print(f"Error: Invalid GT JSON structure in {json_path}.")
            print("Expected: {'images': {'filename1': {...}, 'filename2': {...}}}")
            return None
        return gt_data['images']
    except FileNotFoundError:
        print(f"Error: GT JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}. Check file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading GT JSON: {e}")
        traceback.print_exc()
        return None

def extract_categories_and_sample(gt_images_dict, num_samples=5):
    """Extracts categories, prints them, and samples filenames per category."""
    print("\nExtracting categories and sampling...")
    category_to_files = defaultdict(list)
    for filename in gt_images_dict.keys():
        category = get_category_from_filename(filename)
        category_to_files[category].append(filename)

    # --- Print Discovered Categories ---
    korean_categories = sorted(list(category_to_files.keys()))
    print("\n--- Discovered Korean Categories (Check against CATEGORY_TRANSLATION) ---")
    if korean_categories:
        for cat in korean_categories:
            print(f"- {cat} ({len(category_to_files[cat])} images)")
    else:
        print("No categories found.")
    print("---------------------------------------------------------------------\n")

    # --- Sample Files ---
    sampled_files_by_category = {}
    total_sampled = 0
    for category, files in category_to_files.items():
        sample_size = min(num_samples, len(files))
        if sample_size > 0:
            sampled_files = random.sample(files, sample_size)
            sampled_files_by_category[category] = sampled_files
            total_sampled += len(sampled_files)

    print(f"Total images sampled: {total_sampled}")
    print("Sampling complete.")
    # No longer need to return korean_categories list separately
    return sampled_files_by_category

# --- Removed translate_categories function ---

def find_prediction_csvs(pred_dir):
    """Finds all .csv files directly within the prediction directory."""
    print(f"Scanning for prediction CSV files in: {pred_dir}")
    csv_files = []
    try:
        for item in os.listdir(pred_dir):
            item_path = os.path.join(pred_dir, item)
            if os.path.isfile(item_path) and item.lower().endswith('.csv'):
                csv_files.append(item_path)
        print(f"Found {len(csv_files)} prediction CSV files.")
        if not csv_files:
             print("Warning: No prediction CSV files found. Visualization will only show GT boxes.")
    except FileNotFoundError:
        print(f"Error: Prediction directory not found: {pred_dir}")
        return []
    except Exception as e:
        print(f"Error scanning prediction directory {pred_dir}: {e}")
        return []
    return csv_files

def load_predictions(csv_path):
    """Loads predictions from a single CSV file into a dictionary."""
    predictions = {}
    print(f"  Loading predictions from: {os.path.basename(csv_path)}")
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            try:
                dialect = csv.Sniffer().sniff(csvfile.read(1024*5), delimiters=',\t')
                csvfile.seek(0)
                reader = csv.reader(csvfile, delimiter=dialect.delimiter)
                header = next(reader)
                expected_header = ['filename', 'polygons']
                if header != expected_header:
                     print(f"    Warning: Unexpected header in {os.path.basename(csv_path)}: {header}. Expected {expected_header}. Processing anyway.")
                     csvfile.seek(0)
                     reader = csv.reader(csvfile, delimiter=dialect.delimiter)
            except (csv.Error, StopIteration):
                 print(f"    Warning: Could not automatically detect delimiter or file is empty for {os.path.basename(csv_path)}. Assuming tab ('\\t') and no header.")
                 csvfile.seek(0)
                 reader = csv.reader(csvfile, delimiter='\t')

            processed_rows = 0
            for i, row in enumerate(reader):
                if i == 0 and row == ['filename', 'polygons']:
                    print("    Skipping detected header row again.")
                    continue

                if len(row) == 2:
                    img_name, coords_str = row[0].strip(), row[1].strip()
                    if not img_name:
                        print(f"    Warning: Skipping row {i+1} in {os.path.basename(csv_path)} due to empty filename.")
                        continue
                    predictions[img_name] = coords_str
                    processed_rows += 1
                elif len(row) > 0 and row[0].strip():
                    img_name = row[0].strip()
                    coords_str = ""
                    print(f"    Warning: Row {i+1} in {os.path.basename(csv_path)} has incorrect format (expected 2 columns, got {len(row)}). Assuming no polygons for '{img_name}'. Row: {row}")
                    predictions[img_name] = coords_str
                    processed_rows += 1
                elif row:
                    print(f"    Warning: Skipping row {i+1} in {os.path.basename(csv_path)} due to missing/empty filename in first column. Row: {row}")

            print(f"    Loaded predictions for {processed_rows} images.")

    except FileNotFoundError:
        print(f"  Error: Prediction CSV file not found at {csv_path}")
        return None
    except StopIteration:
         print(f"    Warning: Prediction CSV file {os.path.basename(csv_path)} might be empty.")
    except Exception as e:
        print(f"  An unexpected error occurred while loading prediction CSV {os.path.basename(csv_path)}: {e}")
        traceback.print_exc()
        return None
    return predictions

def parse_gt_points(points_data):
    """Parses GT points (expects [[x1,y1],[x2,y2],...]). Returns list of tuples or None."""
    if not isinstance(points_data, list) or len(points_data) != 4:
        return None
    try:
        parsed_points = []
        for p in points_data:
            if isinstance(p, list) and len(p) == 2:
                 x, y = int(p[0]), int(p[1])
                 if x < 0 or y < 0:
                     x = max(0, x)
                     y = max(0, y)
                 parsed_points.append((x, y))
            else:
                 return None
        return parsed_points
    except (ValueError, TypeError):
        return None

def parse_pred_coords(coords_str):
    """
    Parses prediction coordinate string (x y x y...|...).
    Handles cases where multiple boxes are concatenated without '|' delimiter.
    Returns list of lists of tuples.
    """
    if not coords_str:
        return []

    all_boxes_points = []
    box_strings = coords_str.split('|')

    for box_str in box_strings:
        box_str = box_str.strip()
        if not box_str: continue

        coord_values_str = box_str.split(' ')
        num_coords = len(coord_values_str)

        if num_coords == 0:
            continue

        if num_coords % 8 == 0:
            for i in range(0, num_coords, 8):
                chunk_coords_str = coord_values_str[i:i+8]
                try:
                    coords = [int(float(c)) for c in chunk_coords_str]
                    pred_bbox_points = []
                    for j in range(0, 8, 2):
                         x, y = coords[j], coords[j+1]
                         if x < 0 or y < 0:
                             x = max(0, x)
                             y = max(0, y)
                         pred_bbox_points.append((x, y))
                    all_boxes_points.append(pred_bbox_points)
                except (ValueError, TypeError):
                    print(f"    Warning: Invalid coordinate value found in prediction chunk: '{' '.join(chunk_coords_str)}'. Skipping this box chunk.")
                    continue
        else:
            print(f"    Warning: Malformed coordinate string fragment in prediction: '{box_str}'. Expected a multiple of 8 values, got {num_coords}. Skipping this fragment.")
            continue

    return all_boxes_points

def draw_boxes(draw, points, color, width):
    """Helper function to draw a single polygon."""
    if points and len(points) >= 3:
        try:
            draw.line(points + [points[0]], fill=color, width=width)
            return True
        except Exception as e:
            print(f"    Error drawing box with points {points}: {e}")
            return False
    return False

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting BBox Visualization Script...")

    # --- 1. Load Ground Truth ---
    gt_images_data = load_gt_data(GT_JSON_PATH)
    if gt_images_data is None:
        print("Failed to load Ground Truth data. Exiting.")
        exit()

    # --- 2. Extract Categories, Print List, and Sample ---
    # The keys of this dictionary are the Korean category names
    sampled_files_by_cat = extract_categories_and_sample(
        gt_images_data, NUM_SAMPLES_PER_CATEGORY
    )
    if not sampled_files_by_cat:
         print("No categories found or sampled in GT data. Cannot proceed.")
         exit()

    # --- 3. Find Prediction CSVs ---
    pred_csv_paths = find_prediction_csvs(PRED_CSV_DIR)
    if not pred_csv_paths:
        print("No prediction CSV files found. Visualization will only show GT boxes, but proceeding.")

    # --- 4. Create Output Directories (using English names from dictionary) ---
    print("\nEnsuring all category output directories exist (using English names)...")
    created_dirs = set()
    missing_translations_warned = set() # Track warnings to avoid repetition
    for category_ko in sampled_files_by_cat.keys():
        # --- Look up English name in the dictionary ---
        category_en = CATEGORY_TRANSLATION.get(category_ko)
        if not category_en:
            # Fallback if not found in the dictionary
            category_en = category_ko.replace('_', '-')
            if category_ko not in missing_translations_warned:
                print(f"  Warning: No English translation found for category '{category_ko}' in CATEGORY_TRANSLATION. Using fallback folder name: '{category_en}'. Please update the dictionary.")
                missing_translations_warned.add(category_ko) # Mark as warned

        output_dir = os.path.join(OUTPUT_BASE_FOLDER, category_en)
        if output_dir not in created_dirs:
            try:
                os.makedirs(output_dir, exist_ok=True)
                created_dirs.add(output_dir)
            except OSError as e:
                print(f"  Warning: Could not create directory '{output_dir}': {e}. Files for this category might not be saved.")
    print("Directory check complete.")

    # --- 5. Iterate through models (CSVs) and draw ---
    print("\n--- Starting Visualization Generation ---")
    overall_processed_count = 0
    overall_skipped_count = 0
    overall_error_count = 0

    # Loop through each model (CSV file)
    for pred_csv_path in pred_csv_paths:
        model_name = os.path.splitext(os.path.basename(pred_csv_path))[0]
        print(f"\nProcessing Model: {model_name} (from {os.path.basename(pred_csv_path)})")

        predictions_dict = load_predictions(pred_csv_path)
        if predictions_dict is None:
            print(f"  Skipping model {model_name} due to error loading predictions.")
            est_errors = sum(len(files) for files in sampled_files_by_cat.values())
            overall_error_count += est_errors
            continue

        model_processed_count = 0
        model_skipped_count = 0

        # Iterate through each category (Korean key) and its sampled filenames
        for category_ko, sampled_filenames in sampled_files_by_cat.items():

            # --- Get English category name from the dictionary ---
            category_en = CATEGORY_TRANSLATION.get(category_ko)
            if not category_en:
                category_en = category_ko.replace('_', '-') # Use the same fallback
                # Warning already printed during directory creation

            output_dir = os.path.join(OUTPUT_BASE_FOLDER, category_en)

            if not os.path.isdir(output_dir):
                 print(f"  Skipping category '{category_ko}' (English: '{category_en}') for model '{model_name}' because output directory '{output_dir}' does not exist.")
                 model_skipped_count += len(sampled_filenames)
                 continue

            print(f"  Processing Category: {category_ko} -> {category_en} ({len(sampled_filenames)} images for this model)")

            # Process each sampled image
            for filename in sampled_filenames:
                original_img_path = os.path.join(IMG_BASE_FOLDER, filename)

                # --- Construct output filename using English category ---
                match = re.match(r"^(.*?)_(\d+\.jpg)$", filename, re.IGNORECASE)
                if match:
                    filename_suffix = "_" + match.group(2)
                else:
                    print(f"    Warning: Could not extract suffix from filename '{filename}'. Using original filename structure in output.")
                    if filename.startswith(category_ko):
                         filename_suffix = filename[len(category_ko):]
                    else:
                         filename_suffix = "_" + filename

                output_filename = f"{model_name}_{category_en}{filename_suffix}"
                output_img_path = os.path.join(output_dir, output_filename)

                if not os.path.exists(original_img_path):
                    print(f"    Warning: Original image file not found, skipping: {original_img_path} (for model {model_name})")
                    model_skipped_count += 1
                    continue

                try:
                    image = Image.open(original_img_path).convert("RGB")
                    draw = ImageDraw.Draw(image)
                    gt_boxes_drawn = 0
                    pred_boxes_drawn = 0

                    # --- Draw GT Bounding Boxes (Blue) ---
                    image_details = gt_images_data.get(filename)
                    if image_details and 'words' in image_details:
                        words_data = image_details['words']
                        gt_annotations = []
                        if isinstance(words_data, dict):
                             gt_annotations = words_data.values()
                        elif isinstance(words_data, list):
                             gt_annotations = words_data

                        for annotation in gt_annotations:
                            if isinstance(annotation, dict) and 'points' in annotation:
                                gt_points = parse_gt_points(annotation['points'])
                                if gt_points:
                                    if draw_boxes(draw, gt_points, GT_COLOR, LINE_WIDTH):
                                        gt_boxes_drawn += 1

                    # --- Draw Prediction Bounding Boxes (Red) ---
                    pred_coords_str = predictions_dict.get(filename)
                    if pred_coords_str is not None:
                        pred_boxes_points_list = parse_pred_coords(pred_coords_str)
                        for pred_points in pred_boxes_points_list:
                            if draw_boxes(draw, pred_points, PRED_COLOR, LINE_WIDTH):
                                pred_boxes_drawn += 1

                    # --- Save Combined Image ---
                    image.save(output_img_path)
                    model_processed_count += 1

                except FileNotFoundError:
                    print(f"    Error: Original image disappeared during processing: {original_img_path} (for model {model_name})")
                    model_skipped_count += 1
                except Exception as e:
                    print(f"    Error processing image {filename} for model {model_name}: {e}")
                    traceback.print_exc()
                    model_skipped_count += 1

        print(f"  Finished processing for model {model_name}. Processed: {model_processed_count}, Skipped/Errors: {model_skipped_count}")
        overall_processed_count += model_processed_count
        overall_skipped_count += model_skipped_count

    # --- Final Summary ---
    print("\n--- Visualization Generation Summary ---")
    print(f"Total prediction CSVs processed: {len(pred_csv_paths)}")
    print(f"Total images successfully processed and saved: {overall_processed_count}")
    print(f"Total images skipped due to errors or file not found: {overall_skipped_count + overall_error_count}")
    print(f"Comparison images saved under: {OUTPUT_BASE_FOLDER}")
    print("Structure: ENGLISH_CATEGORY_NAME / MODEL_NAME_ENGLISH_CATEGORY_NAME_NUMBER.jpg")
    print("Script finished.")