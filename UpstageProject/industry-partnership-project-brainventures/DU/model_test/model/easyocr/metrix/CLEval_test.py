# EasyOCR 결과와 ai_hub GT와 결과를 비교함함
import json
import csv
import os
import numpy as np
# Import torch as CLEvalMetric relies on it
import torch
# Import the correct class from the installed package
from cleval import CLEvalMetric
import traceback

# --- Configuration ---
gt_json_path = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/jsons/val.json"
pred_csv_path = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/easyocr/easyocr.csv"

# --- CLEvalMetric Configuration ---
# Map your config to CLEvalMetric's __init__ parameters
# Refer to CLEval-master/cleval/torchmetric.py for exact parameter names
metric_config = {
    'case_sensitive': True, # Default in CLEvalMetric
    'ap_constraint': 0.5, # Your AREA_PRECISION_CONSTRAINT (IoU threshold for matching)
    # Add other relevant parameters from torchmetric.py's __init__ if needed
    # 'recall_gran_penalty': 1.0, # Default
    # 'precision_gran_penalty': 1.0, # Default
    # 'vertical_aspect_ratio_thresh': 0.5, # Default
}

# --- Helper Functions ---

def parse_gt_json(json_path):
    """GT JSON 파일을 파싱하여 CLEval 형식에 맞는 딕셔너리로 변환"""
    print(f"Loading and parsing GT JSON: {json_path}")
    gt_data_formatted = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        if not (isinstance(gt_data, dict) and 'images' in gt_data and isinstance(gt_data['images'], dict)):
            raise ValueError("Unexpected GT JSON structure. Expected {'images': {filename: {...}}}")

        images_dict = gt_data['images']
        for filename, image_details in images_dict.items():
            gt_polygons = []
            # --- MODIFICATION: Also extract text if available, default to "" ---
            gt_texts = []
            words_dict = image_details.get('words', {})
            if not isinstance(words_dict, dict):
                print(f"Warning: 'words' is not a dict for {filename}. Skipping GT annotations.")
                continue

            for word_id, annotation in words_dict.items():
                points = annotation.get('points')
                text = annotation.get('text', "") # Get text, default to empty string

                # Expecting [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] format
                if points and isinstance(points, list) and len(points) == 4 and all(isinstance(p, list) and len(p) == 2 for p in points):
                    try:
                        # Flatten points to [x1, y1, x2, y2, x3, y3, x4, y4] for numpy array
                        # Ensure points are numeric and convert to float
                        formatted_points_flat = [float(coord) for point in points for coord in point]
                        if len(formatted_points_flat) == 8:
                             # Store the flat list for easier conversion later
                            gt_polygons.append(formatted_points_flat)
                            gt_texts.append(text if text is not None else "") # Ensure it's a string
                        else:
                             print(f"Warning: Invalid number of coordinates after flattening GT points for {filename}, word {word_id}. Points: {points}")
                    except (ValueError, TypeError, IndexError):
                        print(f"Warning: Invalid points format/value in GT for {filename}, word {word_id}. Points: {points}")
                else:
                     print(f"Warning: Missing or invalid 'points' format in GT for {filename}, word {word_id}. Points: {points}")

            if gt_polygons: # Add only if valid polygons exist
                # Store both polygons and corresponding texts
                gt_data_formatted[filename] = {'polygons': gt_polygons, 'texts': gt_texts}

        print(f"Parsed GT data for {len(gt_data_formatted)} images.")
        return gt_data_formatted

    except FileNotFoundError:
        print(f"Error: GT JSON file not found at {json_path}")
        return None
    except Exception as e:
        print(f"Error parsing GT JSON: {e}")
        traceback.print_exc()
        return None

def parse_pred_csv(csv_path):
    """예측 CSV 파일을 파싱하여 CLEval 형식에 맞는 딕셔너리로 변환"""
    print(f"Loading and parsing Prediction CSV: {csv_path}")
    pred_data_formatted = {}
    predictions_raw = {}
    file_count = 0
    box_count = 0
    skipped_rows = 0

    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for i, row in enumerate(reader):
                if len(row) == 2:
                    filename, coords_str = row
                    if filename not in predictions_raw:
                        predictions_raw[filename] = []
                        file_count += 1
                    predictions_raw[filename].append(coords_str)
                else:
                    print(f"Warning: Skipping malformed row {i+1} in CSV: {row}")
                    skipped_rows += 1

        print(f"Loaded raw predictions for {file_count} images from CSV.")

        # Raw 데이터를 CLEval 형식으로 변환
        for filename, coord_strs_list in predictions_raw.items():
            pred_polygons = []
            # --- MODIFICATION: Create dummy text list ---
            pred_texts = []
            for coords_str in coord_strs_list:
                 box_strings = coords_str.split('|')
                 for box_str in box_strings:
                    if not box_str: continue
                    coord_values_str = box_str.strip().split(' ')
                    if len(coord_values_str) == 8:
                        try:
                            # 좌표를 float으로 변환 (정밀도 유지) and flatten
                            coords_flat = [float(c) for c in coord_values_str]
                            pred_polygons.append(coords_flat) # Store flat list
                            pred_texts.append("") # Add dummy empty string for text
                            box_count += 1
                        except ValueError:
                            print(f"Warning: Non-numeric coordinate in prediction CSV for {filename}. Box: '{box_str}'")
                    else:
                        print(f"Warning: Malformed coordinate string in prediction CSV for {filename}: '{box_str}'. Expected 8 values.")

            if pred_polygons: # Add only if valid polygons exist
                # Store both polygons and corresponding dummy texts
                pred_data_formatted[filename] = {'polygons': pred_polygons, 'texts': pred_texts}

        print(f"Parsed predictions for {len(pred_data_formatted)} images with a total of {box_count} boxes.")
        if skipped_rows > 0:
            print(f"Skipped {skipped_rows} malformed rows in CSV.")
        return pred_data_formatted

    except FileNotFoundError:
        print(f"Error: Prediction CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error parsing prediction CSV: {e}")
        traceback.print_exc()
        return None

# --- Main Evaluation ---
if __name__ == "__main__":
    # 1. 데이터 로드 및 파싱 (returns dict: {filename: {'polygons': [[x1,y1,...], ...], 'texts': ["",...]} })
    gt_data = parse_gt_json(gt_json_path)
    pred_data = parse_pred_csv(pred_csv_path)

    if gt_data is None or pred_data is None:
        print("Evaluation cannot proceed due to errors in data loading.")
        exit()

    print("\nStarting CLEval evaluation using CLEvalMetric...")
    try:
        # Instantiate the metric
        # Pass config arguments directly
        metric = CLEvalMetric(**metric_config)
        # If running on GPU: metric = metric.to('cuda')

        # Get the union of all filenames to process
        all_filenames = set(gt_data.keys()) | set(pred_data.keys())
        print(f"Processing {len(all_filenames)} unique image entries...")

        # Iterate through images
        processed_files = 0
        for filename in all_filenames:
            # --- Extract polygons and texts (or defaults) ---
            gt_info = gt_data.get(filename, {'polygons': [], 'texts': []})
            pred_info = pred_data.get(filename, {'polygons': [], 'texts': []})

            gt_polygons_flat = gt_info['polygons']
            pred_polygons_flat = pred_info['polygons']
            gt_letters_list = gt_info['texts']
            pred_letters_list = pred_info['texts'] # These are dummy "" from parsing

            # --- Convert data to numpy arrays expected by CLEvalMetric.update ---
            # Shape should be (N, 8) where N is the number of boxes
            gt_quads_np = np.array(gt_polygons_flat, dtype=np.float32) if gt_polygons_flat else np.empty((0, 8), dtype=np.float32)
            pred_quads_np = np.array(pred_polygons_flat, dtype=np.float32) if pred_polygons_flat else np.empty((0, 8), dtype=np.float32)

            # Ensure correct shape even if empty
            if gt_quads_np.ndim == 1 and gt_quads_np.shape[0] == 0: gt_quads_np = gt_quads_np.reshape(0, 8)
            if pred_quads_np.ndim == 1 and pred_quads_np.shape[0] == 0: pred_quads_np = pred_quads_np.reshape(0, 8)

            # --- Call update for each sample ---
            # Pass the lists of strings (empty strings for predictions)
            metric.update(
                det_quads=pred_quads_np,
                gt_quads=gt_quads_np,
                det_letters=pred_letters_list, # Pass the list of empty strings
                gt_letters=gt_letters_list,   # Pass the list of actual (or empty) GT strings
                gt_is_dcs=None # Assuming no 'don't care' labels in your GT
            )
            # --- Modification End ---

            processed_files += 1
            if processed_files % 500 == 0:
                print(f"  Processed {processed_files}/{len(all_filenames)} files...")

        # Compute final results after processing all samples
        print("Computing final metrics...")
        results = metric.compute()

        # 3. 결과 출력 (Focus on Detection Metrics)
        print("\n--- CLEvalMetric Detection Results ---")
        # Keys based on torchmetric.py: det_p, det_r, det_h
        precision = results.get('det_p', 'N/A')
        recall = results.get('det_r', 'N/A')
        hmean = results.get('det_h', 'N/A') # F1-score

        # Convert from potential torch tensors to float for printing
        if isinstance(precision, torch.Tensor): precision = precision.item()
        if isinstance(recall, torch.Tensor): recall = recall.item()
        if isinstance(hmean, torch.Tensor): hmean = hmean.item()

        print(f"Precision: {precision:.4f}" if isinstance(precision, float) else f"Precision: {precision}")
        print(f"Recall:    {recall:.4f}" if isinstance(recall, float) else f"Recall:    {recall}")
        print(f"HMean (F1-score): {hmean:.4f}" if isinstance(hmean, float) else f"HMean (F1-score): {hmean}")

        # Optionally print E2E results if needed, but focus is on detection
        # if 'e2e_h' in results:
        #     e2e_hmean = results.get('e2e_h', 'N/A')
        #     if isinstance(e2e_hmean, torch.Tensor): e2e_hmean = e2e_hmean.item()
        #     print(f"\n(E2E HMean: {e2e_hmean:.4f})" if isinstance(e2e_hmean, float) else f"(E2E HMean: {e2e_hmean})")


        # print("Full results dictionary:", results) # Uncomment to see all returned values

    except ImportError:
        print("\nError: Could not import 'CLEvalMetric' from 'cleval'. Is cleval installed correctly?")
        print("Try: pip install cleval")
    except Exception as e:
        print(f"\nAn error occurred during CLEval evaluation: {e}")
        traceback.print_exc()
