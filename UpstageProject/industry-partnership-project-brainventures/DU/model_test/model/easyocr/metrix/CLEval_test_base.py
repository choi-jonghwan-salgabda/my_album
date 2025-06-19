# baseline 결과를 외부 CLEval로 돌렸을 때 점수를 비교해보기 위함함
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
# Ground Truth JSON 파일 경로
gt_json_path = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/jsons/val.json"
# Prediction JSON 파일 경로 (CSV 대신 JSON 사용)
pred_json_path = "/data/ephemeral/home/outputs/ocr_training/submissions/20250411_092951.json"

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

        # GT JSON 구조 확인
        if not (isinstance(gt_data, dict) and 'images' in gt_data and isinstance(gt_data['images'], dict)):
            raise ValueError("Unexpected GT JSON structure. Expected {'images': {filename: {...}}}")

        images_dict = gt_data['images']
        for filename, image_details in images_dict.items():
            gt_polygons = []
            gt_texts = []
            words_dict = image_details.get('words', {})
            if not isinstance(words_dict, dict):
                print(f"Warning: 'words' is not a dict for GT {filename}. Skipping GT annotations.")
                continue

            for word_id, annotation in words_dict.items():
                points = annotation.get('points')
                text = annotation.get('text', "") # 텍스트 추출, 없으면 빈 문자열

                # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 형식 확인
                if points and isinstance(points, list) and len(points) == 4 and all(isinstance(p, list) and len(p) == 2 for p in points):
                    try:
                        # 좌표를 float 리스트 [x1, y1, x2, y2, x3, y3, x4, y4]로 변환
                        formatted_points_flat = [float(coord) for point in points for coord in point]
                        if len(formatted_points_flat) == 8:
                            gt_polygons.append(formatted_points_flat)
                            gt_texts.append(text if text is not None else "") # 문자열 보장
                        else:
                             print(f"Warning: Invalid number of coordinates after flattening GT points for {filename}, word {word_id}. Points: {points}")
                    except (ValueError, TypeError, IndexError):
                        print(f"Warning: Invalid points format/value in GT for {filename}, word {word_id}. Points: {points}")
                else:
                     print(f"Warning: Missing or invalid 'points' format in GT for {filename}, word {word_id}. Points: {points}")

            if gt_polygons: # 유효한 폴리곤이 있을 때만 추가
                gt_data_formatted[filename] = {'polygons': gt_polygons, 'texts': gt_texts}

        print(f"Parsed GT data for {len(gt_data_formatted)} images.")
        return gt_data_formatted

    except FileNotFoundError:
        print(f"Error: GT JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from GT file {json_path}. Check file format.")
         return None
    except Exception as e:
        print(f"Error parsing GT JSON: {e}")
        traceback.print_exc()
        return None

def parse_pred_json(json_path):
    """Prediction JSON 파일을 파싱하여 CLEval 형식에 맞는 딕셔너리로 변환"""
    print(f"Loading and parsing Prediction JSON: {json_path}")
    pred_data_formatted = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        # Prediction JSON 구조 확인 (GT와 동일한 구조 가정)
        if not (isinstance(pred_data, dict) and 'images' in pred_data and isinstance(pred_data['images'], dict)):
            raise ValueError("Unexpected Prediction JSON structure. Expected {'images': {filename: {...}}}")

        images_dict = pred_data['images']
        for filename, image_details in images_dict.items():
            pred_polygons = []
            pred_texts = [] # 예측된 텍스트 저장
            words_dict = image_details.get('words', {})
            if not isinstance(words_dict, dict):
                print(f"Warning: 'words' is not a dict for Prediction {filename}. Skipping Prediction annotations.")
                continue

            for word_id, annotation in words_dict.items():
                points = annotation.get('points')
                text = annotation.get('text', "") # 예측된 텍스트 추출
                # confidence = annotation.get('confidence') # confidence는 CLEvalMetric.update에서 직접 사용하지 않음

                # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 형식 확인
                if points and isinstance(points, list) and len(points) == 4 and all(isinstance(p, list) and len(p) == 2 for p in points):
                    try:
                        # 좌표를 float 리스트 [x1, y1, x2, y2, x3, y3, x4, y4]로 변환
                        formatted_points_flat = [float(coord) for point in points for coord in point]
                        if len(formatted_points_flat) == 8:
                            pred_polygons.append(formatted_points_flat)
                            pred_texts.append(text if text is not None else "") # 문자열 보장
                        else:
                             print(f"Warning: Invalid number of coordinates after flattening Pred points for {filename}, word {word_id}. Points: {points}")
                    except (ValueError, TypeError, IndexError):
                        print(f"Warning: Invalid points format/value in Pred for {filename}, word {word_id}. Points: {points}")
                else:
                     print(f"Warning: Missing or invalid 'points' format in Pred for {filename}, word {word_id}. Points: {points}")

            if pred_polygons: # 유효한 폴리곤이 있을 때만 추가
                pred_data_formatted[filename] = {'polygons': pred_polygons, 'texts': pred_texts}

        print(f"Parsed Prediction data for {len(pred_data_formatted)} images.")
        return pred_data_formatted

    except FileNotFoundError:
        print(f"Error: Prediction JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from Prediction file {json_path}. Check file format.")
         return None
    except Exception as e:
        print(f"Error parsing Prediction JSON: {e}")
        traceback.print_exc()
        return None

# --- Main Evaluation ---
if __name__ == "__main__":
    # 1. 데이터 로드 및 파싱 (JSON 파서 사용)
    gt_data = parse_gt_json(gt_json_path)
    pred_data = parse_pred_json(pred_json_path) # 수정: JSON 파서 호출

    if gt_data is None or pred_data is None:
        print("Evaluation cannot proceed due to errors in data loading.")
        exit()

    print("\nStarting CLEval evaluation using CLEvalMetric...")
    try:
        # Instantiate the metric
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
            pred_letters_list = pred_info['texts'] # 수정: JSON에서 파싱된 실제 예측 텍스트 사용

            # --- Convert data to numpy arrays expected by CLEvalMetric.update ---
            gt_quads_np = np.array(gt_polygons_flat, dtype=np.float32) if gt_polygons_flat else np.empty((0, 8), dtype=np.float32)
            pred_quads_np = np.array(pred_polygons_flat, dtype=np.float32) if pred_polygons_flat else np.empty((0, 8), dtype=np.float32)

            # Ensure correct shape even if empty
            if gt_quads_np.ndim == 1 and gt_quads_np.shape[0] == 0: gt_quads_np = gt_quads_np.reshape(0, 8)
            if pred_quads_np.ndim == 1 and pred_quads_np.shape[0] == 0: pred_quads_np = pred_quads_np.reshape(0, 8)

            # --- Call update for each sample ---
            metric.update(
                det_quads=pred_quads_np,
                gt_quads=gt_quads_np,
                det_letters=pred_letters_list, # 예측된 텍스트 리스트 전달
                gt_letters=gt_letters_list,   # GT 텍스트 리스트 전달
                gt_is_dcs=None # 'don't care' 라벨 없다고 가정
            )

            processed_files += 1
            if processed_files % 500 == 0:
                print(f"  Processed {processed_files}/{len(all_filenames)} files...")

        # Compute final results after processing all samples
        print("Computing final metrics...")
        results = metric.compute()

        # 3. 결과 출력 (Detection 및 E2E 결과 모두 출력)
        print("\n--- CLEvalMetric Detection Results ---")
        precision = results.get('det_p', 'N/A')
        recall = results.get('det_r', 'N/A')
        hmean = results.get('det_h', 'N/A') # F1-score

        if isinstance(precision, torch.Tensor): precision = precision.item()
        if isinstance(recall, torch.Tensor): recall = recall.item()
        if isinstance(hmean, torch.Tensor): hmean = hmean.item()

        print(f"Detection Precision: {precision:.4f}" if isinstance(precision, float) else f"Detection Precision: {precision}")
        print(f"Detection Recall:    {recall:.4f}" if isinstance(recall, float) else f"Detection Recall:    {recall}")
        print(f"Detection HMean (F1-score): {hmean:.4f}" if isinstance(hmean, float) else f"Detection HMean (F1-score): {hmean}")

        print("\n--- CLEvalMetric End-to-End Results ---")
        e2e_precision = results.get('e2e_p', 'N/A')
        e2e_recall = results.get('e2e_r', 'N/A')
        e2e_hmean = results.get('e2e_h', 'N/A') # E2E F1-score

        if isinstance(e2e_precision, torch.Tensor): e2e_precision = e2e_precision.item()
        if isinstance(e2e_recall, torch.Tensor): e2e_recall = e2e_recall.item()
        if isinstance(e2e_hmean, torch.Tensor): e2e_hmean = e2e_hmean.item()

        print(f"E2E Precision: {e2e_precision:.4f}" if isinstance(e2e_precision, float) else f"E2E Precision: {e2e_precision}")
        print(f"E2E Recall:    {e2e_recall:.4f}" if isinstance(e2e_recall, float) else f"E2E Recall:    {e2e_recall}")
        print(f"E2E HMean (F1-score): {e2e_hmean:.4f}" if isinstance(e2e_hmean, float) else f"E2E HMean (F1-score): {e2e_hmean}")


        # print("Full results dictionary:", results) # Uncomment to see all returned values

    except ImportError:
        print("\nError: Could not import 'CLEvalMetric' from 'cleval'. Is cleval installed correctly?")
        print("Try: pip install cleval")
    except Exception as e:
        print(f"\nAn error occurred during CLEval evaluation: {e}")
        traceback.print_exc()
