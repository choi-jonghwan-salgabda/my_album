import os
import json
import pandas as pd
import numpy as np
import torch
from cleval_metric import CLEvalMetric
from pathlib import Path


def load_test_data(json_path):
    """Load ground truth data from the test JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_data = {}
    for img_name, img_info in data.get('images', {}).items():
        gt_quads = []
        gt_texts = []
        gt_is_dcs = []
        
        for word_id, word_info in img_info.get('words', {}).items():
            if 'points' in word_info:
                # Convert points from [[x1, y1], [x2, y2], ...] to [x1, y1, x2, y2, ...]
                quad = [coord for point in word_info['points'] for coord in point]
                gt_quads.append(quad)
                gt_texts.append(word_info.get('text', ''))
                gt_is_dcs.append(False)  # Assume no "don't care" areas
        
        if gt_quads:
            gt_data[img_name] = {
                'quads': gt_quads,
                'texts': gt_texts,
                'is_dcs': gt_is_dcs
            }
    
    return gt_data

def load_predictions(csv_path):
    """Load MixNet predictions from CSV file."""
    df = pd.read_csv(csv_path)
    
    pred_data = {}
    for _, row in df.iterrows():
        filename = row['filename']
        polygons_str = row['polygons']
        
        # Parse polygons string (format: "x1 y1 x2 y2 ... xn yn|x1 y1 ...")
        polygons = []
        if isinstance(polygons_str, str) and len(polygons_str) > 0:
            for poly_str in polygons_str.split('|'):
                points = [float(p) for p in poly_str.split()]
                if len(points) >= 8:  # Ensure at least 4 points (8 coordinates)
                    polygons.append(points)
        
        if polygons:
            pred_data[filename] = {
                'quads': polygons,
                'texts': [''] * len(polygons)  # No text content in detection-only task
            }
    
    return pred_data

def evaluate_with_cleval(gt_data, pred_data):
    """Evaluate using CLEvalMetric."""
    metric = CLEvalMetric(
        case_sensitive=True,
        recall_gran_penalty=1.0,
        precision_gran_penalty=1.0
    )
    
    # Evaluate each image
    for img_name, gt_info in gt_data.items():
        if img_name in pred_data:
            det_quads = pred_data[img_name]['quads']
            det_texts = None
            gt_quads = gt_info['quads']
            gt_texts = None
            gt_is_dcs = None
            
            # Update metrics with this image's data
            metric.update(det_quads, gt_quads, det_texts, gt_texts, gt_is_dcs)
    
    # Compute final metrics
    results = metric.compute()
    return results

def main():
    # Paths
    test_json_path = "/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/jsons/test.json"
    csv_prediction_path = "/data/ephemeral/home/industry-partnership-project-brainventures/output/mixnet_test_enhanced_tuning.csv"  # Adjust this to your MixNet output CSV

    # Check if files exist
    if not os.path.exists(test_json_path):
        print(f"Error: Ground truth file not found at {test_json_path}")
        return
    if not os.path.exists(csv_prediction_path):
        print(f"Error: Prediction file not found at {csv_prediction_path}")
        return

    print(f"Loading ground truth data from {test_json_path}...")
    gt_data = load_test_data(test_json_path)
    print(f"Loaded {len(gt_data)} images with ground truth annotations")

    print(f"Loading predictions from {csv_prediction_path}...")
    pred_data = load_predictions(csv_prediction_path)
    print(f"Loaded predictions for {len(pred_data)} images")

    # Find common images
    common_images = set(gt_data.keys()) & set(pred_data.keys())
    print(f"Found {len(common_images)} common images for evaluation")
    
    # Evaluate
    print("Evaluating with CLEval metric...")
    results = evaluate_with_cleval(gt_data, pred_data)
    
    # Print results
    print("\n===== CLEval Evaluation Results =====")
    print(f"Detection Recall: {results['det_r'].item():.4f}")
    print(f"Detection Precision: {results['det_p'].item():.4f}")
    print(f"Detection Harmonic Mean (F1): {results['det_h'].item():.4f}")
    print(f"Split Cases: {results['num_splitted'].item()}")
    print(f"Merge Cases: {results['num_merged'].item()}")
    print(f"Character Overlaps: {results['num_char_overlapped'].item()}")
    
    # Save results to file
    results_path = os.path.join(os.path.dirname(csv_prediction_path), "cleval_results.txt")
    with open(results_path, 'w') as f:
        f.write("===== CLEval Evaluation Results =====\n")
        f.write(f"Detection Recall: {results['det_r'].item():.4f}\n")
        f.write(f"Detection Precision: {results['det_p'].item():.4f}\n")
        f.write(f"Detection Harmonic Mean (F1): {results['det_h'].item():.4f}\n")
        f.write(f"Split Cases: {results['num_splitted'].item()}\n")
        f.write(f"Merge Cases: {results['num_merged'].item()}\n")
        f.write(f"Character Overlaps: {results['num_char_overlapped'].item()}\n")
    
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()