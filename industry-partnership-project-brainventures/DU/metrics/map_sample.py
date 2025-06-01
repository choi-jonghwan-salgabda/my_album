import json
import numpy as np
from sklearn.metrics import average_precision_score

# IoU 계산 함수
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# 텍스트 매칭 함수
def match_text(pred_text, gt_text):
    return pred_text.lower() == gt_text.lower()

# mAP 계산 함수
def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    predictions = sorted(predictions, key=lambda x: x.get("score", 1.0), reverse=True)
    y_true = []
    y_scores = []
    used_gt = set()

    for pred in predictions:
        pred_box = pred["bbox"]
        pred_text = pred.get("text", "")
        pred_conf = pred.get("score", 1.0)

        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in used_gt:
                continue
            gt_box = gt["bbox"]
            gt_text = gt.get("text", "")
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou and match_text(pred_text, gt_text):
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            y_true.append(1)
            used_gt.add(best_gt_idx)
        else:
            y_true.append(0)
        y_scores.append(pred_conf)

    if len(y_true) == 0:
        return 0
    ap = average_precision_score(y_true, y_scores)
    return ap

# COCO 형식 JSON 로드
with open("coco_annotations.json", "r") as f:
    coco_data = json.load(f)

# Ground Truth와 예측 데이터 가정 (동일 데이터로 테스트)
ground_truths = coco_data["annotations"]
predictions = coco_data["annotations"]  # 실제로는 예측 데이터로 대체

# mAP 계산
ap = calculate_map(predictions, ground_truths)
print(f"mAP@0.5: {ap:.3f}")