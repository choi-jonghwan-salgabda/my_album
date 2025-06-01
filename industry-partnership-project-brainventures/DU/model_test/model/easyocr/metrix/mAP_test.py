import json
import numpy as np

def calculate_iou(box1, box2):
    """
    두 bbox의 IoU(Intersection over Union)를 계산합니다.
    box1, box2: [x1, y1, x2, y2] 형식의 bbox 좌표
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    iou = intersection / union if union > 0 else 0
    return iou
################################## gt
with open("/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/label/000001.json", "r") as f:
    gt_data = json.load(f)
print(f"GT bbox 갯수: {len(gt_data['annotations'])}")
gt_boxes = []
for ann in gt_data['annotations']:
    x, y, w, h = ann['bbox']
    gt_box = [x, y, x + w, y + h]
    gt_boxes.append(gt_box)
print(f"gt_boxes: {gt_boxes}")
################################## pred
with open("/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/output/easyocr/annotated_책표지_총류_000001.json", "r") as f:
    pred_data = json.load(f)
print(f"EasyOCR bbox 갯수: {len(pred_data)}")

# Pred boxes와 confidence scores 추출
pred_boxes = []
confidence_scores = []
for item in pred_data:
    box = item[0]
    conf = item[2]  # EasyOCR의 confidence score
    x_coords = [coord[0] for coord in box]
    y_coords = [coord[1] for coord in box]
    converted_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    pred_boxes.append(converted_box)
    confidence_scores.append(conf)

print(f"Confidence scores: {confidence_scores}")

################################## IoU 계산
print("\nIoU 계산 결과:")
for i, gt_box in enumerate(gt_boxes):
    print(f"\nGround Truth #{i+1}:")
    for j, pred_box in enumerate(pred_boxes):
        iou = calculate_iou(gt_box, pred_box)
        print(f"- Prediction #{j+1}: IoU = {iou:.4f}")

# 각 GT bbox에 대해 가장 높은 IoU를 가진 예측 bbox 찾기
print("\n최고 IoU 매칭:")
for i, gt_box in enumerate(gt_boxes):
    ious = [calculate_iou(gt_box, pred_box) for pred_box in pred_boxes]
    max_iou = max(ious)
    max_iou_idx = ious.index(max_iou)
    print(f"GT #{i+1} -> Pred #{max_iou_idx+1}: IoU = {max_iou:.4f}")

################################## mAP 계산
def calculate_map(gt_boxes, pred_boxes, confidence_scores, iou_threshold=0.5):
    # confidence scores로 예측 결과 정렬
    indices = np.argsort(-np.array(confidence_scores))
    sorted_preds = [pred_boxes[i] for i in indices]
    sorted_scores = [confidence_scores[i] for i in indices]
    
    tp = np.zeros(len(sorted_preds))
    fp = np.zeros(len(sorted_preds))
    
    # 각 예측에 대해 TP/FP 결정
    for i, pred in enumerate(sorted_preds):
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_boxes):
            iou = calculate_iou(gt, pred)
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
                
        if max_iou >= iou_threshold:
            tp[i] = 1
        else:
            fp[i] = 1
            
    # 누적합 계산
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # precision과 recall 계산
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / len(gt_boxes)
    
    # precision values를 최대값으로 smoothing
    for i in range(len(precisions)-1, 0, -1):
        precisions[i-1] = max(precisions[i-1], precisions[i])
    
    # AP 계산 (11-point interpolation)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap, precisions.tolist(), recalls.tolist()

# mAP 계산 및 출력
ap, precisions, recalls = calculate_map(gt_boxes, pred_boxes, confidence_scores)

print("\nmAP 계산 결과:")
print(f"Average Precision (AP): {ap:.4f}")
print("\nPrecision-Recall 값들:")
for threshold, precision, recall in zip(np.arange(0, 1.1, 0.1), precisions, recalls):
    print(f"Threshold {threshold:.1f}: Precision = {precision:.4f}, Recall = {recall:.4f}")