import json
import numpy as np

def calculate_iou(box1, box2):
    """
    두 bbox의 IoU(Intersection over Union)를 계산합니다.
    box1, box2: [x1, y1, x2, y2] 형식의 bbox 좌표
    """
    # box1과 box2의 교집합 영역 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 교집합 영역의 넓이 계산
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 각 box의 넓이 계산
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 합집합 영역의 넓이 계산
    union = box1_area + box2_area - intersection

    # IoU 계산
    iou = intersection / union if union > 0 else 0
    return iou

################################## gt
with open("/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/label/000001.json", "r") as f:
    gt_data = json.load(f)
print(f"GT bbox 갯수: {len(gt_data['annotations'])}")
gt_boxes = []
for ann in gt_data['annotations']:
    x, y, w, h = ann['bbox']
    gt_box = [x, y, x + w, y + h]  # x,y,w,h -> x1,y1,x2,y2로 변환
    gt_boxes.append(gt_box)
print(f"gt_boxes: {gt_boxes}")

################################## pred
with open("/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/output/easyocr/annotated_책표지_총류_000001.json", "r") as f:
    pred_data = json.load(f)
print(f"EasyOCR bbox 갯수: {len(pred_data)}")
pred_boxes = [item[0] for item in pred_data]  # pred bbox만 추출
print(f"pred_boxes: {pred_boxes}")

################################## IoU 계산
# pred_boxes를 [x1, y1, x2, y2] 형식으로 변환
converted_pred_boxes = []
for box in pred_boxes:
    x_coords = [coord[0] for coord in box]
    y_coords = [coord[1] for coord in box]
    converted_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    converted_pred_boxes.append(converted_box)

# 각 GT bbox와 예측 bbox 쌍에 대해 IoU 계산
print("\nIoU 계산 결과:")
for i, gt_box in enumerate(gt_boxes):
    print(f"\nGround Truth #{i+1}:")
    for j, pred_box in enumerate(converted_pred_boxes):
        iou = calculate_iou(gt_box, pred_box)
        print(f"- Prediction #{j+1}: IoU = {iou:.4f}")

# 각 GT bbox에 대해 가장 높은 IoU를 가진 예측 bbox 찾기
print("\n최고 IoU 매칭:")
for i, gt_box in enumerate(gt_boxes):
    ious = [calculate_iou(gt_box, pred_box) for pred_box in converted_pred_boxes]
    max_iou = max(ious)
    max_iou_idx = ious.index(max_iou)
    print(f"GT #{i+1} -> Pred #{max_iou_idx+1}: IoU = {max_iou:.4f}")