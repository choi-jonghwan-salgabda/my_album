import cv2
import numpy as np
import os

def detect_objects_dnn(net: cv2.dnn.Net, image: np.ndarray, class_names: list, conf_threshold: float = 0.5, nms_threshold: float = 0.4) -> np.ndarray:
    """
    OpenCV DNN 모델을 사용하여 이미지에서 객체를 검출하고 결과를 이미지에 그립니다.

    Args:
        net: 미리 로드된 OpenCV DNN 네트워크 객체.
        image: 객체 검출을 수행할 입력 이미지 (numpy 배열).
        class_names: 클래스 이름 목록 (인덱스가 클래스 ID에 해당).
        conf_threshold: 검출 결과의 신뢰도 임계값. 이 값보다 낮은 결과는 무시됩니다.
        nms_threshold: Non-Maximum Suppression (NMS) 임계값. 중복된 바운딩 박스 제거에 사용됩니다.

    Returns:
        검출 결과가 그려진 이미지 (numpy 배열).
    """
    # 이미지의 높이와 너비를 가져옵니다.
    height, width = image.shape[:2]

    # 이미지에서 Blob을 생성합니다.
    # 사용하는 모델의 요구사항에 맞게 이 부분을 조정해야 합니다.
    # 예시: YOLO 모델의 경우
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    # 네트워크의 입력으로 Blob을 설정합니다.
    net.setInput(blob)

    # 네트워크의 출력 레이어 이름을 가져옵니다. (모델에 따라 다를 수 있습니다)
    # YOLO의 경우 getUnconnectedOutLayersNames()를 사용합니다.
    output_layers_names = net.getUnconnectedOutLayersNames()

    # 순방향 추론(forward pass)을 수행하여 결과를 얻습니다.
    outs = net.forward(output_layers_names)

    # 검출된 객체 정보를 저장할 리스트를 초기화합니다.
    detections = []
    class_ids = []
    confidences = []
    boxes = []

    # 검출 결과를 분석합니다.
    # 결과 형태는 모델(YOLO, SSD 등)에 따라 다릅니다. 여기서는 YOLO 스타일 출력을 가정합니다.
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 설정된 신뢰도 임계값보다 높은 검출 결과만 고려합니다.
            if confidence > conf_threshold:
                # 바운딩 박스 좌표를 계산합니다. (센터 x, 센터 y, 너비, 높이 형태)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 좌상단 좌표로 변환합니다.
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression (NMS)를 적용하여 중복된 박스를 제거합니다. [[2]](https://stackoverflow.com/questions/75349869/opencv-dnn-disable-detecting-coco-dataset-for-some-categories-and-enable-for-oth)
    # cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    # score_threshold는 사실상 conf_threshold와 유사하게 사용됩니다.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # 결과를 이미지에 그립니다.
    output_image = image.copy()
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            confidence = confidences[i]
            class_id = class_ids[i]

            # 클래스 이름과 신뢰도 라벨 생성
            # class_id가 class_names 리스트의 범위를 벗어날 경우를 대비합니다.
            if class_id < len(class_names):
                 label = f"{class_names[class_id]}: {confidence:.2f}"
            else:
                 label = f"Class ID {class_id}: {confidence:.2f}" # 알 수 없는 클래스 ID

            # 바운딩 박스 색상 결정 (클래스 ID에 따라 다르게 할 수도 있습니다)
            color = (0, 255, 0) # 초록색 (B, G, R)

            # 바운딩 박스 그리기
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)

            # 클래스 이름 및 신뢰도 텍스트 추가
            # 텍스트 위치 조정
            text_y = y - 10 if y - 10 > 10 else y + 10
            cv2.putText(output_image, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output_image

# --- 메인 실행 부분 ---
if __name__ == '__main__':
    # --- 설정 변수 ---
    # TODO: 실제 파일 경로로 변경해주세요.
    model_cfg_path = "path/to/your/model.cfg"       # 예: "yolov3.cfg"
    model_weights_path = "path/to/your/model.weights" # 예: "yolov3.weights"
    class_names_path = "/home/owner/SambaData/Backup/FastCamp/Myproject/my_album/config/coco.names"  # 예: "coco.names" (이전 답변의 내용)
    input_image_path = "path/to/your/image.jpg"     # 예: "test_image.jpg"
    output_image_path = "detection_result.jpg"      # 결과를 저장할 파일 이름

    # 검출 신뢰도 및 NMS 임계값
    confidence_threshold = 0.5
    nms_threshold = 0.4

    # --- 1. 클래스 이름 로드 ---
    class_names = []
    if not os.path.exists(class_names_path):
        print(f"오류: 클래스 이름 파일 '{class_names_path}'을(를) 찾을 수 없습니다.")
        # 파일이 없으면 임의의 클래스 ID를 사용하도록 빈 리스트 유지
    else:
        try:
            with open(class_names_path, "r", encoding="utf-8") as f:
                class_names = [line.strip() for line in f.readlines()]
            print(f"클래스 이름 {len(class_names)}개 로드 성공.")
        except Exception as e:
            print(f"오류: 클래스 이름 파일 '{class_names_path}' 로드 중 오류 발생: {e}")
            class_names = [] # 오류 발생 시 빈 리스트로 초기화

    # --- 2. DNN 네트워크 로드 ---
    net = None
    if not os.path.exists(model_cfg_path) or not os.path.exists(model_weights_path):
        print(f"오류: 모델 파일('{model_cfg_path}', '{model_weights_path}') 중 하나 이상을 찾을 수 없습니다.")
        print("모델 설정 파일과 가중치 파일 경로를 확인해주세요.")
    else:
        try:
            # cv2.dnn.readNet 함수를 사용하여 네트워크를 로드합니다. [[3]](https://junstar92.tistory.com/411)
            net = cv2.dnn.readNet(model_weights_path, model_cfg_path)

            # 모델 종류에 따라 파서(Parser) 백엔드를 명시적으로 지정해야 할 수도 있습니다.
            # 예: Caffe 모델의 경우 net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            # TensorFlow 모델의 경우 net = cv2.dnn.readNetFromTensorflow(pb) 등

            # 사용할 백엔드와 타겟을 설정할 수 있습니다. (예: CPU, GPU)
            # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # 기본값
            # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # 기본값

            print("DNN 네트워크 로드 성공.")

        except Exception as e:
            print(f"오류: DNN 네트워크 로드 실패: {e}")
            print("모델 설정 파일 및 가중치 파일이 유효한지 확인해주세요.")
            net = None # 로드 실패 시 net 변수를 None으로 설정

    # --- 3. 이미지 로드 및 객체 검출 실행 ---
    if net is not None: # 네트워크가 성공적으로 로드된 경우에만 진행
        if not os.path.exists(input_image_path):
            print(f"오류: 입력 이미지 파일 '{input_image_path}'을(를) 찾을 수 없습니다.")
        else:
            try:
                # cv2.imread를 사용하여 이미지 파일을 로드합니다.
                image = cv2.imread(input_image_path)

                if image is None:
                    print(f"오류: 이미지 파일 '{input_image_path}'을(를) 읽을 수 없습니다.")
                else:
                    print(f"이미지 '{input_image_path}' 로드 성공. 객체 검출 시작...")

                    # 객체 검출 함수 호출
                    detected_image = detect_objects_dnn(
                        net,
                        image,
                        class_names,
                        conf_threshold=confidence_threshold,
                        nms_threshold=nms_threshold
                    )

                    # --- 4. 결과 저장 또는 표시 ---
                    try:
                        # 결과를 파일로 저장
                        cv2.imwrite(output_image_path, detected_image)
                        print(f"검출 결과 이미지가 '{output_image_path}'로 저장되었습니다.")

                        # 선택적으로 결과를 화면에 표시할 수 있습니다. (GUI 환경에서만 작동)
                        # cv2.imshow("Object Detection Result", detected_image)
                        # print("\n결과 이미지를 화면에 표시했습니다. 아무 키나 누르면 종료됩니다.")
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                    except Exception as e:
                        print(f"오류: 결과 이미지 저장 또는 표시 중 오류 발생: {e}")

            except Exception as e:
                print(f"오류: 이미지 로드 또는 객체 검출 처리 중 오류 발생: {e}")

    else:
        print("네트워크 로드에 실패하여 객체 검출을 수행할 수 없습니다.")

