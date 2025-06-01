import json
import os
import logging
from typing import List, Dict, Any, Union
import cv2
# from mediapipe.tasks import python # MediaPipe 관련 임포트 주석 처리 또는 제거
# from mediapipe.tasks.python import vision # MediaPipe 관련 임포트 주석 처리 또는 제거
from pathlib import Path
from ultralytics import YOLO # Ultralytics YOLO 라이브러리 임포트

# shared_utils 패키지에서 configger 클래스 가져오기
# shared_utils 프로젝트의 src/utility/configger.py에 configger 클래스가 있다고 가정
# object_detector.py 파일 내 임포트 구문
try:
    # my_utils 패키지(../my_utils) 내의 src/my_utils/photo_utils.configger 모듈에서 configger를 임포트
    from my_utils.photo_utils.configger import configger, setup_logging # <-- 이렇게 수정
    from my_utils.photo_utils.object_utils import save_object_json_with_polygon, compute_sha256 # <-- 이렇게 수정
    # configger 클래스 내부에 로거 객체가 있으므로 별도 로거 가져오기 필요 없을 수 있으나,
    # 현재 모듈의 로거도 따로 사용하는 것이 로깅 메시지 추적에 더 용이합니다.
    # ... (나머지 코드) ...
except ImportError as e:
    print(f"my_utils 패키지(../my_utils) 내의 src/my_utils/photo_utils.configger 모듈에서 configger를 임포트 중 오류 발생: {e}")
    sys.exit(1)

# --- 로깅 기본 설정 ---
# # 애플리케이션 시작 시 한 번만 호출
# # 이 설정은 shared_utils 내 configger 로거에도 적용됩니다.
# # 로깅 기본 설정
# logging.basicConfig(
#     level=logging.DEBUG,  # 어느 레벨부터 로그를 출력할지 설정 (INFO, DEBUG, WARNING, ERROR, CRITICAL)
#     # 주의: format 문자열에 오타나 불필요한 문자가 있는 것 같습니다. 수정했습니다.
#     # 원본: format='%(asctime)s - %(name)-10s - %(levelname)-8s - %(funcName)-10mv co   s - %(message)s',
#     format='%(asctime)s - %(levelname)-6s - %(funcName)-10s - %(message)s', # 로그 출력 형식 지정 (오타 수정 및 형식 개선)
#     # stream=sys.stdout # 로그를 어디로 출력할지 설정 (기본값은 sys.stderr)
#     # filename='app.log' # 로그를 파일에 저장하려면 이 줄 사용 (stream과 함께 사용하지 않음)
# ) # <-- 여기서 닫는 괄호 ')' 다음 줄로 이동합니다.

# 현재 모듈의 로거 객체 가져오기
logger = logging.getLogger(__name__)
  

def detect_faces_in_crop_yolo(image_crop: cv2.Mat, obj_bbox_origin: List[int], yolo_face_model, confidence_threshold: float) -> List[Dict]:
    """
    잘라낸 이미지 영역에서 YOLO 모델을 사용하여 얼굴을 검출하고, 원본 이미지 좌표 기준으로 얼굴 bbox를 반환합니다.

    Args:
        image_crop: 얼굴 검출을 수행할 잘라낸 이미지 (OpenCV BGR 형식).
        obj_bbox_origin: 잘라낸 영역의 원본 이미지 내 좌표 [x1, y1, x2, y2].
        yolo_face_model: 로드된 YOLO 얼굴 검출 모델 인스턴스 (Ultralytics).
        confidence_threshold: 얼굴 검출 최소 신뢰도 임계값.

    Returns:
        원본 이미지 좌표 기준의 얼굴 bbox 및 점수 리스트. 예: [{"bbox": [fx1, fy1, fx2, fy2], "score": s}, ...]
    """
    if image_crop is None or image_crop.size == 0:
        logger.warning("얼굴 검출 입력 이미지가 비어 있습니다.")
        return []

    # YOLO 모델 실행
    # source=image_crop 로 직접 이미지를 전달합니다.
    # conf=confidence_threshold 로 최소 신뢰도 설정
    # verbose=False 로 모델 실행 시 상세 로그 억제
    results = yolo_face_model(image_crop, conf=confidence_threshold, verbose=False)

    faces_in_original_coords = []
    # 결과 파싱
    for r in results:
        # r.boxes 객체는 검출된 객체들의 bbox, 클래스, 신뢰도 등을 포함합니다.
        for box in r.boxes:
            # box.xyxy는 [x1, y1, x2, y2] 형식의 bbox 좌표 (잘라낸 이미지 기준)
            # box.conf는 신뢰도 점수
            # box.cls는 클래스 인덱스 (얼굴 모델의 경우 대부분 0 또는 얼굴 클래스 인덱스)

            score = float(box.conf)
            # 이미 confidence_threshold로 필터링되었지만, 다시 확인하거나 로깅에 사용
            # if score < confidence_threshold:
            #     continue # 이미 모델 호출 시 conf= 로 필터링됨

            # 잘라낸 이미지 기준 픽셀 bbox [x1, y1, x2, y2]
            # box.xyxy는 텐서 형태일 수 있으므로 리스트로 변환
            face_crop_bbox_tensor = box.xyxy[0].tolist()
            face_crop_bbox = [int(coord) for coord in face_crop_bbox_tensor] # 정수로 변환

            # 원본 이미지 기준 픽셀 bbox [x1, y1, x2, y2] 계산
            # obj_bbox_origin의 좌상단 좌표를 더해줍니다.
            obj_x1, obj_y1, _, _ = obj_bbox_origin

            face_orig_bbox = [
                obj_x1 + face_crop_bbox[0],
                obj_y1 + face_crop_bbox[1],
                obj_x1 + face_crop_bbox[2],
                obj_y1 + face_crop_bbox[3],
            ]

            # 원본 이미지 경계를 벗어나지 않도록 클리핑 (선택 사항, 필요하다면 추가)
            # img_h, img_w, _ = 원본 이미지 shape 필요... 또는 여기서는 생략
            # face_orig_bbox[0] = max(0, face_orig_bbox[0])
            # ...

            faces_in_original_coords.append({"bbox": face_orig_bbox, "score": score})

    return faces_in_original_coords

if __name__ == "__main__":
    # 1. 설정 파일 로드 (기존 코드와 동일) 및 환경 설정 (config 로딩 및 로깅 설정)
    try:
        config_path = Path(__file__).parent.parent / 'config' / 'my_yolo_tiny.yaml'
        config = setup_logging(config_path)
    except Exception as e:
        logger.exception(f"설정 로딩 중 오류 발생: {e}")
        exit(1)

    logger.info("프로그램 시작")

    json_input_path             = config.get_value("paths.dataset.raw_jsons_dir")
    image_base_dir              = config.get_value("paths.dataset.raw_image_dir")
    output_base_dir             = config.get_value("paths.outputs.outputs_dir")
    output_image_success_dir    = config.get_value("paths.outputs.detected_objects_dir")
    output_image_failure_dir    = config.get_value("paths.outputs.undetect_objects_dir")

    # 출력 디렉토리 생성
    os.makedirs(output_image_success_dir, exist_ok=True)
    os.makedirs(output_image_failure_dir, exist_ok=True)
    logger.info(f"출력 디렉토리 생성 완료: {output_image_success_dir}, {output_image_failure_dir}")

    # 2. JSON 읽기 및 모델 로딩
    json_data = load_json(json_input_path)
    if not json_data:
        logger.error("처리할 JSON 데이터가 없습니다. 프로그램을 종료합니다.")
        exit(1)


    # YOLO 얼굴 검출 모델 로딩
    yolo_model_path = config.get_value("face_detection.model_path")
    min_detection_confidence = config.get_value("face_detection.min_detection_confidence", default=0.5)

    try:
        yolo_face_model = YOLO(yolo_model_path) # YOLO 모델 로딩
        logger.info(f"YOLO Face Detection 모델 로딩 완료: {yolo_model_path} (confidence={min_detection_confidence})")
        # 모델 warmup (처음 추론 시 시간이 걸릴 수 있으므로 작은 더미 이미지로 한 번 실행)
        try:
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            yolo_face_model(dummy_img, verbose=False)
            logger.info("YOLO 모델 warmup 완료.")
        except Exception as e:
             logger.warning(f"YOLO 모델 warmup 중 오류 발생: {e}")

    except Exception as e:
        logger.exception(f"YOLO Face Detection 모델 로딩 중 오류 발생: {e}")
        exit(1)

    # 통계 변수 초기화
    total_images = len(json_data)
    processed_images_count = 0
    images_with_faces_count = 0
    images_without_faces_count = 0
    total_objects_count = 0
    objects_with_faces_count = 0
    objects_without_faces_count = 0
    total_faces_detected = 0

    updated_json_data = [] # 결과를 저장할 새로운 리스트

    # 이미지 처리 루프
    for entry in json_data:
        image_rel_path = entry.get("image_path")
        objects = entry.get("objects", [])

        if not image_rel_path:
            logger.warning(f"image_path가 누락된 항목이 있습니다: {entry}")
            continue

        image_full_path = Path(image_base_dir) / image_rel_path

        # 이미지 읽기
        image = cv2.imread(str(image_full_path))

        if image is None:
            logger.error(f"이미지 파일을 읽을 수 없습니다: {image_full_path}")
            images_without_faces_count += 1 # 이미지 읽기 실패도 얼굴 검출 실패로 간주
            processed_images_count += 1
            updated_json_data.append(entry) # 원본 정보는 유지
            continue

        processed_images_count += 1
        image_has_any_face = False # 이 이미지에서 얼굴이 하나라도 검출되었는지 여부

        # 원본 이미지에 bbox를 그리기 위한 복사본
        image_with_boxes = image.copy()

        # 객체 bbox 만큼 잘라와 얼굴 위치 잡기
        processed_objects = []
        total_objects_count += len(objects) # 이미지 내 모든 객체 수를 전체 통계에 더함

        for obj in objects:
            obj_bbox = obj.get("bbox")
            obj_label = obj.get("label", "unknown") # 원본 객체 레이블

            # 객체 bbox 유효성 검사
            if not obj_bbox or len(obj_bbox) != 4:
                logger.warning(f"잘못된 bbox 형식입니다. 건너뜁니다: {obj_bbox} (image: {image_rel_path})")
                processed_objects.append(obj) # 원본 객체 정보 유지
                continue

            # bbox 좌표가 정수인지 확인 및 변환
            try:
                x1, y1, x2, y2 = map(int, obj_bbox)
            except (ValueError, TypeError):
                 logger.warning(f"bbox 좌표가 숫자가 아닙니다. 건너뜁니다: {obj_bbox} (image: {image_rel_path})")
                 processed_objects.append(obj) # 원본 객체 정보 유지
                 continue

            # 좌표 유효성 검사 (이미지 경계 내에 있는지)
            img_h, img_w, _ = image.shape
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h or x1 >= x2 or y1 >= y2:
                logger.warning(f"bbox 좌표가 이미지 범위를 벗어나거나 잘못되었습니다. 건너뜁니다: {obj_bbox} (image: {image_rel_path}, image_size: {img_w}x{img_h})")
                processed_objects.append(obj) # 원본 객체 정보 유지
                continue

            # 이미지 잘라내기 (bbox 영역)
            image_crop = image[y1:y2, x1:x2]

            # 잘라낸 영역에서 YOLO 모델로 얼굴 검출
            faces_found_in_object = detect_faces_in_crop_yolo(
                image_crop,
                [x1, y1, x2, y2], # 원본 이미지 기준 객체 bbox 좌표 전달
                yolo_face_model,
                min_detection_confidence # 설정된 최소 신뢰도 임계값 전달
            )

            # 얼굴 검출 정보 추가
            if faces_found_in_object:
                obj["faces"] = faces_found_in_object
                image_has_any_face = True
                objects_with_faces_count += 1
                total_faces_detected += len(faces_found_in_object)

                # 검출된 얼굴 bbox를 원본 이미지 복사본에 그리기
                for face_info in faces_found_in_object:
                    fb = face_info["bbox"]
                    # 얼굴 bbox 그리기 (녹색, 두께 2)
                    cv2.rectangle(image_with_boxes, (fb[0], fb[1]), (fb[2], fb[3]), (0, 255, 0), 2)

            else:
                # 얼굴이 검출되지 않은 객체 처리 (옵션: 원본 bbox를 다른 색으로 그릴 수도 있음)
                objects_without_faces_count += 1
                # cv2.rectangle(image_with_boxes, (x1, y1), (y1, y2), (255, 0, 0), 2) # 예: 파란색

            processed_objects.append(obj) # 처리된 객체 정보 (얼굴 정보 추가될 수 있음) 리스트에 추가

        # 업데이트된 객체 리스트를 현재 항목에 할당
        entry["objects"] = processed_objects
        updated_json_data.append(entry) # 업데이트된 항목을 결과 리스트에 추가

        # 원본 사진이미지에 검출된 얼굴 bbox를 그려 넣고 이미지 저장
        image_filename = Path(image_rel_path).name # 파일 이름만 추출
        if image_has_any_face:
            output_image_path = Path(output_image_success_dir) / image_filename
            images_with_faces_count += 1
            logger.info(f"얼굴 검출 성공 이미지 저장: {output_image_path}")
        else:
            output_image_path = Path(output_image_failure_dir) / image_filename
            images_without_faces_count += 1
            logger.info(f"얼굴 검출 실패 이미지 저장: {output_image_path}")

        try:
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            # 이미지 저장 시 파일 경로가 한글을 포함할 경우 문제가 될 수 있으므로,
            # cv2.imencode와 cv2.imwrite를 조합하거나 다른 방식을 사용할 수 있습니다.
            # 여기서는 단순화하여 사용합니다. 실제 환경에서 필요시 수정해주세요.
            cv2.imwrite(str(output_image_path), image_with_boxes)
        except Exception as e:
            logger.error(f"처리된 이미지 저장 중 오류 발생: {output_image_path}, 오류: {e}")

    # 모델 리소스 해제 (YOLO 객체는 명시적으로 해제할 필요는 적으나, 필요시 고려)
    # del yolo_face_model # 필요에 따라 명시적 객체 삭제

    # 업데이트된 JSON 파일 저장
    save_json(updated_json_data, output_json_path)

    # 통계 정보 보여주기
    logger.info("\n--- 처리 결과 통계 ---")
    logger.info(f"전체 JSON 항목 수: {total_images}")
    logger.info(f"처리된 이미지 수 (읽기 성공): {processed_images_count}")
    logger.info(f"   - 얼굴 검출 성공 이미지 (최소 얼굴 1개): {images_with_faces_count}")
    logger.info(f"   - 얼굴 검출 실패 이미지 (얼굴 0개): {images_without_faces_count}")
    logger.info(f"처리된 전체 객체 수: {total_objects_count}")
    logger.info(f"   - 객체 내 얼굴 검출 성공 객체: {objects_with_faces_count}")
    logger.info(f"   - 객체 내 얼굴 검출 실패 객체: {objects_without_faces_count}")
    logger.info(f"전체 검출된 얼굴 수: {total_faces_detected}")
    logger.info("---------------------")

    logger.info("프로그램 완료.")

