# my_yolo_tiny/sorc/object_detector.py
print("--- 스크립트 실행 시작 ---")
import inspect 
import logging # 로깅을 위해 필요
import sys     # 표준 출력 스트림 사용을 위해 필요
from pathlib import Path # 경로 관리를 위해 필요
import torch # torch 모듈을 임포트합니다.
import _bz2
from datetime import datetime

import cv2 # OpenCV 라이브러리
from ultralytics import YOLO # ultralytics의 YOLO 모델

# shared_utils 패키지에서 configger 클래스 가져오기
# shared_utils 프로젝트의 src/utility/configger.py에 configger 클래스가 있다고 가정
# object_detector.py 파일 내 임포트 구문
# 사용자 정의 유틸리티 모듈 임포트
try:
    from my_utils.config_utils.SimpleLogger import logger
    from my_utils.config_utils.configger import configger, get_log_cfg_argument
    from my_utils.photo_utils.object_utils import compute_sha256, load_json, save_object_json_with_polygon, save_cropped_face_image
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

except ImportError as e:
    print(f"my_utils 패키지(../my_utils) 내의 src/my_utils/photo_utils.configger 모듈에서 configger를 임포트 중 오류 발생: {e}")
    sys.exit(1)

def get_dataset_dir(cfg:configger):
    dataset_dir = []
    key_list = [
      "datasets_dir",
      "raw_image_dir",
      "raw_jsons_dir",
      "detected_objects_dir",
      "undetect_objects_dir",
      "detected_face_images_dir",
      "detected_face_json_dir",
      "detected_list_path",
      "undetect_list_path",
      "failed_list_path"
    ]
    try:
        datasets_key = "project.paths.datasets"

        for key in key_list:
            key_list_str = f"{datasets_key}.{key}"
            value = cfg.get_path(key_list_str, ensure_exists=True)
            if value:
                dataset_dir.append(value)
                logger.debug(f"value: {value}")
            else:
                logger.warning(f"{key_list_str} 키를 설정에서 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"datasets_key 설정 값 가져오기 중 오류 발생: {e}")
        sys.exit(1)
    return dataset_dir

# --- 메인 실행 함수 ---
def run_object_detection(cfg: configger):
    # 2. 설정 값 가져오기 (기존 코드와 동일)
    dataset_dir = []
    try:
        dataset_dir = get_dataset_dir(cfg)
        logger.debug(f"dataset_dir: {dataset_dir}")
        sys.exit(1)

        ## dataset설정 값
        datasets_key = "project.paths.datasets"
        datasets_dir = cfg.get_path("project.paths.datasets.datasets_dir", ensure_exists=True) # dataset.datasets_dir 가정
        if datasets_dir:
            logger.debug(f"데이터셋 디렉토리 경로: {datasets_dir}")
        else:
            logger.warning(f"datasets_dir: {datasets_dir} 키를 설정에서 찾을 수 없습니다.")

        input_image_dir = cfg.get_path("project.paths.datasets.raw_image_dir", ensure_exists = True)
        if not input_image_dir or not input_image_dir.is_dir():
            logger.error(f"설정된 input_image_dir '{input_image_dir}'이 유효한 디렉토리가 아닙니다.")
            sys.exit(1)
        logger.debug(f"설정된 input_image_dir: {input_image_dir}")

        output_json_dir = cfg.get_path("project.paths.datasets.raw_jsons_dir", ensure_exists = True)
        if not output_json_dir or not output_json_dir.is_dir():
            logger.error(f"설정된 output_json_dir '{output_json_dir}'이 유효한 디렉토리가 아닙니다.")
        logger.debug(f"output_json_dir: {output_json_dir}")

        ## output 설정갑
        detected_objects_dir = cfg.get_path("project.paths.datasets.detected_objects_dir", ensure_exists = True)
        if not detected_objects_dir or not detected_objects_dir.is_dir():
            logger.error(f"설정된 detected_objects_dir '{detected_objects_dir}'이 유효한 디렉토리가 아닙니다.")
        logger.debug(f"detected_objects_dir: {detected_objects_dir}")

        undetect_objects_dir = cfg.get_path("project.paths.datasets.undetect_objects_dir", ensure_exists = True)
        if not undetect_objects_dir or not detected_objects_dir.is_dir():
            logger.error(f"설정된 undetect_objects_dir '{undetect_objects_dir}'이 유효한 디렉토리가 아닙니다.")
        logger.debug(f"undetect_objects_dir: {undetect_objects_dir}")

         # 지원되는 이미지 확장자 목록 가져오기
        # ... (모델의 일반 설정 값 가져오는 코드) ...
        supported_extensions = cfg.get_value('models.supported_image_extensions', ensure_exists = True)
        if not supported_extensions:
             logger.warning("설정에 supported_image_extensions이 정의되지 않았습니다. 모든 파일을 시도합니다.")
        logger.debug(f"지원되는 이미지 확장자: {supported_extensions}")

        model_selection = int(cfg.get_value('models.model_selection', ensure_exists = True))
        if not model_selection:
             logger.warning("설정에 model_selection이 정의되지 않았습니다. 모든 파일을 시도합니다.")
        logger.debug(f"model_selection: {model_selection}")

        # ... (YOLO 모델 설정 값 가져오는 코드) ...
        model_name = cfg.get_value('models.object_yolo_tiny_model.general_detection_model.model_name', ensure_exists = True)
        logger.debug(f"model_name: {model_name}")

        model_weights_path = cfg.get_path('models.object_yolo_tiny_model.general_detection_model.model_weights_path', ensure_exists = True)
        logger.debug(f"model_weights_path: {model_weights_path}")

        confidence_threshold = float(cfg.get_value('models.object_yolo_tiny_model.general_detection_model.confidence_threshold', ensure_exists = True))
        logger.debug(f"confidence_threshold: {confidence_threshold}")

        iou_threshold = float(cfg.get_value('models.object_yolo_tiny_model.general_detection_model.iou_threshold', ensure_exists = True))
        logger.debug(f"iou_threshold: {iou_threshold}")

        classes_to_detect = cfg.get_value('models.object_yolo_tiny_model.general_detection_model.classes_to_detect', ensure_exists = True)
        logger.debug(f"classes_to_detect: {classes_to_detect}")

        use_cpu = cfg.get_value('models.object_yolo_tiny_model.face_detection_model.use_cpu', ensure_exists = True)
        logger.debug(f"use_cpu: {use_cpu}")

        imgsz = int(cfg.get_value('models.object_yolo_tiny_model.face_detection_model.imgsz', ensure_exists = True))
        logger.debug(f"imgsz: {imgsz}")
        # ... (다른 설정 값 로깅) ...imgsz

    except Exception as e:
        logger.error(f"설정 값 가져오기 중 오류 발생: {e}")
        sys.exit(1)


    # 3. YOLO 모델 로딩 (기존 코드와 동일)
    try:
        # 3-1). 사용할 장치를 결정합니다.
        #    설정에서 use_cpu가 True로 명시되어 있으면 CPU를 우선 사용합니다.
        #    그렇지 않고 CUDA 사용이 가능하면 GPU를 사용하고,
        #    둘 다 아니면 CPU를 사용합니다.
        if use_cpu:
            # Case 1: 설정에서 CPU 사용이 명시된 경우
            selected_device = 'cpu'
            logger.debug("설정에서 CPU 사용이 명시되어 CPU를 사용합니다.")
        else:
            # Case 2: 설정에서 CPU 사용이 명시되지 않은 경우
            if torch.cuda.is_available():
                # Case 2a: CUDA 지원 GPU가 감지된 경우
                selected_device = 'cuda'
                logger.debug("CUDA 지원 GPU가 감지되었습니다. GPU를 사용합니다.")
            else:
                # Case 2b: CUDA 지원 GPU를 찾을 수 없는 경우
                selected_device = 'cpu'
                logger.warning("CUDA 지원 GPU를 찾을 수 없습니다.")
                logger.warning("torch.cuda.is_available(): %s", torch.cuda.is_available())
                # 필요하다면 device_count, CUDA_VISIBLE_DEVICES 등 추가 정보 로깅
                # logger.warning("torch.cuda.device_count(): %s", torch.cuda.device_count())
                # logger.warning("os.environ['CUDA_VISIBLE_DEVICES']: %s", os.environ.get('CUDA_VISIBLE_DEVICES'))
                logger.warning("GPU 사용이 불가능하므로 CPU를 대신 사용합니다.")

        # 3-2). 결정된 장치(selected_device)를 사용하여 YOLO 모델을 로딩합니다.
        logger.debug(f"로컬 모델 파일 로딩: {model_weights_path} (장치: {selected_device})")
        if model_weights_path and model_weights_path.exists():
            # 수정: YOLO() 생성자에서 device 인자를 제거
            model = YOLO(str(model_weights_path))
        elif model_name:
            logger.debug(f"모델 이름으로 로딩 (자동 다운로드): {model_name}")
            # 수정: YOLO() 생성자에서 device 인자를 제거
            model = YOLO(model_name)
        else:
            logger.error("설정에 유효한 'model_name' 또는 'model_weights_path'가 지정되지 않았습니다.")
            sys.exit(1)

        logger.info("YOLO 모델 로딩 성공.")

    except Exception as e:
        logger.error(f"YOLO 모델 로딩 중 오류 발생: {e}")
        sys.exit(1)


    # 4. 이미지 파일 목록 가져오기 및 순회 처리
    image_files = []
    try:
        # input_image_dir 디렉토리 내의 모든 파일에 대해 필터링
        logger.info(f"'{input_image_dir}' 디렉토리에서 이미지 파일 목록 생성 중...")
        for file_path in input_image_dir.iterdir():
            if file_path.is_file(): # 파일인지 확인
                 # 확장자 검사 (설정된 확장자 목록이 있는 경우)
                 if supported_extensions:
                      if file_path.suffix.lower() in [ext.lower() for ext in supported_extensions]:
                           image_files.append(file_path)
                 else:
                      # supported_extensions가 없으면 모든 파일 추가 (주의 필요)
                      image_files.append(file_path)

        image_files.sort() # 파일 이름을 기준으로 정렬하여 일관성 유지
        logger.info(f"총 {len(image_files)}개의 이미지 파일 발견.")

        if not image_files:
             logger.warning("'{input_image_dir}' 디렉토리에 처리할 이미지 파일이 없습니다.")
             # sys.exit(0) # 이미지가 없으면 정상 종료하도록 선택 가능

    except Exception as e:
        logger.error(f"이미지 파일 목록 생성 중 오류 발생: {e}")
        sys.exit(1)


    # 5. 이미지 파일 순회 및 객체 검출 루프 (기존 카메라 루프 대체)

    # --- 통계 변수 초기화 ---
    total_files_read = 0 # 읽기를 시도한 총 파일 수
    files_read_success = 0 # 파일 읽기 성공 수
    files_skipped_read_error = 0 # 파일 읽기 오류로 건너뛴 파일 수
    files_processed_successfully = 0 # 오류 없이 처리 완료된 파일 수
    files_with_objects_detected = 0 # 객체가 1개 이상 검출된 파일 수
    files_with_no_objects_detected = 0 # 객체가 검출되지 않은 파일 수
    total_objects_detected = 0 # 검출된 총 오브젝트 수
    files_processed_with_error = 0 # 처리 중 예외 발생으로 건너뛴 파일 수
    # --- 통계 변수 초기화 끝 ---

    # idx 변수가 어디서 정의되었는지 명확하지 않아 일단 주석 처리합니다.
    # 필요에 따라 idx 변수의 사용 방식을 조정해주세요.
    # if idx:
    #     break
    # else:
    #     logger.info(f"{idx}번째 이미지 파일 처리 완료.")

    for index, image_path in enumerate(image_files):
        total_files_read += 1 # 총 파일 수 증가
        logger.info(f"[{index+1}/{len(image_files)}] 이미지 파일 처리 시작: {image_path}")

        try:
            # 이미지 파일 읽기 (OpenCV 사용)
            frame = cv2.imread(str(image_path)) # Path 객체를 문자열로 변환하여 전달

            if frame is None:
                 logger.warning(f"이미지 파일 '{image_path}'을 읽을 수 없습니다. 건너뜁니다.")
                 files_skipped_read_error += 1 # 읽기 오류 파일 수 증가
                 continue # 파일 읽기 실패 시 다음 파일로 이동

            files_read_success += 1 # 파일 읽기 성공 수 증가

            results = model(
                frame,                       # ① 입력 데이터
                conf=confidence_threshold,   # ② 신뢰도 임계값 (Confidence Threshold)
                iou=iou_threshold,           # ③ IoU 임계값 (IoU Threshold) for NMS
                classes=classes_to_detect,   # ④ 검출할 클래스 목록 (Classes to Detect)
                imgsz=imgsz,                  # ⑤ 입력 이미지 크기 (Image Size)
                device=selected_device,               # ⑥ 실행 장치 (Device)
                verbose=False                # ⑦ 상세 출력 여부 (Verbosity)
            )
            """
            results[0] 객체 속성들.

            boxes: 탐지된 각 객체의 바운딩 박스 정보와 신뢰도(confidence), 클래스 정보
                results[0].boxes는 ultralytics.engine.results.Boxes 타입의 객체 리스트입니다.
                results[0].boxes.xyxy: 각 객체의 바운딩 박스 좌표 (x1, y1, x2, y2) - 이미지 좌상단 기준.
                results[0].boxes.xywh: 각 객체의 바운딩 박스 좌표 (x 중심, y 중심, 너비, 높이).-여기서 중심은 
                results[0].boxes.conf: 각 객체의 탐지 신뢰도 점수 (0.0 ~ 1.0).
                results[0].boxes.cls: 각 객체의 클래스 인덱스.
                len(results[0].boxes): 현재 이미지에서 탐지된 객체의 총 개수.
            masks (시멘틱 분할 모델의 경우): 객체 분할(Segmentation) 모델
                탐지된 각 객체의 픽셀 단위 마스크 정보
                results[0].masks는 ultralytics.engine.results.Masks 타입 객체입니다.
                results[0].masks.xy: 각 객체의 마스크 윤곽선 좌표.
                results[0].masks.data: 각 객체의 마스크 이미지 데이터 (NumPy 배열 형태).
            keypoints (포즈 추정 모델의 경우): 포즈 추정(Pose Estimation) 모델을 사용하는 경우, 
                탐지된 각 사람의 관절(키포인트) 좌표 및 신뢰도 정보
                results[0].keypoints는 ultralytics.engine.results.Keypoints 타입 객체입니다.
            probs (분류 모델의 경우): 이미지 분류(Classification) 모델을 사용하는 경우, 각 클래스에 대한 확률 분포 정보가 담겨 있습니다.
            orig_img: 원본 이미지 데이터 (NumPy 배열 형태).
            orig_shape: 원본 이미지의 형태 (높이, 너비).
            speed: 모델 추론에 소요된 시간 정보 (전처리, 추론, 후처리).
            names: 클래스 인덱스와 실제 클래스 이름(문자열)을 매핑한 딕셔너리. results[0].names를 통해 접근할 수 있으며, 
                예를 들어 results[0].names[class_index] 형태로 클래스 이름을 얻을 수 있습니다.
                모델이 학습된 클래스의 이름과 해당 클래스의 인덱스를 매핑한 딕셔너리입니다. 
                예를 들어 results[0].names는 {0: 'person', 1: 'bicycle', 2: 'car', ...} 와 같은 형태를 가집니다. 
                여기서 키(key)는 클래스 인덱스이고, 값(value)은 해당 클래스의 이름입니다.
            """
            # 결과 처리 및 시각화 (ultralytics plot 기능 사용)
            # results[0].plot()은 시각화된 이미지를 반환합니다.
            # 예시 코드 (사용하시는 라이브러리의 plot 함수 인자명에 따라 수정이 필요할 수 있습니다.)
            # 노란색 (BGR: Blue=0, Green=255, Red=255)
            annotated_color = (0, 255, 255)

            annotated_frame = results[0].plot() #(colors=annotated_color)
            logger.info("이미지에 객체 검출 결과 시각화 완료.") # 로그 메시지 수정

            # 결과 파일 저장
            # 저장 파일 이름 결정 (원본 파일 이름에 접미사 추가 등)
            save_file_name = f"annotated_{image_path.name}"


            # 검출된 객체 수 확인
            detected_objects_in_frame = len(results[0].boxes)
            total_objects_detected += detected_objects_in_frame # 총 오브젝트 수 증가
            files_processed_successfully += 1 # 오류 없이 처리 완료 파일 수 증가


            if detected_objects_in_frame > 0:
                files_with_objects_detected += 1 # 객체 검출 성공 파일 수 증가
                image_hash_value = compute_sha256(frame)

                processed_ditec_obj_data = []
#                if results[0].boxes is not None: # boxes가 있는지 확인
                # If detected_objects_in_frame > 0, results[0].boxes is guaranteed to be non-empty.
                # So, `if results[0].boxes is not None:` is redundant here.
                if results[0].boxes is not None: # This check is technically redundant if detected_objects_in_frame > 0
                    for box in results[0].boxes:
                        # box 객체에서 필요한 정보 추출
                        xyxy_coords = box.xyxy.tolist()[0] # xyxy 좌표 (리스트 형태로 변환)
                        xywh_coords = box.xywh.tolist()[0] # xywh 좌표 (리스트 형태로 변환)
                        confidence = float(box.conf)      # 신뢰도 (float으로 변환)
                        class_id = int(box.cls)           # 클래스 인덱스 (int로 변환)
                        class_name = results[0].names[class_id] # 클래스 이름

                        # 필요한 정보들을 딕셔너리로 구성
                        ditec_obj_info = {
                            "box_xyxy": xyxy_coords,
                            "box_xywh": xywh_coords,
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": class_name
                            # 필요에 따라 다른 정보(예: 마스크, 키포인트 등) 추가
                        }
                        processed_ditec_obj_data.append(ditec_obj_info)

                    output_json_dir.mkdir(parents=True, exist_ok=True) # 디렉토리 미리 생성
                    json_file_name = f"{image_path.stem}.json" # 원본 이미지 이름에서 확장자 제외
                    json_save_path = output_json_dir / json_file_name

                    save_object_json_with_polygon(
                        image_path  = image_path, 
                        image_hash  = image_hash_value,
                        ditec_obj   = processed_ditec_obj_data,
                        json_path   = json_save_path
                    )
                #     logger.info(f"'{image_path.name}'에서 {detected_objects_in_frame}개의 객체 검출됨.")

                #     # 결과 이미지 파일 저장
                #     save_path = detected_objects_dir / save_file_name
                #     logger.info(f"검출된 이미지정보: {save_path}저장됨.")

                # else:
                #     files_with_no_objects_detected += 1 # 객체 검출되지 않은 파일 수 증가
                #     logger.info(f"'{image_path.name}'에서 객체 검출되지 않음.") 

                #     # 결과 이미지 파일 저장
                #     save_path = undetect_objects_dir / save_file_name
                #     logger.info(f"검출못한 이미지정보: {save_path}저장됨.")

                # # cv2.imwrite 함수로 결과 이미지 저장
                # success = cv2.imwrite(str(save_path), annotated_frame)
                # if success:
                #     logger.info(f"결과 이미지 저장 완료: {save_path}")
                # else:
                #     logger.error(f"결과 이미지 저장 실패: {save_path}")
                    logger.info(f"'{image_path.name}'에서 {detected_objects_in_frame}개의 객체 검출됨. JSON 저장: {json_save_path}")
                save_path = detected_objects_dir / save_file_name
                logger.info(f"검출된 객체가 있는 이미지 저장 경로: {save_path}")
            else:
                files_with_no_objects_detected += 1 # 객체 검출되지 않은 파일 수 증가
                # logger.warning("결과 이미지 저장 디렉토리가 설정되지 않았거나 유효하지 않습니다. 이미지 저장을 건너뜁니다.")
                logger.info(f"'{image_path.name}'에서 객체 검출되지 않음.")
                save_path = undetect_objects_dir / save_file_name
                logger.info(f"검출된 객체가 없는 이미지 저장 경로: {save_path}")

            # cv2.imwrite 함수로 결과 이미지 저장
            # Common image saving logic
            Path(save_path).parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            success = cv2.imwrite(str(save_path), annotated_frame)
            if success:
                logger.info(f"결과 이미지 저장 완료: {save_path}")
            else:
                logger.error(f"결과 이미지 저장 실패: {save_path}")

        except Exception as e:
            logger.error(f"이미지 파일 '{image_path}' 처리 중 오류 발생: {e}")
            files_processed_with_error += 1 # 처리 오류 파일 수 증가
            # 오류 발생 시 다음 파일로 이동하거나 종료할 수 있습니다.
            continue # 다음 파일로 이동

        # idx 변수 처리가 필요하다면 여기에 적절히 배치합니다.
        # 현재 코드에서는 idx 변수의 역할이 불분명합니다.
        # if idx: # 이 조건이 의미하는 바에 따라 코드를 유지하거나 수정해야 합니다.
        #     break # 루프를 중간에 종료하는 로직

        logger.info(f"[{index+1}/{len(image_files)}] 이미지 파일 처리 완료.")


    # 9. 모든 이미지 처리 완료 또는 중단 후 자원 해제
    # cv2.destroyAllWindows() # GUI 제거로 필요 없음

    # --- 통계 결과 출력 ---
    logger.info("--- 이미지 파일 처리 통계 ---")
    logger.info(f"총 읽기 시도 파일 수: {total_files_read}")
    logger.info(f"읽기 성공 파일 수: {files_read_success}")
    logger.info(f"읽기 오류 건너뛴 파일 수: {files_skipped_read_error}")
    logger.info(f"오류 없이 처리 완료 파일 수: {files_processed_successfully}")
    logger.info(f"  > 객체 1개 이상 검출된 파일 수: {files_with_objects_detected}")
    logger.info(f"  > 객체 검출되지 않은 파일 수: {files_with_no_objects_detected}")
    logger.info(f"처리 중 오류 발생 파일 수: {files_processed_with_error}")
    logger.info(f"총 검출된 오브젝트 수: {total_objects_detected}")
    logger.info("------------------------")
    # --- 통계 결과 출력 끝 ---


# 스크립트가 직접 실행될 때 run_object_detection 함수 호출
if __name__ == "__main__":
    """
    설정 파일을 로드하고 YOLO 객체 검출을 수행하는 함수에게 일을 시킴
    """
    # 0. 애플리케이션 아귀먼트 있으면 갖오기
    parsed_args = get_log_cfg_argument()

    script_name = Path(__file__).stem # Define script_name early for logging
    try:
        # 1. 애플리케이션 로거 설정
        date_str = datetime.now().strftime("%y%m%d")
        log_file_name = f"{script_name}_{date_str}.log"
        full_log_path = Path(parsed_args.log_dir) / log_file_name
        logger.setup(
            logger_path=str(full_log_path),
            min_level=parsed_args.log_level,
            include_function_name=True,
            pretty_print=True
        )
        logger.info(f"애플리케이션({script_name}) 시작")
        logger.info(f"명령줄 인자로 결정된 경로: {vars(parsed_args)}")
    except Exception as e:
        print(f"치명적 오류: 로거 설정 중 오류 발생 - {e}", file=sys.stderr)
        sys.exit(1)

    # 2. configger 인스턴스 생성
    # configger는 이제 위에서 설정된 공유 logger를 사용합니다.
    # root_dir과 config_path는 실제 프로젝트에 맞게 설정해야 합니다.
    # 설정 파일이 실제로 존재하는지 확인

    logger.info(f"Configger 초기화 시도: root_dir='{parsed_args.root_dir}', config_path='{parsed_args.config_path}'")

    try:
        cfg_object = configger(root_dir=parsed_args.root_dir, config_path=parsed_args.config_path)

        run_object_detection(cfg_object)
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}", exc_info=True)
    finally:
        # 애플리케이션 종료 시 로거 정리 (특히 비동기 사용 시 중요)
        logger.info("my_yolo_tiny 애플리케이션 종료")
        logger.shutdown()
        exit(0)
