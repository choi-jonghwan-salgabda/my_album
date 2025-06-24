# 표준 라이브러리
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# 외부 라이브러리
import numpy as np
import torch

import cv2 # OpenCV 라이브러리
from ultralytics import YOLO # ultralytics의 YOLO 모델

# 사용자 정의 모듈
try:
    from my_utils.config_utils.SimpleLogger import logger, get_argument
    from my_utils.config_utils.configger import configger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    # Basic config for standalone use
    logging.basicConfig(level=logging.INFO)
    logger.warning("Could not import SimpleLogger or configger. Using standard logging.")
    configger = None

class YoloDetector:
    """
    YOLO 모델을 로드하고 객체/얼굴 검출을 수행하는 클래스.
    """
    def __init__(self, cfg_obj: "configger", model_config_key: str):
        """
        YoloDetector를 초기화하고 YOLO 모델을 로드합니다.

        Args:
            cfg_obj (configger): YOLO 모델 경로 및 설정 정보가 포함된 configger 객체.
            model_config_key (str): 설정 파일에서 이 모델의 설정을 찾기 위한 키.
                                    (예: 'models.object_yolo_tiny_model.face_detection_model')
        """
        self.model: Optional[YOLO] = None
        self.model_info: Dict[str, Any] = {}
        self.is_ready: bool = False
        self._load_model(cfg_obj, model_config_key)

    def _load_model(self, cfg_obj: "configger", model_config_key: str):
        """설정 파일에서 모델을 로드하여 인스턴스 변수에 할당합니다. (내부 사용)"""
        if not isinstance(cfg_obj, configger):
            logger.error("configger 객체가 유효하지 않아 YOLO 모델을 로드할 수 없습니다.")
            return

        logger.info(f"YOLO 모델 로딩 시작. 설정 키: '{model_config_key}'")
        try:
            # configger 객체의 get_config 메서드를 사용하여 설정 섹션을 가져옵니다.
            object_yolo_cfg = cfg_obj.get_config(model_config_key)
            if not isinstance(object_yolo_cfg, dict):
                logger.error(f"설정 키 '{model_config_key}'에 해당하는 설정이 딕셔너리 형식이 아닙니다. 실제 타입: {type(object_yolo_cfg)}")
                return

            self.model_name                 = object_yolo_cfg.get('model_name')
            model_weights_path_str          = object_yolo_cfg.get('model_weights_path')
            self.confidence_threshold = float(object_yolo_cfg.get('confidence_threshold', 0.25))
            self.iou_threshold        = float(object_yolo_cfg.get('iou_threshold', 0.45))
            self.classes_to_detect          = object_yolo_cfg.get('classes_to_detect')
            use_cpu                         = object_yolo_cfg.get('use_cpu', True)
            self.imgsz                  = int(object_yolo_cfg.get('imgsz', 640))
            
            self.model_weights_path = Path(model_weights_path_str) if model_weights_path_str else None

            # 장치 결정 로직
            if use_cpu:
                self.device = 'cpu'
                logger.debug("설정에서 CPU 사용이 명시되어 CPU를 사용합니다.")
            else:
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    logger.debug(f"CUDA 지원 GPU가 감지되었습니다. GPU ({torch.cuda.get_device_name(0)})를 사용합니다.")
                    logger.debug(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
                    # 환경 변수는 없을 수도 있으므로 get을 사용하고 기본값 지정
                    logger.debug(f"os.environ['CUDA_VISIBLE_DEVICES']: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
                else:
                    self.device = 'cpu'
                    logger.warning("CUDA 지원 GPU를 찾을 수 없습니다.")
                    logger.warning(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
                    logger.warning(f"GPU 사용이 불가능하므로 CPU를 대신 사용합니다.")

            # YOLO 모델 로딩
            model = None
            self.model_source = None # 로딩 소스 (경로 또는 이름) 추적
            if self.model_weights_path and self.model_weights_path.exists():
                # self.model_weights_path (Path 객체) 사용. device 인자는 predict 시점에 주는 것이 일반적입니다.
                self.model_source = str(self.model_weights_path)
                self.model = YOLO(self.model_source)
                self.model.to(self.device) # 모델을 선택된 장치로 이동
                logger.debug(f"로컬 모델 파일 '{self.model_source}' 로딩 성공.")
            elif self.model_name:
                # self.model_name (문자열) 사용. device 인자는 predict 시점에 주는 것이 일반적입니다.
                self.model_source = self.model_name
                self.model = YOLO(self.model_source)
                self.model.to(self.device) # 모델을 선택된 장치로 이동
                logger.info(f"YOLO 모델 로딩 성공: '{self.model_source}' -> 장치: '{self.device}'")
            else:
                logger.error("설정에 유효한 'self.model_name' 또는 'self.model_weights_path'가 지정되지 않았습니다.")
                sys.exit(1)

            # 모델 워밍업
            logger.debug("YOLO 모델 워밍업 중...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_img, verbose=False)
            logger.debug("YOLO 모델 워밍업 완료.")

            self.model_info = {"confidence_threshold": self.confidence_threshold, "device": self.device, "self.model_source": self.model_source}
            self.is_ready = True

        except Exception as e:
            logger.error(f"YOLO 모델 로드 중 오류 발생: {e}", exc_info=True)
            self.is_ready = False

    def detect(self, image: np.ndarray, conf: Optional[float] = None) -> List[Any]:
        if not self.is_ready or self.model is None:
            logger.error("YOLO 모델이 준비되지 않아 검출을 수행할 수 없습니다.")
            return []
        
        confidence = conf if conf is not None else self.model_info.get("confidence_threshold", 0.25)
        
        try:
            return self.model(image, conf=confidence, verbose=False)
        except Exception as e:
            logger.error(f"YOLO 객체 검출 중 오류 발생: {e}", exc_info=True)
            return []

if __name__ == "__main__":
    # 표준 라이브러리 임포트
    from pathlib import Path
    from datetime import datetime

    # 1. 명령줄 인자 파싱
    args = get_argument()

    # 2. 로거 설정
    log_file_name = f"{Path(__file__).stem}_test_{datetime.now().strftime('%y%m%d_%H%M')}.log"
    log_file_path = Path(args.log_dir) / log_file_name

    # logger는 파일 상단에서 이미 임포트되었으므로 바로 사용
    logger.setup(
        logger_path=str(log_file_path),
        min_level=args.log_level,
        include_function_name=True,
        pretty_print=True
    )
    logger.info(f"--- yolo_utils.py test execution ---")
    logger.info(f"로그 파일 경로: {log_file_path}")

    # 3. configger 인스턴스 생성
    try:
        config_manager = configger(root_dir=args.root_dir, config_path=args.config_path)
        logger.info("configger 인스턴스 생성 완료.")
    except Exception as e:
        logger.critical(f"configger 초기화 실패: {e}", exc_info=True) # configger 초기화 실패
        sys.exit(1)
    
    # 4. YoloDetector 인스턴스 생성 detected object를 이용
    logger.info("YoloDetector 인스턴스 생성 및 모델 로드 시도...")
    object_detector = YoloDetector(config_manager, model_config_key='models.object_yolo_tiny_model.object_detection_model')
    if not object_detector.is_ready:
        logger.critical("YoloDetector 초기화에 실패하여 테스트를 중단합니다.")
        sys.exit(1)

    # 4.1. 테스트 이미지 로드
    # config.yaml에 테스트 이미지 경로를 설정해야 합니다.
    # 예:
    # project:
    #   paths:
    #     datasets:
    #       raw_image_dir: "path/to/your/test_image.jpg"
    test_image_dir_str = config_manager.get_value("project.paths.datasets.raw_image_dir")
    if not test_image_dir_str:
        logger.warning("테스트 이미지 경로가 설정 파일에 없습니다 ('project.paths.datasets.raw_image_dir').")
        logger.info("테스트를 위해 임의의 더미 이미지를 생성합니다. 실제 얼굴 검출은 되지 않을 수 있습니다.")
        # 더미 이미지 생성
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 더미 이미지에 간단한 도형 추가 (YOLO가 탐지할 수 있는 객체는 아님)
        cv2.circle(dummy_image, (320, 240), 50, (0, 0, 255), -1) # 빨간색 원
        cv2.rectangle(dummy_image, (100, 100), (200, 200), (255, 0, 0), 2) # 파란색 사각형
        is_success, image_buffer = cv2.imencode(".jpg", dummy_image)
        if not is_success:
            logger.error("더미 이미지 인코딩 실패.")
            sys.exit(1)
        image_bytes_for_test = image_buffer.tobytes()
        # YOLO는 OpenCV BGR 이미지를 직접 받으므로, 바이트에서 디코딩
        test_image_np = cv2.imdecode(np.frombuffer(image_bytes_for_test, np.uint8), cv2.IMREAD_COLOR)
    else:
        test_image_path = Path(test_image_dir_str) / '025AE14850DDAE952BFDC5.jpg'
        if not test_image_path.is_file():
            logger.critical(f"설정된 테스트 이미지 파일을 찾을 수 없습니다: {test_image_path}")
            sys.exit(1)
        
        logger.info(f"테스트 이미지 로드: {test_image_path}") # 테스트 이미지 로드
        try:
            test_image_np = cv2.imread(str(test_image_path))
            if test_image_np is None:
                logger.critical(f"테스트 이미지 파일을 OpenCV로 로드할 수 없습니다: {test_image_path}")
                sys.exit(1)
        except Exception as e:
            logger.critical(f"테스트 이미지 파일 읽기 오류: {e}", exc_info=True)
            sys.exit(1)

    # 4.2. YOLO 객체 탐지 테스트 (detect 메서드)
    logger.info("\n--- 'detect' 메서드 테스트 시작 ---")
    yolo_results = object_detector.detect(test_image_np)

    if yolo_results:
        # YOLO results 객체는 리스트 형태일 수 있으며, 각 요소는 이미지 하나에 대한 Results 객체입니다.
        # 여기서는 단일 이미지를 처리하므로 첫 번째 Results 객체를 사용합니다.
        results_for_image = yolo_results[0]
        
        detected_count = len(results_for_image.boxes)
        logger.info(f"총 {detected_count}개의 객체를 검출했습니다.")

        # 시각화를 위해 이미지 복사
        annotated_image = test_image_np.copy()

        for i, box in enumerate(results_for_image.boxes):
            xyxy = box.xyxy.tolist()[0] # [x1, y1, x2, y2]
            conf = float(box.conf)
            cls_id = int(box.cls)
            cls_name = results_for_image.names[cls_id]
            
            logger.info(f"  - 객체 {i+1}: Class='{cls_name}' (ID: {cls_id}), Conf={conf:.2f}, BBox={xyxy}")

            # 이미지에 바운딩 박스와 라벨 그리기
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # 녹색 박스
            cv2.putText(annotated_image, f"{cls_name} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 시각화된 이미지 저장
        output_dir = Path(args.log_dir) / "yolo_test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = output_dir / f"detected_image_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(str(output_image_path), annotated_image)
        logger.info(f"탐지 결과가 그려진 이미지 저장: {output_image_path}")

    else:
        logger.warning("이미지에서 얼굴을 검출하지 못했습니다.")
    
    logger.info("--- Object Detection 'detect' 메서드 테스트 종료 ---\n")
    del object_detector # 이전 모델 리소스 해제

    # 5. YoloDetector 인스턴스 생성 detected face를 이용
    logger.info("\n" + "="*50)
    logger.info("YoloDetector(face) 인스턴스 생성 및 모델 로드 시도...")
    object_detector = YoloDetector(config_manager, model_config_key='models.object_yolo_tiny_model.face_detection_model')
    if not object_detector.is_ready:
        logger.critical("YoloDetector(face_detection_model) 초기화에 실패하여 테스트를 중단합니다.")
        sys.exit(1)

    # 5.1. 테스트 이미지 로드
    # config.yaml에 테스트 이미지 경로를 설정해야 합니다.
    # 예:
    # project:
    #   paths:
    #     datasets:
    #       raw_image_dir: "path/to/your/test_image.jpg"
    test_image_dir_str = config_manager.get_value("project.paths.datasets.raw_image_dir")
    if not test_image_dir_str:
        logger.warning("테스트 이미지 경로가 설정 파일에 없습니다 ('project.paths.datasets.raw_image_dir').")
        logger.info("테스트를 위해 임의의 더미 이미지를 생성합니다. 실제 얼굴 검출은 되지 않을 수 있습니다.")
        # 더미 이미지 생성
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 더미 이미지에 간단한 도형 추가 (YOLO가 탐지할 수 있는 객체는 아님)
        cv2.circle(dummy_image, (320, 240), 50, (0, 0, 255), -1) # 빨간색 원
        cv2.rectangle(dummy_image, (100, 100), (200, 200), (255, 0, 0), 2) # 파란색 사각형
        is_success, image_buffer = cv2.imencode(".jpg", dummy_image)
        if not is_success:
            logger.error("더미 이미지 인코딩 실패.")
            sys.exit(1)
        image_bytes_for_test = image_buffer.tobytes()
        # YOLO는 OpenCV BGR 이미지를 직접 받으므로, 바이트에서 디코딩
        test_image_np = cv2.imdecode(np.frombuffer(image_bytes_for_test, np.uint8), cv2.IMREAD_COLOR)
    else:
        test_image_path = Path(test_image_dir_str) / '025AE14850DDAE952BFDC5.jpg'
        if not test_image_path.is_file():
            logger.critical(f"설정된 테스트 이미지 파일을 찾을 수 없습니다: {test_image_path}")
            sys.exit(1)
        
        logger.info(f"테스트 이미지 로드: {test_image_path}") # 테스트 이미지 로드
        try:
            test_image_np = cv2.imread(str(test_image_path))
            if test_image_np is None:
                logger.critical(f"테스트 이미지 파일을 OpenCV로 로드할 수 없습니다: {test_image_path}")
                sys.exit(1)
        except Exception as e:
            logger.critical(f"테스트 이미지 파일 읽기 오류: {e}", exc_info=True)
            sys.exit(1)

    # 5.2. YOLO 객체 탐지 테스트 (detect 메서드)
    logger.info("\n--- 'detect' 메서드 테스트 시작 ---")
    yolo_results = object_detector.detect(test_image_np)

    if yolo_results:
        # YOLO results 객체는 리스트 형태일 수 있으며, 각 요소는 이미지 하나에 대한 Results 객체입니다.
        # 여기서는 단일 이미지를 처리하므로 첫 번째 Results 객체를 사용합니다.
        results_for_image = yolo_results[0]
        
        detected_count = len(results_for_image.boxes)
        logger.info(f"총 {detected_count}개의 객체를 검출했습니다.")

        # 시각화를 위해 이미지 복사
        annotated_image = test_image_np.copy()

        for i, box in enumerate(results_for_image.boxes):
            xyxy = box.xyxy.tolist()[0] # [x1, y1, x2, y2]
            conf = float(box.conf)
            cls_id = int(box.cls)
            cls_name = results_for_image.names[cls_id]
            
            logger.info(f"  - 객체 {i+1}: Class='{cls_name}' (ID: {cls_id}), Conf={conf:.2f}, BBox={xyxy}")

            # 이미지에 바운딩 박스와 라벨 그리기
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # 녹색 박스
            cv2.putText(annotated_image, f"{cls_name} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 시각화된 이미지 저장
        output_dir = Path(args.log_dir) / "yolo_test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = output_dir / f"detected_image_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(str(output_image_path), annotated_image)
        logger.info(f"탐지 결과가 그려진 이미지 저장: {output_image_path}")

    else:
        logger.warning("이미지에서 얼굴을 검출하지 못했습니다.")
    
    logger.info("--- Face Detection 'detect' 메서드 테스트 종료 ---\n")

    logger.info("yolo_utils.py 테스트 실행 완료.")
    if hasattr(logger, 'shutdown') and callable(logger.shutdown):
        logger.shutdown()
    sys.exit(0)
