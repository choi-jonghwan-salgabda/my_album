# 표준 라이브러리
import sys
from typing import Dict, Any, List, Tuple, Optional

# 외부 라이브러리
import dlib
import numpy as np
import cv2

# 사용자 정의 모듈
try:
    from my_utils.config_utils.SimpleLogger import logger, get_argument
    from my_utils.config_utils.configger import configger
except ImportError:
    #     print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    import traceback
    traceback.print_exc() # 전체 트레이스백 출력 (개발 단계에서 유용)
    sys.exit(1)

# --- 유틸리티 함수 ---

def dlib_rect_to_bbox(rect: dlib.rectangle) -> Tuple[int, int, int, int]:
    """dlib.rectangle 객체를 (x, y, w, h) 형식의 튜플로 변환합니다."""
    return rect.left(), rect.top(), rect.width(), rect.height()

def dlib_rect_to_xyxy(rect: dlib.rectangle) -> Tuple[int, int, int, int]:
    """dlib.rectangle 객체를 (x1, y1, x2, y2) 형식의 튜플로 변환합니다."""
    return rect.left(), rect.top(), rect.right(), rect.bottom()

class DlibFaceProcessor:
    """
    dlib를 사용하여 얼굴 검출 및 특징 추출을 수행하는 클래스.
    """
    def __init__(self, cfg_obj: "configger"):
        """
        DlibFaceProcessor를 초기화하고 dlib 모델을 로드합니다.

        Args:
            cfg_obj (configger): dlib 모델 경로 정보가 포함된 configger 객체.
        """
        self.detector: Optional[dlib.fhog_object_detector] = None
        self.predictor: Optional[dlib.shape_predictor] = None
        self.recognizer: Optional[dlib.face_recognition_model_v1] = None
        self.is_ready: bool = False
        self._load_models(cfg_obj)

    def _load_models(self, cfg_obj: "configger"):
        """
        설정 파일에서 모델을 로드하여 인스턴스 변수에 할당합니다. (내부 사용)
        """
        if not isinstance(cfg_obj, configger):
            logger.error("configger 객체가 유효하지 않아 dlib 모델을 로드할 수 없습니다.")
            return
        logger.info("시작 DlibFaceProcessor 초기화")

        try: # models.face_recognition 섹션 자체를 가져옵니다.
            face_recognition_section = cfg_obj.get_config('models.face_recognition')
            if not face_recognition_section:
                logger.error("설정 파일에서 'models.face_recognition' 섹션을 찾을 수 없습니다.")
                return
            
            # photo_album.yaml의 현재 구조에 따라 'face_rec_model_path'와 'landmark_model_path'를 직접 가져옵니다.
            # 이 값들은 이미 전체 경로를 포함하고 있다고 가정합니다.
            face_rec_model_path_str = face_recognition_section.get('face_rec_model_path')
            if not face_rec_model_path_str:
                logger.error("dlib 얼굴 인식 모델 경로('face_rec_model_path')를 설정에서 찾을 수 없습니다.")
                return

            landmark_model_path_str = face_recognition_section.get('landmark_model_path')
            if not landmark_model_path_str:
                logger.error("dlib 랜드마크 모델 경로('landmark_model_path')를 설정에서 찾을 수 없습니다.")
                return

            # Path 객체로 변환합니다.
            face_rec_model_path = Path(face_rec_model_path_str)
            landmark_model_path = Path(landmark_model_path_str)
            logger.info(f"face_rec_model_path : {face_rec_model_path}")
            logger.info(f"landmark_model_path : {landmark_model_path}")
            if not face_rec_model_path.is_file() or not landmark_model_path.is_file():
                logger.error(f"dlib 모델 파일이 존재하지 않습니다:\n- {face_rec_model_path}\n- {landmark_model_path}")
                return

            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(str(landmark_model_path))
            self.recognizer = dlib.face_recognition_model_v1(str(face_rec_model_path))
            
            self.is_ready = True
            logger.info("DlibFaceProcessor가 성공적으로 초기화되었습니다.")

        except Exception as e:
            logger.error(f"dlib 모델 로드 중 오류 발생: {e}", exc_info=True)
            self.is_ready = False

    def _detect_faces(self, rgb_img: np.ndarray, upsample: int = 1) -> List[dlib.rectangle]:
        """이미지에서 얼굴들을 검출합니다. (내부 사용)"""
        if not isinstance(rgb_img, np.ndarray) or self.detector is None:
            return []
        try:
            return self.detector(rgb_img, upsample)
        except Exception as e:
            logger.error(f"dlib 얼굴 검출 중 오류 발생: {e}", exc_info=True)
            return []

    def _extract_embeddings(self, rgb_img: np.ndarray, face_locations: List[dlib.rectangle], num_jitters: int = 1) -> List[np.ndarray]:
        """주어진 위치에서 얼굴 임베딩을 추출합니다. (내부 사용)"""
        if not isinstance(rgb_img, np.ndarray) or not all([self.predictor, self.recognizer]):
            return []
        
        embeddings = []
        try:
            for rect in face_locations:
                shape = self.predictor(rgb_img, rect)
                embedding = self.recognizer.compute_face_descriptor(rgb_img, shape, num_jitters)
                embeddings.append(np.array(embedding, dtype=np.float32))
            return embeddings
        except Exception as e:
            logger.error(f"dlib 얼굴 임베딩 추출 중 오류 발생: {e}", exc_info=True)
            return []

    def process_image(self, image_bytes: bytes, upsample: int = 1, num_jitters: int = 1) -> List[Dict[str, Any]]:
        """
        이미지 바이트에서 모든 얼굴을 검출하고, 각 얼굴의 위치와 임베딩을 추출합니다.

        Args:
            image_bytes (bytes): 이미지 파일의 바이트 데이터.
            upsample (int): 얼굴 검출 시 업샘플링 횟수.
            num_jitters (int): 임베딩 추출 시 지터링 횟수.

        Returns:
            List[Dict[str, Any]]: 각 얼굴의 'bbox_xyxy'와 'embedding'을 포함하는 딕셔너리 리스트.
        """
        if not self.is_ready:
            logger.error("프로세서가 준비되지 않았습니다. 모델 로드를 확인하세요.")
            return []
        if not image_bytes:
            logger.warning("입력된 이미지 바이트가 없습니다.")
            return []

        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("이미지 바이트를 디코딩할 수 없습니다.")
                return []

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_locations = self._detect_faces(rgb_img, upsample)
            if not face_locations:
                return []

            embeddings = self._extract_embeddings(rgb_img, face_locations, num_jitters)
            if len(face_locations) != len(embeddings):
                logger.error("검출된 얼굴 수와 추출된 임베딩 수가 일치하지 않습니다.")
                return []

            return [{'bbox_xyxy': dlib_rect_to_xyxy(rect), 'embedding': emb}
                    for rect, emb in zip(face_locations, embeddings)]

        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생: {e}", exc_info=True)
            return []

    def get_embedding_from_image_crop(self, image_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        잘라낸 얼굴 이미지(BGR)에서 직접 임베딩을 추출합니다.
        이 메서드는 이미 얼굴이 잘려진 이미지를 입력으로 받습니다.

        Args:
            image_crop (np.ndarray): 얼굴 부분만 잘라낸 BGR 이미지.

        Returns:
            Optional[np.ndarray]: 추출된 128차원 임베딩, 실패 시 None.
        """
        if not self.is_ready:
            logger.error("DlibFaceProcessor가 준비되지 않아 임베딩을 추출할 수 없습니다.")
            return None
        if image_crop is None or image_crop.size == 0:
            logger.warning("입력된 얼굴 크롭 이미지가 비어있습니다.")
            return None

        rgb_img = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        # 이미지 전체를 얼굴로 간주하고 경계 상자를 만듭니다.
        face_location = dlib.rectangle(0, 0, rgb_img.shape[1], rgb_img.shape[0])

        embeddings = self._extract_embeddings(rgb_img, [face_location])
        return embeddings[0] if embeddings else None

    def process_largest_face(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        이미지에서 가장 큰 얼굴 하나를 찾아 특징 벡터(임베딩)를 추출합니다.

        Args:
            image_bytes (bytes): 이미지 파일의 바이트 데이터.

        Returns:
            Optional[np.ndarray]: 가장 큰 얼굴의 128차원 임베딩. 얼굴을 찾지 못하면 None.
        """
        if not self.is_ready:
            logger.error("프로세서가 준비되지 않았습니다. 모델 로드를 확인하세요.")
            return None
        if not image_bytes:
            logger.warning("입력된 이미지 바이트가 없습니다.")
            return None

        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("이미지 바이트를 디코딩할 수 없습니다.")
                return None

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_locations = self._detect_faces(rgb_img, upsample=1)
            if not face_locations:
                return None

            largest_face_rect = max(face_locations, key=lambda rect: rect.width() * rect.height())
            
            embeddings = self._extract_embeddings(rgb_img, [largest_face_rect])
            return embeddings[0] if embeddings else None

        except Exception as e:
            logger.error(f"가장 큰 얼굴 처리 중 오류 발생: {e}", exc_info=True)
            return None

if __name__ == "__main__":
    # 표준 라이브러리 임포트 (main 블록 내에서만 사용)
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
    logger.info(f"--- dlib_utils.py test execution ---")
    logger.info(f"로그 파일 경로: {log_file_path}")

    # 3. configger 인스턴스 생성
    try:
        config_manager = configger(root_dir=args.root_dir, config_path=args.config_path)
        logger.info("configger 인스턴스 생성 완료.")
    except Exception as e:
        logger.critical(f"configger 초기화 실패: {e}", exc_info=True)
        sys.exit(1)

    # 4. DlibFaceProcessor 인스턴스 생성
    logger.info("DlibFaceProcessor 인스턴스 생성 및 모델 로드 시도...")
    dlib_processor = DlibFaceProcessor(config_manager)
    if not dlib_processor.is_ready:
        logger.critical("DlibFaceProcessor 초기화에 실패하여 테스트를 중단합니다.")
        sys.exit(1)

    # 5. 테스트 이미지 로드
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
        # 얼굴처럼 보이게 흰 사각형 추가 (검출 보장은 없음)
        cv2.rectangle(dummy_image, (200, 100), (400, 350), (255, 255, 255), -1)
        is_success, image_buffer = cv2.imencode(".jpg", dummy_image)
        if not is_success:
            logger.error("더미 이미지 인코딩 실패.")
            sys.exit(1)
        image_bytes_for_test = image_buffer.tobytes()
    else:
        test_image_path = Path(test_image_dir_str) / '025AE14850DDAE952BFDC5.jpg'
        if not test_image_path.is_file():
            logger.critical(f"설정된 테스트 이미지 파일을 찾을 수 없습니다: {test_image_path}")
            sys.exit(1)
        
        logger.info(f"테스트 이미지 로드: {test_image_path}")
        try:
            with open(test_image_path, "rb") as f:
                image_bytes_for_test = f.read()
        except Exception as e:
            logger.critical(f"테스트 이미지 파일 읽기 오류: {e}", exc_info=True)
            sys.exit(1)

    # 6. 다중 얼굴 처리 테스트 (process_image 메서드)
    logger.info("\n--- 'process_image' 메서드 테스트 시작 ---")
    multi_face_results = dlib_processor.process_image(image_bytes=image_bytes_for_test)

    if multi_face_results:
        logger.info(f"총 {len(multi_face_results)}개의 얼굴을 검출했습니다.")
        for i, result in enumerate(multi_face_results):
            bbox = result.get('bbox_xyxy')
            embedding = result.get('embedding')
            logger.info(f"  - 얼굴 {i+1}: BBox={bbox}, Embedding(5)={embedding[:5]}")
    else:
        logger.warning("이미지에서 얼굴을 검출하지 못했습니다.")
    
    logger.info("--- 'process_image' 메서드 테스트 종료 ---\n")

    # 7. 가장 큰 얼굴 처리 테스트 (process_largest_face 메서드)
    logger.info("--- 'process_largest_face' 메서드 테스트 시작 ---")
    single_face_embedding = dlib_processor.process_largest_face(image_bytes=image_bytes_for_test)

    if single_face_embedding is not None:
        logger.info("가장 큰 얼굴의 임베딩을 성공적으로 추출했습니다.")
        logger.info(f"  - 임베딩 (처음 5개 값): {single_face_embedding[:5]}")
        # single_face_embedding은 이미 임베딩(np.ndarray) 자체이므로, 추가적인 처리가 필요 없습니다.
    else:
        logger.warning("이미지에서 얼굴을 검출하지 못했거나 임베딩 추출에 실패했습니다.")
    
    logger.info("--- 'process_largest_face' 메서드 테스트 종료 ---")

    logger.info("dlib_utils.py 테스트 실행 완료.")
