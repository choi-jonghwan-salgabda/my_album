import os
import pickle
import numpy as np
import cv2
import shutil
import gc
import tempfile
import logging
from pathlib import Path
# config_loader에서 load_config와 get_face_encodings 임포트
from config_loader import load_config, get_face_encodings
import math # math 임포트 추가

import mediapipe as mp
# face_recognition은 get_face_encodings 내부에서 사용되므로 여기서 직접 임포트 필요 없음
# import face_recognition
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-8s %(levelname)-7s %(message)s') # 포맷 수정

# save_index, plot_distribution, save_face 함수는 변경 없음
# ... (save_index, plot_distribution, save_face 함수 정의) ...
def save_index(index_file, encodings, paths):
    try:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=index_file.parent, suffix=".tmp") as temp_f:
            pickle.dump({"encodings": encodings, "paths": paths}, temp_f)
            temp_path = Path(temp_f.name)
        shutil.move(str(temp_path), index_file)
        return True
    except Exception as e:
        logging.error(f"❌ Failed to save index: {e}")
        return False

def plot_distribution(encodings, output_path):
    if len(encodings) < 2:
        logging.info("📉 시각화 생략 (encoding 2개 미만)")
        return
    try:
        reduced = TSNE(n_components=2, random_state=42).fit_transform(np.array(encodings))
        plt.figure(figsize=(16, 10))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7)
        plt.title("Face Index Distribution (t-SNE)")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"📊 분포도 저장 완료: {output_path}")
    except Exception as e:
        logging.error(f"❌ t-SNE 시각화 실패: {e}")

def save_face(image, bbox, save_path):
    h, w, _ = image.shape
    x, y, width, height = bbox
    left = max(int(x), 0)
    top = max(int(y), 0)
    right = min(int(x + width), w)
    bottom = min(int(y + height), h)
    if top >= bottom or left >= right:
        return False
    face = image[top:bottom, left:right]
    # 저장 시 BGR로 변환
    cv2.imwrite(str(save_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    return True

def detect_faces_in_lanmark(img_path, mp_face_detection):
    return

def index_faces(config_path):
    config = load_config(config_path)
    raw_dir = Path(config["data_path"])
    crop_dir = Path(config["cropped_faces_dir"])
    index_file = Path(config["index_output"])
    vis_output = config.get("visualization_output", "static/results/index_distribution.png")
    # --- 설정 파일에서 tolerance 신뢰도 값 읽기 (없으면 기본값 0.6 사용) ---
    tolerance = config.get("tolerance", 0.6)

    crop_dir.mkdir(parents=True, exist_ok=True)
    encodings, paths = [], []

    # Mediapipe 설정은 유지 (크롭을 위해)
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=tolerance)

    images = list(raw_dir.glob("**/*"))
    images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    image_count = len(images)
    if image_count == 0:
        logging.warning(f"⚠️ {raw_dir} 에서 이미지를 찾을 수 없습니다.")
        return
    digit_width = math.floor(math.log10(image_count)) + 1

    logging.info(f"📂 이미지 {image_count}장 탐색됨")

    processed_faces_count = 0 # 처리된 얼굴 수 카운트

    for idx, img_path in enumerate(images, 1):
        logging.info(f"[{idx:0{digit_width}d}/{image_count}] 처리 중: {img_path.name}")

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning(f"⚠️ 이미지 로딩 실패: {img_path}")
                continue
                # Mediapipe는 RGB 이미지를 사용
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = mp_face.process(img_rgb)

            if not result.detections:
                continue

            face_saved_in_image = False # 이미지 당 얼굴 저장 여부 플래그
            for i, detection in enumerate(result.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = (
                    bboxC.xmin * iw,
                    bboxC.ymin * ih,
                    bboxC.width * iw,
                    bboxC.height * ih
                )
                out_path = crop_dir / f"{img_path.stem}_face{i}{img_path.suffix}"

                # 1. 얼굴 크롭 및 저장 (기존과 동일)
                if not save_face(img_rgb, bbox, out_path):
                    logging.warning(f"⚠️ 얼굴 저장 실패 (크기 0?): {out_path}")
                    continue

                # 2. 저장된 크롭 이미지로 인코딩 추출 (공용 함수 사용, CNN 모델 명시)
                #    공용 함수는 경로를 직접 처리 가능
                face_enc_list = get_face_encodings(out_path, model="cnn")

                if face_enc_list:
                    # 첫 번째 인코딩 사용 (크롭된 이미지에는 얼굴 하나만 있을 것으로 가정)
                    encodings.append(face_enc_list[0])
                    paths.append(str(out_path)) # 크롭된 경로 저장
                    processed_faces_count += 1
                    face_saved_in_image = True
                else:
                    # 인코딩 실패 시 저장된 크롭 파일 삭제 (선택 사항)
                    logging.warning(f"⚠️ 크롭된 얼굴({out_path.name}) 인코딩 실패. 파일 삭제 시도.")
                    try:
                        out_path.unlink(missing_ok=True)
                    except OSError as e:
                        logging.error(f"❌ 크롭 파일 삭제 실패 {out_path}: {e}")

            # 이미지 처리 후 메모리 정리 강화
            del img, img_rgb, result
            if face_saved_in_image: # 얼굴이 하나라도 저장된 경우만 gc 실행
                    gc.collect()
        except Exception as e:
            # 오류 발생 시에도 메모리 정리 시도
            logging.warning(f"⚠️ 처리 중 오류 ({img_path.name}): {e}", exc_info=True) # 상세 오류 로깅
            gc.collect()

    logging.info(f"✅ 총 인덱싱된 얼굴 수: {processed_faces_count}") # 실제 처리된 얼굴 수 로깅
    if encodings: # 인코딩된 얼굴이 있을 때만 저장 및 시각화
        if save_index(index_file, encodings, paths):
            logging.info(f"✅ 인덱스 저장 완료: {index_file}")
        plot_distribution(encodings, vis_output)
    else:
        logging.warning("⚠️ 인덱싱된 얼굴이 없어 인덱스 파일 및 분포도를 생성하지 않습니다.")

    logging.info("🎉 인덱싱 완료.")

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/.my_config.yaml"
    # 설정 파일 로딩은 index_faces 내부에서 하므로 여기서 존재 확인 불필요
    # if not os.path.exists(config_path):
    #     logging.critical(f"❌ 구성 파일을 찾을 수 없습니다: {config_path}")
    #     exit(1)
    try:
        index_faces(config_path)
    except FileNotFoundError:
        # load_config에서 발생한 FileNotFoundError 처리
        logging.critical("스크립트 실행 중단: 설정 파일을 찾을 수 없습니다.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"스크립트 실행 중 예상치 못한 오류 발생: {e}", exc_info=True)
        sys.exit(1)

