"""
이 코드는 사용자가 업로드한 이미지에서 얼굴을 찾아,
미리 구축된 데이터베이스(인덱스)와 비교하여
가장 유사한 얼굴들을 보여주는 웹 서비스입니다.

FastAPI 프레임워크를 기반으로 하며,
비동기 처리와 스레드 풀을 활용하여 효율적으로 동작하도록 구성되어 있습니다.
결과 이미지는 고유한 이름으로 복사되어 웹 페이지에 표시됩니다.

"""
import sys
import os
# sys.path 설정은 가급적 프로젝트 구조나 PYTHONPATH 환경 변수로 관리하는 것이 좋습니다.
# sys.path.append(os.path.dirname(os.path.abspath(__file__))) # 주석 처리 또는 제거
import io
import shutil
import pickle
import uuid
import logging
from pathlib import Path
import asyncio # 비동기 처리를 위해 추가

import numpy as np
# face_recognition은 거리 계산 및 얼굴 위치 찾기에 필요
import face_recognition
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates # Jinja2 추가
from fastapi.concurrency import run_in_threadpool # 스레드풀 실행 추가

# config_loader에서 load_config와 get_face_encodings 임포트 (상대 경로 사용)
# 현재 파일 위치를 기준으로 config_loader를 찾기 위해 sys.path 조정이 필요할 수 있음
# 만약 src 디렉토리에서 직접 실행한다면 아래와 같이 수정
# from config_loader import load_config, get_face_encodings
# 만약 프로젝트 루트에서 실행한다면 (예: uvicorn src.face_search_web:app)
from .config_loader import load_config, get_face_encodings
import re

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-6s - %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 로딩 및 경로 정의 ---
try:
    # 설정 파일 경로를 스크립트 위치 기준으로 변경
    CONFIG_FILE_PATH = Path(__file__).resolve().parent.parent / "config" / ".my_config.yaml"
    if not CONFIG_FILE_PATH.exists():
         # 대체 경로 시도 (예: 현재 작업 디렉토리 기준)
         CONFIG_FILE_PATH = Path("config/.my_config.yaml")

    logger.info(f"설정 파일 로드 시도: {CONFIG_FILE_PATH.resolve()}")
    config = load_config(CONFIG_FILE_PATH)

    INDEX_FILE = Path(config["index_output"])
    TOP_K = config.get("top_k", None)
    TOLERANCE = config.get("tolerance", 0.6) # 기본값 0.6 유지
    RAW_DIR = Path(config["data_path"])
    CROP_DIR = Path(config["cropped_faces_dir"])

    # --- 경로 설정 방식 변경 (스크립트 기준) ---
    BASE_DIR = Path(__file__).resolve().parent
    # static 디렉토리 경로 설정 (BASE_DIR 아래)
    STATIC_DIR = BASE_DIR / "static"
    # templates 디렉토리 경로 설정 (BASE_DIR 아래)
    TEMPLATE_DIR = BASE_DIR / "templates"
    # 결과 저장 디렉토리 경로 설정 (STATIC_DIR 아래)
    STATIC_RESULT_DIR = STATIC_DIR / "results"

except KeyError as e:
    logger.critical(f"❌ 설정 파일에 필요한 키가 없습니다: {e}")
    sys.exit(1)
except FileNotFoundError:
    # load_config에서 처리되지만, 여기서도 명시적으로 처리
    logger.critical(f"❌ 설정 파일({CONFIG_FILE_PATH})을 찾을 수 없습니다.")
    sys.exit(1)
except Exception as e:
    logger.critical(f"❌ 설정 로딩 또는 경로 설정 중 오류 발생: {e}", exc_info=True)
    sys.exit(1)

# --- FastAPI 앱 생성 및 정적/템플릿 설정 ---
app = FastAPI()

# 디렉토리 생성 (앱 시작 시)
try:
    STATIC_DIR.mkdir(exist_ok=True)
    STATIC_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATE_DIR.mkdir(exist_ok=True)
    logger.info(f"정적 파일 디렉토리: {STATIC_DIR.resolve()}")
    logger.info(f"템플릿 디렉토리: {TEMPLATE_DIR.resolve()}")
    logger.info(f"결과 저장 디렉토리: {STATIC_RESULT_DIR.resolve()}")
except OSError as e:
     logger.critical(f"❌ 필수 디렉토리 생성 실패: {e}")
     sys.exit(1)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# --- 인덱스 로딩 ---
INDEX_ENCODINGS = np.array([])
INDEX_PATHS = []
if INDEX_FILE.exists():
    try:
        logger.info(f"인덱스 파일 로드 시도: {INDEX_FILE.resolve()}")
        with open(INDEX_FILE, "rb") as f:
            data = pickle.load(f)
            if "encodings" in data and "paths" in data:
                INDEX_ENCODINGS = np.array(data["encodings"])
                INDEX_PATHS = data["paths"]
                if INDEX_ENCODINGS.ndim != 2 or (INDEX_ENCODINGS.size > 0 and INDEX_ENCODINGS.shape[1] != 128):
                    logger.warning(f"⚠️ 인덱스 인코딩 배열의 형태가 예상과 다릅니다: {INDEX_ENCODINGS.shape}")
                logger.info(f"✅ 인덱스 로드 완료: {len(INDEX_PATHS)}개 항목")
            else:
                logger.error("❌ 인덱스 파일에 'encodings' 또는 'paths' 키가 없습니다.")
    except (pickle.UnpicklingError, EOFError, Exception) as e:
        logger.error(f"❌ 인덱스 파일 로드 실패 ({INDEX_FILE}): {e}", exc_info=True)
else:
    logger.warning(f"⚠️ 인덱스 파일({INDEX_FILE})을 찾을 수 없습니다. 빈 인덱스로 시작합니다.")

# --- Helper 함수 ---
def get_original_path_from_cropped(cropped_path_str: str) -> Path | None:
    """
    크롭된 얼굴 이미지 경로로부터 원본 이미지 경로를 추정합니다.
    (주의: face_indexer_landmark.py의 저장 방식에 의존적입니다)
    """
    try:
        cropped_path = Path(cropped_path_str)
        # '_face' 뒤의 숫자 제거
        original_stem = re.sub(r'_face\d+$', '', cropped_path.stem)
        original_filename = original_stem + cropped_path.suffix
        # 원본 이미지는 RAW_DIR 아래에 있다고 가정
        # RAW_DIR이 절대 경로가 아닐 경우를 대비해 resolve() 사용 고려
        potential_original_path = RAW_DIR.resolve() / original_filename
        logger.debug(f"크롭 경로 '{cropped_path_str}' -> 추정 원본 경로: {potential_original_path}")
        # 실제 파일 존재 여부는 이 함수에서 확인하지 않음
        return potential_original_path
    except Exception as e:
        logger.error(f"원본 경로 변환 중 오류 ({cropped_path_str}): {e}")
        return None

def find_similar_faces(upload_image: Image.Image):
    """
    업로드된 이미지에서 얼굴을 찾아 인코딩하고,
    미리 로드된 인덱스와 비교하여 유사한 얼굴 목록을 반환합니다.
    """
    if INDEX_ENCODINGS.size == 0:
        logger.warning("인덱스가 비어있어 검색을 수행할 수 없습니다.")
        return []

    try:
        # 1. 업로드된 이미지에서 얼굴 인코딩 추출 (공용 함수 사용)
        #    인덱싱 시 사용한 모델과 동일하게 설정하는 것이 좋음 (예: "cnn")
        logger.info("업로드된 이미지에서 얼굴 인코딩 시작 (모델: cnn)...")
        face_encodings_list = get_face_encodings(upload_image, model="cnn")

        if not face_encodings_list:
            logger.warning("업로드된 이미지에서 얼굴을 찾거나 인코딩하지 못했습니다.")
            return []

        # 첫 번째 찾은 얼굴의 인코딩 사용 (필요시 여러 얼굴 처리 로직 추가 가능)
        query_enc = face_encodings_list[0]
        logger.info("얼굴 인코딩 완료. 인덱스와 비교 시작...")

        # 2. 인덱스와 거리 계산
        distances = face_recognition.face_distance(INDEX_ENCODINGS, query_enc)

        if distances.size > 0:
            min_dist = np.min(distances)
            avg_dist = np.mean(distances)
            logger.info(f"계산된 거리: 개수={distances.size}, 최소={min_dist:.4f}, 평균={avg_dist:.4f}")
        else:
            logger.info("계산된 거리가 없습니다 (인덱스가 비어있을 수 있음).")
            return [] # 거리가 없으면 매칭 결과도 없음

        # 3. Tolerance 기반 매칭 및 결과 생성
        match_indices = np.where(distances <= TOLERANCE)[0]
        logger.info(f"매칭된 인덱스 개수 (Tolerance {TOLERANCE}): {len(match_indices)}")

        results = []
        for i in match_indices:
            if i < len(INDEX_PATHS):
                # 결과에는 (거리, 크롭된 이미지 경로) 저장
                results.append((float(distances[i]), INDEX_PATHS[i]))
            else:
                # 이 경우는 데이터 불일치를 의미하므로 심각한 오류일 수 있음
                logger.error(f"인덱스 불일치 오류: 거리 계산 결과 인덱스 {i}가 경로 목록 크기({len(INDEX_PATHS)})를 벗어납니다.")

        # 4. 결과 정렬 및 Top-K 제한
        results.sort(key=lambda x: x[0]) # 거리가 짧은 순으로 정렬
        if TOP_K is not None and TOP_K > 0:
            logger.info(f"결과를 상위 {TOP_K}개로 제한합니다.")
            results = results[:TOP_K]

        logger.info(f"최종 반환 결과 개수: {len(results)}")
        return results

    except Exception as e:
        logger.error(f"유사 얼굴 검색 중 예상치 못한 오류 발생: {e}", exc_info=True)
        return []

# --- FastAPI 엔드포인트 ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def main_page(request: Request):
    """메인 HTML 페이지를 렌더링합니다."""
    template_path = TEMPLATE_DIR / "search.html"
    if not template_path.exists():
         logger.error(f"템플릿 파일({template_path})을 찾을 수 없습니다.")
         raise HTTPException(status_code=500, detail="서버 설정 오류: UI 템플릿 파일을 찾을 수 없습니다.")
    return templates.TemplateResponse("search.html", {"request": request})

@app.post("/find_faces/", response_class=JSONResponse)
async def find_faces_api(image: UploadFile = File(...)):
    """
    이미지를 업로드받아 유사한 얼굴을 검색하고,
    결과 원본 이미지들의 정보(고유 URL, 원본 파일명, 거리)를 반환합니다.
    CPU 집약적인 작업은 스레드 풀에서 비동기적으로 처리합니다.
    """
    request_start_time = asyncio.get_event_loop().time()
    logger.info(f"'/find_faces/' API 요청 수신: {image.filename} ({image.content_type})")

    try:
        image_data = await image.read()
        # 이미지 열기 및 RGB 변환 (CPU/IO 작업이므로 스레드풀)
        logger.info("이미지 데이터 로드 및 변환 시작...")
        uploaded_image = await run_in_threadpool(Image.open(io.BytesIO(image_data)).convert, "RGB")
        logger.info("이미지 데이터 로드 및 변환 완료.")
    except Exception as e:
        logger.error(f"이미지 읽기 또는 변환 실패: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아니거나 처리 중 오류가 발생했습니다.")
    finally:
        # UploadFile 객체는 사용 후 닫아주는 것이 좋음
        await image.close()

    # 유사 얼굴 검색 (CPU 집약적이므로 스레드풀)
    logger.info("유사 얼굴 검색 시작...")
    search_start_time = asyncio.get_event_loop().time()
    # find_similar_faces 함수 자체가 동기 함수이므로 run_in_threadpool 사용
    results = await run_in_threadpool(find_similar_faces, uploaded_image)
    search_duration = asyncio.get_event_loop().time() - search_start_time
    logger.info(f"유사 얼굴 검색 완료 ({search_duration:.2f}초 소요). 결과 처리 시작...")

    served_result_images = []
    processed_originals = set() # 이번 요청에서 이미 처리한 원본 이미지 경로 추적

    copy_tasks = [] # 파일 복사 작업을 모을 리스트

    for dist, cropped_img_path_str in results:
        # 원본 이미지 경로 추정 (동기 함수)
        original_path = get_original_path_from_cropped(cropped_img_path_str)
        if not original_path:
            logger.warning(f"크롭 경로 '{cropped_img_path_str}'에 대한 원본 경로 추정 실패.")
            continue # 원본 경로 추정 실패 시 건너뛰기

        original_path_str = str(original_path.resolve()) # 정규화된 경로 사용
        if original_path_str in processed_originals:
            logger.debug(f"이미 처리된 원본 이미지 건너뛰기: {original_path.name}")
            continue # 이미 처리된 원본이면 건너뛰기

        # 파일 존재 여부 확인 (디스크 I/O 이므로 스레드풀)
        # is_file()은 동기 함수이므로 run_in_threadpool 사용
        is_file = await run_in_threadpool(original_path.is_file)

        if is_file:
            # 고유 ID 생성 및 결과 파일 경로 설정
            unique_id = uuid.uuid4()
            # 원본 파일 확장자 유지
            static_filename = f"{unique_id}{original_path.suffix}"
            dest_path = STATIC_RESULT_DIR / static_filename

            # 파일 복사 작업을 비동기 태스크로 추가 (shutil.copy는 동기 함수)
            copy_task = run_in_threadpool(shutil.copy, original_path, dest_path)
            copy_tasks.append(copy_task)

            # 결과 목록에는 복사 *전*에 정보 추가 (URL은 예측 가능)
            served_result_images.append({
                "url": f"/static/results/{static_filename}", # 고유 URL
                "original_name": original_path.name,       # 원본 파일명
                "distance": f"{dist:.4f}"                  # 유사도 거리
            })
            processed_originals.add(original_path_str) # 처리된 원본으로 기록
            logger.info(f"결과 추가: {original_path.name} (거리: {dist:.4f}), 복사 예정: {dest_path.name}")
        else:
            logger.warning(f"추정된 원본 이미지가 존재하지 않음: {original_path_str}")

    # 모든 파일 복사 작업이 완료될 때까지 기다림
    if copy_tasks:
        logger.info(f"{len(copy_tasks)}개의 파일 복사 작업 시작...")
        copy_start_time = asyncio.get_event_loop().time()
        try:
            await asyncio.gather(*copy_tasks)
            copy_duration = asyncio.get_event_loop().time() - copy_start_time
            logger.info(f"파일 복사 작업 완료 ({copy_duration:.2f}초 소요).")
        except Exception as e:
            # 파일 복사 중 오류 발생 시 로깅 (개별 오류는 이미 처리되었을 수 있음)
            logger.error(f"파일 복사 작업 중 오류 발생: {e}", exc_info=True)
            # 오류가 발생해도 일단 수집된 결과는 반환

    request_duration = asyncio.get_event_loop().time() - request_start_time
    logger.info(f"'/find_faces/' API 요청 처리 완료. 총 {request_duration:.2f}초 소요.")

    return JSONResponse(content={
        "message": "유사 얼굴 검색 성공",
        "uploaded_filename": image.filename, # 원본 파일명 반환
        "found_similar_images": served_result_images # 가공된 결과 리스트 반환
    })

# --- 서버 실행 (예시, 실제 실행은 uvicorn 명령 사용) ---
# 이 스크립트를 직접 실행할 때 사용 (python src/face_search_web.py)
if __name__ == "__main__":
    import uvicorn
    logger.info("FastAPI 서버 시작 (개발 모드)...")
    # reload=True는 코드 변경 시 서버 자동 재시작 (개발 시 유용)
    # host="0.0.0.0"은 모든 네트워크 인터페이스에서 접속 허용
    # port=8000은 사용할 포트 번호
    # app="face_search_web:app" 은 실행할 FastAPI 앱 객체 지정 (파일명:앱객체명)
    uvicorn.run("face_search_web:app", host="0.0.0.0", port=8000, reload=True)
    # 운영 환경에서는 reload=False 또는 Gunicorn 등 WSGI 서버 사용 권장
