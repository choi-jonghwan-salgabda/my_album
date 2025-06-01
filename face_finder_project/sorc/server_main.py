# -*- coding: utf-8 -*-
# ~/face_finder_project/sorc/server_main.py
import os
import shutil
import subprocess
import uuid
import logging # 로깅 기능 사용
from pathlib import Path # 파일 경로를 객체 지향적으로 다루기 위한 라이브러리
from typing import List # 타입 힌팅용

from fastapi import FastAPI, File, UploadFile, HTTPException # FastAPI 관련 클래스 임포트
from fastapi.staticfiles import StaticFiles # 정적 파일 서빙 기능
from fastapi.responses import HTMLResponse # HTML 응답 반환 기능

# --- 로거 설정 ---
# 콘솔과 파일에 로그를 남기도록 설정합니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-6s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__) # 로거 객체 생성

# --- 설정 ---
# 기본 경로 설정: 스크립트 파일 위치를 기준으로 프로젝트 관련 경로들을 정의합니다.
BASE_DIR = Path(__file__).resolve().parent # 이 스크립트(server_main.py)가 있는 디렉토리 (예: .../face_finder_project/sorc)
PROJECT_ROOT = BASE_DIR.parent # 프로젝트 루트 디렉토리 (BASE_DIR의 부모, 예: .../face_finder_project)

UPLOAD_DIR = PROJECT_ROOT / "uploads" # 업로드된 이미지를 임시 저장할 디렉토리
RESULT_DIR = PROJECT_ROOT / "results" # find_similar_faces.py 스크립트가 결과를 저장할 수 있는 디렉토리 (오류 수정: 주석 해제)
STATIC_DIR = PROJECT_ROOT / "static" # 웹 서버가 정적 파일(CSS, JS, 결과 이미지 등)을 제공할 루트 디렉토리
STATIC_RESULT_DIR = STATIC_DIR / "results" # 검색 결과 이미지를 복사하여 웹에서 접근 가능하게 할 디렉토리

# 실행할 파이썬 스크립트 경로 설정
FIND_SIMILAR_FACES_SCRIPT = BASE_DIR / "find_similar_faces.py" # server_main.py와 같은 디렉토리에 있는 스크립트

# 파이썬 실행 파일 경로 설정 (Poetry 가상환경 사용 시 보통 'python'으로 충분)
PYTHON_EXECUTABLE = "python"

# --- 디렉토리 생성 (없으면) ---
# 스크립트 실행에 필요한 디렉토리들이 없으면 자동으로 생성합니다.
UPLOAD_DIR.mkdir(exist_ok=True) # exist_ok=True: 디렉토리가 이미 있어도 오류 발생 안 함
RESULT_DIR.mkdir(exist_ok=True) # 스크립트가 사용할 경우 대비하여 생성 (오류 수정: 주석 해제 후 사용)
STATIC_DIR.mkdir(exist_ok=True)
STATIC_RESULT_DIR.mkdir(parents=True, exist_ok=True) # parents=True: 중간 경로가 없어도 생성

# --- FastAPI 앱 생성 ---
app = FastAPI(title="유사 얼굴 검색 API", description="이미지를 업로드하여 유사한 얼굴 이미지를 검색합니다.")

# --- 정적 파일 서빙 설정 ---
# '/static' URL 경로로 요청이 오면 STATIC_DIR 디렉토리의 파일을 제공합니다.
# 예: /static/results/image.jpg -> PROJECT_ROOT/static/results/image.jpg 파일 제공
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def parse_script_output(stdout: str) -> List[str]:
    """
    find_similar_faces.py 스크립트의 표준 출력(stdout) 문자열을 분석하여,
    검색된 유사 이미지 파일 경로 목록을 추출합니다.

    Args:
        stdout (str): 스크립트의 표준 출력 내용.

    Returns:
        List[str]: 추출된 이미지 파일 경로 문자열 리스트.
    """
    found_paths = [] # 찾은 경로를 저장할 리스트
    in_results_section = False # 결과 섹션 안에 있는지 여부 플래그
    for line in stdout.splitlines(): # 출력을 한 줄씩 처리
        line = line.strip() # 양 끝 공백 제거
        # ">> 검색 결과:" 로 시작하고 "발견" 이 포함된 라인을 만나면 결과 섹션 시작
        if line.startswith(">> 검색 결과:") and "발견" in line:
            in_results_section = True
            continue # 다음 라인으로
        # 결과 섹션 안에서 구분선("---")을 만나면 파싱 종료 (결과가 하나라도 있었을 경우)
        if in_results_section and line.startswith("---"):
             if len(found_paths) > 0:
                 break
        # 결과 섹션 안이고, 라인이 비어있지 않고, 숫자로 시작하며 ':' 가 포함된 경우 (예: "  1: /path/image.jpg")
        if in_results_section and line and line[0].isdigit() and ':' in line:
            try:
                # ':' 기준으로 나누고 뒷부분(경로) 추출 후 공백 제거
                path_part = line.split(':', 1)[1].strip()
                found_paths.append(path_part) # 리스트에 추가
            except IndexError:
                # 예상치 못한 형식의 라인이면 경고 로깅
                logger.warning(f"결과 라인 파싱 실패: {line}")
    logger.info(f"스크립트 출력에서 {len(found_paths)}개의 경로 파싱됨.")
    return found_paths

@app.post("/find_faces/", summary="유사 얼굴 검색", description="이미지 파일을 업로드하면 유사한 얼굴이 포함된 이미지들의 URL 목록을 반환합니다.")
async def find_faces_endpoint(image: UploadFile = File(..., description="검색 기준으로 사용할 얼굴 이미지 파일")):
    """
    이미지를 업로드받아 find_similar_faces.py 스크립트를 실행하고,
    결과 이미지들의 웹 접근 가능 URL 목록을 JSON 형태로 반환하는 API 엔드포인트입니다.
    """
    # 1. 고유한 파일 이름 생성 및 업로드된 이미지 저장
    ext = Path(image.filename).suffix.lower() # 파일 확장자 추출 및 소문자 변환
    # 지원하는 이미지 확장자 확인
    if ext not in ['.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp']:
        logger.warning(f"지원하지 않는 이미지 형식 업로드 시도: {image.filename}")
        raise HTTPException(status_code=400, detail=f"지원하지 않는 이미지 형식입니다. ({ext})")

    unique_filename = f"{uuid.uuid4()}{ext}" # UUID를 사용하여 고유한 파일명 생성
    upload_path = UPLOAD_DIR / unique_filename # 저장될 전체 경로
    try:
        # 파일을 바이너리 쓰기 모드('wb')로 열고, 업로드된 파일 내용을 복사하여 저장
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        logger.info(f"이미지 저장 완료: {upload_path}")
    except Exception as e:
        logger.error(f"파일 저장 실패 ({upload_path}): {e}", exc_info=True) # 상세 오류 로깅
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")
    finally:
        # 파일 처리가 끝나면 항상 파일 핸들을 닫아줍니다.
        image.file.close()

    # 2. find_similar_faces.py 스크립트 실행 (search 모드)
    #    업로드된 이미지를 --query_image 인자로 전달합니다.
    cmd = [
        PYTHON_EXECUTABLE, # 파이썬 실행 파일
        str(FIND_SIMILAR_FACES_SCRIPT), # 실행할 스크립트 경로
        "search", # 실행 모드: 검색
        "--query_image", str(upload_path), # 검색할 이미지 경로 전달
        # 필요시 추가 인자 전달 가능 (예: --tolerance 0.45)
        "--tolerance", "0.45"
    ]

    logger.info(f"실행 명령어: {' '.join(cmd)}") # 실행할 명령어 로그 기록

    try:
        # subprocess.run을 사용하여 외부 스크립트 실행
        process = subprocess.run(
            cmd, # 실행할 명령어 리스트
            capture_output=True, # 표준 출력과 표준 에러를 캡처
            text=True, # 출력을 텍스트(문자열)로 처리
            check=True, # 실행 실패 시(종료 코드 0 아님) CalledProcessError 발생시킴
            cwd=PROJECT_ROOT # 스크립트 실행 디렉토리를 프로젝트 루트로 설정 (.my_config.yaml 등 접근 위함)
        )
        # 스크립트 실행 결과 로깅
        logger.info(f"스크립트 실행 완료 (stdout 길이: {len(process.stdout)}, stderr 길이: {len(process.stderr)})")
        if process.stderr: # 표준 에러에 내용이 있으면 경고 로깅 (오류가 아니더라도 경고 메시지 출력 가능)
             logger.warning(f"스크립트 stderr:\n{process.stderr}")

        # 3. 결과 처리: 스크립트의 표준 출력(stdout) 파싱하여 경로 리스트 얻기
        found_image_paths_str = parse_script_output(process.stdout)
        found_image_paths = [Path(p) for p in found_image_paths_str] # 문자열 경로를 Path 객체로 변환

        # 4. 결과 이미지를 static 디렉토리로 복사하고 웹 URL 생성
        served_result_urls = [] # 클라이언트에게 전달할 URL 목록
        copied_files_count = 0 # 복사 성공한 파일 개수
        for img_path in found_image_paths:
            # 스크립트가 반환한 경로가 실제 파일인지 확인
            if img_path.is_file():
                try:
                    # 복사될 파일명 결정 (여기서는 원본 파일명 사용)
                    # 파일명 충돌 방지를 위해 고유 ID 추가 고려 가능:
                    # target_filename = f"{unique_filename.split('-')[0]}_{img_path.name}"
                    target_filename = img_path.name
                    # 복사될 최종 경로 (static/results/파일명)
                    static_path = STATIC_RESULT_DIR / target_filename
                    # 파일 복사
                    shutil.copy(img_path, static_path)
                    # 클라이언트가 접근할 수 있는 웹 URL 생성 (/static/results/파일명)
                    served_result_urls.append(f"/static/results/{target_filename}")
                    copied_files_count += 1
                except Exception as e:
                    # 파일 복사 중 오류 발생 시 경고 로깅 (전체 요청 실패는 아님)
                    logger.warning(f"결과 파일 복사/URL 생성 실패 ({img_path}): {e}")
            else:
                # 스크립트가 반환한 경로가 파일이 아닌 경우 경고 로깅
                logger.warning(f"스크립트가 반환한 경로가 파일이 아님: {img_path}")

        logger.info(f"{copied_files_count}개의 결과 파일을 static 경로로 복사 완료.")

        # 5. 임시 업로드 파일 삭제 (선택 사항)
        #    검색이 완료되었으므로 원본 업로드 파일은 삭제해도 됩니다.
        try:
            upload_path.unlink() # 파일 삭제
            logger.info(f"업로드 파일 삭제 완료: {upload_path}")
        except OSError as e:
            # 파일 삭제 실패 시 경고 로깅
            logger.warning(f"업로드 파일 삭제 실패 ({upload_path}): {e}")

        # 6. 결과 반환 (JSON 형식)
        return {
            "message": "유사 얼굴 검색 성공",
            "uploaded_filename": image.filename, # 사용자가 업로드한 원본 파일명
            # "script_stdout": process.stdout.strip(), # 디버깅 시 스크립트 전체 출력 확인용 (주석 처리)
            "found_similar_images": served_result_urls # 결과 이미지 URL 리스트
        }

    except subprocess.CalledProcessError as e:
        # 스크립트 실행 실패 시 (check=True 로 인해 발생)
        logger.error(f"스크립트 실행 오류 발생 (Return code: {e.returncode})")
        logger.error(f"  Command: {' '.join(e.cmd)}")
        logger.error(f"  Stderr:\n{e.stderr}") # 스크립트의 표준 에러 출력 로깅
        logger.error(f"  Stdout:\n{e.stdout}") # 스크립트의 표준 출력 로깅 (오류 원인 파악에 도움될 수 있음)
        # 클라이언트에게 간단한 오류 메시지 전달 (보안상 상세 내용 노출 최소화)
        raise HTTPException(status_code=500, detail=f"얼굴 검색 스크립트 실행 실패: {e.stderr[:200]}...") # 에러 메시지 일부만 포함
    except Exception as e:
        # 그 외 예측하지 못한 서버 내부 오류 발생 시
        logger.error(f"처리 중 예기치 않은 오류 발생: {e}", exc_info=True) # 상세 오류 정보 로깅
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {e}")

# 간단한 HTML 업로드 폼 제공 (테스트용)
@app.get("/", response_class=HTMLResponse, include_in_schema=False) # 스키마에는 포함 안 함
async def main_page():
    """
    웹 브라우저에서 접속 시 간단한 파일 업로드 폼과 결과 표시 영역을 보여주는 HTML 페이지를 반환합니다.
    결과는 목록 형태로 표시되며, 클릭 시 큰 이미지 미리보기 및 전체 화면 확대 기능을 제공합니다.
    """
    # HTML, CSS, JavaScript 코드를 포함하는 문자열
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>유사 얼굴 찾기</title>
            <meta charset="UTF-8">
            <style>
                body { font-family: sans-serif; margin: 20px; line-height: 1.6; display: flex; flex-direction: column; }
                .container { display: flex; flex-direction: row; margin-top: 20px; gap: 20px; flex-grow: 1; }
                .left-panel { width: 40%; display: flex; flex-direction: column; }
                .right-panel { width: 60%; border-left: 1px solid #ccc; padding-left: 20px; }

                #result-list { list-style: none; padding: 0; margin: 0; max-height: 60vh; overflow-y: auto; border: 1px solid #eee; }
                #result-list li {
                    padding: 8px 12px; cursor: pointer; border-bottom: 1px solid #eee; display: flex; align-items: center;
                }
                #result-list li:hover { background-color: #f0f0f0; }
                #result-list li.selected { background-color: #d0e0ff; font-weight: bold; }
                #result-list img.thumbnail {
                    width: 40px; height: 40px; margin-right: 10px; object-fit: cover; border: 1px solid #ccc;
                }

                #image-preview { text-align: center; margin-top: 10px; }
                #image-preview img { /* 미리보기 이미지 스타일 */
                    max-width: 100%; max-height: 70vh; border: 1px solid #ccc; margin-top: 10px;
                    cursor: zoom-in; /* 클릭 가능함을 나타내는 커서 */
                }
                #preview-placeholder { color: #888; margin-top: 20px; }

                /* --- 전체 화면 모달 스타일 --- */
                #fullscreen-modal {
                    display: none; /* 평소에는 숨김 */
                    position: fixed; /* 화면에 고정 */
                    z-index: 1000; /* 다른 요소 위에 표시 */
                    left: 0; top: 0; width: 100%; height: 100%;
                    background-color: rgba(0, 0, 0, 0.85); /* 반투명 검은 배경 */
                    justify-content: center; /* 가로 중앙 정렬 */
                    align-items: center; /* 세로 중앙 정렬 */
                }
                #fullscreen-image {
                    max-width: 90%; /* 화면 너비의 90% */
                    max-height: 90%; /* 화면 높이의 90% */
                    object-fit: contain; /* 이미지 비율 유지 */
                }
                #close-modal { /* 닫기 버튼 스타일 */
                    position: absolute;
                    top: 20px; right: 35px;
                    color: #f1f1f1; font-size: 40px; font-weight: bold;
                    cursor: pointer;
                }
                #close-modal:hover { color: #bbb; }
                /* --- --- */

                #spinner {
                    display: none; border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%;
                    width: 30px; height: 30px; animation: spin 1s linear infinite; margin-top: 15px;
                }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                button { padding: 8px 15px; cursor: pointer; }
                input[type="file"] { margin-right: 10px; }
                #error-message { color: red; margin-top: 10px; }
                h2 { margin-bottom: 10px; }
            </style>
        </head>
        <body>
            <h1>유사 얼굴 찾기</h1>
            <p>얼굴 사진을 업로드하면 데이터셋에서 유사한 얼굴이 포함된 다른 사진들을 찾아줍니다.</p>

            <form id="uploadForm" enctype="multipart/form-data">
                <input name="image" type="file" accept="image/*" required>
                <button type="submit">찾기</button>
            </form>

            <div id="spinner"></div>
            <div id="error-message"></div>

            <div class="container">
                <div class="left-panel">
                    <h2>결과 목록:</h2>
                    <ul id="result-list"></ul>
                </div>
                <div class="right-panel">
                    <h2>미리보기:</h2>
                    <div id="image-preview">
                        <p id="preview-placeholder">목록에서 이미지를 선택하세요.</p>
                    </div>
                </div>
            </div>

            <!-- 전체 화면 이미지 표시용 모달 -->
            <div id="fullscreen-modal">
                <span id="close-modal">&times;</span> <!-- 닫기 버튼 (X) -->
                <img id="fullscreen-image" src="" alt="전체 화면 이미지">
            </div>

            <script>
                const form = document.getElementById('uploadForm');
                const resultList = document.getElementById('result-list');
                const previewDiv = document.getElementById('image-preview');
                const previewPlaceholder = document.getElementById('preview-placeholder');
                const spinner = document.getElementById('spinner');
                const errorMessageDiv = document.getElementById('error-message');
                const fullscreenModal = document.getElementById('fullscreen-modal'); // 모달 요소
                const fullscreenImage = document.getElementById('fullscreen-image'); // 모달 안 이미지 요소
                const closeModalButton = document.getElementById('close-modal'); // 닫기 버튼 요소

                // 미리보기 업데이트 함수
                function showPreview(imageUrl, filename) {
                    previewDiv.innerHTML = ''; // 이전 미리보기 지우기
                    const imgPreview = document.createElement('img');
                    imgPreview.src = imageUrl;
                    imgPreview.alt = "미리보기: " + filename;

                    // 미리보기 이미지 클릭 시 전체 화면 모달 열기
                    imgPreview.addEventListener('click', () => {
                        fullscreenImage.src = imageUrl; // 모달 이미지 소스 설정
                        fullscreenModal.style.display = 'flex'; // 모달 보이기 (flex로 설정해야 정렬됨)
                    });

                    previewDiv.appendChild(imgPreview);
                    previewPlaceholder.style.display = 'none';
                }

                // 모달 닫기 함수
                function closeModal() {
                    fullscreenModal.style.display = 'none'; // 모달 숨기기
                    fullscreenImage.src = ''; // 이미지 소스 초기화 (메모리 관리 도움)
                }

                // 닫기 버튼 클릭 시 모달 닫기
                closeModalButton.addEventListener('click', closeModal);

                // 모달 배경 클릭 시 모달 닫기 (이미지 클릭 시는 닫히지 않음)
                fullscreenModal.addEventListener('click', (event) => {
                    if (event.target === fullscreenModal) { // 클릭된 요소가 모달 배경 자체일 때만
                        closeModal();
                    }
                });

                // ESC 키 눌렀을 때 모달 닫기
                document.addEventListener('keydown', (event) => {
                    if (event.key === 'Escape' && fullscreenModal.style.display === 'flex') {
                        closeModal();
                    }
                });


                // 폼 제출 이벤트 처리
                form.onsubmit = async (event) => {
                    event.preventDefault();
                    resultList.innerHTML = '';
                    previewDiv.innerHTML = '';
                    previewPlaceholder.style.display = 'block';
                    errorMessageDiv.innerHTML = '';
                    spinner.style.display = 'block';

                    const formData = new FormData(form);

                    try {
                        const response = await fetch('/find_faces/', {
                            method: 'POST',
                            body: formData,
                        });

                        spinner.style.display = 'none';

                        if (!response.ok) {
                            const errorData = await response.json();
                            errorMessageDiv.innerHTML = `오류: ${response.status} - ${errorData.detail || '알 수 없는 오류가 발생했습니다.'}`;
                            console.error('Server Error:', errorData);
                            return;
                        }

                        const data = await response.json();
                        if (data.found_similar_images && data.found_similar_images.length > 0) {
                            data.found_similar_images.forEach(url => {
                                const li = document.createElement('li');
                                const filename = url.split('/').pop();

                                const imgThumb = document.createElement('img');
                                imgThumb.src = url;
                                imgThumb.alt = "썸네일";
                                imgThumb.className = 'thumbnail';

                                const span = document.createElement('span');
                                span.textContent = filename;

                                li.appendChild(imgThumb);
                                li.appendChild(span);
                                li.dataset.imageUrl = url;
                                li.dataset.filename = filename;

                                li.addEventListener('click', () => {
                                    document.querySelectorAll('#result-list li').forEach(item => item.classList.remove('selected'));
                                    li.classList.add('selected');
                                    showPreview(li.dataset.imageUrl, li.dataset.filename);
                                });

                                resultList.appendChild(li);
                            });
                        } else {
                            resultList.innerHTML = '<li>유사한 얼굴을 포함한 이미지를 찾지 못했습니다.</li>';
                        }
                    } catch (error) {
                        spinner.style.display = 'none';
                        errorMessageDiv.innerHTML = `요청 실패: ${error}. 서버 연결 상태를 확인하세요.`;
                        console.error('Fetch Error:', error);
                    }
                };
            </script>
        </body>
    </html>
   """

# --- 서버 실행 (터미널에서 직접 실행 시 참고) ---
# 아래 주석은 터미널에서 uvicorn 서버를 실행하는 명령어 예시입니다.
# poetry run uvicorn sorc.server_main:app --host 0.0.0.0 --port 8000 --reload
# - poetry run: Poetry 가상환경 내에서 명령어 실행
# - sorc.server_main:app: sorc 디렉토리의 server_main.py 파일 안에 있는 app 객체를 실행
# - --host 0.0.0.0 : 모든 네트워크 인터페이스에서 접속 허용 (외부 접속 가능하게 함)
# - --port 8000 : 8000번 포트 사용 (다른 포트 사용 가능)
# - --reload : 코드 변경 시 서버 자동 재시작 (개발 시 유용)
