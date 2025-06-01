# ~/face_finder_project/main.py
import os
import shutil
import subprocess
import uuid
import logging # 로깅 추가
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse # 간단한 HTML 폼 제공용

# --- 로거 설정 ---
# find_similar_faces.py와 유사한 로거 설정 (선택 사항)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-6s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 ---
# 이 경로들은 .my_config.yaml에서 읽어오거나 환경 변수로 관리하는 것이 더 좋을 수 있습니다.
# 여기서는 간단하게 main.py 내에 정의합니다.
BASE_DIR = Path(__file__).resolve().parent # main.py가 있는 디렉토리
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results" # find_similar_faces.py가 사용할 수도 있는 경로 (현재는 사용 안 함)
STATIC_DIR = BASE_DIR / "static"
STATIC_RESULT_DIR = STATIC_DIR / "results" # 웹으로 제공될 결과 이미지 저장 경로
FIND_SIMILAR_FACES_SCRIPT = BASE_DIR / "sorc" / "find_similar_faces.py" # 스크립트 경로 수정

# 서버에서 Poetry 가상 환경의 Python 경로 확인 필요 시
# 예: PYTHON_EXECUTABLE = "/home/your_user/.cache/pypoetry/virtualenvs/your_project_env-py3.x/bin/python"
# 보통 poetry run 으로 실행하면 자동 적용됩니다.
PYTHON_EXECUTABLE = "python" # poetry run 사용 시 기본값

# --- 디렉토리 생성 (없으면) ---
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True) # 스크립트가 사용할 경우 대비
STATIC_DIR.mkdir(exist_ok=True)
STATIC_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# --- FastAPI 앱 생성 ---
app = FastAPI()

# --- 정적 파일 서빙 설정 ---
# /static URL 경로로 static 디렉토리의 파일을 제공
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def parse_script_output(stdout: str) -> List[str]:
    """find_similar_faces.py의 표준 출력을 파싱하여 이미지 경로 리스트를 추출합니다."""
    found_paths = []
    in_results_section = False
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(">> 검색 결과:") and "발견" in line:
            in_results_section = True
            continue
        if in_results_section and line.startswith("---"): # 구분선 만나면 종료
             if len(found_paths) > 0: # 결과 라인이 하나라도 있었으면 종료
                 break
        if in_results_section and line and line[0].isdigit() and ':' in line:
            # "  1: /path/to/image.jpg" 형식에서 경로 부분 추출
            try:
                path_part = line.split(':', 1)[1].strip()
                found_paths.append(path_part)
            except IndexError:
                logger.warning(f"결과 라인 파싱 실패: {line}")
    logger.info(f"스크립트 출력에서 {len(found_paths)}개의 경로 파싱됨.")
    return found_paths

@app.post("/find_faces/")
async def find_faces_endpoint(image: UploadFile = File(...)):
    """
    이미지를 업로드받아 find_similar_faces.py를 실행하고 결과를 반환하는 엔드포인트
    """
    # 1. 고유한 파일 이름 생성 및 업로드된 이미지 저장
    ext = Path(image.filename).suffix.lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp']:
        raise HTTPException(status_code=400, detail="지원하지 않는 이미지 형식입니다.")

    unique_filename = f"{uuid.uuid4()}{ext}"
    upload_path = UPLOAD_DIR / unique_filename
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        logger.info(f"이미지 저장 완료: {upload_path}")
    except Exception as e:
        logger.error(f"파일 저장 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")
    finally:
        image.file.close() # 파일 핸들 닫기

    # 2. find_similar_faces.py 스크립트 실행 (search 모드)
    #    업로드된 이미지를 --query_image 인자로 전달
    cmd = [
        PYTHON_EXECUTABLE,
        str(FIND_SIMILAR_FACES_SCRIPT),
        "search", # mode 인자
        "--query_image", str(upload_path),
        # 필요시 다른 인자 추가 가능 (예: --index_file, --tolerance)
        # "--index_file", "/path/to/your/index.pkl",
        # "--tolerance", "0.55"
    ]

    logger.info(f"실행 명령어: {' '.join(cmd)}")

    try:
        # poetry run 환경에서 실행되도록 subprocess 실행
        process = subprocess.run(
            cmd,
            capture_output=True, # stdout, stderr 캡처
            text=True, # 출력을 텍스트로 처리
            check=True, # 실행 오류 시 CalledProcessError 발생
            cwd=BASE_DIR # 프로젝트 루트 디렉토리에서 실행 (설정 파일 등 상대 경로 기준)
        )
        logger.info(f"스크립트 실행 완료 (stdout 길이: {len(process.stdout)}, stderr 길이: {len(process.stderr)})")
        if process.stderr: # 표준 에러에 내용이 있으면 경고 로깅
             logger.warning(f"스크립트 stderr:\n{process.stderr}")

        # 3. 결과 처리: 스크립트의 표준 출력(stdout) 파싱
        found_image_paths_str = parse_script_output(process.stdout)
        found_image_paths = [Path(p) for p in found_image_paths_str] # Path 객체로 변환

        # 4. 결과 이미지를 static 디렉토리로 복사하고 웹 URL 생성
        served_result_urls = []
        copied_files_count = 0
        for img_path in found_image_paths:
            if img_path.is_file():
                try:
                    # 고유성을 위해 원본 파일명 앞에 UUID 일부 추가 (선택 사항)
                    # target_filename = f"{unique_filename.split('-')[0]}_{img_path.name}"
                    target_filename = img_path.name # 간단하게 원본 파일명 사용
                    static_path = STATIC_RESULT_DIR / target_filename
                    shutil.copy(img_path, static_path)
                    # 클라이언트가 접근할 URL 생성 (/static/results/이미지파일명)
                    served_result_urls.append(f"/static/results/{target_filename}")
                    copied_files_count += 1
                except Exception as e:
                    logger.warning(f"결과 파일 복사/URL 생성 실패 ({img_path}): {e}")
            else:
                logger.warning(f"스크립트가 반환한 경로가 파일이 아님: {img_path}")

        logger.info(f"{copied_files_count}개의 결과 파일을 static 경로로 복사 완료.")

        # 5. 임시 업로드 파일 삭제 (선택 사항)
        try:
            upload_path.unlink()
            logger.info(f"업로드 파일 삭제 완료: {upload_path}")
        except OSError as e:
            logger.warning(f"업로드 파일 삭제 실패: {e}")

        # 6. 결과 반환
        return {
            "message": "유사 얼굴 검색 성공",
            "uploaded_filename": image.filename,
            # "script_stdout": process.stdout.strip(), # 디버깅 시 필요하면 포함
            "found_similar_images": served_result_urls # 웹 URL 리스트
        }

    except subprocess.CalledProcessError as e:
        # 스크립트 실행 중 오류 발생 시
        logger.error(f"스크립트 실행 오류 발생 (Return code: {e.returncode})")
        logger.error(f"  Command: {' '.join(e.cmd)}")
        logger.error(f"  Stderr:\n{e.stderr}")
        logger.error(f"  Stdout:\n{e.stdout}")
        raise HTTPException(status_code=500, detail=f"얼굴 검색 스크립트 실행 실패: {e.stderr[:200]}") # 에러 일부만 노출
    except Exception as e:
        # 기타 서버 내부 오류
        logger.error(f"처리 중 예기치 않은 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {e}")

# 간단한 HTML 업로드 폼 제공 (테스트용)
@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>유사 얼굴 찾기</title>
            <style>
                body { font-family: sans-serif; margin: 20px; }
                #results img { max-width: 200px; max-height: 200px; margin: 5px; border: 1px solid #ccc; }
                #spinner { display: none; border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin-top: 10px; }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            </style>
        </head>
        <body>
            <h1>유사 얼굴 찾기</h1>
            <form id="uploadForm" enctype="multipart/form-data">
                <input name="image" type="file" accept="image/*" required>
                <button type="submit">찾기</button>
            </form>
            <div id="spinner"></div>
            <h2>결과:</h2>
            <div id="results"></div>
            <script>
                const form = document.getElementById('uploadForm');
                const resultsDiv = document.getElementById('results');
                const spinner = document.getElementById('spinner');

                form.onsubmit = async (event) => {
                    event.preventDefault();
                    resultsDiv.innerHTML = ''; // 이전 결과 지우기
                    spinner.style.display = 'block'; // 스피너 보이기

                    const formData = new FormData(form);
                    try {
                        const response = await fetch('/find_faces/', {
                            method: 'POST',
                            body: formData,
                        });

                        spinner.style.display = 'none'; // 스피너 숨기기

                        if (!response.ok) {
                            const errorData = await response.json();
                            resultsDiv.innerHTML = `<p style="color: red;">오류: ${response.status} - ${errorData.detail || '알 수 없는 오류'}</p>`;
                            return;
                        }

                        const data = await response.json();
                        if (data.found_similar_images && data.found_similar_images.length > 0) {
                            data.found_similar_images.forEach(url => {
                                const img = document.createElement('img');
                                img.src = url;
                                resultsDiv.appendChild(img);
                            });
                        } else {
                            resultsDiv.innerHTML = '<p>유사한 얼굴을 찾지 못했습니다.</p>';
                        }
                    } catch (error) {
                        spinner.style.display = 'none'; // 스피너 숨기기
                        resultsDiv.innerHTML = `<p style="color: red;">요청 실패: ${error}</p>`;
                        console.error('Error:', error);
                    }
                };
            </script>
        </body>
    </html>
    """

# --- 서버 실행 (터미널에서 직접 실행) ---
# poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# --host 0.0.0.0 : 모든 네트워크 인터페이스에서 접속 허용
# --port 8000 : 사용할 포트 번호 (방화벽에서 열려 있어야 함)
# --reload : 코드 변경 시 자동 재시작 (개발 시 유용)
