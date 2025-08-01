# 1. 가상 환경 디렉토리(.venv)를 삭제합니다.
echo "가상 환경 디렉토리(.venv)를 삭제합니다..."
rm -rf .venv

# 2. lock 파일을 삭제합니다.
echo "poetry.lock 파일을 삭제합니다..."
rm -f poetry.lock

echo "기존 환경 삭제 완료."
