#!/bin/bash

# 이 스크립트는 pyproject.toml 파일에 정의된 의존성을 사용하여
# Poetry 가상 환경을 설정하고 필요한 패키지를 설치합니다.

echo "pyproject.toml 파일과 poetry.lock 파일 동기화..."
# pyproject.toml이 변경되었을 수 있으므로, lock 파일을 먼저 업데이트합니다.
# 'poetry lock' 명령어는 pyproject.toml의 내용을 기반으로 lock 파일을 재생성합니다.
if ! poetry lock; then
    echo "오류: poetry.lock 파일 생성/업데이트에 실패했습니다."
    exit 1
fi

echo "poetry.lock 파일에 명시된 의존성을 설치합니다..."
# 이제 동기화된 lock 파일을 기반으로 안전하게 설치합니다.
if ! poetry install; then
    echo "오류: 의존성 설치에 실패했습니다. pyproject.toml 파일을 확인하세요."
    exit 1
fi

echo "Poetry 환경 설정 및 의존성 설치가 완료되었습니다."
echo "가상 환경을 활성화하려면 'poetry shell'을 실행하세요."
echo "또는 'poetry run python src/object_detector.py'와 같이 스크립트를 실행할 수 있습니다."
