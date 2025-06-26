#!/bin/bash
# 이 스크립트는 개발 환경을 설정하고, 현재 작업 디렉토리의 안정적인 버전을 백업합니다.
# 오류 발생 시 즉시 중단되도록 설정합니다.
set -e

# 1. 현재 작업 디렉토리로 이동합니다.
cd  ~/SambaData/Backup/FastCamp/Myproject/my_utils/
 
echo ">>> 현재 작업 디렉토리의 내용을 백업합니다..."
# 2. 현재 작업 디렉토리의 내용을 source_backup 디렉토리로 백업합니다.
#    -a 플래그는 아카이브 모드로, 권한과 타임스탬프를 보존하며 재귀적으로 복사합니다.
cp -a ./. ~/SambaData/OwnerData/source_backup/my_utils/
echo ">>> 백업 완료."

echo ">>> Poetry 의존성을 설정합니다..."
# 3. Poetry 의존성을 관리합니다.
poetry lock
poetry add PyYAML tqdm # tqdm 추가
poetry install
echo ">>> 의존성 설정 완료."

echo ">>> 테스트 스크립트를 실행합니다..."
# 4. 주요 모듈을 실행하여 정상 동작하는지 테스트합니다.
poetry run python -m my_utils.config_utils.configger
poetry run python -m my_utils.config_utils.SimpleLogger
echo ">>> 테스트 완료."

echo ">>> 현재 디렉토리 목록:"
# 5. 현재 디렉토리의 파일 목록을 출력합니다.
ls -l

echo ">>> 스크립트 실행 완료."
