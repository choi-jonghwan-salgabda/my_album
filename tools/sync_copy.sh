#!/bin/bash

# 서버 정보 (필요시 수정)
USER="owner"
SERVER="jongsook.iptime.org"
PORT="5410"

# 명령줄 인자 확인
if [ "$#" -ne 2 ]; then
    echo "사용법: $0 <원격 소스 디렉토리> <로컬 목적지 디렉토리>"
    echo "예시: $0 ./SambaData/Backup/FastCamp/ /data/ephemeral/home/downloaded_backup/"
    exit 1
fi

# 명령줄 인자에서 소스 및 목적지 디렉토리 설정
REMOTE_SRC_DIR="$1" # 원격 소스 디렉토리
LOCAL_DEST_DIR="$2" # 로컬 목적지 디렉토리

# 로컬 목적지 디렉토리 생성 (없으면)
# rsync가 자동으로 생성해주기도 하지만, 명시적으로 확인/생성하는 것이 좋을 수 있습니다.
mkdir -p "$LOCAL_DEST_DIR"
if [ ! -d "$LOCAL_DEST_DIR" ]; then
    echo "오류: 로컬 목적지 디렉토리 '$LOCAL_DEST_DIR'를 생성할 수 없습니다."
    exit 1
fi

echo "========== Sync Start (Remote to Local) : $(date) =========="
echo "Source (Remote): ${USER}@${SERVER}:${REMOTE_SRC_DIR}"
echo "Destination (Local): ${LOCAL_DEST_DIR}"

# rsync 명령어: 원격 -> 로컬
rsync -az --progress --stats \
--exclude='.cache/' \
--exclude='.git/' \
--exclude='checkpoints/' \
--exclude='__pycache__/' \
--exclude='*.log' \
--exclude='.venv/' \
--exclude='dist/' \
-e "ssh -p $PORT" "${USER}@${SERVER}:${REMOTE_SRC_DIR}" "${LOCAL_DEST_DIR}"

# rsync 종료 코드 확인
if [ $? -eq 0 ]; then
    echo "========== Sync Done Successfully : $(date) =========="
else
    echo "!!!!!!!!!! Sync Failed : $(date) !!!!!!!!!! "
    exit 1 # 오류 발생 시 스크립트 종료
fi

exit 0
