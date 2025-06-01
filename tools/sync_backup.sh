#!/bin/bash

# 서버 정보
USER="owner"
SERVER="jongsook.iptime.org"
PORT="5410"

# 동기화할 디렉토리
SRC_DIR="/data/ephemeral/home/Upstage/"
DEST_DIR="./SambaData/Backup/FastCamp/Myproject/"

# 반복 주기 (초) — 1800초 = 30분
INTERVAL=100

while true; do
    echo "========== Backup Start : $(date) =========="

    rsync -az --progress --stats \
    --exclude='.cache/' \
    --exclude='.git/' \
    --exclude='checkpoints/' \
    --exclude='__pycache__/' \
    --exclude='*.log' \
    --exclude='.venv/' \
    --exclude='dist/' \
    -e "ssh -p $PORT" "${SRC_DIR}" "${USER}@${SERVER}:${DEST_DIR}"

    echo "========== Backup Done : $(date) =========="

    echo "다음 백업까지 대기 중... (${INTERVAL}초)"
    sleep $INTERVAL
done
