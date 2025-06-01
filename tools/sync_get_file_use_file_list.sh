#!/bin/bash

# --- 설정 ---

# 원격 서버 정보
REMOTE_USER="owner"                 # 원격 서버 사용자 이름
REMOTE_SERVER="jongsook.iptime.org" # 원격 서버 주소
REMOTE_PORT="5410"                  # 원격 서버 SSH 포트
# 중요: 원격 서버에서 파일을 찾을 기준 디렉토리 경로입니다.
# LOCAL_FILE_LIST 파일 안의 경로들이 이 디렉토리를 기준으로 합니다.
# 예: /home/owner/source_images/ 또는 /path/to/data/
REMOTE_BASE_DIR="./SambaData/Backup/FastCamp/aihub_data/datafiles/1.traing/image/" # <---- 실제 원격 서버 경로로 수정하세요!

# 로컬 설정
# 가져올 파일 목록이 적힌 로컬 파일 경로
# 중요: 이 파일 안의 파일명 대소문자가 원격 서버와 일치해야 합니다!
LOCAL_FILE_LIST="/data/ephemeral/home/Upstage/brainventures/data/jsons/train_nonexist_list.lst"
# 원격에서 가져온 파일을 저장할 로컬 디렉토리 경로
LOCAL_DEST_DIR="/data/ephemeral/home/data/aihub/Fastcampus_project/images/" # <---- 파일을 저장할 로컬 경로를 지정하세요.

# SSH 명령어 정의
SSH_CMD="ssh -p ${REMOTE_PORT}"

# --- 스크립트 시작 ---

echo "========== 원격 파일 가져오기 시작 : $(date) =========="
echo "원격 서버      : ${REMOTE_USER}@${REMOTE_SERVER}:${REMOTE_PORT}"
echo "원격 기준 경로 : ${REMOTE_BASE_DIR}"
echo "로컬 파일 목록 : ${LOCAL_FILE_LIST}"
echo "로컬 저장 경로 : ${LOCAL_DEST_DIR}"
echo "-----------------------------------------------------"

# 1. 로컬 파일 목록 존재 확인
if [ ! -f "${LOCAL_FILE_LIST}" ]; then
    echo "오류: 로컬 파일 목록을 찾을 수 없습니다: '${LOCAL_FILE_LIST}'" >&2 # 오류는 표준 에러(stderr)로 출력
    echo "========== 가져오기 실패 : $(date) =========="
    exit 1
fi

# --- 추가: 처리할 파일 수 로깅 ---
# wc -l : 파일의 라인 수를 셉니다. awk는 공백 제거 후 숫자만 추출합니다.
NUM_FILES_TO_FETCH=$(wc -l < "${LOCAL_FILE_LIST}" | awk '{print $1}')
echo "정보: 파일 목록 '${LOCAL_FILE_LIST}'에서 ${NUM_FILES_TO_FETCH}개의 파일을 가져올 예정입니다."
# --- 처리할 파일 수 로깅 끝 ---

# 2. 로컬 대상 디렉토리 생성 (없으면)
# mkdir -p 는 중간 경로가 없어도 생성해주고, 이미 있어도 오류를 내지 않습니다.
mkdir -p "${LOCAL_DEST_DIR}"
if [ $? -ne 0 ]; then
    echo "오류: 로컬 저장 경로를 생성할 수 없습니다: '${LOCAL_DEST_DIR}'" >&2
    echo "========== 가져오기 실패 : $(date) =========="
    exit 1
fi
echo "정보: 로컬 저장 경로 확인/생성 완료: ${LOCAL_DEST_DIR}"

# 3. rsync를 사용하여 파일 가져오기
# -a: 아카이브 모드 (권한, 시간 등 유지)
# -z: 전송 중 데이터 압축
# -v: 상세 정보 출력
# --files-from=FILE: 지정된 파일에서 가져올 파일 목록(상대 경로)을 읽음
#                    REMOTE_BASE_DIR 를 기준으로 이 목록의 파일들을 찾습니다. (대소문자 구분 주의!)
# --progress: 전송 진행 상태를 보여줍니다 (선택 사항).
# -e "ssh -p PORT": 사용할 SSH 명령어 지정
echo "정보: '${LOCAL_FILE_LIST}' 목록의 파일들에 대해 rsync 동기화를 시작합니다..."
rsync -az --progress \
    --files-from="${LOCAL_FILE_LIST}" \
    -e "${SSH_CMD}" \
    "${REMOTE_USER}@${REMOTE_SERVER}:${REMOTE_BASE_DIR}" \
    "${LOCAL_DEST_DIR}"

# 4. rsync 실행 결과 확인
RSYNC_EXIT_CODE=$?

echo "-----------------------------------------------------" # 결과 구분을 위한 라인
if [ ${RSYNC_EXIT_CODE} -eq 0 ]; then
    echo "성공: Rsync 작업이 성공적으로 완료되었습니다."
    echo "정보: '${LOCAL_FILE_LIST}' 목록의 파일들을 '${REMOTE_BASE_DIR}'에서 '${LOCAL_DEST_DIR}'(으)로 가져왔습니다."
    echo "========== 가져오기 완료 : $(date) =========="
else
    # 오류 메시지는 표준 에러(stderr)로 출력
    >&2 echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    >&2 echo "!!! 오류: Rsync 작업 실패 (종료 코드: ${RSYNC_EXIT_CODE}) !!!"
    >&2 echo "!!! 파일 목록의 대소문자, 원격 기준 경로 및 네트워크 연결을 확인하세요. !!!" # 안내 메시지 수정
    >&2 echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "========== 가져오기 실패 : $(date) =========="
fi

# 스크립트를 rsync 종료 코드로 종료
exit ${RSYNC_EXIT_CODE}
