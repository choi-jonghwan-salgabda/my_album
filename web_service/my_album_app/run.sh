#!/bin/bash

# 스크립트 실행 중 오류 발생 시 즉시 종료
set -e

# 스크립트 파일의 실제 경로 가져오기
SCRIPT_DIR_RUN_SH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 현재 스크립트 위치(web_service/my_album_app)를 프로젝트 루트의 기본값으로 설정
PROJECT_ROOT_DEFAULT="$SCRIPT_DIR_RUN_SH"

# --- 명령줄 인자 파싱을 위한 기본값 설정 ---
ARG_ROOT_DIR=""
ARG_CONFIG_PATH=""
ARG_LOG_DIR=""
ARG_LOG_LEVEL=""

# --- getopt를 사용하여 명령줄 인자 파싱 ---
# ':'가 붙은 옵션은 인자를 필요로 함
# GNU getopt를 사용하는 경우 -o 다음에 짧은 옵션, --long 다음에 긴 옵션을 명시합니다.
# 옵션 문자열에서 각 옵션 뒤에 콜론(:)을 붙이면 해당 옵션이 인자를 필요로 함을 의미합니다.
TEMP=$(getopt -o r:c:l:L:h --long root-dir:,config-path:,log-dir:,log-level:,help \
              -n 'run.sh' -- "$@")

if [ $? != 0 ] ; then echo "인자 파싱 중 오류 발생..." >&2 ; exit 1 ; fi

# getopt에 의해 재정렬된 인자들을 셸 변수에 할당
eval set -- "$TEMP"

while true ; do
    case "$1" in
        -r|--root-dir) ARG_ROOT_DIR="$2" ; shift 2 ;;
        -c|--config-path) ARG_CONFIG_PATH="$2" ; shift 2 ;;
        -l|--log-dir) ARG_LOG_DIR="$2" ; shift 2 ;;
        -L|--log-level) ARG_LOG_LEVEL="$2" ; shift 2 ;;
        -h|--help)
            echo "사용법: $0 [-r ROOT_DIR] [-c CONFIG_PATH] [-l LOG_DIR] [-L LOG_LEVEL]"
            echo "  -r, --root-dir      프로젝트 루트 디렉토리"
            echo "  -c, --config-path   설정 파일 경로"
            echo "  -l, --log-dir       로그 디렉토리"
            echo "  -L, --log-level     로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
            exit 0 ;;
        --) shift ; break ;; # 인자 끝
        *) echo "내부 오류! 알 수 없는 옵션: $1" ; exit 1 ;;
    esac
done

# --- 환경 변수 설정 ---
# 명령줄 인자가 있으면 사용하고, 없으면 기본값 사용
# realpath를 사용하여 상대 경로를 절대 경로로 변환
export MY_ALBUM_ROOT_DIR=$( [ -n "$ARG_ROOT_DIR" ] && realpath -m "$ARG_ROOT_DIR" || echo "$PROJECT_ROOT_DEFAULT" )
export MY_ALBUM_CONFIG_PATH=$( [ -n "$ARG_CONFIG_PATH" ] && realpath -m "$ARG_CONFIG_PATH" || echo "${MY_ALBUM_ROOT_DIR}../../config/photo_album.yaml" )
export MY_ALBUM_LOG_DIR=$( [ -n "$ARG_LOG_DIR" ] && realpath -m "$ARG_LOG_DIR" || echo "${MY_ALBUM_ROOT_DIR}/logs/my_album_app_gunicorn" )
export MY_ALBUM_LOG_LEVEL=${ARG_LOG_LEVEL:-INFO} # 인자가 없으면 INFO 사용

# 로그 디렉토리 생성 (없으면)
mkdir -p "${MY_ALBUM_LOG_DIR}"

echo "환경 변수 설정:"
echo "MY_ALBUM_ROOT_DIR=${MY_ALBUM_ROOT_DIR}"
echo "MY_ALBUM_CONFIG_PATH=${MY_ALBUM_CONFIG_PATH}"
echo "MY_ALBUM_LOG_DIR=${MY_ALBUM_LOG_DIR}"
echo "MY_ALBUM_LOG_LEVEL=${MY_ALBUM_LOG_LEVEL}"
echo "Gunicorn 실행 디렉토리 (run.sh 위치): ${SCRIPT_DIR_RUN_SH}"
echo "Gunicorn --chdir 대상 디렉토리 (app.py 위치): ${SCRIPT_DIR_RUN_SH}/src/my_album_labeling_app" # MY_ALBUM_ROOT_DIR 대신 SCRIPT_DIR_RUN_SH 기준

# Gunicorn 실행
# --chdir 옵션은 poetry run gunicorn 명령어의 일부로 poetry가 실행되는 현재 디렉토리 기준입니다.
# SSL 인증서 경로는 이 스크립트 파일 위치(web_service/my_album_app)를 기준으로 프로젝트 루트(../../)에 있는 파일을 가리킵니다.
poetry run gunicorn --workers 4 --bind 192.168.219.10:5001 \
--certfile ../../192.168.219.10.pem \
--keyfile ../../192.168.219.10-key.pem \
--chdir ./src/my_album_labeling_app app:app
