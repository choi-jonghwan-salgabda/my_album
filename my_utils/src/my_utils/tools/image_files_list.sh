#!/bin/bash

usage() {
    cat <<EOF
사용법: $0 <검색_디렉토리> [제외_디렉토리]

지정된 디렉토리와 그 하위 디렉토리에서 일반적인 이미지 파일(JPG, PNG 등)을 찾고
출력합니다.

  [제외_디렉토리] : 검색에서 제외할 특정 하위 디렉토리 경로 (선택 사항)
EOF
    exit 1
}

# --- 명령 매개변수 처리 ---
# 디렉토리 경로가 인자로 제공되었는지 확인
if [ -z "$1" ]; then
    usage
fi

# 검색할 디렉토리 경로를 변수에 저장
SEARCH_DIR="$1"
# 제외할 디렉토리 경로 (제공된 경우)
EXCLUDE_DIR="$2"


# --- 디렉토리 유효성 검사 ---
# 제공된 경로가 유효한 디렉토리인지 확인
if [ ! -d "$SEARCH_DIR" ]; then
    echo "오류: '$SEARCH_DIR'는 유효한 디렉토리가 아닙니다."
    usage
fi

# --- 검색 시작 안내 ---
echo "---"
echo "경로 '$SEARCH_DIR'에서 이미지 파일 이름을 검색하고 정렬합니다."
echo "---"

# 제외할 경로가 지정되었는지 확인하고 find 명령어 옵션을 구성
EXCLUDE_OPTS=""
if [ -n "$EXCLUDE_DIR" ]; then
    # 검색 경로와 제외 경로에서 끝의 '/'를 제거하여 경로를 정규화합니다.
    SEARCH_DIR_CLEANED="${SEARCH_DIR%/}"
    EXCLUDE_DIR_CLEANED="${EXCLUDE_DIR%/}"
    # find 명령어의 -path 옵션은 전체 경로를 기준으로 동작하므로, 검색 시작 경로를 포함한 전체 제외 경로를 지정해야 합니다.
    EXCLUDE_OPTS="-path \"$SEARCH_DIR_CLEANED/$EXCLUDE_DIR_CLEANED\" -prune -o"
    echo "제외 경로: \"$SEARCH_DIR_CLEANED/$EXCLUDE_DIR_CLEANED\""
    echo "---"
fi

# 검색할 이미지 확장자 목록을 배열로 정의하여 관리 용이성 향상
image_extensions=(
    '*.jpg' '*.jpeg' '*.png' '*.bmp' '*.tiff'
    '*.gif' '*.webp' '*.heic'
)

# find 명령어의 조건을 동적으로 생성
find_conditions=()
for ext in "${image_extensions[@]}"; do
    # 배열이 비어있지 않으면 OR 연산자 추가
    if [ ${#find_conditions[@]} -gt 0 ]; then
        find_conditions+=(-o)
    fi
    find_conditions+=(-iname "$ext")
done

# --- 파일 검색, 이름 출력 및 정렬 ---
# -type f: 파일만 검색
# -iname: 대소문자 구분 없이 검색 (예: .jpg, .JPG 모두)
# find의 결과를 tee를 사용하여 화면과 임시 파일로 동시에 보냅니다.
temp_file=$(mktemp)
trap 'rm -f "$temp_file"' EXIT HUP INT QUIT TERM

find "$SEARCH_DIR" $EXCLUDE_OPTS -type f \( "${find_conditions[@]}" \) -printf "%p\n" | tee "$temp_file"

echo "---"
# 임시 파일의 줄 수를 세어 파일 개수를 구합니다.
file_count=$(wc -l < "$temp_file")
echo "검색 완료. 총 ${file_count}개의 파일을 찾았습니다."
