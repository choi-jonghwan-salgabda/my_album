#!/bin/sh

# --- 스크립트 설정 ---
# 원본 파일들을 검색할 디렉토리를 지정하세요.
SOURCE_DIR="./source_directory"

# 파일을 복사할 대상 디렉토리를 지정하세요.
DEST_DIR="./destination_directory"

# 파일 목록이 담긴 파일의 경로를 지정하세요.
FILE_LIST_PATH="./image_test_data.lst"

# --------------------

# 대상 디렉토리가 존재하지 않으면 생성합니다.
mkdir -p "$DEST_DIR"

echo "파일 목록을 읽어 '$SOURCE_DIR'에서 파일을 찾아 '$DEST_DIR'로 복사합니다."
echo "원본 디렉토리 구조가 유지됩니다."
echo "-------------------------------------"

# 임시 플래그 파일 생성 (파일을 찾았는지 여부 확인용)
FOUND_FLAG_FILE=$(mktemp)
trap 'rm -f "$FOUND_FLAG_FILE"' EXIT HUP INT QUIT TERM # 스크립트 종료/중단 시 임시 파일 자동 삭제

# 파일 목록을 한 줄씩 읽습니다.
# IFS= 줄바꿈 문자로 설정하여 공백이 있는 파일명도 처리할 수 있도록 함.
# find 명령어를 사용하여 $SOURCE_DIR 아래에서 해당 파일을 찾습니다.
while read -r filename; do
    # 파일명의 좌우 공백을 제거합니다.
    clean_filename=$(echo "$filename" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    if [ -z "$clean_filename" ]; then
        continue # 빈 줄은 건너뜁니다.
    fi

    # 루프 시작 전 플래그 파일 비우기
    > "$FOUND_FLAG_FILE"

    # find 명령어로 원본 디렉토리에서 파일을 찾습니다.
    # -print0 옵션과 xargs -0 옵션을 사용하여 파일명에 공백이 포함되어도 안전하게 처리합니다.
    find "$SOURCE_DIR" -type f -name "$clean_filename" -print0 | while IFS= read -r -d '' source_path; do
        echo 1 > "$FOUND_FLAG_FILE" # 파일을 하나라도 찾으면 플래그 파일에 내용 기록
        # 원본 디렉토리 경로를 제거하여 상대 경로를 추출합니다.
        relative_path="${source_path#$SOURCE_DIR/}"
        
        # 파일명을 제외한 디렉토리 경로만 추출합니다.
        relative_dir=$(dirname "$relative_path")
        
        # 대상 디렉토리에 동일한 디렉토리 구조를 만듭니다.
        mkdir -p "$DEST_DIR/$relative_dir"
        
        # 파일 복사
        cp "$source_path" "$DEST_DIR/$relative_path"
        
        echo "복사됨: $relative_path"
    done

    # 플래그 파일이 비어있으면, find 명령이 아무 결과도 반환하지 않은 것이므로 파일을 못 찾은 경우임
    if [ ! -s "$FOUND_FLAG_FILE" ]; then
        echo "경고: '$clean_filename' 파일을 '$SOURCE_DIR' 에서 찾을 수 없습니다."
    fi
done < "$FILE_LIST_PATH"

echo "-------------------------------------"
echo "작업 완료. '$DEST_DIR'을 확인하세요."