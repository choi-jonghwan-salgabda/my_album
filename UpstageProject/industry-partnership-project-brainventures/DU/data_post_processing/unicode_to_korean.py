import json
import os

def fix_broken_korean_paths_in_json(input_json_path, output_json_path):
    """
    JSON 파일 내의 깨진 한글 경로 (키)를 복구합니다.

    Args:
        input_json_path (str): 깨진 경로가 포함된 원본 JSON 파일 경로.
        output_json_path (str): 복구된 경로를 포함하여 저장할 새 JSON 파일 경로.
    """
    try:
        # 1. 원본 JSON 파일 읽기 (파일 자체는 UTF-8로 가정)
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"성공: 원본 JSON 파일 로드 완료 ({input_json_path})")

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {input_json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"오류: JSON 파싱 실패 - {input_json_path}. 내용: {e}")
        return
    except Exception as e:
        print(f"오류: 파일 읽기 중 예외 발생 - {e}")
        return

    corrected_data = {}
    fixed_count = 0
    skipped_count = 0

    # 2. JSON 데이터 구조 확인 및 경로 복구 시도 (딕셔너리 형태 가정)
    if not isinstance(data, dict):
        print("오류: JSON 데이터가 예상한 딕셔너리 형태가 아닙니다.")
        print("데이터 구조를 확인하고 코드 수정이 필요할 수 있습니다.")
        # 만약 리스트 형태라면 아래 주석처럼 수정 필요
        # corrected_data = []
        # for item in data:
        #     if isinstance(item, dict) and 'image_path_key' in item: # 'image_path_key'는 실제 키 이름으로 변경
        #         broken_path = item['image_path_key']
        #         # ... 복구 로직 ...
        #         item['image_path_key'] = corrected_path
        #         corrected_data.append(item)
        #     else:
        #         corrected_data.append(item) # 다른 형태의 아이템은 그대로 추가
        return

    print("경로 복구를 시작합니다...")
    for broken_path_key, value in data.items():
        try:
            # 3. 깨진 문자열 복구 시도:
            #    - 깨진 문자열을 latin-1 바이트로 인코딩
            #    - 그 바이트를 다시 utf-8로 디코딩
            corrected_path = broken_path_key.encode('latin-1').decode('utf-8')

            # 복구가 실제로 일어났는지 확인 (원본과 다른 경우)
            if corrected_path != broken_path_key:
                print(f"  [복구됨] '{broken_path_key}' -> '{corrected_path}'")
                corrected_data[corrected_path] = value
                fixed_count += 1
            else:
                # 이미 정상적인 경로이거나 다른 이유로 깨진 경우
                # print(f"  [변경없음] '{broken_path_key}' (이미 정상이거나 다른 인코딩 문제일 수 있음)")
                corrected_data[broken_path_key] = value # 원본 키 유지
                skipped_count +=1

        except UnicodeEncodeError:
            print(f"  [오류] '{broken_path_key}' 경로를 latin-1로 인코딩할 수 없습니다. 원본 유지.")
            corrected_data[broken_path_key] = value # 오류 시 원본 키 유지
            skipped_count += 1
        except UnicodeDecodeError:
            print(f"  [오류] '{broken_path_key}' 경로를 UTF-8로 디코딩할 수 없습니다 (latin-1 변환 후). 원본 유지.")
            corrected_data[broken_path_key] = value # 오류 시 원본 키 유지
            skipped_count += 1
        except Exception as e:
            print(f"  [오류] '{broken_path_key}' 처리 중 예외 발생: {e}. 원본 유지.")
            corrected_data[broken_path_key] = value # 오류 시 원본 키 유지
            skipped_count += 1


    # 4. 복구된 데이터로 새 JSON 파일 저장
    try:
        # 출력 디렉토리 생성 (필요한 경우)
        output_dir = os.path.dirname(output_json_path)
        if output_dir: # 경로에 디렉토리가 포함된 경우
            os.makedirs(output_dir, exist_ok=True)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 옵션으로 한글이 유니코드 이스케이프(\uXXXX) 없이 저장되도록 함
            json.dump(corrected_data, f, ensure_ascii=False, indent=2)

        print("\n----------------------------------------")
        print(f"성공: 복구 작업 완료!")
        print(f"  - 총 {len(data)}개의 항목 처리")
        print(f"  - {fixed_count}개의 경로 복구됨")
        print(f"  - {skipped_count}개의 경로 변경 없음 또는 오류")
        print(f"결과가 다음 파일에 저장되었습니다: {output_json_path}")
        print("----------------------------------------")

    except IOError as e:
        print(f"\n오류: 결과를 파일에 쓰는 중 오류 발생 - {output_json_path}. 내용: {e}")
    except Exception as e:
        print(f"\n오류: 파일 저장 중 예외 발생 - {e}")

# --- 사용 예시 ---
# 입력 파일 경로 (깨진 경로가 있는 JSON)
input_file = "/data/ephemeral/home/outputs/ocr_training/submissions/20250412_101529.json"
# 출력 파일 경로 (복구된 결과를 저장할 파일)
# 원본 파일을 덮어쓰지 않도록 새 이름을 지정하는 것이 좋습니다.
output_file = "/data/ephemeral/home/outputs/ocr_training/submissions/20250412_101529_korean.json"

# 함수 실행
fix_broken_korean_paths_in_json(input_file, output_file)
