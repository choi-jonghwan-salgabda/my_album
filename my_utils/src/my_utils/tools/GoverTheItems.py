# src/GoverTheItems.py

# 표준 라이브러리 임포트
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Set
import traceback

# 서드파티 라이브러리 임포트
from tqdm import tqdm

# 사용자 정의 유틸리티 모듈 임포트
try:
    from my_utils.config_utils.SimpleLogger import logger
    from my_utils.config_utils.arg_utils import get_argument
    from my_utils.config_utils.configger import configger
    from my_utils.object_utils.photo_utils import rotate_image_if_needed, JsonConfigHandler, _get_string_key_from_config # JsonConfigHandler 임포트
    from my_utils.config_utils.file_utils import calculate_sha256, safe_move, safe_copy, DiskFullError, get_exif_date_taken, get_original_filename
    from my_utils.config_utils.display_utils import calc_digit_number, get_display_width, truncate_string, visual_length
except ImportError as e:
    # 실제 발생한 예외 e를 출력하여 원인 파악
    print(f"모듈 임포트 중 오류 발생: {e}")
    print(f"자세한 오류 정보:")
    import traceback
    traceback.print_exc() # 전체 트레이스백 출력 (개발 단계에서 유용)
    sys.exit(1)

# --- 전역 변수: 통계용 변수 사용을 위해해 ---
DEFAULT_STATUS_TEMPLATE  = {
    "total_input_found":         {"value": 0,  "msg": "총 입력 파일 수 (지원 확장자 기준)"}, # 찾은 총 입력 파일 수
    "error_input_file_read":        {"value": 0,  "msg": "입력 파일 읽기 오류 수"}, # 입력 파일 읽기 실패 수
    "req_process_count":         {"value": 0,  "msg": "총 처리 시도 파일 수"}, # 처리 시도한 총 파일 수
    "error_extension":   {"value": 0,  "msg": "지원되지 않는 확장자로 건너뛴 파일 수"}, # 확장자 오류로 건너뛴 파일 수
    "error_image_rotation":          {"value": 0,  "msg": "이미지 회전중 오류 발생 파일 수"}, # 이미지 회전 중 오류 발생 수
    "error_target_file_get":        {"value": 0,  "msg": "처리대상(image or json) 파일 읽기 오류 수"}, # 대상 파일(이미지 또는 JSON) 읽기 오류 수
    "error_input_file_process":        {"value": 0,  "msg": "입력파일 처리 중 오류 수"}, # 입력 파일 처리 중 발생한 오류 수
    "request_embedding_processing":        {"value": 0,  "msg": "임베딩 처리 요청 수"}, # 임베딩 처리 요청된 횟수
    "error_embedding_empty_target":        {"value": 0,  "msg": "임베딩 오류 - 처리 대상 없음"}, # 임베딩 대상이 없는 경우의 오류 수
    "error_embedding_array_empty":        {"value": 0,  "msg": "임베딩 오류 - 빈 임베딩 배열"}, # 임베딩 배열이 비어있는 경우의 오류 수
    "error_embedding_config_missing":        {"value": 0,  "msg": "임베딩 오류 - 설정 값 없음"}, # 임베딩 관련 설정 값이 누락된 경우
    "error_embedding_config_mismatch":        {"value": 0,  "msg": "임베딩 오류 - 설정 값과 차원 불일치"}, # 설정된 임베딩 차원과 실제 데이터 차원이 다른 경우
    "error_embedding_data_shape_mismatch": {"value": 0,  "msg": "임베딩 오류 - 데이터 규격(shape) 불일치"}, # 임베딩 데이터의 형태(shape)가 예상과 다른 경우
    "error_embedding_training_data_missing": {"value": 0,  "msg": "임베딩 오류 - 학습 데이터 없음 (IVF 계열)"}, # IVF 계열 인덱스 학습 데이터가 없는 경우
    "error_embedding_dimension_m_mismatch":  {"value": 0,  "msg": "임베딩 오류 - 차원이 M의 배수 아님 (IVFPQ)"}, # IVFPQ 인덱스에서 차원이 M의 배수가 아닌 경우
    "error_embedding_object_creation_failed": {"value": 0,  "msg": "임베딩 오류 - FAISS 인덱스 객체 생성 실패"}, # FAISS 인덱스 객체 생성 실패 수
    "error_embedding_index_read_failed":      {"value": 0,  "msg": "임베딩 오류 - FAISS 인덱스 파일 읽기 실패"}, # FAISS 인덱스 파일 읽기 실패 수
    "error_embedding_general":           {"value": 0,  "msg": "임베딩 추출/처리 중 일반 오류 수"}, # 임베딩 관련 일반 오류 수
    "request_save_index":        {"value": 0,  "msg": "인덱스 저장 요청 수"}, # 인덱스 저장 요청 횟수
    "total_object_count":      {"value": 0,  "msg": "검출된 총 객체 수"}, # 검출된 전체 객체 수
    "files_with_detected_objects":     {"value": 0,  "msg": "객체가 1개 이상 검출된 파일 수"}, # 하나 이상의 객체가 검출된 파일 수
    "get_object_crop":           {"value": 0,  "msg": "객체가 검출된 객체 수"}, # 성공적으로 크롭된 객체 수
    "error_object_crop":           {"value": 0,  "msg": "객체 크롭(crop) 처리 오류 수"}, # 객체 크롭 중 오류 발생 수
    "error_object_bbox_format":           {"value": 0,  "msg": "객체 바운딩 박스 형식 오류 수"}, # 객체 바운딩 박스 형식이 잘못된 경우
    "error_object_bbox_count_mismatch":    {"value": 0,  "msg": "객체 바운딩 박스 개수 불일치 오류 수"}, # 바운딩 박스 개수가 예상과 다른 경우
    "error_object_bbox_position":           {"value": 0,  "msg": "객체 바운딩 박스 좌표 오류 수"}, # 바운딩 박스 좌표값이 잘못된 경우
    "undetection_object":   {"value": 0,  "msg": "객체가 검출되지 않은 파일 수"}, # 객체가 검출되지 않은 파일 수
    "error_copied_input_file": {"value": 0, "msg": "오류 발생 입력파일 보관 실패 수"}, # 오류 발생 시 입력 파일 백업 실패 수
    "detect_faces_in_object":    {"value": 0,  "msg": "객체에서 얼굴검출을 성공한 수"}, # 객체 내에서 얼굴 검출 성공 수
    "error_faces_in_object":    {"value": 0,  "msg": "객체에서 얼굴검출 실패 수"}, # 객체 내에서 얼굴 검출 실패 수
    "unmatched_object_number":   {"value": 0,  "msg": "검출 대상 object수와 검출한 object의 수가 다른 파일수"}, # 예상 객체 수와 실제 검출된 객체 수가 다른 파일 수
    "total_output_files":        {"value": 0,  "msg": "총 출력 파일수"}, # 생성된 총 출력 파일 수
    "read_input_files_success":          {"value": 0,  "msg": "읽은 입력 파일 수 (detect_object 기준)"}, # (detect_object 기준) 성공적으로 읽은 입력 파일 수
    "read_input_files_error":          {"value": 0,  "msg": "읽은 입력 파일 수 (detect_object 기준)"}, # (detect_object 기준) 읽기 실패한 입력 파일 수
    "files_json_load":           {"value": 0,  "msg": "JSON 정보 읽은 파일 수"}, # JSON 정보를 성공적으로 읽은 파일 수
    "files_json_update":         {"value": 0,  "msg": "JSON 파일 덧씌우기 성공 파일 수"}, # JSON 파일 업데이트 성공 수
    "error_json_update":         {"value": 0,  "msg": "JSON 파일 덧씌우기 실패 수"}, # JSON 파일 업데이트 실패 수
    "get_image_path_in_json":    {"value": 0,  "msg": "IMAGE 파일 경로 가져온 파일 수"}, # JSON에서 이미지 경로를 성공적으로 가져온 파일 수
    "undetected_image_copied_success": {"value": 0, "msg": "미검출 이미지 복사 성공 수"}, # 미검출 이미지를 성공적으로 복사한 수
    "undetected_image_copied_error": {"value": 0, "msg": "미검출 이미지 복사 실패 수"}, # 미검출 이미지 복사 실패 수
    "undetection_object_file":   {"value": 0,  "msg": "객체가 검출되지 않은 파일 수"}, # (중복된 의미, 위 undetection_object와 통합 가능성 검토)
    "num_detected_objects":      {"value": 0,  "msg": "검출된 총 객체 수"}, # (중복된 의미, 위 total_object_count와 통합 가능성 검토)
    "files_object_crop":         {"value": 0,  "msg": "객체가 있는 파일 수"}, # (중복된 의미, 위 files_with_detected_objects와 통합 가능성 검토)
    "error_faild_file_backup":        {"value": 0,  "msg": "읽을때 오류가 난 입력 파일을 보관하는데 오류발생 수"}, # 읽기 오류난 입력 파일 백업 실패 수
    "files_skipped_extension":   {"value": 0,  "msg": "지원되지 않는 확장자로 건너뛴 파일 수"}, # (중복된 의미, 위 error_extension과 통합 가능성 검토)
    "files_processed_for_log":   {"value": 0,  "msg": "로그용으로 처리 시도한 파일 수"}, # 최종 통계에는 보통 표시되지 않음
    "files_processed_main_error":{"value": 0,  "msg": "메인 루프에서 처리 중 오류 발생 파일 수"} # 메인 처리 루프에서 오류가 발생한 파일 수
} 


def run_main(cfg: configger):
    # 1. 입력 JSON 파일 목록 가져오기 및 처리

    # 1.1. JSON 파일 목록 가져오기 및 총 개수 세기 (메모리 효율적)
    logger.info(f"'{input_dir}' 디렉토리에서 JSON 파일 탐색 시작...")

    # glob 결과 이터레이터를 생성 (개수 세기용)
    # 이 이터레이터는 아래 sum() 함수에 의해 소모됩니다.
    json_file_iterator_for_counting = input_dir.glob("**/*.json")

    # 1.2. is_file() 필터링을 적용하면서 개수를 셉니다.
    # sum(1 for ...) 구문은 이터레이터를 순회하며 각 요소에 대해 1을 더하여 총 개수를 계산합니다.
    # 이 과정에서 전체 경로를 메모리에 리스트로 저장하지 않습니다.
    total_input_found = sum(1 for p in json_file_iterator_for_counting if p.is_file())

    if total_input_found == 0:
        logger.warning(f"1.2. '{input_dir}' 디렉토리에서 인덱싱할 JSON 파일을 찾을 수 없습니다.")
        logger.info("✅ 최종 통계:")
        logger.info(f"   - 탐색된 JSON 파일 총 개수: {total_input_found}")
        logger.info(f"   - 인덱싱된 총 얼굴 개수: 0")
        logger.info(f"{Path(__file__).name} 정상 종료 (처리할 파일 없음).")
        sys.exit(0)
    status["total_input_found"]["value"] = total_input_found

    logger.info(f'✅ 인덱싱할 JSON 파일 {status["total_input_found"]["value"]}개 발견.')

    digit_width = calc_digit_number(total_input_found) # 로그 출력 시 숫자 너비 계산
    # 1.2. 모든 파일에서 얼굴 정보 누적 (메모리 효율적인 파일 순회)
    # glob 결과를 다시 생성 (실제 처리용)
    json_file_iterator_for_processing = input_dir.glob("**/*.json")

    all_embeddings: List[np.ndarray] = [] # 모든 파일의 임베딩을 누적할 리스트
    all_metadatas: List[Dict[str, Any]] = [] # 모든 파일의 메타데이터를 누적할 리스트
    total_faces_processed = 0 # 성공적으로 처리된 총 얼굴 개수
    # batch_index = 0  # 배치 처리 시 사용 (현재는 모든 데이터 수집 후 일괄 처리)
    # total_faces_failed 변수는 필요에 따라 추가

     # 1.3. JSON 파일 내 얼굴 정보 수집 시작...
    logger.info("JSON 파일 내 얼굴 정보 수집 시작...")
    # tqdm을 사용하여 진행률 바 추가
    with tqdm(total=total_input_found, desc="JSON 파일 처리 중", unit="파일", dynamic_ncols=True) as pbar:
        for json_file_path in json_file_iterator_for_processing: # 이터레이터를 순회
            if not json_file_path.is_file():
                status["error_input_file_read"]["value"] += 1 # 파일이 아닌 경우 오류 카운트
                pbar.update(1)
                continue

            status["req_process_count"]["value"] += 1
            # tqdm이 있으므로, 상세 로그는 debug 레벨로 유지합니다.
            logger.debug(f"1.3. [{status['req_process_count']['value']:>{digit_width}}/{status['total_input_found']['value']}] JSON 파일 처리 중: {json_file_path.name}")

            # 2. 현재 파일(json_file_path)에서 얼굴의 임베딩과 메타데이터 가져와서 indexing
            # json_key_config_data를 전달하도록 수정
            embeddings_from_file, metadatas_from_file = get_all_face_data_from_json_alone(
                cfg,
                json_file_path,
                json_handler           # JsonConfigHandler 인스턴스 전달
                )

            if embeddings_from_file: # 파일에서 유효한 임베딩이 추출된 경우
                all_embeddings.extend(embeddings_from_file) # 누적 리스트에 추가
                all_metadatas.extend(metadatas_from_file) # 누적 리스트에 추가
                total_faces_processed += len(embeddings_from_file) # 총 얼굴 개수 누적
                # 진행률 바에 표시될 수 있도록 info 레벨 로그는 간결하게 유지하거나 debug로 변경
                logger.debug(f"  '{json_file_path.name}' 파일에서 얼굴 {len(embeddings_from_file)}개 정보 추출 완료. (누적 {total_faces_processed}개)")

            else:
                # 파일 내에 유효한 얼굴 정보가 없거나 오류 발생 시
                status["error_embedding_general"]["value"] += 1 # 보다 일반적인 에러 카운터 사용
                logger.warning(f"  '{json_file_path.name}' 파일에서 유효한 얼굴 정보를 추출하지 못했습니다.")
            pbar.update(1)

    # 3. 모든 파일 처리 후, 수집된 전체 임베딩으로 FAISS 인덱스 구축
    if not all_embeddings:
        logger.warning("3. 수집된 얼굴 임베딩이 없어 FAISS 인덱스를 생성하지 않습니다.")
        status["error_embedding_empty_target"]["value"] +=1
    else:
        logger.info(f"총 {len(all_embeddings)}개의 얼굴 임베딩을 사용하여 FAISS 인덱스 구축 및 저장 시작...")
        build_and_save_index_alone(all_embeddings, all_metadatas, cfg)

    # 9. 모든 이미지 처리 완료 또는 중단 후 자원 해제
    # 9-1. 통계 결과 출력 ---
    logger.info("--- JSON 파일 처리 및 인덱싱 통계 ---") # 헤더 메시지 변경
    # 통계 메시지 중 가장 긴 것을 기준으로 출력 너비 조절 (visual_length 사용)
    max_visual_msg_len = 0
    if DEFAULT_STATUS_TEMPLATE: # DEFAULT_STATUS_TEMPLATE이 비어있지 않은 경우에만 실행
        # status 딕셔너리에 있는 키들 중 DEFAULT_STATUS_TEMPLATE에도 있는 키의 메시지만 고려
        valid_msgs = [
            DEFAULT_STATUS_TEMPLATE[key]["msg"]
            for key in status.keys()
            if key in DEFAULT_STATUS_TEMPLATE and "msg" in DEFAULT_STATUS_TEMPLATE[key]
        ]
        if valid_msgs: # 유효한 메시지가 있을 경우에만 max 계산
            max_visual_msg_len = max(visual_length(msg) for msg in valid_msgs)

    # 통계 출력 시 사용할 숫자 너비 계산 (가장 큰 값 기준)
    # status 딕셔너리의 모든 value들을 가져와서 그 중 최대값을 기준으로 너비 계산
    all_values = [data["value"] for data in status.values() if isinstance(data.get("value"), int)]
    max_val_for_width = max(all_values) if all_values else 0 # 모든 값이 0이거나 없을 경우 대비

    # 로그 출력 시 숫자 너비는 위에서 total_input_found 기준으로 계산된 digit_width를 사용합니다.
    # 여기서는 run_main 초반에 계산된 digit_width를 그대로 사용합니다.

    fill_char = '-' # 채움 문자 변경
    logger.info("--- 이미지 파일 처리 통계 ---")
    for key, data in status.items():
        # DEFAULT_STATUS_TEMPLATE에 해당 키가 있을 경우에만 메시지를 가져오고, 없으면 키 이름을 사용
        msg = DEFAULT_STATUS_TEMPLATE.get(key, {}).get("msg", key)
        value = data["value"]
        # f-string의 기본 정렬은 문자 개수 기준이므로, visual_length에 맞춰 수동으로 패딩 추가
        padding_spaces = max(0, max_visual_msg_len - visual_length(msg))
        logger.info(f"{msg}{fill_char * padding_spaces} : {value:>{digit_width}}") # 기존 digit_width 사용
    logger.info("------------------------------------") # 구분선 길이 조정
    # --- 통계 결과 출력 끝 ---

if __name__ == "__main__":
    # 0. 애플리케이션 시작 및 인자 파싱
    logger.info(f"애플리케이션 시작")
    parsed_args = get_argument()

    # 1. 로거 설정
    script_name = Path(__file__).stem # 로깅 및 파일 이름에 사용할 스크립트 이름
    try:
        # 1. 애플리케이션 로거 설정
        date_str = datetime.now().strftime("%y%m%d")
        log_file_name = f"{script_name}_{date_str}.log"
        full_log_path = Path(parsed_args.log_dir) / log_file_name
        logger.setup(
            logger_path=str(full_log_path),
            min_level=parsed_args.log_level,
            include_function_name=True,
            pretty_print=True
        )
        logger.info(f"애플리케이션({script_name}) 시작")
        logger.info(f"명령줄 인자로 결정된 경로: {vars(parsed_args)}")
    except Exception as e:
        print(f"치명적 오류: 로거 설정 중 오류 발생 - {e}", file=sys.stderr)
        sys.exit(1)

    # 2. 설정(configger) 초기화
    # configger는 위에서 설정된 로거를 내부적으로 사용할 수 있습니다.
    logger.info(f"Configger 초기화 시도: root_dir='{parsed_args.root_dir}', config_path='{parsed_args.config_path}'")

    try:
        cfg_object = configger(root_dir=parsed_args.root_dir, config_path=parsed_args.config_path)
        logger.info(f"Configger 초기화 완료")
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}")
        sys.exit(1) # 설정 로드 실패 시 종료

    # 3. 메인 로직 실행
    logger.info(f"메인 처리 로직 시작...")
    try:
        run_main(cfg_object) # 설정 객체를 run_main 함수에 전달
    except KeyError as e:
        logger.critical(f"설정 파일에서 필수 경로 키를 찾을 수 없습니다: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"설정 파일에서 경로 정보 처리 중 오류 발생: {e}")
        sys.exit(1)
    finally:
        # 애플리케이션 종료 시 로거 정리 (필요시)
        logger.info(f"{script_name} 애플리케이션 종료")
        logger.shutdown()
        exit(0)

    # 최종 print 문은 로깅으로 대체되었으므로 제거하거나 주석 처리합니다.
    # print("모든 JSON 파일 경로 처리가 완료되었습니다.")
