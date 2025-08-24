# my_utils/photo_utils/object_utils.py

import sys
import hashlib
import json
import io
import warnings
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
import dlib
import cv2
import numpy as np
from PIL import Image, ExifTags, UnidentifiedImageError

# shared_utils 패키지에서 configger 클래스 가져오기
# shared_utils 프로젝트의 src/utility/configger.py에 configger 클래스가 있다고 가정
# object_detector.py 파일 내 임포트 구문
# 사용자 정의 유틸리티 모듈 임포트
try:
    from my_utils.config_utils.SimpleLogger import logger
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)
# 이 파일 내에서 직접적인 로깅은 최소화하고, 호출하는 쪽에서 로깅을 처리한다고 가정합니다.
# 필요시 logger 객체를 함수 인자로 받거나 전역 로거를 사용할 수 있습니다.

class ExifReadError(Exception):
    """EXIF 데이터를 읽는 중 오류가 발생했을 때 사용하는 사용자 정의 예외입니다."""
    pass


def is_image_valid_debug(
    img_path_or_stream: Union[Path, io.BytesIO]
    ) -> Tuple[bool, str]:
    """
    이미지 유효성을 검사하고, 결과와 에러 메시지를 반환합니다.

    Args:
        img_path_or_stream (Union[Path, io.BytesIO]): 이미지 경로 또는 스트림.

    Returns:
        Tuple[bool, str]: (유효성 여부, 에러 메시지 또는 빈 문자열)
    """
    path_info = f"'{img_path_or_stream.name}'" if isinstance(img_path_or_stream, Path) else "a stream object"
    try:
        with warnings.catch_warnings():
            # Pillow의 TiffImagePlugin 등에서 발생하는 'Truncated File Read' 경고를 무시합니다.
            # 이 경고는 파일이 손상되었을 수 있음을 나타내지만, 스크립트는 이미
            # UnidentifiedImageError와 같은 예외를 통해 손상된 파일을 처리하고 있습니다.
            # tqdm 진행률 표시줄이 깨지는 것을 방지하기 위해 경고를 숨깁니다.
            warnings.filterwarnings("ignore", "Truncated File Read")
            with Image.open(img_path_or_stream) as img:
                img.load()
        return True, ""
    except UnidentifiedImageError:
        return False, f"❌ 이미지 형식 인식 실패: {path_info}"
    except OSError as e:
        return False, f"❌ 이미지 디코딩 실패: {path_info} - {e}"
    except Exception as e:
        return False, f"❗ 예상치 못한 이미지 오류: {path_info} - {e}"

def calculate_sha256(
    file_path_or_stream: Union[Path, io.BytesIO]
    ) -> Tuple[Optional[str], Optional[str]]:
    """
    파일 경로 또는 메모리 스트림에 대해 SHA256 해시 값을 계산합니다.
    스트림 객체가 입력되면, 다른 처리를 위해 스트림의 위치를 다시 처음으로 되돌립니다.

    Args:
        file_path_or_stream (Union[Path, io.BytesIO]): 파일 경로 또는 io.BytesIO와 같은 스트림 객체.

    Returns:
        Optional[str]: 계산된 SHA256 해시 문자열. 오류 발생 시 None.
    """
    sha256_hash = hashlib.sha256()
    try:
        if isinstance(file_path_or_stream, Path):
            # 입력이 파일 경로인 경우
            with open(file_path_or_stream, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
        else:
            # 입력이 스트림인 경우 (e.g., io.BytesIO)
            # 다른 함수에서 스트림을 읽었을 수 있으므로, 시작 위치로 되돌립니다.
            file_path_or_stream.seek(0)
            for byte_block in iter(lambda: file_path_or_stream.read(4096), b""):
                sha256_hash.update(byte_block)
            # 다음 함수가 스트림을 다시 읽을 수 있도록 위치를 되돌려줍니다.
            file_path_or_stream.seek(0)
        return sha256_hash.hexdigest(), None
    except (IOError, OSError) as e:
        # 경로 객체일 때만 파일 경로를 로그에 남깁니다.
        path_info = f"'{file_path_or_stream}'" if isinstance(file_path_or_stream, Path) else "a stream object"
        #logger.error(f"해시 계산 중 파일/스트림 읽기 오류 ({path_info}): {e}")
        return None, f"해시 계산 중 파일/스트림 읽기 오류가 발생했습니다 ({path_info}): {e}"

def get_exif_date_taken(
    image_path_or_stream: Union[Path, io.BytesIO]
    ) -> Tuple[Optional[str], Optional[str]]: # (날짜 문자열, 오류 메시지)
    """
    이미지 파일 경로 또는 스트림에서 EXIF 촬영 날짜를 읽어 반환합니다.
    'DateTimeOriginal', 'DateTimeDigitized', 'DateTime' 태그를 순서대로 확인합니다.
    EXIF 데이터가 없거나 날짜 정보가 없으면 (None, 메시지) 튜플을 반환합니다.
    """

    # path_info를 try 블록 시작 전에 안전하게 정의
    path_info = "알 수 없는 입력 객체" # 기본값 설정

    if isinstance(image_path_or_stream, Path):
        path_info = f"'{image_path_or_stream.name}'"
    elif isinstance(image_path_or_stream, io.BytesIO):
        path_info = "스트림 객체"
        try:
            # io.BytesIO의 경우 스트림의 위치를 처음으로 되돌립니다.
            # Image.open이 내부적으로 처리하는 경우가 많지만, 다른 곳에서 이미 읽은 스트림일 수 있기 때문입니다.
            image_path_or_stream.seek(0)
        except Exception:
            # seek이 불가능하거나 이미 닫힌 스트림일 수 있으므로 예외 처리 (로그 필요시 추가)
            pass

    try:
        # Image.open()은 Path 객체와 스트림 객체 모두 처리 가능
        with Image.open(image_path_or_stream) as img:
            # getexif()는 EXIF 데이터가 없으면 None을 반환하므로 _getexif() 대신 사용 권장
            exif_data = img.getexif()

            if not exif_data:
                # EXIF 데이터 자체가 없는 경우
                return None, f"EXIF 데이터가 없음 ({path_info})"

            # 확인할 날짜 관련 EXIF 태그 ID 목록 (우선순위 순)
            # 36867: DateTimeOriginal (촬영일)
            # 36868: DateTimeDigitized (디지털화된 날짜)
            # 306: DateTime (파일 수정일)
            date_tags_to_check = [36867, 36868, 306]
            
            date_taken_str = None
            found_tag_id = None
            for tag_id in date_tags_to_check:
                date_taken_str = exif_data.get(tag_id)
                if date_taken_str:
                    found_tag_id = tag_id
                    break # 첫 번째로 찾은 날짜 정보를 사용

            if date_taken_str:
                # 값은 'YYYY:MM:DD HH:MM:SS' 형식이므로, 날짜 부분만 추출하여 포맷 변경
                try:
                    formatted_date = date_taken_str.split(' ')[0].replace(':', '-')
                    # 필요에 따라 날짜 문자열의 유효성 검사 추가 (예: re.match 또는 datetime.strptime)
                    return formatted_date, None # 날짜와 오류 없음(None) 반환
                except (IndexError, TypeError): # split 결과 오류 또는 예상치 못한 타입
                    return None, f"EXIF 날짜 형식 오류 (Tag: {found_tag_id}, Path: {path_info}): '{date_taken_str}'"
            else:
                # 어떤 날짜 태그도 찾지 못한 경우
                return None, f"EXIF에서 유효한 날짜 태그(36867, 36868, 306)를 찾을 수 없음 ({path_info})"

    # 이 함수 내에서 발생할 수 있는 주요 예외들을 포괄적으로 잡습니다.
    # ExifReadError는 이 함수가 자체적으로 raise하는 부분이 없다면 except에 넣을 필요는 없습니다.
    # FileNotFoundError, IOError, OSError는 파일 시스템 관련 오류.
    except (UnidentifiedImageError, FileNotFoundError, IOError, OSError, Exception) as e:
        # 모든 예외를 포착하여 오류 메시지와 함께 반환합니다.
        return None, f"EXIF 날짜를 읽는 중 오류 발생 ({path_info}): {e}"

def _get_string_key_from_config(config_dict: Optional[Dict[str, Any]], key_name: str, default_value: str, logger_instance) -> str:
    """
    설정 딕셔너리에서 문자열 값을 가져오며, 문자열이 아닐 경우 오류를 로깅하고 기본값을 반환합니다.
    config_dict가 None인 경우에도 안전하게 기본값을 반환합니다.
    """
    if config_dict is None:
        logger_instance.warning(
            f"설정 딕셔너리(config_dict)가 None입니다. 키 '{key_name}'에 대해 기본값 '{default_value}'을(를) 사용합니다."
        )
        return default_value

    value = config_dict.get(key_name, default_value)
    if not isinstance(value, str):
        logger_instance.error(
            f"설정 오류 ('json_keys'): 키 '{key_name}'에 대해 문자열이 예상되었으나, "
            f"타입 {type(value)} (값: '{value}')이(가) 반환되었습니다. 기본값 '{default_value}'을(를) 사용합니다."
        )
        return default_value
    return value

def rotate_image_if_needed(image_path_str: str) -> bool:
    """
    이미지 파일 경로를 받아 EXIF 정보에 따라 이미지를 회전시키고 원본에 덮어씁니다.

    Args:
        image_path_str (str): 이미지 파일의 경로 문자열.

    Returns:
        bool: 이미지가 성공적으로 회전되었거나 회전이 필요 없었으면 True,
              오류 발생 또는 파일을 열 수 없으면 False.
    """
    try:
        img = Image.open(image_path_str)
        # 이미지 객체에서 EXIF 데이터를 가져옵니다. 없을 경우 None을 반환합니다.
        exif = img._getexif()

        if exif is None:
            logger.debug(f"이미지 '{image_path_str}'에 EXIF 정보가 없습니다. 회전을 건너뜁니다.")
            img.close()
            return True # EXIF 정보가 없으면 회전 불필요, 성공으로 간주

        orientation_tag_id = None
        # EXIF 태그 이름 'Orientation'에 해당하는 숫자 ID를 찾습니다.
        for tag_id, tag_name in ExifTags.TAGS.items():
            if tag_name == 'Orientation':
                orientation_tag_id = tag_id
                break
        
        if orientation_tag_id is None or orientation_tag_id not in exif:
            logger.debug(f"이미지 '{image_path_str}'에 Orientation 태그가 없습니다. 회전을 건너뜁니다.")
            img.close()
            return True # Orientation 태그가 없으면 회전 불필요, 성공으로 간주

        orientation_value = exif[orientation_tag_id]
        logger.debug(f"이미지 '{image_path_str}'의 Orientation 값: {orientation_value}")

        rotated_img = None
        if orientation_value == 1: # Normal (정상)
            logger.debug(f"이미지 '{image_path_str}'가 이미 정상 방향입니다. 회전하지 않습니다.")
            img.close()
            return True
        elif orientation_value == 2: # Flipped horizontally (좌우 반전)
            rotated_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation_value == 3: # Rotated 180 degrees (180도 회전)
            rotated_img = img.rotate(180)
        elif orientation_value == 4: # Flipped vertically (상하 반전)
            rotated_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation_value == 5: # Rotated 90 degrees CCW and flipped vertically (반시계 90도 회전 후 상하 반전)
            rotated_img = img.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation_value == 6: # Rotated 90 degrees CW (시계 방향 90도 회전)
            rotated_img = img.rotate(-90, expand=True)
        elif orientation_value == 7: # Rotated 90 degrees CW and flipped vertically (시계 방향 90도 회전 후 상하 반전)
            rotated_img = img.rotate(-90, expand=True).transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation_value == 8: # Rotated 90 degrees CCW (반시계 방향 90도 회전)
            rotated_img = img.rotate(90, expand=True)
        else:
            logger.debug(f"이미지 '{image_path_str}'의 Orientation 값({orientation_value})을 알 수 없습니다. 회전하지 않습니다.")
            img.close()
            return True # 알 수 없는 값이면 회전하지 않음, 성공으로 간주

        # 회전된 이미지가 있다면 원본 파일에 덮어쓰기
        if rotated_img:
            # 원본 EXIF 정보를 유지하려면 img.info.get('exif')를 사용해야 하지만,
            # PIL에서 회전 시 EXIF의 Orientation 값을 자동으로 1로 업데이트하지 않을 수 있습니다.
            # 따라서, 여기서는 EXIF 정보를 명시적으로 전달하지 않고 저장하여,
            # 회전 후 Orientation 정보가 문제를 일으키지 않도록 합니다.
            # 필요하다면, 저장 후 EXIF를 수정하는 로직을 추가할 수 있습니다.
            rotated_img.save(image_path_str)
            rotated_img.close()
            logger.debug(f"이미지 '{image_path_str}'를 EXIF 정보에 따라 회전하고 저장했습니다.")
        
        img.close() # 원본 이미지 객체 닫기
        return True

    except FileNotFoundError:
        logger.error(f"이미지 파일 '{image_path_str}'을(를) 찾을 수 없습니다.")
        return False
    except Exception as e:
        logger.error(f"이미지 '{image_path_str}' 회전 중 오류 발생: {e}")
        if 'img' in locals() and img: # img 객체가 열려있으면 닫기
            img.close() # type: ignore
        if 'rotated_img' in locals() and rotated_img: # rotated_img 객체가 있으면 닫기
            rotated_img.close()
        return False

def crop_image_from_path_to_buffer(
    image_path: Union[str, Path],
    bbox: List[Union[int, float]],
    output_format: str = '.jpg'
) -> Optional[io.BytesIO]:
    """
    이미지 경로와 바운딩 박스를 받아 이미지를 자르고 메모리 버퍼로 반환합니다.

    Args:
        image_path (Union[str, Path]): 원본 이미지 파일의 경로.
        bbox (List[Union[int, float]]): [x1, y1, x2, y2] 형식의 바운딩 박스.
        output_format (str): 출력 이미지 형식 (예: '.jpg', '.png').

    Returns:
        Optional[io.BytesIO]: 잘린 이미지가 담긴 BytesIO 버퍼. 실패 시 None.
    """
    try:
        # 1. 이미지 경로 확인 및 로드
        p = Path(image_path)
        if not p.exists():
            logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return None
        
        # OpenCV는 유니코드 경로 문제를 일으킬 수 있으므로, numpy를 통해 읽습니다.
        img_array = np.fromfile(str(p), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            logger.error(f"이미지를 로드할 수 없습니다: {image_path}")
            return None

        # 2. 바운딩 박스 좌표를 정수로 변환
        x1, y1, x2, y2 = map(int, bbox)

        # 3. 이미지 자르기
        # 좌표가 이미지 경계를 벗어나지 않도록 조정
        h, w, _ = image.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x1 >= x2 or y1 >= y2:
            logger.warning(f"유효하지 않은 바운딩 박스 좌표로 인해 빈 이미지가 생성됩니다: {bbox}")
            return None

        cropped_image = image[y1:y2, x1:x2]

        if cropped_image.size == 0:
            logger.warning(f"자르기 결과 이미지가 비어있습니다. Bbox: {bbox}, Image shape: {image.shape}")
            return None

        # 4. 메모리 버퍼로 인코딩
        is_success, buffer = cv2.imencode(output_format, cropped_image)
        if not is_success:
            logger.error(f"이미지를 '{output_format}' 형식으로 인코딩하는 데 실패했습니다.")
            return None

        return io.BytesIO(buffer)

    except Exception as e:
        logger.error(f"이미지 자르기 중 예외 발생 (경로: {image_path}, bbox: {bbox}): {e}", exc_info=True)
        return None

def extract_face_features_from_face_crop(
    face_img: np.ndarray,
    shape_predictor_path: str,
    face_rec_model_path: str
    ) -> Optional[np.ndarray]:
    """
    잘린 얼굴 이미지에서 dlib을 사용해 얼굴 특징 벡터(128차원)를 추출합니다.
    
    Args:
        face_img (np.ndarray): 얼굴 영역만 포함된 이미지 (BGR 또는 RGB 가능)
        shape_predictor_path (str): 랜드마크 모델 경로 (.dat)
        face_rec_model_path (str): 얼굴 인식 모델 경로 (.dat)
    
    Returns:
        Optional[np.ndarray]: 128차원 얼굴 벡터. 실패 시 None 반환.
    """

    if face_img is None or not isinstance(face_img, np.ndarray):
        print("[오류] face_img가 유효하지 않습니다.")
        return None

    try:
        # 모델 로딩
        shape_predictor = dlib.shape_predictor(shape_predictor_path)
        face_recognizer = dlib.face_recognition_model_v1(face_rec_model_path)

        # dlib은 RGB 이미지 사용
        if face_img.shape[2] == 3:
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = face_img  # 이미 RGB일 경우 그대로

        # 얼굴 위치를 이미지 전체로 설정
        h, w = rgb_img.shape[:2]
        rect = dlib.rectangle(left=0, top=0, right=w, bottom=h)

        shape = shape_predictor(rgb_img, rect)
        face_descriptor = face_recognizer.compute_face_descriptor(rgb_img, shape)

        return np.array(face_descriptor, dtype=np.float32)

    except Exception as e:
        print(f"[예외] 얼굴 특징 추출 중 오류: {e}")
        return None


# def compute_sha256_from_file(file_path: Path) -> Optional[str]:
#     """
#     주어진 파일 경로에 대해 SHA256 해시 값을 계산하여 문자열로 반환합니다.

#     Args:
#         file_path (Path): 해시를 계산할 파일의 경로.

#     Returns:
#         str | None: 계산된 SHA256 해시 값 (16진수 문자열). 파일 읽기 오류 시 None을 반환합니다.
#     """
#     sha256_hash = hashlib.sha256()
#     try:
#         # 파일을 바이너리 읽기 모드('rb')로 열어서 처리
#         with open(file_path, "rb") as f:
#             for byte_block in iter(lambda: f.read(4096), b""):
#                 sha256_hash.update(byte_block) # 파일 내용을 조금씩 읽어 해시 업데이트
#         return sha256_hash.hexdigest()
#     except IOError as e:
#         logger.error(f"파일 읽기 오류 {file_path}: {e}")
#         return None

# def compute_sha256_from_memory(image_data: np.ndarray) -> Optional[str]:
#     """
#     이미지 데이터(NumPy 배열)의 SHA256 해시 값을 계산합니다.

#     Args:
#         image_data (np.ndarray): OpenCV 등으로 읽은 이미지 데이터 (NumPy 배열).

#     Returns:
#         Optional[str]: 계산된 SHA256 해시 문자열. 오류 발생 시 None을 반환합니다.
#     """
#     if not isinstance(image_data, np.ndarray):
#         logger.error("입력된 이미지 데이터가 NumPy 배열이 아닙니다.")
#         return None
#     try:
#         # NumPy 배열을 바이트 문자열로 변환합니다.
#         # 이미지의 경우, 내용이 변경되지 않도록 C-contiguous array로 만드는 것이 좋습니다.
#         image_bytes = image_data.tobytes()
        
#         hasher = hashlib.sha256()
#         hasher.update(image_bytes)
#         return hasher.hexdigest()
#     except Exception as e:
#         logger.error(f"SHA256 해시 계산 중 오류 발생: {e}")
#         return None
