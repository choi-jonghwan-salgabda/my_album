# my_utils/photo_utils/object_utils.py

import sys
import hashlib
import json
import io
from pathlib import Path
from typing import List, Dict, Any, Union, Optional # Optional 추가
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

def calculate_sha256(file_path_or_stream: Union[Path, io.BytesIO]) -> Optional[str]:
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
        return sha256_hash.hexdigest()
    except (IOError, OSError) as e:
        # 경로 객체일 때만 파일 경로를 로그에 남깁니다.
        path_info = f"'{file_path_or_stream}'" if isinstance(file_path_or_stream, Path) else "a stream object"
        logger.error(f"해시 계산 중 파일/스트림 읽기 오류 ({path_info}): {e}")
        return None


def get_exif_date_taken(image_path_or_stream: Union[Path, io.BytesIO]) -> Optional[str]:
    """
    이미지 파일 경로 또는 스트림에서 EXIF 'DateTimeOriginal' 태그를 읽어 촬영 날짜를 반환합니다.
    EXIF 데이터가 없거나 날짜 정보가 없으면 None을 반환합니다.
    """
    try:
        with Image.open(image_path_or_stream) as img:
            # exif 데이터가 없는 경우도 있으므로 _getexif() 사용
            exif_data = img._getexif()
            if not exif_data:
                return None

            # 'DateTimeOriginal' 태그 (Tag ID 36867)를 찾습니다.
            date_taken_str = exif_data.get(36867)
            if date_taken_str:
                # 값은 'YYYY:MM:DD HH:MM:SS' 형식이므로, 날짜 부분만 추출하여 포맷 변경
                return date_taken_str.split(' ')[0].replace(':', '-')
    except Exception as e:
        # Pillow가 처리할 수 없는 파일이거나 다른 EXIF 관련 오류 발생 시 예외를 발생시킵니다.
        path_info = f"'{image_path_or_stream.name}'" if isinstance(image_path_or_stream, Path) else "a stream object"
        raise ExifReadError(f"EXIF 날짜를 읽는 중 오류 발생 ({path_info}): {e}") from e
    return None

def is_image_valid_debug(img_path_or_stream: Union[Path, io.BytesIO]) -> bool:
    """
    이미지 파일 경로 또는 스트림의 유효성을 검사하고, 구체적인 에러 메시지를 로그로 남깁니다.

    Args:
        img_path_or_stream (Union[Path, io.BytesIO]): 이미지 경로 또는 스트림.

    Returns:
        bool: 유효하면 True, 오류 발생 시 False
    """
    path_info = f"'{img_path_or_stream.name}'" if isinstance(img_path_or_stream, Path) else "a stream object"
    try:
        with Image.open(img_path_or_stream) as img:
            img.load()  # 이미지 전체 디코딩 시도
        logger.debug(f"✅ 유효한 이미지: {path_info}")
        return True
    except UnidentifiedImageError:
        logger.warning(f"❌ 이미지 형식 인식 실패: {path_info}")
        return False
    except OSError as e:
        logger.warning(f"❌ 이미지 디코딩 실패: {path_info} - {e}")
        return False
    except Exception as e:
        logger.error(f"❗ 예상치 못한 이미지 오류: {path_info} - {e}")
        return False


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
            logger.warning(f"이미지 '{image_path_str}'의 Orientation 값({orientation_value})을 알 수 없습니다. 회전하지 않습니다.")
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
            logger.info(f"이미지 '{image_path_str}'를 EXIF 정보에 따라 회전하고 저장했습니다.")
        
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


def compute_sha256_from_file(file_path: Path) -> Optional[str]:
    """
    주어진 파일 경로에 대해 SHA256 해시 값을 계산하여 문자열로 반환합니다.

    Args:
        file_path (Path): 해시를 계산할 파일의 경로.

    Returns:
        str | None: 계산된 SHA256 해시 값 (16진수 문자열). 파일 읽기 오류 시 None을 반환합니다.
    """
    sha256_hash = hashlib.sha256()
    try:
        # 파일을 바이너리 읽기 모드('rb')로 열어서 처리
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block) # 파일 내용을 조금씩 읽어 해시 업데이트
        return sha256_hash.hexdigest()
    except IOError as e:
        logger.error(f"파일 읽기 오류 {file_path}: {e}")
        return None

def compute_sha256_from_memory(image_data: np.ndarray) -> Optional[str]:
    """
    이미지 데이터(NumPy 배열)의 SHA256 해시 값을 계산합니다.

    Args:
        image_data (np.ndarray): OpenCV 등으로 읽은 이미지 데이터 (NumPy 배열).

    Returns:
        Optional[str]: 계산된 SHA256 해시 문자열. 오류 발생 시 None을 반환합니다.
    """
    if not isinstance(image_data, np.ndarray):
        logger.error("입력된 이미지 데이터가 NumPy 배열이 아닙니다.")
        return None
    try:
        # NumPy 배열을 바이트 문자열로 변환합니다.
        # 이미지의 경우, 내용이 변경되지 않도록 C-contiguous array로 만드는 것이 좋습니다.
        image_bytes = image_data.tobytes()
        
        hasher = hashlib.sha256()
        hasher.update(image_bytes)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"SHA256 해시 계산 중 오류 발생: {e}")
        return None

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

class JsonConfigHandler:
    """
    설정 파일(@photo_album.yaml의 json_keys)을 기반으로 JSON 파일을 읽고 쓰는 클래스입니다.
    """
    def __init__(self, json_keys_config: Dict[str, Any]):
        """
        JsonConfigHandler를 초기화합니다.

        Args:
            json_keys_config (Dict[str, Any]): YAML 설정의 'json_keys' 섹션에 해당하는 딕셔너리.
        """
        logger.info(f"JsonConfigHandler 초기화 시작: root_dir='{json_keys_config}'")

        if not isinstance(json_keys_config, dict):
            logger.error(f"JsonConfigHandler 초기화 오류: json_keys_config가 딕셔너리가 아닙니다 (타입: {type(json_keys_config)}).")
            # 또는 기본 설정을 사용하거나 예외를 발생시킬 수 있습니다.
            # 여기서는 주요 키들을 기본값으로 설정합니다.
            json_keys_config = {} # 빈 딕셔너리로 설정하여 아래 _get_string_key_from_config가 기본값을 사용하도록 함

        self.json_keys_config = json_keys_config

        self.user_profile_lst       = self.json_keys_config.get("user_profile", {})
        self.user_profile_key       = self.user_profile_lst.get("key", "user_profile")
        self.username_key           = self.user_profile_lst.get("username", {}).get("key", "username")
        self.username_val           = self.user_profile_lst.get("username", {}).get("name", "salgabda")
        self.email_key              = self.user_profile_lst.get("email", {}).get("key", "email")
        self.email_val              = self.user_profile_lst.get("email", {}).get("name", "salgasalgaba@naver.combda")

        logger.debug(f"user_profile_key: {self.user_profile_key}, "
                     f"username_key: {self.username_key}, username_val: {self.username_val}, "
                     f"email_key: {self.email_key}, email_val: {self.email_val}")

        self.image_info_lst         = self.json_keys_config.get("image_info_lst", {})
        self.image_info_key         = self.image_info_lst.get("key", "image_info")
        self.image_resolution_key   = self.image_info_lst.get("resolution", {}).get("key", "resolution")
        self.image_width_key        = self.image_info_lst.get("resolution", {}).get("width_key", "width")
        self.image_height_key       = self.image_info_lst.get("resolution", {}).get("height_key", "height")
        self.image_channels_key     = self.image_info_lst.get("resolution", {}).get("channels_key", "channels")
        self.image_name_key         = self.image_info_lst.get("image_name_key", "image_name")
        self.image_path_key         = self.image_info_lst.get("image_path_key", "image_path")
        self.image_path_val         = self.image_info_lst.get(self.image_path_key, "")
        self.image_hash_key         = self.image_info_lst.get("image_hash_key", "image_hash")

        logger.debug(f"image_info_key: {self.image_info_lst}, "
                     f"image_resolution_key: {self.image_resolution_key}, "
                     f"image_width_key: {self.image_width_key}, image_height_key: {self.image_height_key}, "
                     f"image_channels_key: {self.image_channels_key}, image_name_key: {self.image_name_key}, "
                     f"image_path_key: {self.image_path_key}, image_hash_key: {self.image_hash_key}")

        self.object_info_lst        = self.json_keys_config.get("object_info_lst", {})
        self.object_info_key        = self.object_info_lst.get("key", "detected_obj")
        self.object_label_mask      = self.object_info_lst.get("label_mask", "***")
        self.object_box_xyxy_key    = self.object_info_lst.get("object_box_xyxy_key", "box_xyxy")
        self.object_box_xywh_key    = self.object_info_lst.get("object_box_xywh_key", "box_xywh")
        self.object_confidence_key  = self.object_info_lst.get("object_confidence_key", "confidence")
        self.object_class_id_key    = self.object_info_lst.get("object_class_id_key", "class_id")
        self.object_class_name_key  = self.object_info_lst.get("object_class_name_key", "class_name")
        self.object_label_key       = self.object_info_lst.get("object_label_key", "label")
        self.object_index_key       = self.object_info_lst.get("object_index_key", "index")
        logger.debug(f"object_info_key: {self.object_info_key}, object_label_mask: {self.object_label_mask}"
                     f"object_box_xyxy_key: {self.object_box_xyxy_key}, object_box_xywh_key: {self.object_box_xywh_key}, "
                     f"object_confidence_key: {self.object_confidence_key}, object_class_id_key: {self.object_class_id_key}, "
                     f"object_class_name_key: {self.object_class_name_key}, object_label_key: {self.object_label_key}, "
                     f"object_index_key: {self.object_index_key}")

        self.face_info_lst                    = self.object_info_lst.get("face_info_lst", {})
        self.face_info_key          = self.face_info_lst.get("key", "detected_face")
        self.face_label_mask        = self.face_info_lst.get("label_mask", "***")
        self.face_box_xyxy_key      = self.face_info_lst.get("face_box_xyxy_key", "box_xyxy")
        self.face_confidence_key    = self.face_info_lst.get("face_confidence_key", "confidence")
        self.face_class_id_key      = self.face_info_lst.get("face_class_id_key", "class_id")
        self.face_class_name_key    = self.face_info_lst.get("face_class_name_key", "class_name")
        self.face_label_key         = self.face_info_lst.get("face_label_key", "label")
        self.face_embedding_key     = self.face_info_lst.get("face_embedding_key", "embedding")
        self.face_id_key            = self.face_info_lst.get("face_id_key", "face_id")
        self.face_box_key           = self.face_info_lst.get("face_box_key", "box")
        logger.debug(f"face_info_key: {self.face_info_key}, face_label_mask: {self.face_label_mask}"
                     f"face_box_xyxy_key: {self.face_box_xyxy_key}, face_confidence_key: {self.face_confidence_key}, "
                     f"face_class_id_key: {self.face_class_id_key}, face_class_name_key: {self.face_class_name_key}, "
                     f"face_label_key: {self.face_label_key}, face_embedding_key: {self.face_embedding_key}, "
                     f"face_id_key: {self.face_id_key}, face_box_key: {self.face_box_key}")
        logger.debug(f"JsonConfigHandler가 성공적으로 초기화되었습니다.")

    def write_json(
        self,
        image_path: Path,
        image_hash: Optional[str],
        width: int,
        height: int,
        channels: int,
        detected_objects: List[Dict[str, Any]],
        json_path: Path
    ) -> bool:
        """
        초기화 시 로드된 설정을 사용하여 이미지 및 객체 감지 정보를 JSON 파일로 저장합니다.
        """
        try:
            output_data = {}
            user_profile_information = {
                self.username_key: self.username_val,
                self.email_key: self.email_val
            }
            image_information = {
                self.image_resolution_key: {
                    self.image_width_key: width,       # self.w_key -> self.image_width_key
                    self.image_height_key: height,     # self.h_key -> self.image_height_key
                    self.image_channels_key: channels  # self.ch_key -> self.image_channels_key
                },
                self.image_name_key: image_path.name,
                self.image_path_key: str(image_path), # Path 객체를 문자열로 변환
                self.image_hash_key: image_hash
            }
            output_data[self.user_profile_key] = user_profile_information
            output_data[self.image_info_key] = image_information # 리스트 대신 딕셔너리 할당
            output_data[self.object_info_key] = detected_objects

            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            logger.debug(f"JSON 데이터가 '{json_path}'에 성공적으로 저장되었습니다.")
            return True
        except Exception as e:
            logger.error(f"JSON 파일 '{json_path}' 저장 중 오류 발생: {e}", exc_info=True)
            return False

    def read_json(self, json_path: Path) -> Optional[Dict[str, Any]]:
        """
        JSON 파일을 읽고, 초기화 시 로드된 설정을 사용하여 주요 키들이 존재하는지 검증합니다.
        """
        if not json_path.exists():
            logger.warning(f"JSON 파일을 찾을 수 없습니다: {json_path}")
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파일 디코딩 오류 '{json_path}': {e}")
            return None
        except Exception as e:
            logger.error(f"JSON 파일 읽기 중 예기치 않은 오류 발생 '{json_path}': {e}")
            return None

        missing_keys = []
        if self.image_info_key not in data: # self.image_info_key
            missing_keys.append(self.image_info_key)
        if self.object_info_key not in data: # self.objs_list_key -> self.objectobject_info_key_info_lst_key (실제 저장된 키)
            missing_keys.append(self.object_info_key)
        # self.obj_count_key는 write_json에서 저장하지 않으므로 검증에서 제외하거나, 필요시 write_json에서 추가해야 함

        if missing_keys:
            logger.warning(f"JSON 파일 '{json_path}'에 필수 키가 누락되었습니다: {', '.join(missing_keys)}")
            return None

        # 추가적인 내부 키 검증 (예: image_info 내의 file_name_key 등)은 필요에 따라 여기에 추가할 수 있습니다.
        # 예를 들어, image_info 그룹 내의 키 검증:
        # image_info_data = data.get(self.image_info_key, {})
        # if self.fname_key not in image_info_data:
        #     logger.warning(f"JSON 파일 '{json_path}'의 '{self.image_info_key}'에 '{self.fname_key}' 키가 누락되었습니다.")
        #     return None
        # ... (다른 내부 키 검증)

        logger.debug(f"JSON 데이터 '{json_path}' 로드 및 검증 완료.")
        return data

def load_photo_data_from_sources(
    logger_instance, # 로거 인스턴스
    raw_jsons_dir_path: Optional[str] = None,
    json_save_file_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
    """
    지정된 소스에서 사진 메타데이터를 로드합니다.
    1. raw_jsons_dir_path 디렉토리 내의 모든 JSON 파일을 로드 시도.
    2. 1번에서 데이터가 없거나 경로가 없으면 json_save_file_path (대표 파일) 로드 시도.
    JSON 파일의 내용은 리스트 또는 단일 딕셔너리일 수 있습니다.

    Args:
        logger_instance: 로깅을 위한 로거 객체.
        raw_jsons_dir_path (Optional[str]): 여러 JSON 파일이 있는 디렉토리 경로.
        json_save_file_path (Optional[str]): 단일 대표 JSON 파일 경로.

    Returns:
        List[Dict[str, Any]]: 로드된 사진 메타데이터의 리스트.
    """
    all_photo_data: List[Dict[str, Any]] = []

    # 1. raw_jsons_dir_path가 설정되어 있고 디렉토리가 존재하면 모든 JSON 파일 로드
    if raw_jsons_dir_path and os.path.isdir(raw_jsons_dir_path):
        logger_instance.info(f"{raw_jsons_dir_path} 디렉토리에서 모든 JSON 파일을 로드합니다.")
        for filename in os.listdir(raw_jsons_dir_path):
            if filename.endswith(".json"):
                file_path = os.path.join(raw_jsons_dir_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_photo_data.extend(data)
                        elif isinstance(data, dict): # JSON 파일 내용이 단일 객체(딕셔너리)인 경우
                            all_photo_data.append(data)
                        else:
                            logger_instance.warning(f"파일 {file_path}의 내용이 리스트 또는 딕셔너리 형식이 아닙니다. 건너<0xEB><0><0x8E>니다.")
                except json.JSONDecodeError:
                    logger_instance.warning(f"파일 {file_path}의 형식이 잘못되었습니다. 건너<0xEB><0><0x8E>니다.")
                except Exception as e:
                    logger_instance.error(f"파일 {file_path} 로드 중 오류 발생: {e}")
        if not all_photo_data and os.path.exists(raw_jsons_dir_path): # 디렉토리는 있지만 유효한 데이터가 없는 경우
            logger_instance.warning(f"{raw_jsons_dir_path} 디렉토리에서 유효한 JSON 데이터를 찾지 못했습니다.")

    # 2. 만약 raw_jsons_dir_path에서 데이터를 로드하지 못했거나, 해당 경로가 없다면
    #    json_save_file_path (대표 파일) 로드를 시도합니다.
    if not all_photo_data and json_save_file_path and os.path.exists(json_save_file_path):
        logger_instance.info(f"대표 JSON 파일({json_save_file_path}) 로드를 시도합니다.")
        try:
            with open(json_save_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_photo_data = data
                elif isinstance(data, dict): # 대표 파일도 단일 객체일 수 있음
                    all_photo_data = [data]
        except json.JSONDecodeError:
            logger_instance.warning(f"대표 JSON 파일({json_save_file_path})의 형식이 잘못되었습니다.")
        except Exception as e:
            logger_instance.error(f"대표 JSON 파일({json_save_file_path}) 로드 중 오류 발생: {e}")
    elif not all_photo_data and json_save_file_path and not os.path.exists(json_save_file_path):
        logger_instance.warning(f"대표 JSON 파일({json_save_file_path})을 찾을 수 없습니다.")

    if not all_photo_data:
        logger_instance.warning("로드할 수 있는 사진 데이터가 없습니다. 빈 리스트를 반환합니다.")
    return all_photo_data

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
