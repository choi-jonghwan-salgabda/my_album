import os
import sys
import yaml
import json
import io
from pathlib import Path
from datetime import datetime
import copy
import shutil # For rmtree in tests
from typing import List, Dict, Union, Any, Optional
import traceback

#import logging # 로깅을 위해 추가
# SimpleLogger.py로부터 공유 logger 인스턴스를 가져옵니다.
try:
    from my_utils.config_utils.SimpleLogger import logger, get_argument
    from my_utils.config_utils.configger import configger
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    import traceback
    traceback.print_exc() # 전체 트레이스백 출력 (개발 단계에서 유용)
    sys.exit(1)

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

# class JsonManager:
#     """
#     설정 파일(@photo_album.yaml의 json_keys)을 기반으로 JSON 파일을 읽고 쓰고,
#     외부에서 값 접근 및 수정이 가능하도록 지원하는 클래스입니다.
#     """

#     def __init__(self, json_keys_config: Dict[str, Any], json_path: Optional[Path] = None):
#         """
#         JsonManager 초기화하고, 선택적으로 JSON 파일을 로딩합니다.

#         Args:
#             json_keys_config (Dict[str, Any]): YAML 설정의 'json_keys' 섹션.
#             json_path (Optional[Path]): 로드할 JSON 파일 경로 (선택).
#         """
#         if not isinstance(json_keys_config, dict):
#             logger.error(f"설정 초기화 오류: json_keys_config가 딕셔너리가 아닙니다 (타입: {type(json_keys_config)}).")
#             json_keys_config = {}

#         self.json_keys_config = json_keys_config
#         self._data: Dict[str, Any] = {}  # 실제 JSON 내용을 저장

#         if json_path:
#             self.read_json(json_path)

#         # Helper to safely get nested values from config
#         def _get_nested_key(config_dict: Dict[str, Any], path: List[str], default: Any, log_error: bool = True) -> Any:
#             current = config_dict
#             for i, key in enumerate(path):
#                 if not isinstance(current, dict):
#                     if log_error:
#                         logger.error(
#                             f"설정 오류: 경로 '{'.'.join(path[:i])}'의 값이 딕셔너리가 아닙니다 (타입: {type(current)}). "
#                             f"키 '{key}'를 찾을 수 없습니다. 기본값 '{default}'을(를) 사용합니다."
#                         )
#                     return default
#                 current = current.get(key, default)
#                 if current == default and i < len(path) - 1: # If default is returned mid-path, it means key was missing
#                     if log_error:
#                         logger.error(
#                             f"설정 오류: 경로 '{'.'.join(path[:i+1])}'에서 키 '{key}'를 찾을 수 없습니다. 기본값 '{default}'을(를) 사용합니다."
#                         )
#                     return default
#             return current

#         # Parse json_keys_config and set attributes, similar to JsonConfigHandler
#         # User Profile Keys
#         user_profile_lst = _get_nested_key(json_keys_config, ["user_profile"], {})
#         self.user_profile_key = _get_nested_key(user_profile_lst, ["key"], "user_profile")
#         self.username_key = _get_nested_key(user_profile_lst, ["username", "key"], "username")
#         self.username_val = _get_nested_key(user_profile_lst, ["username", "name"], "salgabda")
#         self.email_key = _get_nested_key(user_profile_lst, ["email", "key"], "email")
#         self.email_val = _get_nested_key(user_profile_lst, ["email", "name"], "salgasalgaba@naver.combda")

#         logger.debug(f"user_profile_key: {self.user_profile_key}, "
#                      f"username_key: {self.username_key}, username_val: {self.username_val}, "
#                      f"email_key: {self.email_key}, email_val: {self.email_val}")

#         # Image Info Keys
#         image_info_lst = _get_nested_key(json_keys_config, ["image_info_lst"], {})
#         self.image_info_key = _get_nested_key(image_info_lst, ["key"], "image_info")
#         self.image_resolution_key = _get_nested_key(image_info_lst, ["resolution", "key"], "resolution")
#         self.image_width_key = _get_nested_key(image_info_lst, ["resolution", "width_key"], "width")
#         self.image_height_key = _get_nested_key(image_info_lst, ["resolution", "height_key"], "height")
#         self.image_channels_key = _get_nested_key(image_info_lst, ["resolution", "channels_key"], "channels")
#         self.image_name_key = _get_nested_key(image_info_lst, ["image_name_key"], "image_name")
#         self.image_path_key = _get_nested_key(image_info_lst, ["image_path_key"], "image_path")
#         # self.image_path_val = _get_nested_key(image_info_lst, [self.image_path_key], "") # This line is problematic in JsonConfigHandler too. It tries to get a value using a key that is itself a key name.
#         self.image_hash_key = _get_nested_key(image_info_lst, ["image_hash_key"], "image_hash")

#         logger.debug(f"image_info_key: {self.image_info_key}, "
#                      f"image_resolution_key: {self.image_resolution_key}, "
#                      f"image_width_key: {self.image_width_key}, image_height_key: {self.image_height_key}, "
#                      f"image_channels_key: {self.image_channels_key}, image_name_key: {self.image_name_key}, "
#                      f"image_path_key: {self.image_path_key}, image_hash_key: {self.image_hash_key}")

#         # Object Info Keys
#         object_info_lst = _get_nested_key(json_keys_config, ["object_info_lst"], {})
#         self.object_info_key = _get_nested_key(object_info_lst, ["key"], "detected_obj")
#         self.object_label_mask = _get_nested_key(object_info_lst, ["label_mask"], "***")
#         self.object_box_xyxy_key = _get_nested_key(object_info_lst, ["object_box_xyxy_key"], "box_xyxy")
#         self.object_box_xywh_key = _get_nested_key(object_info_lst, ["object_box_xywh_key"], "box_xywh")
#         self.object_confidence_key = _get_nested_key(object_info_lst, ["object_confidence_key"], "confidence")
#         self.object_class_id_key = _get_nested_key(object_info_lst, ["object_class_id_key"], "class_id")
#         self.object_class_name_key = _get_nested_key(object_info_lst, ["object_class_name_key"], "class_name")
#         self.object_label_key = _get_nested_key(object_info_lst, ["object_label_key"], "label")
#         self.object_index_key = _get_nested_key(object_info_lst, ["object_index_key"], "index")
#         logger.debug(f"object_info_key: {self.object_info_key}, object_label_mask: {self.object_label_mask}"
#                      f"object_box_xyxy_key: {self.object_box_xyxy_key}, object_box_xywh_key: {self.object_box_xywh_key}, "
#                      f"object_confidence_key: {self.object_confidence_key}, object_class_id_key: {self.object_class_id_key}, "
#                      f"object_class_name_key: {self.object_class_name_key}, object_label_key: {self.object_label_key}, "
#                      f"object_index_key: {self.object_index_key}")

#         # Face Info Keys (nested under object_info_lst)
#         face_info_lst = _get_nested_key(object_info_lst, ["face_info_lst"], {})
#         self.face_info_key = _get_nested_key(face_info_lst, ["key"], "detected_face")
#         self.face_label_mask = _get_nested_key(face_info_lst, ["label_mask"], "***")
#         self.face_box_xyxy_key = _get_nested_key(face_info_lst, ["face_box_xyxy_key"], "box_xyxy")
#         self.face_confidence_key = _get_nested_key(face_info_lst, ["face_confidence_key"], "confidence")
#         self.face_class_id_key = _get_nested_key(face_info_lst, ["face_class_id_key"], "class_id")
#         self.face_class_name_key = _get_nested_key(face_info_lst, ["face_class_name_key"], "class_name")
#         self.face_label_key = _get_nested_key(face_info_lst, ["face_label_key"], "label")
#         self.face_embedding_key = _get_nested_key(face_info_lst, ["face_embedding_key"], "embedding")
#         self.face_id_key = _get_nested_key(face_info_lst, ["face_id_key"], "face_id")
#         self.face_box_key = _get_nested_key(face_info_lst, ["face_box_key"], "box")
#         self.cropped_image_dir_key = _get_nested_key(face_info_lst, ["cropped_image_dir_key"], "cropped_image_dir")

#         logger.debug(f"face_info_key: {self.face_info_key}, face_label_mask: {self.face_label_mask}"
#                      f"face_box_xyxy_key: {self.face_box_xyxy_key}, face_confidence_key: {self.face_confidence_key}, "
#                      f"face_class_id_key: {self.face_class_id_key}, face_class_name_key: {self.face_class_name_key}, "
#                      f"face_label_key: {self.face_label_key}, face_embedding_key: {self.face_embedding_key}, "
#                      f"face_id_key: {self.face_id_key}, face_box_key: {self.face_box_key}, "
#                      f"cropped_image_dir_key: {self.cropped_image_dir_key}")

#         logger.debug(f"JsonManager가 성공적으로 초기화되었습니다.")

#     def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
#         """
#         특정 섹션의 값을 반환합니다.

#         Args:
#             section (str): JSON의 최상위 키 (예: user_profile).
#             key (Optional[str]): 해당 섹션 내의 키.
#             default (Any): 기본값.

#         Returns:
#             Any: 해당 값 또는 기본값.
#         """
#         section_data = self._data.get(section, {})
#         if key is None:
#             return section_data
#         return section_data.get(key, default)

#     def build_full_json(
#         self,
#         image_path: Path,
#         image_hash: Optional[str],
#         width: int,
#         height: int,
#         channels: int,
#         detected_objects: List[Dict[str, Any]],
#     ):
#         """
#         주어진 정보로 내부 데이터(_data)를 완전히 새로 구성합니다.
#         기존 _data 내용은 덮어쓰여집니다.
#         """
#         output_data = {}
#         user_profile_information = {
#             self.username_key: self.username_val,
#             self.email_key: self.email_val
#         }
#         image_information = {
#             self.image_resolution_key: {
#                 self.image_width_key: width,
#                 self.image_height_key: height,
#                 self.image_channels_key: channels
#             },
#             self.image_name_key: image_path.name,
#             self.image_path_key: str(image_path),
#             self.image_hash_key: image_hash
#         }
#         output_data[self.user_profile_key] = user_profile_information
#         output_data[self.image_info_key] = image_information
#         output_data[self.object_info_key] = detected_objects

#         self._data = output_data # 내부 데이터 갱신
#         logger.debug("내부 JSON 데이터가 새 정보로 재구성되었습니다.")

#     def set(self, section: str, key: str, value: Any):
#         """
#         특정 섹션 내 키의 값을 설정합니다.

#         Args:
#             section (str): JSON의 최상위 키.
#             key (str): 내부 키.
#             value (Any): 설정할 값.
#         """
#         if section not in self._data or not isinstance(self._data[section], dict):
#             self._data[section] = {}
#         self._data[section][key] = value
#         logger.debug(f"[SET] {section}.{key} = {value}")

#     def read_json(self, json_path: Path) -> bool:
#         """
#         JSON 파일을 읽고 내부 데이터로 로드합니다.

#         Args:
#             json_path (Path): JSON 파일 경로.

#         Returns:
#             bool: 성공 여부.
#         """
#         if not json_path.exists():
#             logger.warning(f"JSON 파일이 존재하지 않습니다: {json_path}")
#             return False
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 self._data = json.load(f)
#             logger.debug(f"JSON 데이터가 성공적으로 로드되었습니다: {json_path}")
#             return True
#         except json.JSONDecodeError as e:
#             logger.error(f"JSON 파싱 오류: {e}")
#         except Exception as e:
#             logger.error(f"JSON 로딩 중 예외 발생: {e}")
#         return False

#     def write_json(self, json_path: Path) -> bool:
#         """
#         현재 데이터를 JSON 파일로 저장합니다.

#         Args:
#             json_path (Path): 저장할 JSON 파일 경로.

#         Returns:
#             bool: 성공 여부.
#         """
#         try:
#             json_path.parent.mkdir(parents=True, exist_ok=True)
#             with open(json_path, 'w', encoding='utf-8') as f:
#                 json.dump(self._data, f, ensure_ascii=False, indent=4)
#             logger.debug(f"JSON이 성공적으로 저장되었습니다: {json_path}")
#             return True
#         except Exception as e:
#             logger.error(f"JSON 저장 중 오류 발생: {e}")
#             return False

#     @staticmethod
#     def load_json_from_path(json_path: Path) -> Optional[Dict[str, Any]]:
#         """
#         주어진 경로에서 JSON 파일을 읽어 내용을 반환합니다.
#         이 메서드는 정적(static)이므로, 특정 인스턴스의 상태(self._data)를 변경하지 않습니다.
#         파일 로딩에 실패하면 None을 반환합니다.

#         Args:
#             json_path (Path): 읽을 JSON 파일의 경로.

#         Returns:
#             Optional[Dict[str, Any]]: 로드된 JSON 데이터 또는 실패 시 None.
#         """
#         if not json_path.exists():
#             logger.warning(f"JSON 파일이 존재하지 않습니다: {json_path}")
#             return None
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 return json.load(f)
#         except (json.JSONDecodeError, Exception) as e:
#             logger.error(f"'{json_path}' 파일 로딩/파싱 중 오류 발생: {e}")
#             return None

#     def update_from_dict(self, updates: Dict[str, Dict[str, Any]]):
#         """
#         외부 딕셔너리를 통해 내부 데이터를 일괄 갱신합니다.

#         Args:
#             updates (Dict[str, Dict[str, Any]]): 섹션별 갱신할 값들.
#         """
#         for section, values in updates.items():
#             if not isinstance(values, dict):
#                 continue
#             for key, val in values.items():
#                 self.set(section, key, val)

#     def dump(self) -> Dict[str, Any]:
#         """
#         현재 저장된 전체 데이터를 반환합니다.

#         Returns:
#             Dict[str, Any]: 내부 JSON 데이터 전체.
#         """
#         return self._data

# === 사용 예시 ===
if __name__ == "__main__":
    # 0. 지금 내가 일하는 곳은"
    direction_dir = os.getcwd()
    print(f"지금 쥔계서 계신곳(direction_dir)      : {direction_dir}")
    worker_path_obj = Path(__file__).resolve()
    project_root_path = worker_path_obj.parent.parent
    print(f"지금 일꾼이 일하는곳(worker_dir_name)  : {project_root_path}")

   # 1. 명령줄 인자 파싱
    # get_argument 함수는 SimpleLogger.py에서 임포트됩니다.
    # 이 함수는 --root-dir, --config-path, --log-dir, --log-level 인자를 처리합니다.
    args = get_argument()

    # 2. 로거 설정
    # SimpleLogger.py에서 임포트된 공유 logger 인스턴스를 사용합니다.
    # 로그 파일 이름에 타임스탬프를 추가하여 실행 시마다 고유한 로그 파일을 생성합니다.
    log_file_name = f"configger_standalone_test_{datetime.now().strftime('%y%m%d_%H%M%S')}.log"
    # args.log_dir은 get_argument에 의해 기본값(<root_dir>/logs) 또는 명령줄 값으로 설정됩니다.
    log_file_path = os.path.join(args.log_dir, log_file_name)

    logger.setup(
        logger_path=log_file_path,
        min_level=args.log_level,
        include_function_name=True, # 테스트 실행 시 함수 이름 포함이 유용합니다.
        pretty_print=True
    )
    logger.info(f"--- configger.py test execution ---")
    logger.info(f"파싱된 인자: root_dir='{args.root_dir}', config_path='{args.config_path}', log_dir='{args.log_dir}', log_level='{args.log_level}'")
    logger.info(f"로그 파일 경로: {log_file_path}")
    logger.show_config()


    # 3. configger 인스턴스 생성
    try:
        # 파싱된 명령줄 인자(args.root_dir, args.config_path)를 사용합니다.
        logger.info(f"configger 인스턴스 생성 시도: root_dir='{args.root_dir}', config_path='{args.config_path}'")
        config_manager = configger(root_dir=args.root_dir, config_path=args.config_path)
        logger.info(f"configger 인스턴스 생성 완료.")
    except (KeyError, TypeError, AttributeError) as e:
        logger.error(f"Configger 모델 변수값 가저오기 오류 발생: {e}")
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

    # 4. JsonManager 초기화
    logger.info(f"--- JsonManager.py test execution ---")
    try:
        json_keys_config = config_manager.get_config("json_keys")
        logger.info(f"json_keys_config :{json_keys_config}")
        if json_keys_config:
            json_path = Path('/home/owner/SambaData/OwnerData/train/jsons/1525956891490.json')
            # --- 방법 1: 새 JSON 데이터 생성 및 저장 ---
            logger.info("\n--- 방법 1: 새 JSON 데이터 생성 및 저장 ---")
            json_manager_new = JsonManager(json_keys_config=json_keys_config)
            logger.info(f"새 JsonManager 인스턴스 생성 완료.")

            dummy_detected_objects = [
                {
                    json_manager_new.object_class_id_key: 1,
                    json_manager_new.object_class_name_key: "cat",
                    json_manager_new.object_box_xyxy_key: [50, 60, 200, 220],
                    json_manager_new.object_confidence_key: 0.98,
                    json_manager_new.object_label_key: "고양이"
                }
            ]

            dummy_image_path = Path("sample_new.jpg")

            # build_full_json으로 내부 데이터 구성
            json_manager_new.build_full_json(
                image_path=dummy_image_path,
                image_hash="new_hash_xyz",
                width=1024, height=768, channels=3,
                detected_objects=dummy_detected_objects
            )

            # write_json으로 저장
            test_json_path_new = Path(args.root_dir) / "test_data_new.json"
            success_new = json_manager_new.write_json(test_json_path_new)
            if success_new:
                logger.info(f"[성공] 새 JSON 파일이 저장되었습니다: {test_json_path_new}")
                logger.info(json.dumps(json_manager_new.dump(), ensure_ascii=False, indent=2))

            # --- 방법 2: 기존 JSON 읽고, 수정하고, 저장 ---
            logger.info("\n--- 방법 2: 기존 JSON 읽기, 수정, 저장 ---")
            existing_json_path = Path('/home/owner/SambaData/OwnerData/train/jsons/1525956891490.json')
            if existing_json_path.exists():
                json_manager_existing = JsonManager(json_keys_config=json_keys_config, json_path=existing_json_path)
                logger.info(f"[정보] 기존 JSON 로드 완료: {existing_json_path}")

                # 데이터 수정 (set 메소드 사용)
                logger.info("[정보] 이미지 이름과 사용자 이메일 수정 중...")
                json_manager_existing.set(json_manager_existing.image_info_key, json_manager_existing.image_name_key, "MODIFIED_IMAGE_NAME.jpg")
                json_manager_existing.set(json_manager_existing.user_profile_key, json_manager_existing.email_key, "modified.email@example.com")

                # 수정된 내용 저장
                test_json_path_modified = Path(args.root_dir) / "test_data_modified.json"
                success_modified = json_manager_existing.write_json(test_json_path_modified)
                if success_modified:
                    logger.info(f"[성공] 수정된 JSON 파일이 저장되었습니다: {test_json_path_modified}")
                    logger.info(json.dumps(json_manager_existing.dump(), ensure_ascii=False, indent=2))
            else:
                logger.warning(f"[경고] 기존 JSON 파일이 없어 방법 2 테스트를 건너뜁니다: {existing_json_path}")

    except (KeyError, TypeError, AttributeError) as e:
        logger.error(f"변수값 가저오기 오류 발생: {e}")
        sys.exit(1)
    except Exception as e:
        sys.exit(1)
