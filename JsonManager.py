import os
import sys
import yaml
import json
import re
from pathlib import Path
from datetime import datetime
import copy
import shutil # For rmtree in tests
from typing import List, Dict, Union, Any, Optional

#import logging # 로깅을 위해 추가
# SimpleLogger.py로부터 공유 logger 인스턴스를 가져옵니다.
try:
    from my_utils.config_utils.SimpleLogger import logger, get_argument
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

class JsonManager:
    """
    설정 파일(@photo_album.yaml의 json_keys)을 기반으로 JSON 파일을 읽고 쓰고,
    외부에서 값 접근 및 수정이 가능하도록 지원하는 클래스입니다.
    """

    def __init__(self, json_keys_config: Dict[str, Any], json_path: Optional[Path] = None):
        """
        JsonManager 초기화하고, 선택적으로 JSON 파일을 로딩합니다.

        Args:
            json_keys_config (Dict[str, Any]): YAML 설정의 'json_keys' 섹션.
            json_path (Optional[Path]): 로드할 JSON 파일 경로 (선택).
        """
        if not isinstance(json_keys_config, dict):
            logger.error(f"설정 초기화 오류: json_keys_config가 딕셔너리가 아닙니다 (타입: {type(json_keys_config)}).")
            json_keys_config = {}

        self.json_keys_config = json_keys_config
        self._data: Dict[str, Any] = {}  # 실제 JSON 내용을 저장

        if json_path:
            self.read_json(json_path)

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        특정 섹션의 값을 반환합니다.

        Args:
            section (str): JSON의 최상위 키 (예: user_profile).
            key (Optional[str]): 해당 섹션 내의 키.
            default (Any): 기본값.

        Returns:
            Any: 해당 값 또는 기본값.
        """
        section_data = self._data.get(section, {})
        if key is None:
            return section_data
        return section_data.get(key, default)

    def set(self, section: str, key: str, value: Any):
        """
        특정 섹션 내 키의 값을 설정합니다.

        Args:
            section (str): JSON의 최상위 키.
            key (str): 내부 키.
            value (Any): 설정할 값.
        """
        if section not in self._data or not isinstance(self._data[section], dict):
            self._data[section] = {}
        self._data[section][key] = value
        logger.debug(f"[SET] {section}.{key} = {value}")

    def read_json(self, json_path: Path) -> bool:
        """
        JSON 파일을 읽고 내부 데이터로 로드합니다.

        Args:
            json_path (Path): JSON 파일 경로.

        Returns:
            bool: 성공 여부.
        """
        if not json_path.exists():
            logger.warning(f"JSON 파일이 존재하지 않습니다: {json_path}")
            return False
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            logger.debug(f"JSON 데이터가 성공적으로 로드되었습니다: {json_path}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
        except Exception as e:
            logger.error(f"JSON 로딩 중 예외 발생: {e}")
        return False

    def write_json(self, json_path: Path) -> bool:
        """
        현재 데이터를 JSON 파일로 저장합니다.

        Args:
            json_path (Path): 저장할 JSON 파일 경로.

        Returns:
            bool: 성공 여부.
        """
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=4)
            logger.debug(f"JSON이 성공적으로 저장되었습니다: {json_path}")
            return True
        except Exception as e:
            logger.error(f"JSON 저장 중 오류 발생: {e}")
            return False

    def update_from_dict(self, updates: Dict[str, Dict[str, Any]]):
        """
        외부 딕셔너리를 통해 내부 데이터를 일괄 갱신합니다.

        Args:
            updates (Dict[str, Dict[str, Any]]): 섹션별 갱신할 값들.
        """
        for section, values in updates.items():
            if not isinstance(values, dict):
                continue
            for key, val in values.items():
                self.set(section, key, val)

    def dump(self) -> Dict[str, Any]:
        """
        현재 저장된 전체 데이터를 반환합니다.

        Returns:
            Dict[str, Any]: 내부 JSON 데이터 전체.
        """
        return self._data

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
    logger.info(f"--- configger.py standalone test execution ---")
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
        logger.error(f"모델 변수값 가저오기 오류 발생: {e}")
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

    # 4. JsonManager 초기화
    json_manager = JsonManager(json_keys_config=json_keys_config)

    # 5. 테스트용 JSON 경로 설정
    test_json_path = Path(args.root_dir) / "test_data.json"

    # 6. 더미 객체 리스트 및 이미지 정보 정의
    dummy_detected_objects = [
        {
            json_manager.object_class_id_key: 1,
            json_manager.object_class_name_key: "cat",
            json_manager.object_box_xyxy_key: [50, 60, 200, 220],
            json_manager.object_confidence_key: 0.98,
            json_manager.object_label_key: "고양이"
        }
    ]

    dummy_image_path = Path("sample.jpg")
    dummy_image_hash = "abc123hash"
    width, height, channels = 1024, 768, 3

    # 7. JSON 파일 쓰기 테스트
    success = json_manager.write_json(
        image_path=dummy_image_path,
        image_hash=dummy_image_hash,
        width=width,
        height=height,
        channels=channels,
        detected_objects=dummy_detected_objects,
        json_path=test_json_path
    )

    if success:
        print(f"[성공] JSON 파일이 저장되었습니다: {test_json_path}")
    else:
        print(f"[실패] JSON 파일 저장에 실패했습니다.")

    # 8. JSON 파일 읽기 테스트
    data = json_manager.read_json(test_json_path)
    if data:
        print(f"[성공] JSON 파일을 성공적으로 읽었습니다:\n{json.dumps(data, ensure_ascii=False, indent=2)}")
    else:
        print("[실패] JSON 파일 읽기에 실패했습니다.")
