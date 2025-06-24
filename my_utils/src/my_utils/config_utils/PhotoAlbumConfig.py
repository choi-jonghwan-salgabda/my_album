import os
import sys
import copy
import re
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, List, Union, Optional

try:
    from my_utils.config_utils.SimpleLogger import logger, get_argument
except ImportError as e:
    print(f"치명적 오류: SimpleLogger를 임포트할 수 없습니다: {e}")
    sys.exit(1)

PLACEHOLDER_PATTERN = re.compile(r'\${([^}]+)}|\$([a-zA-Z_][a-zA-Z0-9_]*)')

class PhotoAlbumConfig:
    def __init__(self, root_dir: Union[str, Path], config_path: Union[str, Path]):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()

        logger.info(f"PhotoAlbumConfig 초기화 시작: root_dir='{self.root_dir}', config_path='{self.config_path}'")

        self.cfg = self._load_yaml()
        if self.cfg is None:
            logger.error("설정 파일 로드 실패. 초기화를 중단합니다.")
            sys.exit(1)

        self._perform_placeholder_resolution()
        self.current_cfg = self.cfg
        logger.info("설정 초기화 완료.")

    def _load_yaml(self) -> Optional[dict]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"설정 파일 로딩 오류: {e}")
            return None

    def _perform_placeholder_resolution(self):
        max_passes = 10
        for i in range(max_passes):
            logger.debug(f"치환 패스 {i+1}/{max_passes}")
            cfg_before = copy.deepcopy(self.cfg)
            self._resolve_placeholders(self.cfg)
            if cfg_before == self.cfg:
                break

    def _resolve_placeholders(self, data: Any):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    self._resolve_placeholders(value)
                elif isinstance(value, str):
                    resolved = self._resolve_single_value(value)
                    if key.endswith('_dir') or key.endswith('_path'):
                        resolved = os.path.normpath(os.path.expanduser(resolved))
                        resolved = str(Path(resolved).resolve())
                    data[key] = resolved
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    self._resolve_placeholders(item)
                elif isinstance(item, str):
                    resolved = self._resolve_single_value(item)
                    data[idx] = resolved

    def _resolve_single_value(self, value: str) -> str:
        def replace(match):
            key = match.group(1) or match.group(2)
            if key == 'PWD':
                return str(self.root_dir)
            val = self._get_value_from_key_path(self.cfg, key)
            if val is not None:
                return str(val)
            logger.warning(f"[치환 실패] '${{{key}}}' 를 치환할 수 없습니다.")
            return match.group(0)

        return PLACEHOLDER_PATTERN.sub(replace, value)

    def _get_value_from_key_path(self, data: Any, key_path: str) -> Any:
        keys = key_path.split('.')
        current = data
        try:
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                elif isinstance(current, list):
                    key = int(key)
                    current = current[key]
                else:
                    return None
            return current
        except Exception as e:
            logger.debug(f"키 경로 해석 실패: {key_path} → {e}")
            return None

    def get(self, key: str, default: Any = None) -> Any:
        return self._get_value_from_key_path(self.cfg, key) or default

    def keys(self):
        return self.cfg.keys()

    def as_dict(self) -> dict:
        return copy.deepcopy(self.cfg)

if __name__ == "__main__":
    # 0. 현재 위치 정보
    direction_dir = os.getcwd()
    print(f"지금 쥔계서 계신곳(direction_dir)      : {direction_dir}")
    worker_path_obj = Path(__file__).resolve()
    project_root_path = worker_path_obj.parent.parent
    print(f"지금 일꾼이 일하는곳(worker_dir_name)  : {project_root_path}")

    # 1. 인자 파싱
    args = get_argument()

    # 2. 로그 설정
    log_file_name = f"photo_config_test_{datetime.now().strftime('%y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(args.log_dir, log_file_name)

    logger.setup(
        logger_path=log_file_path,
        min_level=args.log_level,
        include_function_name=True,
        pretty_print=True
    )
    logger.info(f"--- photo_config_main.py 실행 시작 ---")
    logger.info(f"파싱된 인자: root_dir='{args.root_dir}', config_path='{args.config_path}', log_dir='{args.log_dir}', log_level='{args.log_level}'")

    # 3. 설정 파일 로드
    try:
        config = PhotoAlbumConfig(config_path=args.config_path, root_dir=args.root_dir)
        logger.info(f"PhotoAlbumConfig 인스턴스 생성 완료.")
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}", exc_info=True)
        sys.exit(1)

    # 4. 주요 키값 출력 (예: 경로 확인)
    try:
        logger.info(f"--- 주요 설정 값 확인 ---")
        for key in config.keys():
            val = config.get(key)
            logger.info(f"{key} : {val}")
    except Exception as e:
        logger.error(f"설정값 로드 중 오류 발생: {e}", exc_info=True)

    # 5. 테스트 함수 예시
    def test_path(key: str):
        logger.info(f"--- 테스트: '{key}' 키의 경로 출력 ---")
        path = config.get(key)
        if isinstance(path, Path):
            logger.info(f"{key} → {path}")
        else:
            logger.warning(f"{key}는 유효한 Path가 아닙니다: {path}")

    def test_list(key: str):
        logger.info(f"--- 리스트 값 테스트: {key} ---")
        val = config.get(key)
        if isinstance(val, List):
            logger.info(f"{key} 리스트 항목 수: {len(val)} → {val}")
        else:
            logger.warning(f"{key} 값은 리스트가 아닙니다: {val}")

    # 6. 테스트 실행 (원하는 키 지정)
    test_path("target_root")  # 예시: ${ROOT}/photos → 실제 경로
    test_path("backup_dir")   # 예시: ${HOME}/backup
    test_list("active_plugins")  # 예시: ['resize', 'filter']
    test_list("some_numbers")    # 예시: [1, 2, 3] 등

    logger.info(f"--- 설정 테스트 완료 ---")
