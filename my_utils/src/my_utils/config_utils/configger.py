# shared_utils/src/utility/configger.py

import os
import sys
import yaml
import re
from pathlib import Path
from datetime import datetime
import copy
import shutil # For rmtree in tests
from typing import List, Dict, Union, Any, Optional

#import logging # 로깅을 위해 추가
# SimpleLogger.py로부터 공유 logger 인스턴스를 가져옵니다.
try:
    from my_utils.config_utils.SimpleLogger import logger, calc_digit_number, get_argument
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

# 플레이스홀더 정규 표현식: ${VAR} 또는 $VAR 형태를 찾습니다.
# 여기서 VAR는 변수 이름 [a-zA-Z_][a-zA-Z0-9_]* 에 해당합니다.
PLACEHOLDER_PATTERN = re.compile(r'\${([^}]+)}|\$([a-zA-Z_][a-zA-Z0-9_]*)')
class configger:
    """
    arg1 : root_dir -> 입력으로 프로젝트의 기준이되는 을 받는다.
    arg2 : config_path -> config를 구성하는 yaml파일.
    """
    def __init__(self, root_dir:str, config_path: str):
        logger.info(f"(init): 메서드 시작")

        # root_dir 처리: Path 객체로 만들고 사용자 홈 디렉토리 기준 절대 경로로 확장
        self.root_dir = Path(root_dir).expanduser()
        logger.debug(f"(init): 경로 받음 및 확정(root_dir): {self.root_dir} ")

        # config_path 처리: 절대 경로이면 그대로, 상대 경로이면 root_dir과 결합하여 절대 경로 Path 객체 생성
        if os.path.isabs(config_path):
            self.config_path = Path(config_path)
            logger.debug(f"(init): config_path는 절대 경로입니다: {self.config_path}")
        else:
            # os.path.join으로 문자열 결합 후 Path 객체로 변환
            self.config_path = Path(os.path.join(self.root_dir, config_path))
            # pathlib.Path 객체로 변환
            self.config_path = Path(self.config_path).resolve()

            logger.debug(f"(init): config_path는 상대 경로이므로 root_dir과 결합하여 절대 경로를 만듭니다: {self.config_path}")

        logger.debug(f"(init):경로 확정(config_path):   {self.config_path} ")
        logger.debug(f"(init): _load_yaml 호출 직전")

        # YAML 파일 로드
        self.cfg = self._load_yaml()
        if self.cfg is None:
            # YAML 로드 실패 시 처리 로직 (예: 오류 발생 또는 기본 설정 사용)
            # 여기서는 간단히 None으로 설정하고 종료할 수 있습니다.
            logger.error("설정 파일 로드 실패. 초기화를 중단합니다.")
            return

        logger.debug(f"(init): YAML 로드 완료. self.cfg 내용 일부: {str(self.cfg)[:500]}...")

        # 플레이스홀더 치환 시작 (전체 cfg 딕셔너리를 대상으로 재귀 호출)
        # 치환은 원본 self.cfg 객체를 직접 수정하도록 구현할 수 있습니다.
        # 또는 치환된 새로운 딕셔너리를 만들어서 반환받을 수도 있습니다.
        # 여기서는 원본 self.cfg를 직접 수정하는 방식으로 구현해 보겠습니다.
        logger.debug(f"(init): 플레이스홀더 반복 치환 시작")
        
        max_passes = 1 # 최대 반복 횟수 (무한 루프 방지)
        for i in range(max_passes):
            logger.debug(f"(init): 플레이스홀더 치환 패스 {i+1}/{max_passes}")
            # 이전 상태를 복사하여 변경 여부 확인
            cfg_before_pass = copy.deepcopy(self.cfg)
            
            self._traverse_and_resolve(self.cfg, []) # 빈 키 경로 리스트로 시작 (최상위)
            
            # 현재 패스에서 더 이상 변경이 없으면 반복 중단
            if self.cfg == cfg_before_pass:
                logger.info(f"(init): 패스 {i+1}에서 더 이상 변경 사항 없음. 플레이스홀더 치환 완료.")
                break
            else: # for 루프가 break 없이 정상적으로 완료된 경우 (max_passes 만큼 실행된 경우)
                if self.cfg != cfg_before_pass: # 마지막 패스에서도 변경이 있었다면 경고
                    logger.warning(f"(init): 최대 플레이스홀더 치환 패스 {max_passes}회 도달. 아직 해결되지 않은 중첩 플레이스홀더가 있을 수 있습니다.")
                else: # 마지막 패스에서 변경이 없었다면 정상 완료
                    logger.info(f"(init): 플레이스홀더 치환 최대 패스({max_passes}) 내 완료 또는 변경 없음.")

        # 치환이 완료된 self.cfg가 최종 설정 데이터가 됩니다.
        self.current_cfg = self.cfg
        self.next_cfg = self.current_cfg # 현재는 동일하게 설정

        logger.info(f"++++++++++++++++++++++++++++++++++") # 최종 치환 결과 로깅
        logger.info(f"Final self.cfg (resolved):{self.cfg}") # 최종 치환 결과 로깅
        logger.info(f"++++++++++++++++++++++++++++++++++") # 최종 치환 결과 로깅

        # self.yaml_list 및 관련 변수들은 이 방식에서는 필요 없습니다.
        # self.list_depth = 0 # 이제 필요 없을 가능성이 높습니다.


    def _traverse_and_resolve(self, data, current_key_path_list):
        """
        딕셔너리 또는 리스트 내의 문자열에서 플레이스홀더 ${...}를 찾아
        full_config에서 값을 조회하여 치환합니다.
        재귀적으로 동작하며, data 객체를 직접 수정합니다.

        Args:
            data (dict or list or any): 현재 처리할 데이터 (self.cfg의 일부 또는 전체)
            current_key_path_list (list): 현재 위치까지의 키 경로 세그먼트 리스트 (예: ['project', 'paths', 'dataset'])
        """
        # 현재 경로를 점(.)으로 구분된 문자열로 만듭니다 (로깅/디버깅용)
        current_key_path_str = ".".join(map(str, current_key_path_list)) if current_key_path_list else "root"
        logger.info(f"_traverse_and_resolve start-------------------------------------------------------")
        logger.info(f"_현재 경로: {current_key_path_str}")
        logger.info(f"_traverse_and_resolve ------------------------------------------------------------\n")

        if isinstance(data, dict):
            # 데이터가 딕셔너리인 경우, 각 항목을 순회하며 재귀 호출
            for key, value in data.items():
                logger.debug(f"_traverse_and_resolve-isinstance(data, dict), key:{key}, \nvalue:{value}")
                # 다음 레벨의 키 경로 리스트를 생성
                next_key_path_list = current_key_path_list + [key]
                logger.debug(f"_traverse_and_resolve-isinstance(data, dict), key:{key}, \nnext_key_path_list:{next_key_path_list} ")
                # 값(value)에 대해 재귀 호출. 반환 값을 받아서 원래 딕셔너리에 업데이트
                # 이는 불변성을 유지하는 방식이며, 직접 수정하려면 다른 로직이 필요합니다.
                # 현재는 재귀 호출 자체에서 하위 딕셔너리를 수정하도록 설계합니다.
                # 즉, 재귀 호출된 함수 내에서 data[key]의 내용을 직접 변경하는 방식입니다.
                # 하지만 Python에서는 함수 인자로 딕셔너리를 넘기면 참조가 전달되므로
                # 함수 내부에서 data[key]의 하위 내용을 수정하면 원본 딕셔너리도 수정됩니다.
                # 따라서 여기서는 값을 다시 할당하는 대신 그냥 재귀 호출만 합니다.
                self._traverse_and_resolve(value, next_key_path_list)

        elif isinstance(data, list):
            # 데이터가 리스트인 경우, 각 요소를 순회하며 재귀 호출 (인덱스를 키 경로에 포함)
            for index, item in enumerate(data):
                logger.debug(f"_traverse_and_resolve-isinstance(data, list), index:{index}, item:{item}")
                # 다음 레벨의 키 경로 리스트를 생성 (리스트 인덱스를 키처럼 취급)
                next_key_path_list = current_key_path_list + [index] # 리스트 인덱스는 숫자로 저장될 수 있습니다. 필요에 따라 str(index)로 변환.
                # 값(item)에 대해 재귀 호출
                self._traverse_and_resolve(item, next_key_path_list)

        elif isinstance(data, str):
            # 데이터가 문자열인 경우, 플레이스홀더를 찾아 치환
            # 플레이스홀더 패턴: ${...}
            placeholders = re.findall(r'\$\{(.*?)\}', data)
            resolved_string = data # 치환 작업을 할 문자열 복사
            logger.debug(f"_traverse_and_resolve-isinstance(data, str), '{resolved_string}'에서 '{placeholders}'를 찾음")
            if placeholders: # 플레이스홀더가 하나라도 있는 경우에만 처리
                for placeholder_key_path in placeholders:
                    logger.debug(f"_traverse_and_resolve-isinstance(data, str), placeholder_key_path:'{placeholder_key_path}'")
                    lookup_value = None
                    trimmed_key_path = placeholder_key_path.strip() # 플레이스홀더 내용의 앞뒤 공백 제거
                    logger.debug(f"_traverse_and_resolve-trimmed_key_path:'{trimmed_key_path}' <- placeholder_key_path:'{placeholder_key_path}'")

                    # 플레이스홀더 내용이 'PWD'인지 확인
                    if trimmed_key_path == "PWD":
                        try:
                            lookup_value = os.getcwd() # 시스템의 현재 작업 디렉토리 경로 가져오기
                        except Exception as e:
                            logger.error(f"_traverse_and_resolve- PWD 값 조회 오류: {e}")
                            lookup_value = None
                    # PWD가 아닌 경우, 기존처럼 self.cfg에서 키 경로로 값 조회
                    else:
                        # 플레이스홀더 내의 키 경로를 사용하여 전체 설정 데이터(self.cfg)에서 실제 값을 찾습니다.
                        lookup_value = self._get_value_from_key_path(self.cfg, trimmed_key_path)

                    if lookup_value is not None:
                        # 찾은 값이 None이 아니면 문자열에서 플레이스홀더를 실제 값으로 치환
                        # 값의 타입에 따라 문자열로 변환하여 치환해야 합니다.
                        # 주의: replace()는 모든 일치 항목을 치환합니다.
                        # f-string을 사용하는 방식은 한 번에 하나의 플레이스홀더만 치환하므로 부적합합니다.
                        # replace()가 이 경우 더 적합합니다.
                        try:
                            # 치환될 값을 문자열로 변환
                            value_to_replace_with = str(lookup_value)
                            # 플레이스홀더 문자열 (예: "${project.root_dir}")
                            placeholder_str = f"${{{placeholder_key_path}}}"
                            # 문자열 치환 수행
                            resolved_string = resolved_string.replace(placeholder_str, value_to_replace_with)
                            logger.debug(f"_traverse_and_resolve- 치환 성공: '{placeholder_key_path}' -> '{lookup_value}' (in '{data}')")

                        except Exception as e:
                            logger.debug(f"_traverse_and_resolve- 치환 중 오류 발생 (값 변환 또는 치환): '{placeholder_key_path}' with '{lookup_value}' in '{data}' - {e}")
                            # 오류 발생 시 해당 플레이스홀더는 치환되지 않은 채로 남을 수 있습니다.

                    else:
                        logger.debug(f"_traverse_and_resolve-  플레이스홀더 값 찾기 실패: '{placeholder_key_path}' (in '{data}') at path '{current_key_path_str}'")
                        # 값을 찾지 못한 경우 어떻게 처리할지 결정해야 합니다.
                        # (예: 플레이스홀더를 그대로 남겨두거나, 오류 발생, 기본값 설정 등)
                        # 현재는 그대로 남겨둡니다.

                # 문자열 값을 담고 있는 부모 객체 (딕셔너리 또는 리스트)를 찾아서
                # 이 resolved_string으로 해당 값을 업데이트해야 합니다.
                # 재귀 호출 방식에서는 부모 객체의 키/인덱스를 알아야 합니다.
                # 이를 위해 재귀 호출 시 부모 객체와 자신의 키/인덱스를 함께 넘겨주는 방식도 가능합니다.
                # 또는 값을 반환하는 방식으로 구현할 수도 있습니다.
                # 현재 코드는 값을 반환하지 않고 재귀 호출만 하므로, 문자열 치환 결과를
                # 부모 객체에 반영하려면 data 객체를 직접 수정해야 합니다.

                # 문자열이 딕셔너리의 값인 경우
                if current_key_path_list and isinstance(self._get_parent_from_key_path(self.cfg, current_key_path_list[:-1]), dict):
                     parent_obj = self._get_parent_from_key_path(self.cfg, current_key_path_list[:-1])
                     last_key = current_key_path_list[-1]
                     parent_obj[last_key] = resolved_string # 부모 딕셔너리의 값 업데이트
                     logger.debug(f"_traverse_and_resolve- 딕셔너리 값 업데이트 at '{current_key_path_str}'\n")

                # 문자열이 리스트의 요소인 경우
                elif current_key_path_list and isinstance(self._get_parent_from_key_path(self.cfg, current_key_path_list[:-1]), list):
                     parent_obj = self._get_parent_from_key_path(self.cfg, current_key_path_list[:-1])
                     last_index = current_key_path_list[-1] # 리스트 인덱스
                     # 인덱스가 유효한지 확인
                     if isinstance(last_index, int) and 0 <= last_index < len(parent_obj):
                          parent_obj[last_index] = resolved_string # 부모 리스트의 요소 업데이트
                          logger.debug(f"_traverse_and_resolve- 리스트 요소 업데이트 at '{current_key_path_str}'\n")
                     else:
                          logger.debug(f"_traverse_and_resolve- 리스트 인덱스 오류: '{last_index}' at path '{current_key_path_str}'\n")

                # 최상위 문자열인 경우는 거의 없겠지만, 필요하다면 처리
                elif not current_key_path_list and isinstance(self.cfg, str):
                     self.cfg = resolved_string
                     logger.debug(f"_traverse_and_resolve-최상위 문자열 업데이트\n")

            else:
                logger.debug(f"_traverse_and_resolve: 문자열 '{data}'에서 플레이스홀더(${'{...}'})를 찾지 못했습니다.") # 로그 레벨 변경 및 메시지 명확화

            # 문자열인 경우 치환 여부와 관계없이 여기서 처리가 끝납니다.
            # 값을 반환하는 방식이라면 여기서 resolved_string을 반환해야 합니다.
            # 하지만 여기서는 data 객체를 직접 수정하므로 별도 반환이 없습니다.
            pass # 문자열 처리가 완료되었으므로 재귀 중단

        else:
            # 딕셔너리, 리스트, 문자열이 아닌 다른 타입의 데이터는 순회할 필요 없음
            logger.info(f"_traverse_and_resolve-스칼라 값 (처리 건너뜀) in _traverse_and_resolve 호출 - {data}, 현재 경로: {current_key_path_str}\n")
            logger.info(f"_traverse_and_resolve-재귀 끝---------------------------------------------------------------------")
            pass # 재귀 중단

    def _get_value_from_key_path(self, data, key_path):
        """
        점(.)으로 구분된 키 경로를 사용하여 딕셔너리 또는 리스트에서 값을 찾습니다.
        (예: 'project.paths.dataset.datasets_dir' 또는 'list_key.0.nested_key')

        Args:
            data (dict or list): 값을 찾을 데이터 구조 (보통 최상위 self.cfg)
            key_path (str): 점으로 구분된 키 경로

        Returns:
            any: 찾은 값, 또는 키 경로가 유효하지 않으면 None
        """
        keys = key_path.split('.')
        current_data = data
        logger.debug(f">>> DEBUG: 값 조회 시작 for path: {key_path}")
        try:
            for i, key in enumerate(keys):
                logger.debug(f">>> DEBUG: 현재 데이터 타입: {type(current_data)}, 찾을 키: {key}")
                if isinstance(current_data, dict) and key in current_data:
                    # 딕셔너리이고 키가 존재하면 다음 단계로 이동
                    current_data = current_data[key]
                elif isinstance(current_data, list):
                    # 현재 데이터가 리스트인 경우, 키는 인덱스(숫자 문자열)여야 함
                    try:
                        index = int(key)
                        if 0 <= index < len(current_data):
                            # 리스트 인덱스가 유효하면 다음 단계로 이동
                            current_data = current_data[index]
                        else:
                            # 유효하지 않은 리스트 인덱스
                            logger.warning(f"  값 조회 실패: 유효하지 않은 리덱스 '{key}' at path segment '{'.'.join(keys[:i+1])}'")
                            return None
                    except ValueError:
                        # 리스트인데 키가 숫자로 변환되지 않음
                        logger.warning(f"  값 조회 실패: 리스트에서 숫자가 아닌 키 '{key}' 사용 at path segment '{'.'.join(keys[:i+1])}'")
                        return None
                else:
                    # 현재 데이터가 딕셔너리나 리스트가 아니거나 키/인덱스가 존재하지 않음
                    logger.warning(f"  값 조회 실패: 경로 '{'.'.join(keys[:i+1])}'에서 유효한 키/인덱스 '{key}'를 찾을 수 없음. 현재 타입: {type(current_data)}")
                    return None # 경로 중간에 실패

            # 모든 키/인덱스를 따라 성공적으로 도달
            logger.debug(f">>> DEBUG: 값 조회 성공: {current_data}")
            return current_data
        except Exception as e:
            logger.error(f"값 조회 중 예외 발생 for path '{key_path}': {e}")
            return None # 예외 발생 시 실패

    def _get_parent_from_key_path(self, data, key_path_list):
        """
        키 경로 리스트를 사용하여 데이터 구조에서 최종 부모 객체(딕셔너리 또는 리스트)를 찾습니다.

        Args:
            data (dict or list): 탐색 시작 데이터 (보통 최상위 self.cfg)
            key_path_list (list): 부모까지의 키 경로 세그먼트 리스트

        Returns:
            dict or list or None: 찾은 부모 객체, 또는 경로가 유효하지 않으면 None
        """
        current_data = data
        try:
            # 마지막 키/인덱스 이전까지 탐색
            for key in key_path_list:
                if isinstance(current_data, dict) and key in current_data:
                    current_data = current_data[key]
                elif isinstance(current_data, list) and isinstance(key, int) and 0 <= key < len(current_data):
                    current_data = current_data[key]
                else:
                    # 경로 중간에 실패
                    return None
            # 최종적으로 도달한 객체가 부모 객체입니다.
            if isinstance(current_data, (dict, list)):
                 return current_data
            else:
                 # 경로의 마지막이 딕셔너리나 리스트가 아닌 경우 (예: project.name까지 탐색)
                 return None
        except Exception as e:
            logger.error(f"부모 객체 조회 중 예외 발생 for path '{'.'.join(map(str, key_path_list))}': {e}")
            return None

    def _load_yaml(self) -> dict:
        # 이전과 동일 (파일 로드 및 기본 예외 처리)
        """설정 파일 내용을 YAML 형식으로 로드합니다."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"설정 파일 구문 오류: {self.config_path} - {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"설정 파일 로딩 중 예상치 못한 오류 발생: {e}")
            sys.exit(1)

    def _resolve_single_value(self, value: Any, context: dict) -> Any: # value 타입을 Any로 변경하는 것이 더 안전합니다.
        """단일 값에서 플레이스홀더 (${...})를 context 또는 환경 변수를 사용하여 치환합니다."""
        # 값이 문자열이 아니면 치환할 필요가 없으므로 그대로 반환

        if not isinstance(value, str):
            return value

        original_value = value # 치환 전 원본 값 저장
        resolved_value = value # 치환 후 값을 저장할 변수 초기화

        # ${변수} 형태의 플레이스홀더를 찾는 정규 표현식
        pattern = re.compile(r"\$\{([^}]+)\}")

        # 여러 번 치환이 필요할 수 있으므로 (예: 중첩 플레이스홀더) 최대 5회 반복
        # 참고: 현재 로직은 중첩 플레이스홀더를 완벽하게 처리하지는 못할 수 있습니다.
        for _ in range(10):
            made_change = False
            temp_value = resolved_value # 현재까지 치환된 값을 임시 변수에 복사
            matches = pattern.findall(temp_value) # 임시 변수에서 플레이스홀더 찾기

            if not matches:
                # 더 이상 플레이스홀더가 없으면 반복 중단
                break

            for match in matches:
                placeholder_key = match # 플레이스홀더 변수 이름 (예: 'root_dir')

                # context에서 값을 찾거나, 필요하다면 환경 변수에서도 찾도록 로직 추가 가능
                replacement = context.get(placeholder_key) # context에서 값 가져오기
                logger.debug(f"match: {match}, placeholder_key: {placeholder_key}, replacement: {replacement}")

                if replacement is not None:
                    # context에 해당 키가 있으면 값 치환
                    placeholder_str = f"${{{placeholder_key}}}" # 예: "${root_dir}"
                    if placeholder_str in temp_value:
                         # 실제 플레이스홀더 문자열이 존재할 경우에만 치환 수행
                        temp_value = temp_value.replace(placeholder_str, str(replacement))
                        made_change = True # 변경이 발생했음을 표시
                else:
                    # context에서 값을 찾을 수 없는 경우 (필요시 경고 로깅)
                    logger.warning(f"Context에서 플레이스홀더 '{placeholder_key}'에 대한 값을 찾을 수 없습니다. (원본 값: '{original_value}')")
                    # 이 경우 해당 플레이스홀더는 치환되지 않은 채로 남습니다.

            resolved_value = temp_value # 이번 루프에서 변경된 내용을 반영
            if not made_change:
                # 이번 루프에서 어떤 치환도 일어나지 않았으면 더 이상 바뀔 것이 없으므로 중단
                break

        # --- 로깅 코드를 for 루프 바깥으로 이동 ---
        # 치환 전 값과 최종 치환 후 값이 다를 경우에만 로깅
        if original_value != resolved_value:
            logger.debug(f"플레이스홀더 치환: '{original_value:20s}' -> '{resolved_value:35s}'")
        # -----------------------------------------

        # --- return 문도 for 루프 바깥으로 이동 ---
        return resolved_value # 최종 치환된 값을 반환


    def _resolve_placeholders(self, config: dict, context: dict) -> dict:
         # 이 함수는 이제 _load_and_resolve_config에서 최종적으로 만들어진 context를 받아
         # 전달받은 config 객체 전체를 대상으로 치환 및 Path 변환을 수행합니다.
        """
        [비공개 메서드] 전체 설정 딕셔너리를 순회하며 플레이스홀더 치환 및 Path 객체 변환
        """
        def resolve_value_and_path(key, value, current_context):
            # 이 내부 함수는 raw_config를 사용하지 않습니다. (이전과 동일)
            # ... (resolve_value_and_path 함수 내용) ...
            resolved_value = value

            if isinstance(value, str):
                # 플레이스홀더 치환 로직 (current_context 사용)
                # self._resolve_single_value 함수가 original_value_for_debug를 사용한다면
                # 해당 변수를 resolve_value_and_path 함수의 인자로 전달하거나
                # 다른 방식으로 사용 가능하게 해야 합니다. 현재 코드에서는 original_value_for_debug가
                # 정의되지 않았을 수 있습니다. 이 부분을 확인하고 수정해야 합니다.
                original_value_for_debug = value # <-- 디버그용 변수 정의 추가
                resolved_value = self._resolve_single_value(value, current_context) # context 인자 사용
                # 치환이 발생했는지 로깅
                if resolved_value != original_value_for_debug:
                     logger.debug(f"[resolve_value] 키 '{key:20s}': 값이 바뀜 '{original_value_for_debug:35s}' -> '{resolved_value}'")


            # 키 이름 규칙에 따라 Path 객체로 변환 (이전과 동일)
            if isinstance(resolved_value, str) and key is not None:
                if key.endswith("_dir") or key.endswith("_path"):
                    try:
                        path_obj = Path(resolved_value).expanduser().resolve()
                        return path_obj
                    except Exception as e:
                        logger.error(f"_resolve_placeholders에서 경로 문자열을 Path 객체로 변환 중 오류 ('{key}': '{resolved_value}'): {e}")
                        return resolved_value

            return resolved_value


        def recursive_resolve(obj, current_context):
            # 이 내부 함수도 raw_config를 사용하지 않습니다. (이전과 동일)
            """설정 객체를 재귀적으로 순회하며 resolve_value_and_path 적용"""
            if isinstance(obj, dict):
                resolved_dict = {}
                for k, v in obj.items():
                     # resolved_value_and_path를 호출할 때 필요한 인자들을 정확히 전달합니다.
                    resolved_dict[k] = recursive_resolve(resolve_value_and_path(k, v, current_context), current_context) # 재귀 호출
                return resolved_dict
            elif isinstance(obj, list):
                return [recursive_resolve(v, current_context) for v in obj] # 재귀 호출
            else:
                return obj # dict나 list가 아니면 그대로 반환


        # 최종적으로 완성된 context를 사용하여 전달받은 config 객체 재귀적 치환 및 Path 객체 변환
        # raw_config를 사용하는 아래 라인을 제거합니다.
        # resolved_config_raw = recursive_resolve(copy.deepcopy(raw_config), context) # 이 라인 제거

        # 대신 전달받은 config 객체를 처리합니다.
        final_resolved_config = recursive_resolve(copy.deepcopy(config), context) # config 매개변수 사용

        # --- 추가했던 디버그 print들은 최종 결과인 final_resolved_config에 대해 수행합니다 ---
        logger.debug(f">>> DEBUG PRINT (resolve): _resolve_placeholders 반환 값 타입: {type(final_resolved_config)}")
        # 너무 크다면 일부만 출력
        logger.debug(f">>> DEBUG PRINT (resolve): _resolve_placeholders 반환 값 내용 (처음 200자): {str(final_resolved_config)[:200]}...")
        # --- 추가 끝 ---

        # 최종 결과를 반환합니다.
        return final_resolved_config # final_resolved_config 변수 반환

    def get_value(self, key: str, default: Any = None, ensure_exists: bool = True) -> Union[Path, Any]:
        """
        중첩된 키 이름(ex: 'dataset.raw_image_dir')을 기반으로 Path 객체 또는 값를 반환합니다.
        """
        keys = key.split(".")
        current_level = self.cfg # 현재 탐색 중인 딕셔너리/리스트

        # logger.error(f"values:{self.cfg}, keys:{keys}") # 이 로그는 except 블록 안에서 찍혔으므로 제거 또는 이동

        logger.debug(f"get_value called for key: '{key}', split keys: {keys}")

        try:
            # 마지막 딕셔너리를 찾는다.
            for i, k in enumerate(keys):
                logger.debug(f"  --- get_value 루프 {i}번째 반복 ---")
                logger.debug(f"  현재 레벨 (타입: {type(current_level)}): {current_level}") # <-- 현재 레벨 딕셔너리/값 출력
                logger.debug(f"  접근 시도 키: '{k}'") # <-- 현재 접근하려는 키 출력

                if isinstance(current_level, Dict):
                    # 이 라인(current_level[k])에서 KeyError가 발생할 수 있습니다.
                    current_level = current_level[k]
                # 리스트 인덱싱 지원이 필요하다면 여기에 elif isinstance(current_level, List)... 추가
                else:
                    # 중간 단계에서 딕셔너리/리스트가 아닌데 아직 최종 키가 아닌 경우
                    if i < len(keys) - 1:
                         logger.error(f"경로 탐색 실패: 키 '{key}'의 중간 경로 '{k}'가 딕셔너리나 리스트가 아닙니다. (현재 타입: {type(current_level)})")
                         raise TypeError(f"Intermediate key '{k}' for path '{key}' leads to non-container type ({type(current_level)}).")
                    # 최종 키라면, current_level이 최종 값 그대로입니다.

                logger.debug(f"    접근 성공: 키 '{k}'. 새로운 현재 값/레벨 (타입: {type(current_level)}): {current_level}")

            # 루프가 끝까지 돌면 current_level이 최종 값입니다.
            final_value = current_level
            logger.debug(f"키 탐색 완료: 최종 값 '{key}' -> {final_value}")

            # Path 객체 변환 로직 (필요하다면 추가 - 현재 config는 이미 Path 객체를 포함하는 것으로 보임)
            # ensure_exists 로직 (get_value에서는 제거하고 get_path에서 처리 권장)


            return final_value # 최종 값 반환

        except (KeyError, TypeError) as e:
            # current_level[k] 접근 실패 시 이 블록으로 옵니다.
            # logger.error(f"get_value 내부 오류: {e}", exc_info=True) # 중복 로깅 방지
            if default is not None:
                logger.warning(f"키 '{key}' 탐색 실패. 기본값 반환: {default}")
                return default
            else:
                logger.warning(f"키 '{key}' 탐색 실패 (기본값 제공되지 않음). None 반환. 키 경로: {keys}")
                return None

        except Exception as e:
             logger.error(f"get_value 중 예상치 못한 오류 발생: {e}", exc_info=True)
             if default is not None:
                 logger.warning(f"키 '{key}'에 대한 예상치 못한 오류 발생. 기본값 반환: {default}")
                 return default
             else:
                 raise
    # get_path, get_project_config, get_dataset_config 등 나머지 getter 메서드는
    # _load_and_resolve_config가 반환하는 fully_resolved_config 객체에 대해
    # 이전처럼 동일하게 작동할 것입니다.
    # (fully_resolved_config는 raw_config와 동일한 구조를 가지지만, 값이 치환되고 Path 객체로 변환됨)

    def get_path(self, key: str, default: Any = None, ensure_exists: bool = True) -> Union[Path, Any]:
        """
        설정에서 경로 키에 해당하는 Path 객체를 직접 반환합니다.
        get_value를 사용하여 값을 가져오고, ensure_exists=True일 경우,
        키 이름이 '_dir'로 끝나면 해당 디렉토리를, '_path'로 끝나면 부모 디렉토리를 생성 시도합니다.
        """
        # # 마지막 마침표의 위치를 찾습니다.
        # last_dot_index = key.rfind('.')
        # key_base = full_string[:last_dot_index]
        # last_key = full_string[last_dot_index + 1:]
        # last_cfg = self.cfg.get_config(key_base)
        # return last_cfg.get(last_key, None)


        raw_value = None
        try:
            # get_value를 호출하여 원시 값을 가져옵니다.
            # get_value 내부의 ensure_exists는 False로 하여, get_path에서 존재 유무를 최종 결정합니다.
            raw_value = self.get_value(key, default=None, ensure_exists=False)
        except (KeyError, TypeError) as e:
            if default is not None:
                logger.warning(f"키 '{key}'를 가져오는 중 오류 발생 (get_path): {e}. 기본값 반환: {default}")
                return default
            logger.error(f"키 '{key}'를 가져오는 중 오류 발생 (get_path): {e}. 기본값이 제공되지 않았습니다.")
            raise

        if raw_value is None: # get_value가 None을 반환 (키를 찾지 못했거나 값이 None)
            if default is not None: # get_path에 제공된 기본값 사용
                logger.debug(f"get_path: 키 '{key}'를 찾을 수 없거나 값이 None입니다. get_path의 기본값 '{default}' 반환.")
                return default
            else: # get_path에 제공된 기본값이 없으면 None 반환 (호출자가 처리)
                logger.warning(f"get_path: 키 '{key}'를 찾을 수 없거나 값이 None입니다. 기본값이 제공되지 않았으므로 None을 반환합니다.")
                return None

        path_obj = None
        if isinstance(raw_value, Path):
            path_obj = raw_value
        elif isinstance(raw_value, str):
            if not raw_value.strip(): # 빈 문자열 또는 공백만 있는 문자열 처리
                logger.warning(f"경로 키 '{key}'의 값이 빈 문자열입니다.")
                if default is not None: return default
                if ensure_exists: raise ValueError(f"경로 키 '{key}'의 값이 빈 문자열이고 ensure_exists=True입니다.")
                return None # 또는 빈 Path 객체 Path('')를 반환할 수도 있습니다.
            try:
                path_obj = Path(raw_value).expanduser()
            except Exception as e:
                logger.error(f"문자열 '{raw_value}'을 Path 객체로 변환 중 오류 (키: '{key}'): {e}")
                if default is not None: return default
                raise ValueError(f"유효하지 않은 경로 문자열 (키: '{key}'): '{raw_value}'") from e
        else:
            logger.warning(f"경로 키 '{key}'의 값이 문자열이나 Path 객체가 아닙니다 (타입: {type(raw_value)}). 값: '{raw_value}'")
            return default if default is not None else raw_value

        if path_obj is not None and ensure_exists:
            try:
                target_dir_to_create = None
                if key.endswith("_dir"):
                    target_dir_to_create = path_obj
                elif key.endswith("_path"):
                    target_dir_to_create = path_obj.parent # 파일 경로의 경우 부모 디렉토리
                else:
                    logger.debug(f"키 '{key}'는 '_dir' 또는 '_path'로 끝나지 않습니다. '{path_obj}'에 대한 자동 디렉토리 생성은 적용되지 않을 수 있습니다.")

                if target_dir_to_create:
                    if not target_dir_to_create.exists():
                        logger.debug(f"디렉토리 '{target_dir_to_create}' (키: '{key}')가 존재하지 않습니다. 생성합니다...")
                        target_dir_to_create.mkdir(parents=True, exist_ok=True)
                        logger.debug(f"디렉토리 '{target_dir_to_create}' 생성 완료.")
                    elif not target_dir_to_create.is_dir():
                        logger.error(f"경로 '{target_dir_to_create}' (키: '{key}')는 존재하지만 디렉토리가 아닙니다.")
                        raise NotADirectoryError(f"경로 '{target_dir_to_create}' (키: '{key}')는 존재하지만 디렉토리가 아닙니다.")
                    else:
                        logger.debug(f"디렉토리 '{target_dir_to_create}' (키: '{key}')가 이미 존재합니다.")
            
            except OSError as e:
                logger.warning(f"디렉토리 '{target_dir_to_create}' (키: '{key}') 생성 실패 (OSError): {e}. 권한을 확인하세요.")
                raise # ensure_exists가 True이므로 오류를 다시 발생시켜 호출자가 알 수 있도록 함
            except Exception as e:
                logger.error(f"디렉토리 확인/생성 중 예기치 않은 오류 발생 (키: '{key}', 경로 객체: '{path_obj}'): {e}")
                raise # 예기치 않은 오류도 다시 발생

        return path_obj

    def get_path_list(self, key: str, default: Optional[List[str]] = None, ensure_exists: bool = True) -> List[str]:
        """
        설정에서 경로 키에 해당하는 디렉토리의 하위 디렉토리 이름 목록을 반환합니다.
        Args:
            key (str): 설정에서 찾을 점으로 구분된 키 경로 (예: 'project.paths.data_dir').
            default (Optional[List[str]], optional): 키를 찾지 못하거나 오류 발생 시 반환할 기본값.
                                                     기본값은 None이며, 이 경우 빈 리스트가 반환됩니다.
            ensure_exists (bool, optional): True이면 get_path를 통해 경로를 가져올 때
                                            경로(또는 파일의 부모 디렉토리)가 존재하도록 시도합니다. 기본값은 True.
        Returns:
            List[str]: 하위 디렉토리 이름의 리스트. 경로를 찾지 못하거나, 경로가 디렉토리가 아니거나,
                       오류 발생 시 'default'가 None이면 빈 리스트를 반환하고, 아니면 'default' 값을 반환합니다.
        """
        logger.debug(f"get_path_list 호출됨: key='{key}', default={default}, ensure_exists={ensure_exists}")
        try:
            # 1. get_path를 사용하여 기본 경로 객체를 가져옵니다.
            #    get_path_list의 ensure_exists를 get_path에 전달합니다.
            #    get_path에서 키를 못 찾으면 예외가 발생하거나 get_path의 default가 반환될 수 있습니다.
            #    여기서는 get_path가 None을 반환할 수 있도록 default=None으로 호출하고, 여기서 최종 default를 처리합니다.
            base_path_obj = self.get_path(key, default=None, ensure_exists=ensure_exists)

            # 2. base_path_obj 유효성 검사
            if base_path_obj is None:
                logger.warning(f"get_path_list: 키 '{key}'에 해당하는 경로를 찾을 수 없습니다. default 값 '{default}' 반환.")
                return default if default is not None else []

            if not isinstance(base_path_obj, Path):
                logger.warning(f"get_path_list: 키 '{key}'의 값은 Path 객체가 아니지만 '{type(base_path_obj)}' 타입입니다. default 값 '{default}' 반환.")
                return default if default is not None else []

            if not base_path_obj.is_dir():
                logger.warning(f"get_path_list: 경로 '{base_path_obj}' (키: '{key}')는 디렉토리가 아닙니다. 빈 리스트 반환.")
                return [] # 디렉토리가 아니면 빈 리스트 (default를 반환하지 않음)

            # 3. 하위 디렉토리 이름 목록 가져오기
            sub_directories = [item.name for item in base_path_obj.iterdir() if item.is_dir()]
            logger.info(f"get_path_list: 경로 '{base_path_obj}' (키: '{key}')에서 다음 하위 디렉토리들을 찾았습니다: {sub_directories}")
            return sub_directories

        except Exception as e: # get_path 내부에서 발생한 예외 또는 여기서 발생한 경로 관련 예외 포함
            logger.error(f"get_path_list: 키 '{key}' 처리 중 오류 발생: {e}", exc_info=True)
            if default is not None:
                logger.warning(f"get_path_list: 오류로 인해 default 값 '{default}' 반환.")
                return default
            return [] # 오류 발생 시 default가 없으면 빈 리스트 반환

    """
    # 이 메소드는 get_path_list와 유사하지만, 파일 시스템의 디렉토리 대신 
    # 설정(YAML) 구조 내의 딕셔너리 키를 다룹니다.
    # 다음은 configger.py 파일에 적용될 변경 사항입니다.
    """
    def get_key_list(self, key: str, default: Optional[List[str]] = None) -> List[str]:
        """
        설정에서 특정 키 경로에 해당하는 딕셔너리의 하위 키 이름 목록을 반환합니다.
        Args:
            key (str): 설정에서 찾을 점으로 구분된 키 경로 (예: 'project.paths').
            default (Optional[List[str]], optional): 키를 찾지 못하거나, 해당 값이 딕셔너리가 아니거나,
                                                     오류 발생 시 반환할 기본값.
                                                     기본값은 None이며, 이 경우 빈 리스트가 반환됩니다.
        Returns:
            List[str]: 하위 키 이름의 리스트. 조건을 만족하지 못하면 'default'가 None일 경우
                       빈 리스트를, 아니면 'default' 값을 반환합니다.
        """
        logger.debug(f"get_key_list 호출됨: key='{key}', default={default}")
        try:
            # 1. get_value를 사용하여 해당 키의 값을 가져옵니다.
            target_value = self.get_value(key, default=None) # ensure_exists는 get_value에서 처리

            # 2. target_value 유효성 검사
            if target_value is None:
                logger.warning(f"get_key_list: 키 '{key}'에 해당하는 값을 찾을 수 없습니다. default 값 '{default}' 반환.")
                return default if default is not None else []

            if not isinstance(target_value, dict):
                logger.warning(f"get_key_list: 키 '{key}'의 값은 딕셔너리가 아니지만 '{type(target_value)}' 타입입니다. default 값 '{default}' 반환.")
                return default if default is not None else []

            # 3. 하위 키 이름 목록 가져오기
            sub_keys = list(target_value.keys())
            logger.info(f"get_key_list: 키 '{key}'에서 다음 하위 키들을 찾았습니다: {sub_keys}")
            return sub_keys
        except Exception as e: # get_value 내부에서 발생한 예외 또는 여기서 발생한 예외 포함
            logger.error(f"get_key_list: 키 '{key}' 처리 중 오류 발생: {e}", exc_info=True)
            return default if default is not None else []

    def get_value_list(self, key: str, default: Optional[List[Any]] = None) -> List[Any]:
        """
        설정에서 특정 키 경로에 해당하는 리스트 값을 반환합니다.
        해당 키의 값이 리스트가 아니면 default 값을 반환합니다.

        Args:
            key (str): 설정에서 찾을 점으로 구분된 키 경로 (예: 'project.features').
            default (Optional[List[Any]], optional): 키를 찾지 못하거나, 해당 값이 리스트가 아니거나,
                                                     오류 발생 시 반환할 기본값.
                                                     기본값은 None이며, 이 경우 빈 리스트가 반환됩니다.

        Returns:
            List[Any]: 설정에서 가져온 리스트. 조건을 만족하지 못하면 'default'가 None일 경우
                       빈 리스트를, 아니면 'default' 값을 반환합니다.
        """
        logger.debug(f"get_value_list 호출됨: key='{key}', default={default}")
        try:
            # 1. get_value를 사용하여 해당 키의 값을 가져옵니다.
            # get_value는 키가 없거나 값이 null이면 None을 반환할 수 있습니다 (자신의 default가 None일 때).
            target_value = self.get_value(key, default=None)

            # 2. target_value 유효성 검사
            if target_value is None: # 키를 찾지 못했거나, 키의 값이 명시적으로 null인 경우
                logger.warning(f"get_value_list: 키 '{key}'에 해당하는 값을 찾을 수 없거나 값이 null입니다. default 값 '{default}' 반환.")
                return default if default is not None else []

            if not isinstance(target_value, list):
                logger.warning(f"get_value_list: 키 '{key}'의 값은 리스트가 아니지만 '{type(target_value)}' 타입입니다. default 값 '{default}' 반환.")
                return default if default is not None else []

            # 3. 값이 리스트인 경우 해당 리스트 반환
            logger.info(f"get_value_list: 키 '{key}'에서 다음 리스트 값을 찾았습니다: {target_value}")
            return target_value # target_value is already a list
        except Exception as e: # get_value 내부에서 발생한 예외 또는 여기서 발생한 예외 포함
            logger.error(f"get_value_list: 키 '{key}' 처리 중 오류 발생: {e}", exc_info=True)
            return default if default is not None else []

    def get_config(self, key: str) -> dict | None: # None을 반환할 수도 있음을 명시
        """
        주어진 점(.)으로 구분된 키 문자열에 해당하는 중첩된 설정 딕셔너리 또는 값을 가져옵니다.
        키가 존재하지 않으면 None을 반환합니다.
        """
        keys = key.split(".")
        cur_cfg = self.cfg

        try:
            for i, key_name in enumerate(keys):
                logger.debug(f"순번: {i}, '{key_name}'")
                if not isinstance(cur_cfg, dict):
                    # 현재 레벨이 딕셔너리가 아닌데 계속 접근하려고 하면 오류
                    logger.warning(f"키 경로 '{key}'의 '{key_name}' 접근 중: 이전 레벨이 딕셔너리가 아닙니다 (타입: {type(cur_cfg).__name__}).")
                    return None # 딕셔너리가 아닌 곳에서 더 이상 내려갈 수 없음

                # 따옴표 없이 key_name 변수 사용, 기본값 None 사용 권장
                # 마지막 키가 아니라면 다음 레벨은 딕셔너리일 것으로 예상하고 기본값을 {} 대신 None으로 설정
                # 마지막 키라면 어떤 값이든 될 수 있으므로 기본값은 None
                next_cfg = cur_cfg.get(key_name)

                if next_cfg is None and i < len(keys) - 1:
                    # 중간 경로에 해당하는 키가 없으면 나머지 키는 찾을 수 없음
                    logger.warning(f"키 경로 '{key}'의 '{key_name}'를 찾을 수 없습니다.")
                    return None

                logger.debug(f"next_cfg: {next_cfg}")
                cur_cfg = next_cfg

            # 루프 완료 후 최종 값 반환
            # 최종 값이 None일 수 있습니다 (키가 존재하지 않았거나 값이 실제로 None인 경우)
            return cur_cfg

        except Exception as e:
            # 예상치 못한 다른 오류 발생 시
            logger.error(f"키 경로 '{key}' 값 가져오는 중 예상치 못한 오류 발생: {e}")
            # 필요에 따라 sys.exit(1) 호출
            return None


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

        # 4. 설정 값 사용 예시 (테스트용)
        project_name = config_manager.get_value("project.name", default="N/A")
        logger.info(f"설정 파일 ('project.project_name') 값: {project_name}")

        # --- Prepare config_manager.cfg for specific tests ---
        # Ensure 'project' key exists and is a dictionary
        if not isinstance(config_manager.cfg.get('project'), dict):
            config_manager.cfg['project'] = {}
        
        # Ensure 'project.paths' key exists and is a dictionary for get_path_list tests
        # This might be created by the loaded YAML, but ensure it for tests
        if not isinstance(config_manager.cfg['project'].get('paths'), dict):
            config_manager.cfg['project']['paths'] = {}

        # Add/ensure keys for get_key_list tests
        # 'project.name' is likely a string from loaded config, used for non_dict_keys test.
        config_manager.cfg['project']['empty_dict_key'] = {}
        config_manager.cfg['project']['list_key'] = []
        # 'project.non_existent_section' is tested as non-existent, so no need to add.
        # 'json_keys' is expected to be a top-level key from your YAML for the first get_key_list test.
        if 'json_keys' not in config_manager.cfg:
            config_manager.cfg['json_keys'] = {"sample_json_key": "sample_value"} # Add a dummy if not present

        # Add/ensure keys for get_value_list tests
        config_manager.cfg['project']['active_plugins'] = ['plugin_a', 'plugin_b', 'plugin_c']
        config_manager.cfg['project']['thresholds'] = [0.1, 0.5, 0.95]
        config_manager.cfg['project']['empty_list_example'] = []
        config_manager.cfg['project']['max_attempts'] = 10 # Example non-list value for testing

    except (KeyError, TypeError, AttributeError) as e:
        logger.error(f"모델 변수값 가저오기 오류 발생: {e}")
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

    # 4. get_key_list 테스트
    try:
        logger.info(f"--- get_key_list 테스트 시작 ---")
        project_keys = config_manager.get_key_list("json_keys")
        logger.info(f"'json_keys' 하위 키: {project_keys} (예상: 실제 YAML의 json_keys 내용 또는 ['sample_json_key'])")

        path_keys = config_manager.get_key_list("project.paths")
        logger.info(f"'project.paths' 하위 키: {path_keys} (예상: 실제 YAML의 project.paths 내용 또는 테스트로 추가된 키)")

        non_dict_keys = config_manager.get_key_list("project.name", default=["name_is_not_dict"])
        logger.info(f"'project.name' (문자열) 하위 키: {non_dict_keys} (예상: ['name_is_not_dict'])")

        empty_dict_subkeys = config_manager.get_key_list("project.empty_dict_key")
        logger.info(f"'project.empty_dict_key' 하위 키: {empty_dict_subkeys} (예상: [])")

        list_val_keys = config_manager.get_key_list("project.list_key", default=["list_is_not_dict"])
        logger.info(f"'project.list_key' (리스트 값) 하위 키: {list_val_keys} (예상: ['list_is_not_dict'])")

        non_existent_keys = config_manager.get_key_list("project.non_existent_section", default=["default_keys"])
        logger.info(f"'project.non_existent_section' 하위 키: {non_existent_keys} (예상: ['default_keys'])")

        # 5. get_path_list 테스트용 임시 디렉토리 및 파일 생성
        test_root_dir_str = config_manager.get_value("project.root_dir", default="test_temp_config_root")
        if isinstance(test_root_dir_str, Path): # get_value가 Path 객체를 반환할 수 있음
            test_root_dir = test_root_dir_str
        else:
            test_root_dir = Path(test_root_dir_str)

        if test_root_dir.exists() and str(test_root_dir) != str(Path.home()) and str(test_root_dir) != "/": # 안전장치
            shutil.rmtree(test_root_dir)
        test_root_dir.mkdir(parents=True, exist_ok=True)

        data_dir = test_root_dir / "data"
        data_dir.mkdir()
        (data_dir / "sub_dir1").mkdir()
        (data_dir / "sub_dir2").mkdir()
        (data_dir / "not_a_dir.txt").touch()

        empty_dir = test_root_dir / "empty"
        empty_dir.mkdir()

        some_file = test_root_dir / "some_file.txt"
        some_file.touch()

        logger.info(f"테스트용 디렉토리 구조 생성 완료: {test_root_dir}")

    except (KeyError, TypeError, AttributeError) as e:
        logger.error(f"모델 변수값 가저오기 오류 발생: {e}")
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

    # 5. get_path_list 테스트
    try:
        logger.info(f"--- get_path_list 테스트 시작 ---")
        # 실제 YAML에 project.paths.data_dir 등이 정의되어 있다고 가정하고 테스트
        # 이 예제에서는 config_manager.cfg에 직접 값을 넣어 테스트하거나,
        # 테스트용 YAML을 사용해야 합니다. 여기서는 get_value로 가져온 경로를 사용합니다.

        # 테스트 1: 하위 디렉토리가 있는 경우
        # YAML에 "project.paths.test_data_dir": "${project.root_dir}/data" 와 같이 정의되어 있어야 함
        # 여기서는 직접 경로를 만들어 테스트합니다. config_manager.cfg['project']['paths']['data_dir_for_test'] = str(data_dir)
        config_manager.cfg["project"]["paths"]["data_dir_for_test"] = str(data_dir)
        sub_dirs = config_manager.get_path_list("project.paths.data_dir_for_test")
        logger.info(f"project.paths.data_dir_for_test 하위 디렉토리: {sub_dirs} (예상: ['sub_dir1', 'sub_dir2'] 또는 순서 다름)")

        # 테스트 2: 하위 디렉토리가 없는 경우
        config_manager.cfg["project"]["paths"]["empty_dir_for_test"] = str(empty_dir)
        empty_sub_dirs = config_manager.get_path_list("project.paths.empty_dir_for_test")
        logger.info(f"project.paths.empty_dir_for_test 하위 디렉토리: {empty_sub_dirs} (예상: [])")

        # 테스트 3: 경로가 파일인 경우
        config_manager.cfg["project"]["paths"]["file_path_for_test"] = str(some_file)
        file_path_subs = config_manager.get_path_list("project.paths.file_path_for_test")
        logger.info(f"project.paths.file_path_for_test (파일 경로) 하위 디렉토리: {file_path_subs} (예상: [])")

        # 테스트 4: 존재하지 않는 키
        non_existent_subs = config_manager.get_path_list("project.paths.no_such_key_for_test", default=["default_val"])
        logger.info(f"project.paths.no_such_key_for_test 하위 디렉토리: {non_existent_subs} (예상: ['default_val'])")

        # 테스트 종료 후 임시 디렉토리 삭제
        if test_root_dir.exists() and "test_temp_config_root" in str(test_root_dir): # 안전장치 추가
            logger.info(f"get_path_list 테스트 완료. 임시 디렉토리 삭제: {test_root_dir}")
            shutil.rmtree(test_root_dir)

    except (KeyError, TypeError, AttributeError) as e:
        logger.error(f"모델 변수값 가저오기 오류 발생: {e}")
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

    # 6. get_value_list 테스트
    try:
        logger.info(f"--- get_value_list 테스트 시작 ---")

        # 'project.active_plugins' (문자열 리스트) 가져오기
        plugins = config_manager.get_value_list("project.active_plugins")
        logger.info(f"Active plugins: {plugins} (예상: ['plugin_a', 'plugin_b', 'plugin_c'])")

        # 'project.thresholds' (숫자 리스트) 가져오기
        threshold_values = config_manager.get_value_list("project.thresholds")
        logger.info(f"Thresholds: {threshold_values} (예상: [0.1, 0.5, 0.95])")

        # 'project.empty_list_example' (빈 리스트) 가져오기
        empty_list = config_manager.get_value_list("project.empty_list_example")
        logger.info(f"Empty list example: {empty_list} (예상: [])")

        # 값이 리스트가 아닌 경우 (기본값 사용)
        attempts_list = config_manager.get_value_list("project.max_attempts", default=[])
        logger.info(f"Max attempts (as list, project.max_attempts is int): {attempts_list} (예상: [])")

        # 존재하지 않는 키 (기본값 사용)
        unknown_list = config_manager.get_value_list("project.unknown_values", default=["default_item"])
        logger.info(f"Unknown values list: {unknown_list} (예상: ['default_item'])")

        # 존재하지 않는 키 (기본값 None, 빈 리스트 반환 예상)
        unknown_list_no_default = config_manager.get_value_list("project.unknown_values_no_default")
        logger.info(f"Unknown values list (no default): {unknown_list_no_default} (예상: [])")

    except Exception as e:
        logger.error(f"Error getting value list: {e}")

