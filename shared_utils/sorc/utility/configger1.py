# shared_utils/src/utility/configger.py

import yaml
import re
from pathlib import Path
import copy
import logging # 로깅을 위해 추가

# 로거 설정 (shared_utils 패키지 내부 로거)
# 로깅 기본 설정
logging.basicConfig(
    level=logging.DEBUG,  # 어느 레벨부터 로그를 출력할지 설정 (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    # 주의: format 문자열에 오타나 불필요한 문자가 있는 것 같습니다. 수정했습니다.
    # 원본: format='%(asctime)s - %(name)-10s - %(levelname)-8s - %(funcName)-10mv co   s - %(message)s',
    format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(funcName)-20s - %(message)s', # 로그 출력 형식 지정 (오타 수정 및 형식 개선)
    # stream=sys.stdout # 로그를 어디로 출력할지 설정 (기본값은 sys.stderr)
    # filename='app.log' # 로그를 파일에 저장하려면 이 줄 사용 (stream과 함께 사용하지 않음)
) # <-- 여기서 닫는 괄호 ')' 다음 줄로 이동합니다.

# 현재 모듈의 로거 객체 가져오기
logger = logging.getLogger(__name__)
# 핸들러와 포매터는 이 코드를 사용하는 주 애플리케이션(my_yolo_tiny)에서 설정해야 합니다.
# 예: logging.basicConfig(level=logging.INFO)

class configger(config_path):
    """
    YAML 설정 파일을 로드하고, 플레이스홀더를 치환하며,
    경로 요청 시 해당 디렉토리가 존재하도록 보장하는 클래스.
    """
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_and_resolve_config()
        logger.denug(f"1. __init__ config_path : {config_path}")

    def _load_yaml(self) -> dict:
        # ... (기존과 동일) ...
        logger.denug(f"config_path : {config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # _resolve_placeholders 메서드는 이전 답변의 코드를 사용하면 됩니다.
    # (플레이스홀더 치환 후 Path 객체 변환)
    def _resolve_placeholders(self, config: dict, context: dict) -> dict:
        # ... (이전 답변의 _resolve_placeholders 코드) ...
        pattern = re.compile(r"\$\{([^}]+)\}")

        logger.denug(f"0. raw_config : {raw_config}")
        def resolve_value(key, value):
            """값 치환 및 Path 객체 변환"""
            resolved_value = value
            original_value_for_debug = value # 로깅을 위해 원본 저장

            if isinstance(value, str):
                # 플레이스홀더 치환 로직 (context는 이미 완전히 해석된 값들을 가짐)
                resolved_value = self._resolve_single_value(value, context) # 수정된 도우미 함수 사용
                # 치환이 발생했는지 로깅
                # if resolved_value != original_value_for_debug:
                #      print(f"[resolve_value] 키 '{key:20s}': 값이 바뀜 '{original_value_for_debug:35s}' -> '{resolved_value}'")

            # 키 이름 규칙에 따라 Path 객체로 변환
            if isinstance(resolved_value, str) and key is not None:
                if key.endswith("_dir") or key.endswith("_path"):
                    try:
                        path_obj = Path(resolved_value).expanduser() # 이제 resolved_value는 완전한 경로 문자열
                        return path_obj
                    except Exception as e:
                        logger.error(f"경로 문자열을 Path 객체로 변환 중 오류 ('{key}': '{resolved_value}'): {e}")
                        return resolved_value
            return resolved_value

        def recursive_resolve(obj):
            if isinstance(obj, dict):
                return {k: recursive_resolve(resolve_value(k, v)) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_resolve(v) for v in obj]
            else:
                return resolve_value(None, obj)

        return recursive_resolve(config)

    def _resolve_single_value(self, value: str, context: dict) -> str:
        """단일 문자열 값 내의 플레이스홀더를 치환합니다."""
        pattern = re.compile(r"\$\{([^}]+)\}")
        matches = pattern.findall(value)
        resolved_value = value
        # 여러 플레이스홀더가 있을 수 있으므로 반복 치환
        # (주의: 순환 참조가 있으면 무한 루프 가능성 있음)
        for _ in range(5): # 최대 5번 반복하여 중첩된 플레이스홀더 처리 시도
            made_change = False
            temp_value = resolved_value
            matches = pattern.findall(temp_value)
            if not matches:
                break
            for match in matches:
                if match in context:
                    replacement = str(context[match])
                    if f"${{{match}}}" in temp_value: # 실제 치환이 일어나는지 확인
                        temp_value = temp_value.replace(f"${{{match}}}", replacement)
                        made_change = True
            resolved_value = temp_value
            if not made_change: # 더 이상 치환이 일어나지 않으면 종료
                break
        return resolved_value

    def _load_and_resolve_config(self) -> dict:
        """
        [비공개 메서드] 설정 파일 로드 후, 플레이스홀더 치환까지 완료
        """
        logger.denug(f"_load_and_resolve_config : {self.config_path}")
        raw_config = self._load_yaml()
        context = {}

        try:
            # 1. root_dir 먼저 결정 (절대 경로로)
            raw_root_dir = raw_config.get("project", {}).get("root_dir")
            if raw_root_dir:
                # expanduser()와 resolve()를 사용하여 절대 경로 Path 객체 생성
                resolved_root_dir_path = Path(raw_root_dir).expanduser().resolve()
                context["root_dir"] = str(resolved_root_dir_path) # context에는 문자열로 저장
            else:
                # root_dir이 없으면 다른 경로 해석 불가, 오류 발생 또는 기본값 설정 필요
                raise ValueError("설정 파일에 project.root_dir이 정의되지 않았습니다.")

            # 2. root_dir을 기반으로 다른 기본 경로들 결정 (절대 경로 문자열로)
            base_context_for_paths = {"root_dir": context["root_dir"]} # root_dir만 있는 context

            raw_dataset_dir = raw_config.get("project", {}).get("dataset", {}).get("dataset_dir")
            if raw_dataset_dir:
                resolved_dataset_dir_str = self._resolve_single_value(raw_dataset_dir, base_context_for_paths)
                context["dataset_dir"] = str(Path(resolved_dataset_dir_str).expanduser().resolve())

            raw_outputs_dir = raw_config.get("project", {}).get("outputs", {}).get("outputs_dir")
            if raw_outputs_dir:
                resolved_outputs_dir_str = self._resolve_single_value(raw_outputs_dir, base_context_for_paths)
                context["outputs_dir"] = str(Path(resolved_outputs_dir_str).expanduser().resolve())

            raw_utility_dir = raw_config.get("project", {}).get("utility", {}).get("utility_dir")
            if raw_utility_dir:
                resolved_utility_dir_str = self._resolve_single_value(raw_utility_dir, base_context_for_paths)
                context["utility_dir"] = str(Path(resolved_utility_dir_str).expanduser().resolve())

            raw_sorc_dir = raw_config.get("project", {}).get("source", {}).get("sorc_dir")
            if raw_sorc_dir:
                resolved_sorc_dir_str = self._resolve_single_value(raw_sorc_dir, base_context_for_paths)
                context["sorc_dir"] = str(Path(resolved_sorc_dir_str).expanduser().resolve())

            # context에 None 값 제거 (이미 절대 경로이므로 None이 없을 것으로 예상)
            context = {k: v for k, v in context.items() if v is not None}

        except KeyError as e:
            raise ValueError(f"YAML에서 context 생성을 위한 키 누락: {e}") from e
        except Exception as e:
            raise

        # 3. 최종적으로 완성된 context를 사용하여 전체 설정 재귀적 치환 및 Path 객체 변환
        logger.denug(f"1. raw_config : {raw_config}")
        return self._resolve_placeholders(raw_config, context)

    # ======= 외부에 제공하는 메서드 =======

    def get_path(self, key: str, default: Any = None, ensure_exists: bool = True) -> Union[Path, Any]:
        """
        config, dataset, outputs, templates
        설정에서 경로 키에 해당하는 Path 객체를 직접 반환합니다.
        ensure_exists=True일 경우, _dir 키는 해당 디렉토리를, _path 키는 부모 디렉토리를 생성 시도합니다.
        검색 순서: dataset -> outputs -> source -> models -> project (최상위)
        """
        value = None
        found_in_section = False

        search_sections = [
            self.get_dataset_config(),
            self.get_outputs_config(),
            self.get_source_config(),
            self.get_utility_config(),
            self.get_models_config(),
            self.get_project_config()
        ]

        for section in search_sections:
            if isinstance(section, dict) and key in section:
                value = section[key]
                found_in_section = True
                logger.error(f"get_path value: {value}")
                break # 첫 번째 섹션에서 찾으면 중단

        # 최상위 project 키 바로 아래에도 있는지 확인 (섹션에서 못 찾았을 경우)
        if not found_in_section:
            project_config = self.get_project_config()
            if key in project_config:
                 value = project_config[key]

        # 값을 찾았는지 확인 및 Path 객체 처리
        print_log(self.func_name, f"value: {value}")
        if value is not None:
            path_obj = None
            if isinstance(value, Path):
                path_obj = value
            # _resolve_placeholders에서 변환 실패했을 경우 대비 문자열 체크 추가
            elif isinstance(value, str) and (key.endswith("_dir") or key.endswith("_path")):
                 try:
                     path_obj = Path(value).expanduser()
                 except Exception as e:
                     logger.error(f"get_path에서 경로 변환 중 오류 ('{key}'): {e}")
                     return default

            if path_obj is not None:
                if ensure_exists: # 디렉토리 존재 확인 및 생성 로직
                    try:
                        target_dir_to_create = None
                        if key.endswith("_dir"):
                            target_dir_to_create = path_obj
                        elif key.endswith("_path"):
                            target_dir_to_create = path_obj.parent

                        if target_dir_to_create and not target_dir_to_create.exists():
                            target_dir_to_create.mkdir(parents=True, exist_ok=True)
                        # else: # 이미 존재하는 경우
                        #     print(f"  [get_path] 키 '{key}' 관련 디렉토리 이미 존재: {target_dir_to_create}") # 디버깅용

                    except OSError as e:
                        logger.warning(f"경로 자동 생성 실패 (권한 확인 필요): {target_dir_to_create} - {e}")
                    except Exception as e:
                        logger.error(f"경로 확인/생성 중 오류 발생 ('{key}'): {e}")
                        # 생성 실패 시에도 일단 경로 객체는 반환하거나, default 반환 결정 필요
                        # return default # 오류 시 기본값 반환하도록 변경 가능

                return path_obj # 최종 Path 객체 반환
            else:
                # 경로 키가 아니거나 Path 변환 실패 시 원본 값 반환
                return value

        # 모든 섹션에서 찾지 못하면 기본값 반환
        return default

    # get_project_config, get_dataset_config 등 다른 getter는 그대로 유지
    def get_project_config(self) -> dict:
        return self.config.get("project", {})
    def get_dataset_config(self) -> dict:
        return self.config["project"].get("dataset", {})
    def get_outputs_config(self) -> dict:
        return self.config["project"].get("outputs", {})
    def get_utility_config(self) -> dict:
        return self.config["project"].get("utility", {})
    def get_source_config(self) -> dict:
        return self.config["project"].get("source", {})
    def get_models_config(self) -> dict:
        return self.config.get("models", {})

# === 사용 예시 ===
if __name__ == "__main__":
    func_name = "main"
    print_log(func_name, "시작")

    # 0. 기즘 내가 일하는 곳은"
    direction_dir = os.getcwd()
    logger.info(f"지금 쥔계서 계신곳(direction_dir)      : {direction_dir}")
    worker_path_obj = Path(__file__).resolve()
    project_root_path = worker_path_obj.parent.parent
    logger.info(f"지금 일꾼이 일하는곳(worker_dir_name)  : {project_root_path}")

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    config_path = f"{project_root_path}/config/{project_root_path.name}.yaml"
    logger.info(f"받은 구성파일 경로(config_path)        : {config_path}")

    try:
        config = ProjectConfig(config_path)
        dir_obj = config.get_project_config()
        # Path 객체로 제대로 가져왔는지 확인
        project_root_dir = dir_obj.get("root_dir", "")
        logger.info(f"프로젝트 루트: {project_root_dir}")

        # get_path를 사용하여 로그 디렉토리 Path 객체 가져오기
        # ensure_exists=True (기본값)이므로, get_path 내부에서 디렉토리 생성 시도

        # 최종 로그 파일 경로 생성

        log_dir_path_obj = config.get_path("root_dir")
        log_dir_path_obj = config.get_path("config_path")
        log_dir_path_obj = config.get_path("raw_image_dir")
        log_dir_path_obj = config.get_path("face_detector")
        log_dir_path_obj = config.get_path("search_html")
        log_dir_path_obj = config.get_path("configger")
        log_dir_path_obj = config.get_path("worker_logs_dir")

        # Path 객체로 제대로 가져왔는지 확인
        if not isinstance(log_dir_path_obj, Path):
            logger.error(f"worker_logs_dir'를 Path 객체로 가져오지 못했습니다.")
            logger.error(f"log_dir_path_obj의 타입: {type(log_dir_path_obj)}")
            sys.exit(1)

        # 최종 로그 파일 경로 생성
        log_file_path = log_dir_path_obj / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        logger.info(f"로그 파일 경로: {log_file_path}")

        # --- 설정 파일에서 tolerance 신뢰도 값 읽기 ---
        models_config =     config.get_models_config()
        min_detection_confidence = float(models_config.get("min_detection_confidence", 0.6)) # models 섹션에서 가져오기
        target_size_tuple = tuple(models_config.get("target_size", [224, 224])) # 기본값 [224, 224]
        logger.info(f"사용할 정밀도(min_detection_confidence): {min_detection_confidence}, target_size: {target_size_tuple}") # 로깅 추가 (선택 사항)

        # get_path를 사용하여 로그 디렉토리 Path 객체 가져오기
        # ensure_exists=True (기본값)이므로, get_path 내부에서 디렉토리 생성 시도
        log_dir_path_obj = config.get_path("worker_logs_dir")

        # Path 객체로 제대로 가져왔는지 확인
        if not isinstance(log_dir_path_obj, Path):
            logger.error(f"worker_logs_dir'를 Path 객체로 가져오지 못했습니다.")
            logger.error(f"log_dir_path_obj의 타입: {type(log_dir_path_obj)}")
            sys.exit(1)

        # 최종 로그 파일 경로 생성
        log_file_path = log_dir_path_obj / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        logger.info(f"로그 파일 경로: {log_file_path}")

        # 수정된 add_file_logger 호출 (최종 파일 경로 전달)
#        add_file_logger(log_file_path)

        logger.info(f"발자국 그리기 : {log_file_path}")

    except (KeyError, TypeError, AttributeError) as e:
        logger.error(f"모델 변수값 가저오기 오류 발생: {e}")
        sys.exit(1)
    except Exception as e:
        sys.exit(1)
    
