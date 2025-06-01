class processing_config:
    """
    YAML 설정 파일을 로드하고,
    ${root_dir}, ${dataset_dir}, ${output_dir} 같은 플레이스홀더를 실제 경로로 치환한 뒤
    필요한 구성 단위(dataset, output, source, models 등)를 반환하는 클래스
    """

    def __init__(self, config_path: str):
        """
        [생성자]
        - 입력: config_path (str) -> YAML 설정 파일 경로
        - 출력: 없음 (클래스 내부에 config 저장)
        - 기능: 설정 파일 로드 및 경로 플레이스홀더 치환
        """
        self.config_path = Path(config_path)
        self.config = self._load_and_resolve_config()

    def _load_yaml(self) -> dict:
        """
        [비공개 메서드] YAML 파일을 읽어서 Python 딕셔너리로 변환
        - 입력: 없음 (self.config_path 사용)
        - 출력: config (dict)
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_placeholders(self, config: dict, context: dict) -> dict:
        """
        [비공개 메서드] 설정 딕셔너리 내 플레이스홀더(${var})를 실제 값으로 치환
        - 입력:
          - config: 원본 설정 딕셔너리
          - context: 치환할 키-값 매핑(dict) (ex: {"root_dir": "/home/user/project"})
        - 출력: 치환이 완료된 설정 딕셔너리
        """
        pattern = re.compile(r"\$\{([^}]+)\}")  # ${} 안의 변수를 찾는 정규식 패턴

        def resolve_value(value):
            if isinstance(value, str):
                matches = pattern.findall(value)
                for match in matches:
                    if match in context:
                        value = value.replace(f"${{{match}}}", str(context[match]))
            return value

        def recursive_resolve(obj):
            if isinstance(obj, dict):
                return {k: recursive_resolve(resolve_value(v)) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_resolve(v) for v in obj]
            else:
                return resolve_value(obj)

        return recursive_resolve(config)

    def _load_and_resolve_config(self) -> dict:
        """
        [비공개 메서드] 설정 파일 로드 후, 플레이스홀더 치환까지 완료
        - 입력: 없음
        - 출력: 치환이 완료된 전체 설정 딕셔너리
        """
        raw_config = self._load_yaml()
        context = {
            "root_dir": raw_config["project"]["root_dir"],
            "dataset_dir": raw_config["project"]["dataset"]["dataset_dir"],
            "output_dir": raw_config["project"]["output"]["output_dir"],
            "src_dir": raw_config["project"]["source"]["src_dir"]
        }
        return self._resolve_placeholders(raw_config, context)

    # ======= 외부에 제공하는 메서드 =======

    def get_project_config(self) -> dict:
        """
        [공개 메서드] project 전체 정보 반환
        - 입력: 없음
        - 출력: project 섹션 (dict)
        """
        return self.config.get("project", {})

    def get_dataset_config(self) -> dict:
        """
        [공개 메서드] dataset 구성 정보 반환
        - 입력: 없음
        - 출력: dataset 섹션 (dict)
        """
        return self.config["project"].get("dataset", {})

    def get_output_config(self) -> dict:
        """
        [공개 메서드] output 구성 정보 반환
        - 입력: 없음
        - 출력: output 섹션 (dict)
        """
        return self.config["project"].get("output", {})

    def get_source_config(self) -> dict:
        """
        [공개 메서드] source 구성 정보 반환
        - 입력: 없음
        - 출력: source 섹션 (dict)
        """
        return self.config["project"].get("source", {})

    def get_models_config(self) -> dict:
        """
        [공개 메서드] models 설정 정보 반환
        - 입력: 없음
        - 출력: models 섹션 (dict)
        """
        return self.config.get("models", {})
