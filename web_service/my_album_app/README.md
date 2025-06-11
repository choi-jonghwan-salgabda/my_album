# My Album Labeling App

## 프로젝트 설명

이 프로젝트는 사용자가 사진 앨범 이미지를 보고 라벨링할 수 있는 Flask 기반의 웹 애플리케이션입니다. 이미지 라벨링 작업을 웹 인터페이스를 통해 효율적으로 수행할 수 있도록 설계되었습니다.

## 프로젝트 구조

프로젝트는 다음과 같은 주요 디렉토리 및 파일로 구성됩니다.

my_album_app/
├── src/
│ ├── my_album_labeling_app/ # 주 애플리케이션 코드
│ │ └── app.py # Flask 애플리케이션 진입점
│ ├── static/ # 정적 파일 (CSS, JS, 이미지 등)
│ │ └── css/
│ │ └── style.css # 스타일시트
│ └── templates/ # HTML 템플릿 파일
│ ├── index.html # 메인 페이지 템플릿
│ └── label_image.html # 이미지 라벨링 페이지 템플릿
├── poetry.lock # Poetry가 생성하는 잠금 파일
├── pyproject.toml # Poetry 프로젝트 설정 및 의존성 관리
└── README.md # 이 파일



`my_utils`와 같은 로컬 의존성은 이 프로젝트의 `pyproject.toml`에 정의되어 있습니다.

## 설치

이 프로젝트는 [Poetry](https://python-poetry.org/)를 사용하여 의존성을 관리합니다. 프로젝트를 로컬 환경에 설정하려면 다음 단계를 따르세요.

1.  Poetry가 설치되어 있는지 확인합니다. 설치되어 있지 않다면 [Poetry 설치 가이드](https://python-poetry.org/docs/#installation)를 참고하세요.
2.  프로젝트 루트 디렉토리로 이동합니다.
    ```bash
    cd /path/to/your/Myproject/web_service/my_album_app
    ```
3.  프로젝트 의존성을 설치합니다. 이 과정에서 `pyproject.toml`에 정의된 `my_utils`와 같은 로컬 의존성도 함께 설정됩니다.
    ```bash
    poetry install
    ```

## 사용법

프로젝트 설치가 완료되면 다음 명령을 사용하여 Flask 개발 서버를 실행할 수 있습니다.

1.  프로젝트 루트 디렉토리에서 Poetry 가상 환경을 활성화합니다.
    ```bash
    poetry shell
    ```
2.  Flask 애플리케이션을 실행합니다.
    ```bash
    python src/my_album_labeling_app/app.py
    ```
    (또는 Flask 실행 방식에 따라 `flask run` 등을 사용할 수 있습니다. `app.py` 내부에 Flask 앱 객체가 정의되어 있고 `if __name__ == '__main__':` 블록에 `app.run()`이 포함되어 있다면 위의 `python` 명령으로 실행 가능합니다.)
3.  웹 브라우저를 열고 애플리케이션이 실행되는 주소(일반적으로 `http://127.0.0.1:5000/`)로 접속합니다.

## 기능

*   사진 앨범 이미지 목록 보기
*   개별 이미지 선택 및 상세 정보 표시
*   이미지에 대한 라벨링 수행 및 저장 (구현 예정 또는 현재 기능 설명 추가)
*   (`my_utils`를 통해) 설정 관리 및 기타 유틸리티 기능 활용

(구현된 또는 구현 예정인 구체적인 라벨링 기능, 이미지 처리 기능 등이 있다면 여기에 상세 설명을 추가하십시오.)

## 의존성

주요 의존성은 `pyproject.toml` 파일에 명시되어 있습니다.
*   Flask: 웹 프레임워크
*   PyYAML: 설정 파일 로딩
*   ultralytics: (객체 탐지 또는 다른 이미지 처리 기능에 사용될 경우)
*   my-utils: 로컬 유틸리티 함수 모음

## 기여

프로젝트에 기여하고 싶으시면 다음 지침을 따르세요.

1.  프로젝트를 포크합니다.
2.  기능 브랜치를 생성합니다 (`git checkout -b feature/your-feature`).
3.  변경 사항을 커밋합니다 (`git commit -m 'Add your feature'`).
4.  브랜치에 푸시합니다 (`git push origin feature/your-feature`).
5.  풀 리퀘스트를 생성합니다.

## 라이선스

이 프로젝트는 [라이선스 이름]에 따라 라이선스가 부여됩니다. (예: MIT License, Apache License 2.0 등)

## 작성자

[본인 이름] - 이메일 또는 웹사이트 주소

---
이 README는 프로젝트 초기 구조에 맞춰 작성되었습니다. 프로젝트 개발이 진행됨에 따라 내용을 최신 상태로 유지해주세요.
