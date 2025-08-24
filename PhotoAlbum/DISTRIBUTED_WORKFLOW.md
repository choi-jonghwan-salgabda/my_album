# 분산 처리를 통한 사진 앨범 인덱싱 워크플로우

이 문서는 보유하고 있는 여러 컴퓨팅 자원(Windows 데스크톱, Linux 개발 서버 등)을 효율적으로 활용하여 사진 앨범 인덱싱 파이프라인의 전체 처리 시간을 단축하기 위한 분산 처리 워크플로우를 안내합니다.

## 핵심 전략: 역할 분담을 통한 "데이터 처리 공장" 모델

각 시스템의 강점을 기반으로 역할을 분담하여 파이프라인을 병렬로 실행합니다.

- **중앙 데이터 허브 (`jongshomeserver`):**
  - **역할:** 모든 데이터(원본 이미지, 중간 결과물 JSON, 최종 인덱스)를 저장하는 중앙 스토리지입니다.
  - **요구사항:** Samba 공유 폴더(`SambaData`)가 모든 시스템에서 접근 가능해야 합니다.

- **GPU 가속 유닛 (Windows 데스크톱 - RTX 2080 Ti):**
  - **역할:** 가장 강력한 GPU를 활용하여, 시간이 가장 많이 소요되는 무거운 연산(객체 탐지, 얼굴 탐지 및 임베딩 추출)을 전담합니다.

- **CPU/인덱싱 유닛 (Linux 개발 서버 - 12GB RAM):**
  - **역할:** GPU가 필요 없으면서 메모리와 CPU 자원이 중요한 FAISS 인덱스 구축 작업을 담당합니다.

---

## 단계별 실행 가이드

### 0단계: 사전 준비

1. 모든 시스템에서 `jongshomeserver`의 Samba 공유 폴더가 동일한 경로로 마운트되었는지 확인합니다.
2. 각 시스템의 `photo_album.yaml` 설정 파일이 중앙 데이터 허브의 경로를 올바르게 참조하고 있는지 확인합니다.
3. 각 시스템에 `poetry` 환경 및 필수 라이브러리가 설치되어 있는지 확인합니다. (`poetry install`)

### 1단계: 객체 탐지 (Batch Processing)

- **담당 시스템:** **Windows 데스크톱** (GPU 활용)
- **목표:** 수십만 장의 원본 이미지에서 객체(주로 '사람')를 탐지하고, 각 이미지에 대한 JSON 파일을 생성합니다. 새로 도입된 배치 처리 기능으로 GPU 성능을 극대화합니다.
- **스크립트:** `src/object_detector.py`
- **설정 (`photo_album.yaml`):**
  ```yaml
  models:
    object_yolo_tiny_model:
      object_detection_model:
        use_cpu: false # GPU 사용
  processing:
    batch_size: 32 # 또는 64, GPU 메모리에 맞춰 조정
  ```
- **실행 명령어:**
  ```bash
  poetry run python src/object_detector.py --parallel --max-workers 8
  ```
  > **팁:** `--max-workers`는 CPU 코어 수에 맞춰, `batch_size`는 GPU VRAM에 맞춰 조절하여 최적의 성능을 찾으세요.

### 2단계: 얼굴 탐지 및 임베딩 추출

- **담당 시스템:** **Windows 데스크톱** (GPU 활용)
- **목표:** 1단계에서 생성된 JSON 파일을 읽어 '사람' 객체 영역을 잘라내고, 그 안에서 얼굴을 찾아 특징(임베딩)을 추출한 후 JSON 파일을 업데이트합니다.
- **스크립트:** `src/my_album_indexer.py` (또는 `face_detector_with_json.py`)
- **설정 (`photo_album.yaml`):** `models.object_yolo_tiny_model.face_detection_model.use_cpu`를 `false`로 설정합니다.
- **실행 명령어:**
  ```bash
  poetry run python src/my_album_indexer.py
  ```

### 3단계: FAISS 인덱스 구축

- **담당 시스템:** **Linux 개발 서버** (CPU 및 RAM 활용)
- **목표:** 2단계까지 완료된 모든 얼굴 임베딩 데이터를 모아 최종 검색을 위한 FAISS 인덱스 파일과 메타데이터 파일을 생성합니다.
- **스크립트:** `src/indexer_from_face.py`
- **실행 명령어:**
  ```bash
  poetry run python src/indexer_from_face.py
  ```

### 4단계: 웹 서비스 실행

- **담당 시스템:** **Linux 개발 서버** 또는 **노트북 서버**
- **목표:** 완성된 인덱스를 사용하여 얼굴 검색 웹 서비스를 실행합니다.
- **스크립트:** `src/my_album_labeling_app/app.py`
- **실행 명령어:**
  ```bash
  poetry run python src/my_album_labeling_app/app.py
  ```
