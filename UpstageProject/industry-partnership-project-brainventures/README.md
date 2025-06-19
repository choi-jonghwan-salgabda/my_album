# 프로젝트 이름

<br>

## 프로젝트 소개
### <프로젝트 소개>
- _이번 프로젝트에 대해 소개를 작성해주세요_

### <작품 소개>
- _만드신 작품에 대해 간단한 소개를 작성해주세요_

<br>

## 팀 구성원   

| ![최종환](https://github.com/UpstageAILab5/upstageailab-ir-competition-ir_s3/blob/main/docs/pic/JH.JPG) | ![조혜인](https://github.com/UpstageAILab5/upstageailab-ir-competition-ir_s3/blob/main/docs/pic/HI.png) | ![박지은](...) | ![이다언](https://github.com/UpstageAILab5/upstageailab-ir-competition-ir_s3/blob/main/docs/pic/DU.PNG) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [최종환]            |            [조혜인]    |         [박지은은]   |       [이다언]    |      
|                            팀장, ...|.|..|...                   

<br>

## 1. 개발 환경 및 기술 스택
- 주 언어 : Python
- 버전 및 이슈관리 : 3.10.14 ~ 3.10.16
- 협업 툴 : _ex) github, notion_

<br>

## 2. 프로젝트 구조
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

<br>

## 3. 구현 기능
### 기능1
- _작품에 대한 주요 기능을 작성해주세요_
### 기능2
- _작품에 대한 주요 기능을 작성해주세요_
### 기능3
- _작품에 대한 주요 기능을 작성해주세요_

<br>

## 4. 작품 아키텍처(필수X)
- #### _아래 이미지는 예시입니다_
![이미지 설명](https://www.cadgraphics.co.kr/UPLOAD/editor/2024/07/04//2024726410gH04SyxMo3_editor_image.png)

<br>

## 5. 트러블 슈팅
### 1. OOO 에러 발견

#### 설명
- _프로젝트 진행 중 발생한 트러블에 대해 작성해주세요_

#### 해결
- _프로젝트 진행 중 발생한 트러블 해결방법 대해 작성해주세요_

<br>

## 6. 프로젝트 회고
### 박패캠
- _프로젝트 회고를 작성해주세요_

<br>

## 7. 참고자료
- _참고자료를 첨부해주세요_


# 데이터 다운로드
# aihub 의 api 키로 데이터 요청 
다운받으려는 데이터 신청 접수( 다운로드 클릭후 팝업창 확인 요망 ) 
https://www.aihub.or.kr/devsport/apishell/list.do?currMenu=403&topMenu=100 에서 api 키 발급 

# AIHUB 데이터 다운 명령어
curl -o "aihubshell" https://api.aihub.or.kr/api/aihubshell.do

chmod +x aihubshell

 ls -al | grep aihubshell

aihubshell -mode l -datasetkey 105

aihubshell -mode d -datasetkey 105 -aihubapikey 'API_KEY키 발급'

# 특정 버전만 다운로드
aihubshell -mode d -datasetkey 105 -filekey 68070,68071,68072,68073 -aihubapikey 'API_KEY'

sudo apt update
sudo apt install locales
sudo locale-gen ko_KR.UTF-8
export LANG=ko_KR.UTF-8

unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/1.Training/label/TL1.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/1.Training/label
unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/1.Training/origin/TS1.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/1.Training/origin

unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/2.Validation/label/VL1.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/2.Validation/label
unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/2.Validation/origin/VS1.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/2.Validation

unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/02.raw_data/1.Trainging/[라벨]Training.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/02.raw_data
unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/02.raw_data/Validation.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/02.raw_data/2.Validation

37317 
# 데이터 압축풀기 
tar xvfz ./download.tar

# ai hub 명령어
aihubshell -help

