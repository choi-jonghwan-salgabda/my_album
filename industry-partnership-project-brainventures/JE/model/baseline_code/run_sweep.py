import os
import sys
import wandb
import yaml
import subprocess
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)  # 현재 작업 디렉토리를 baseline_code로 설정

# sweep 설정 로드
with open('sweep_config.yaml', 'r') as f:
    sweep_config = yaml.safe_load(f)

# 프로젝트 이름 설정
project_name = "dbnet-baseline-sweep"

# sweep ID 생성
sweep_id = wandb.sweep(sweep_config, project=project_name)

def sweep_agent():
    """sweep 에이전트 함수 - 하이퍼파라미터 세트마다 실행됨"""
    # wandb 실행 초기화
    run = wandb.init()
    
    # 하이퍼파라미터를 hydra 명령줄 인수로 변환
    param_list = []
    for key, value in wandb.config.items():
        if '.' in key:  # 중첩된 파라미터 (예: models.optimizer.lr)
            param_list.append(f"{key}={value}")
        elif isinstance(value, str):
            # 문자열 값은 따옴표로 묶기
            param_list.append(f"{key}='{value}'")
        else:
            param_list.append(f"{key}={value}")
    
    # 훈련 스크립트 실행
    cmd = f"python runners/train.py {' '.join(param_list)} wandb=true"
    print(f"실행 명령어: {cmd}")
    
    # subprocess로 실행
    process = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # 실시간 로그 출력
    for line in process.stdout:
        print(line, end='')
    
    # 프로세스 완료 대기
    process.wait()
    
    # 오류 확인
    if process.returncode != 0:
        print(f"프로세스가 오류 코드 {process.returncode}로 종료되었습니다.")

# sweep 실행 - count는 실험 횟수
wandb.agent(sweep_id, function=sweep_agent, count=10)  

print(f"Sweep 완료! W&B 대시보드에서 결과를 확인하세요: https://wandb.ai/에서 프로젝트 {project_name}를 확인하세요.")