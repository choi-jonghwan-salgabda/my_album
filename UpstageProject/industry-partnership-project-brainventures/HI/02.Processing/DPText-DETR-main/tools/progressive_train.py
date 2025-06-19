import os
import argparse
import subprocess
import gc
import json
from datetime import datetime
import torch
import logging

def get_args():
    parser = argparse.ArgumentParser(description="점진적 파인튜닝 스크립트")
    parser.add_argument("--config", default="configs/DPText_DETR/FastCampus/R_50_poly.yaml", 
                      help="설정 파일 경로")
    parser.add_argument("--batch-size", type=int, default=1000, 
                      help="각 배치당 이미지 수")
    parser.add_argument("--num-batches", type=int, default=10, 
                      help="처리할 배치 수 (0=전체)")
    parser.add_argument("--output-dir", default="output/progressive_finetune",
                      help="출력 디렉토리")
    parser.add_argument("--initial-weights", default="/data/ephemeral/home/DPText-DETR-main/pretrain.pth",
                      help="초기 가중치 파일 경로")
    parser.add_argument("--gpu", type=int, default=1,
                      help="사용할 GPU 수")
    parser.add_argument("--resume", action="store_true",
                      help="이전 학습을 이어서 진행할지 여부")
    return parser.parse_args()

def create_batch_config(base_config_path, output_path, batch_idx, batch_output_dir):
    # 기본 설정 파일 경로 추출
    base_path = os.path.relpath(base_config_path, os.path.dirname(output_path))
    
    # 새 설정 파일 내용 작성
    config_content = f'''_BASE_: "{base_path}"

DATASETS:
  TRAIN: ("fastcampus_train_poly_pos_batch_{batch_idx}",)
  TEST: ("fastcampus_valid_poly_pos",)

OUTPUT_DIR: "{batch_output_dir}"

SOLVER:
  AMP:
    ENABLED: false
'''
    
    # 파일 저장
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"배치 {batch_idx}의 설정 파일이 생성되었습니다: {output_path}")

def get_successful_batches(output_dir):
    """성공한 배치 인덱스와 체크포인트 파일을 반환합니다."""
    
    # 상태 파일 경로
    status_file = os.path.join(output_dir, "training_status.json")
    
    # 상태 파일이 없으면 빈 상태 반환
    if not os.path.exists(status_file):
        return [], None
    
    # 상태 파일 로드
    with open(status_file, 'r') as f:
        status = json.load(f)
    
    successful_batches = status.get("successful_batches", [])
    latest_checkpoint = status.get("latest_checkpoint", None)
    
    return successful_batches, latest_checkpoint

def update_training_status(output_dir, successful_batches, latest_checkpoint):
    """학습 상태 정보를 파일에 저장합니다."""
    
    # 상태 파일 경로
    status_file = os.path.join(output_dir, "training_status.json")
    
    # 상태 정보
    status = {
        "successful_batches": successful_batches,
        "latest_checkpoint": latest_checkpoint,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 파일 저장
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"학습 상태가 업데이트되었습니다: {status_file}")

def main():
    args = get_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 이전 학습 상태 확인
    successful_batches, latest_checkpoint = get_successful_batches(args.output_dir)
    
    # 이어서 학습할지 결정
    if args.resume and latest_checkpoint and os.path.exists(latest_checkpoint):
        current_weights = latest_checkpoint
        print(f"이전 학습을 이어서 진행합니다. 가중치: {current_weights}")
        print(f"성공한 배치: {successful_batches}")
    else:
        current_weights = args.initial_weights
        successful_batches = []
        print(f"새 학습을 시작합니다. 초기 가중치: {current_weights}")
    
    # 현재 시간 기록 (실행 시간 측정용)
    start_time = datetime.now()
    
    for batch_idx in range(1, args.num_batches + 1):
        # 이미 성공한 배치는 건너뜀
        if batch_idx in successful_batches:
            print(f"배치 {batch_idx}는 이미 성공적으로 처리되었습니다. 건너뜁니다.")
            continue
        
        # 배치별 출력 디렉토리
        batch_output_dir = os.path.join(args.output_dir, f"batch_{batch_idx}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        # 배치별 설정 파일 생성
        batch_config = os.path.join(batch_output_dir, "config.yaml")
        create_batch_config(args.config, batch_config, batch_idx, batch_output_dir)
        
        # 설정 파일 내용 확인
        print(f"\n----- 배치 {batch_idx} 설정 파일 내용 -----")
        with open(batch_config, 'r') as f:
            print(f.read())
        print("-------------------------------------\n")
        
        # 학습 명령 실행
        cmd = [
            "python", "tools/train_net.py",
            "--config-file", batch_config,
            "--num-gpus", str(args.gpu),
            "MODEL.WEIGHTS", current_weights
        ]
        
        print(f"\n===== 배치 {batch_idx} 학습 시작 =====")
        print(f"설정 파일: {batch_config}")
        print(f"가중치: {current_weights}")
        print(f"출력 디렉토리: {batch_output_dir}")
        print(f"명령어: {' '.join(cmd)}")
        
        gc.collect()
        torch.cuda.empty_cache()

        # 학습 실행
        process = subprocess.run(cmd)
        
        # 학습 후 GPU 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()

        # 새 체크포인트 경로
        new_checkpoint = os.path.join(batch_output_dir, "model_final.pth")
        
        # 학습 성공 여부 판단
        if process.returncode == 0 and os.path.exists(new_checkpoint):
            print(f"배치 {batch_idx} 학습 성공!")
            current_weights = new_checkpoint
            successful_batches.append(batch_idx)
            
            # 학습 상태 업데이트
            update_training_status(args.output_dir, successful_batches, current_weights)
        else:
            print(f"경고: 배치 {batch_idx} 학습 중 오류 발생 (반환 코드: {process.returncode})")
            
            # 체크포인트 존재 확인
            if os.path.exists(new_checkpoint):
                print(f"체크포인트는 생성되었지만 오류가 발생했습니다.")
                current_weights = new_checkpoint
                successful_batches.append(batch_idx)
                update_training_status(args.output_dir, successful_batches, current_weights)
            else:
                print(f"체크포인트가 생성되지 않았습니다. 이 배치는 건너뜁니다.")
                possible_weights = [f for f in os.listdir(batch_output_dir) if f.endswith('.pth')]
                if possible_weights:
                    latest_weight = os.path.join(batch_output_dir, sorted(possible_weights)[-1])
                    print(f"배치 내 최신 체크포인트 발견: {latest_weight}")
                    current_weights = latest_weight
                    successful_batches.append(batch_idx)
                    update_training_status(args.output_dir, successful_batches, current_weights)
        
        # 진행 상황 및 소요 시간 출력
        elapsed = datetime.now() - start_time
        print(f"현재까지 소요 시간: {elapsed}")
        
        # 처리된 배치 수 기준으로 예상 시간 계산
        processed_batches = len(successful_batches)
        if processed_batches > 0:
            estimated_total = elapsed * args.num_batches / processed_batches
            print(f"예상 총 소요 시간: {estimated_total}")
            print(f"남은 예상 시간: {estimated_total - elapsed}")
    
    # 최종 결과 출력
    print("\n===== 점진적 파인튜닝 완료 =====")
    print(f"총 소요 시간: {datetime.now() - start_time}")
    print(f"성공한 배치: {successful_batches}")
    print(f"최종 모델 가중치: {current_weights}")

if __name__ == "__main__":
    main()