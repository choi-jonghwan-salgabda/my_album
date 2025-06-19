import os
import shutil
import glob

def move_images_to_parent_directory():
    """
    images 디렉토리의 하위 폴더들에 있는 모든 이미지 파일을
    image 디렉토리로 이동시킵니다.
    """
    # 기본 경로 설정
    base_dir = '/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project'
    images_dir = os.path.join(base_dir, 'images')  # 하위 디렉토리가 있는 폴더
    target_dir = os.path.join(base_dir, 'image')   # 이동할 대상 폴더
    
    # 대상 디렉토리가 없으면 생성
    os.makedirs(target_dir, exist_ok=True)
    
    # 이미지 파일 확장자 목록
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']
    
    # 이동된 파일 카운트
    moved_count = 0
    
    # images 하위의 모든 디렉토리 순회
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            # 파일 확장자 체크
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # 원본 파일과 대상 파일의 경로
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)
                
                # 이미 대상 폴더에 같은 이름의 파일이 있는 경우 처리
                if os.path.exists(target_file):
                    # 파일명과 확장자 분리
                    file_name, file_ext = os.path.splitext(file)
                    # 새 파일명 생성 (파일명_숫자.확장자)
                    counter = 1
                    while os.path.exists(os.path.join(target_dir, f"{file_name}_{counter}{file_ext}")):
                        counter += 1
                    target_file = os.path.join(target_dir, f"{file_name}_{counter}{file_ext}")
                
                try:
                    # 파일 이동
                    shutil.move(source_file, target_file)
                    moved_count += 1
                    
                    # 진행 상황 출력 (100개마다)
                    if moved_count % 100 == 0:
                        print(f"이동된 파일 수: {moved_count}")
                        
                except Exception as e:
                    print(f"파일 이동 중 오류 발생: {source_file} -> {target_file}")
                    print(f"오류 내용: {e}")
    
    print(f"\n작업 완료! 총 {moved_count}개 파일이 이동되었습니다.")
    return moved_count

# 함수 실행
if __name__ == "__main__":
    move_images_to_parent_directory()