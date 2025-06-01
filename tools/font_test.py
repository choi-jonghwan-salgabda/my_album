import matplotlib
import os
import shutil # shutil 추가

try:
    # matplotlib 캐시 디렉토리 경로 가져오기
    cache_dir = matplotlib.get_cachedir()
    if cache_dir and os.path.exists(cache_dir):
        print(f"Matplotlib 캐시 디렉토리: {cache_dir}")
        # 캐시 디렉토리 내의 fontlist 파일 삭제 (fontlist-vXXX.json 형태)
        # 또는 디렉토리 전체를 삭제하는 것도 일반적임
        # 여기서는 디렉토리 전체 삭제 시도
        try:
            shutil.rmtree(cache_dir) # 디렉토리와 내용물 모두 삭제
            print(f"Matplotlib 캐시 디렉토리를 삭제했습니다: {cache_dir}")
            print("다음에 matplotlib을 사용할 때 캐시가 자동으로 재생성됩니다.")
            # 필요시 디렉토리를 다시 생성할 수도 있지만, 보통은 matplotlib이 자동으로 생성함
            # os.makedirs(cache_dir, exist_ok=True)
        except OSError as e:
            print(f"캐시 디렉토리 삭제 중 오류 발생: {e}")
            print("수동으로 다음 디렉토리를 삭제해 보세요:")
            print(cache_dir)
            # 또는 개별 fontlist 파일만 삭제 시도
            # for filename in os.listdir(cache_dir):
            #     if filename.startswith('fontlist') and filename.endswith('.json'):
            #         file_path = os.path.join(cache_dir, filename)
            #         try:
            #             os.remove(file_path)
            #             print(f"폰트 캐시 파일 삭제: {file_path}")
            #         except OSError as e_file:
            #             print(f"폰트 캐시 파일 삭제 중 오류 발생 ({file_path}): {e_file}")

    else:
        print("Matplotlib 캐시 디렉토리를 찾을 수 없습니다.")

except Exception as e:
    print(f"캐시 디렉토리 처리 중 오류 발생: {e}")