# FindDuplicatePhotos.py
"""
이 스크립트는 지정된 소스 디렉토리와 그 하위 디렉토리의 모든 이미지 파일을 스캔하여,
파일 내용(해시)이 동일한 중복 사진들을 찾아 화면에 시각적으로 보여주는 도구입니다.

주요 기능:
1.  콘텐츠 기반 중복 탐지:
    각 이미지 파일의 내용에 기반한 SHA256 해시 값을 계산하여, 파일명이 다르거나
    다른 폴더에 있더라도 내용이 동일한 파일을 정확하게 찾아냅니다.

2.  중복 이미지 시각화:
    동일한 해시 값을 가진 이미지 그룹이 발견되면, Matplotlib을 사용하여 해당 이미지들을
    한 창에 모두 표시합니다. 각 이미지 아래에는 파일 경로가 표시되어 사용자가
    어떤 파일들이 중복인지 직관적으로 확인할 수 있습니다.

3.  진행 상황 표시:
    tqdm 라이브러리를 사용하여 많은 수의 파일을 처리할 때 현재 진행 상황을
    시각적인 막대 바로 보여주어 사용자가 작업이 얼마나 남았는지 알 수 있습니다.

4.  유연한 소스 디렉토리 지정:
    `--source-dir` 인자를 통해 중복을 검사할 디렉토리를 지정할 수 있습니다.

사용법 예시:
    python FindDuplicatePhotos.py --source-dir ~/SambaData/Backup/FastCamp/data/my_photos/

참고:
- 이 스크립트를 실행하려면 matplotlib 라이브러리가 필요합니다. (pip install matplotlib)
- GUI 환경이 없는 서버에서는 이미지 표시가 실패할 수 있습니다.
"""

import sys
import math
import textwrap
from pathlib import Path
from collections import defaultdict

# 이미지 표시를 위한 라이브러리
try:
    import matplotlib
    # GUI가 없는 환경(예: SSH 터미널)에서 오류가 발생하는 것을 방지
    # DISPLAY 환경 변수가 설정되지 않은 경우, 'Agg' 백엔드를 사용하도록 설정
    import os
    if os.environ.get('DISPLAY', '') == '':
        print("경고: GUI 환경이 감지되지 않았습니다. Matplotlib 백엔드를 'Agg'로 설정합니다. 이미지는 화면에 표시되지 않습니다.")
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, UnidentifiedImageError
except ImportError:
    print("오류: 이 스크립트를 실행하려면 'matplotlib'와 'Pillow' 라이브러리가 필요합니다.")
    print("pip install matplotlib Pillow")
    sys.exit(1)

from tqdm import tqdm

# 프로젝트 공용 유틸리티 임포트
try:
    from my_utils.config_utils.arg_utils import get_argument
    from my_utils.config_utils.SimpleLogger import logger
    from my_utils.config_utils.configger import configger
    from my_utils.config_utils.file_utils import calculate_sha256, scan_files_in_batches
    from my_utils.config_utils.display_utils import calc_digit_number, get_display_width, truncate_string, visual_length
    from my_utils.object_utils.photo_utils import rotate_image_if_needed, JsonConfigHandler, _get_string_key_from_config # JsonConfigHandler 임포트
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)

# 지원하는 이미지 확장자 목록 (소문자로 통일)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic"}

def display_images(image_paths: list[Path], file_hash: str):
    """
    주어진 이미지 경로 리스트를 Matplotlib을 사용하여 화면에 표시합니다.

    Args:
        image_paths (list[Path]): 표시할 이미지 파일의 경로 리스트.
        file_hash (str): 이미지들의 공통 해시 값 (창 제목에 표시용).
    """
    num_images = len(image_paths)
    if num_images == 0:
        return

    # 이미지를 표시할 그리드 크기 계산 (대략 정사각형 형태)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(f"중복 사진 그룹 (해시: {file_hash[:12]}...)\n총 {num_images}개 발견", fontsize=16)
    
    axes = axes.flatten() if num_images > 1 else [axes]

    for i, img_path in enumerate(image_paths):
        try:
            with Image.open(img_path) as img:
                ax = axes[i]
                ax.imshow(img)
                # 경로가 너무 길면 textwrap을 사용하여 보기 좋게 줄 바꿈
                title = textwrap.fill(str(img_path), width=50)
                ax.set_title(title, fontsize=8)
                ax.axis('off')
        except (UnidentifiedImageError, IOError) as e:
            logger.warning(f"이미지 표시 오류 '{img_path}': {e}")
            axes[i].set_title(f"오류: {img_path.name}", fontsize=8, color='red')
            axes[i].axis('off')

    # 남는 subplot 숨기기
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitle과 겹치지 않도록 조정

    # 현재 백엔드가 'Agg'인지 확인하여 파일로 저장하거나 화면에 표시
    if matplotlib.get_backend().lower() == 'agg':
        output_filename = f"duplicates_{file_hash[:12]}.png"
        plt.savefig(output_filename)
        logger.info(f"중복 이미지 그룹을 '{output_filename}' 파일로 저장했습니다.")
        plt.close(fig) # 메모리 해제를 위해 창을 닫습니다.
    else:
        plt.show() # GUI 환경에서는 화면에 표시

def find_and_display_duplicates_logic(source_dir: Path):
    """
    지정된 디렉토리에서 중복 이미지를 찾아내고 화면에 표시하는 핵심 로직.

    Args:
        source_dir (Path): 검색을 시작할 소스 디렉토리.
    """
    batch_size = 2
    total_count, batch_gen = scan_files_in_batches(
        root_dir=source_dir,
        batch_size=batch_size
    )

    if total_count == 0:
        logger.info("처리할 파일 없습니다.")
        return

    # 제너레이터(batch_gen)는 len()으로 길이를 알 수 없으므로, 직접 계산합니다.
    num_batches = math.ceil(total_count / batch_size)
    logger.info(f"총 {total_count}개의 이미지를 찾았습니다. {batch_size}개씩 {int(num_batches)}개의 배치로 나누어 처리합니다.")

    hashes = defaultdict(list)  # 기본값이 빈 리스트([])인 딕셔너리
    with tqdm(total=total_count, desc="해시 계산 중", unit="파일", file=sys.stdout) as pbar:
        for batch in batch_gen:
            for file_path in batch:
                file_hash = calculate_sha256(file_path)
                if file_hash:
                    hashes[file_hash].append(file_path)
                pbar.update(1)

    duplicates = {key: paths for key, paths in hashes.items() if len(paths) > 1}

    if not duplicates:
        logger.info("중복된 사진을 찾지 못했습니다.")
        return

    logger.warning(f"총 {len(duplicates)}개의 중복 사진 그룹을 찾았습니다. 그룹별로 결과를 표시합니다.")
    
    for i, (file_hash, paths) in enumerate(duplicates.items(), 1):
        logger.warning(f"\n--- 중복 그룹 {i}/{len(duplicates)} ---")
        for p in paths:
            logger.warning(f"  - {p}")
        display_images(paths, file_hash)
    
    logger.info("모든 중복 그룹 확인 완료.")

def run_main():
    """스크립트의 메인 실행 함수."""
    # 이 스크립트는 특정 인자를 필수로 요구하지 않으므로, 인자 리스트 없이 get_argument()를 호출합니다.
    parsed_args = get_argument() # 경고를 유발하는 arg_list를 제거합니다.

    # Matplotlib에서 한글이 깨지지 않도록 폰트 설정
    try:
        from matplotlib import font_manager

        font_name = None
        # 선호하는 폰트 순서대로 리스트를 만듭니다.
        font_preferences = ['NanumGothic', 'Malgun Gothic', 'AppleGothic']
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]

        for font in font_preferences:
            if font in available_fonts:
                font_name = font
                break

        if font_name:
            plt.rcParams['font.family'] = font_name
            logger.info(f"Matplotlib 한글 폰트를 '{font_name}'(으)로 설정했습니다.")
        else:
            logger.warning(f"한글 지원 폰트({', '.join(font_preferences)})를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
        # 마이너스 부호가 깨지는 문제 해결
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        logger.warning(f"Matplotlib 폰트 설정 중 오류 발생: {e}")
    
    input_dir_path = parsed_args.source_dir
    if not input_dir_path:
        default_path = Path('~/SambaData/Backup/FastCamp/data/my_photos/').expanduser()
        logger.warning(f"소스 디렉토리(--source-dir)가 제공되지 않았습니다. 기본 경로 '{default_path}'를 사용합니다.")
        input_dir_path = str(default_path)

    source_dir = Path(input_dir_path).expanduser().resolve()

    try:
        find_and_display_duplicates_logic(source_dir)
    except KeyboardInterrupt:
        logger.warning("\n사용자에 의해 작업이 중단되었습니다.")
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"애플리케이션 ({Path(__file__).stem}) 종료")

if __name__ == "__main__":
    if hasattr(logger, "setup"):
        logger.setup(console_min_level="INFO", file_min_level="DEBUG")
    run_main()