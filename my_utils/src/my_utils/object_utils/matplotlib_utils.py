# src/my_utils/plot_utils/matplotlib_utils.py
"""
Matplotlib 관련 유틸리티 함수를 모아놓은 모듈입니다.
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    # 프로젝트의 공용 로거를 사용합니다.
    from my_utils.config_utils.SimpleLogger import logger
except ImportError:
    # 공용 로거를 찾을 수 없을 경우, 표준 로깅으로 대체합니다.
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("SimpleLogger를 임포트할 수 없어 표준 로깅을 사용합니다.")

def setup_korean_font():
    """
    Matplotlib에서 한글이 깨지지 않도록 시스템에 설치된 한글 폰트를 찾아 설정합니다.

    선호하는 폰트 순서: 'NanumGothic', 'Malgun Gothic', 'AppleGothic'.
    설정 성공 또는 실패 시 정보/경고 로그를 남깁니다.
    또한, 마이너스 부호가 깨지는 현상을 방지합니다.
    """
    try:
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