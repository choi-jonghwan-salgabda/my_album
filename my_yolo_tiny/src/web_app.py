# my_album/web_app.py

from flask import Flask, render_template, request, redirect, url_for
import json
import os
# from my_album.config import config

# from my_album.indexing import load_index # 색인 로드 함수 임포트

app = Flask(__name__)

# 색인 데이터 로드 (어플리케이션 시작 시 한번 로드)
# index_data = load_index(config['data_paths']['index_file'])

@app.route('/')
def index():
    """메인 페이지 - 모든 얼굴 이미지를 보여주거나 검색 폼을 제공"""
    # 여기서는 간단히 색인된 모든 이미지를 보여주는 예시
    # 실제 구현 시 페이징 또는 검색 기능을 추가해야 합니다.
    # images_to_display = index_data # 전체 데이터 사용 시

    # 템플릿에 이미지 정보 전달
    return render_template('index.html', images=images_to_display)

@app.route('/search', methods=['GET', 'POST'])
def search():
    """얼굴 검색 페이지"""
    if request.method == 'POST':
        # 사용자로부터 얼굴 이미지 파일 또는 카메라 입력 받기
        # 입력된 얼굴에서 임베딩 추출
        # 저장된 index_data에서 유사한 임베딩 검색
        # 검색 결과 이미지를 템플릿에 전달
        search_results = [] # 검색 결과 이미지 목록
        return render_template('search_results.html', results=search_results)
    else:
        # 검색 폼 페이지 표시
        return render_template('search.html')

@app.route('/image/<path:filename>')
def serve_image(filename):
    """저장된 얼굴 이미지 파일을 웹에서 제공"""
    # detected_faces_dir = config['data_paths']['detected_faces_dir']
    # 이미지 파일의 실제 경로 계산 및 전송
    # from flask import send_from_directory
    # return send_from_directory(detected_faces_dir, filename)
    pass # 실제 구현 필요

# 이 함수는 run.py 등에서 호출될 수 있습니다.
# if __name__ == '__main__':
#     app.run(host=config['web']['host'], port=config['web']['port'], debug=config['web']['debug'])
