import os
import json
import yaml
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# 애플리케이션 루트 및 프로젝트 루트 경로 설정
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT를 실제 프로젝트의 최상위 디렉토리를 가리키도록 수정합니다.
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, '..', '..', '..')) # /home/owner/SambaData/Backup/FastCamp/Myproject/

# 설정 파일 로드 함수
def load_config():
    config_path = os.path.join(PROJECT_ROOT, 'config', 'photo_album.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"오류: 설정 파일({config_path})을 찾을 수 없습니다.")
        exit(1)
    except Exception as e:
        print(f"오류: 설정 파일 로드 중 문제 발생 - {e}")
        exit(1)

config = load_config()
data_config = config.get('data', {})
web_config = config.get('web_server', {})

# 설정에서 절대 경로 구성
JSON_FILE_PATH = os.path.join(PROJECT_ROOT, data_config.get('json_file', 'my_album/data/indexed_photos.json'))
IMAGE_SOURCE_DIR = os.path.join(PROJECT_ROOT, data_config.get('image_source_directory', 'my_album/data/images'))
TEMPLATE_FOLDER = os.path.join(APP_ROOT, web_config.get('template_folder', 'templates'))
STATIC_FOLDER = os.path.join(APP_ROOT, web_config.get('static_folder', 'static'))
IMAGE_SERVE_PREFIX = web_config.get('image_serve_prefix', '/images_data')

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
# Make the primary JSON file path available in templates via config.data.json_file
# This aligns with how the index.html template tries to access it.
if JSON_SAVE_FILE_PATH:
    app.config['DATA'] = {'json_file': JSON_SAVE_FILE_PATH}

# 데이터 로드 및 저장 함수
def load_photo_data():
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"데이터 파일({JSON_FILE_PATH})을 찾을 수 없습니다. 빈 리스트를 반환합니다.")
        return []
    except json.JSONDecodeError:
        print(f"데이터 파일({JSON_FILE_PATH})의 형식이 잘못되었습니다. 빈 리스트를 반환합니다.")
        return []

def save_photo_data(data):
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(JSON_FILE_PATH), exist_ok=True)
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"데이터 저장 중 오류 발생: {e}")

@app.route('/')
def index():
    photos_data = load_photo_data()
    # 각 사진에 ID (인덱스) 추가하여 템플릿에서 사용
    # 이미지 경로도 함께 전달하여 썸네일 등에 활용 가능
    photos_with_ids = [
        {**photo, 'id': i, 'image_url': url_for('serve_image', filename=photo['image_path'])}
        for i, photo in enumerate(photos_data)
    ]
    return render_template('index.html', photos=photos_with_ids)

@app.route('/image/<int:image_id>')
def label_image(image_id):
    photos_data = load_photo_data()
    if 0 <= image_id < len(photos_data):
        image_data = photos_data[image_id]
        # 이미지 URL 생성
        image_url = url_for('serve_image', filename=image_data['image_path'])
        return render_template('label_image.html', image_data=image_data, image_id=image_id, image_url=image_url)
    return "이미지를 찾을 수 없습니다.", 404

@app.route('/save_labels/<int:image_id>', methods=['POST'])
def save_labels(image_id):
    photos_data = load_photo_data()
    if 0 <= image_id < len(photos_data):
        image_to_update = photos_data[image_id]
        for i, face in enumerate(image_to_update['faces']):
            face_name = request.form.get(f'face_{i}_name')
            if face_name is not None:
                face['name'] = face_name.strip()
        
        save_photo_data(photos_data)
        return redirect(url_for('label_image', image_id=image_id))
    return "이미지를 찾을 수 없습니다.", 404

# 설정된 경로에서 이미지 파일 제공
@app.route(f'{IMAGE_SERVE_PREFIX}/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_SOURCE_DIR, filename)

if __name__ == '__main__':
    # YAML 설정에서 호스트 및 포트 사용
    host = web_config.get('host', '127.0.0.1')
    port = web_config.get('port', 5000)
    
    # JSON 파일 및 이미지 디렉토리 존재 여부 확인 (선택적)
    if not os.path.exists(JSON_FILE_PATH):
        print(f"경고: JSON 데이터 파일 '{JSON_FILE_PATH}'를 찾을 수 없습니다. 애플리케이션이 정상 동작하지 않을 수 있습니다.")
        # 테스트용 빈 JSON 파일 생성
        # save_photo_data([]) 
        # print(f"'{JSON_FILE_PATH}'에 빈 JSON 배열을 생성했습니다.")

    if not os.path.isdir(IMAGE_SOURCE_DIR):
        print(f"경고: 이미지 소스 디렉토리 '{IMAGE_SOURCE_DIR}'를 찾을 수 없습니다. 이미지가 표시되지 않을 수 있습니다.")

    app.run(host=host, port=port, debug=True)