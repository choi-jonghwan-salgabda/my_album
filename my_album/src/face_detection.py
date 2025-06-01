import cv2
import dlib

def initialize_face_detector():
    """
    얼굴 검출기를 초기화합니다.
    Returns:
        detector: Dlib의 HOG 기반 얼굴 검출기
    """
    detector = dlib.get_frontal_face_detector()
    return detector

def detect_faces(image):
    """
    주어진 이미지에서 얼굴을 검출합니다.

    Args:
        image: 얼굴을 검출할 이미지 (NumPy 배열).

    Returns:
        List[Dict]: 검출된 얼굴의 위치 정보 (x, y, width, height).
    """
    # Dlib의 얼굴 검출기 초기화
    detector = initialize_face_detector()

    # 얼굴 검출
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환
    faces = detector(gray_image)

    face_locations = []
    for i, face in enumerate(faces):
        # 얼굴의 위치 정보 저장
        face_info = {
            "face_id": i,
            "box": {
                "x": face.left(),
                "y": face.top(),
                "width": face.right() - face.left(),
                "height": face.bottom() - face.top()
            }
        }
        face_locations.append(face_info)

    return face_locations

if __name__ == "__main__":
    # 테스트용 코드 (예시)
    image_path = "path/to/your/image.jpg"  # 테스트할 이미지 경로
    image = cv2.imread(image_path)

    if image is not None:
        detected_faces = detect_faces(image)
        print(f"검출된 얼굴 수: {len(detected_faces)}")
        for face in detected_faces:
            print(f"얼굴 ID: {face['face_id']}, 위치: {face['box']}")
    else:
        print("이미지를 로드할 수 없습니다.")
