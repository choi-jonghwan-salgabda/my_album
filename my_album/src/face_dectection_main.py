import cv2
from util.face_detection import detect_faces

def main():
    # 이미지 로드
    img_path = "images/LEE01024.jpg"  # 테스트할 이미지 경로
    image = cv2.imread(img_path)

    if image is not None:
        # 얼굴 검출
        detected_faces = detect_faces(image)
        print(f"검출된 얼굴 수: {len(detected_faces)}")

        # 검출된 얼굴 정보 출력
        for face in detected_faces:
            print(f"얼굴 ID: {face['face_id']}, 위치: {face['box']}")
    else:
        print("이미지를 로드할 수 없습니다.")

if __name__ == "__main__":
    main()
