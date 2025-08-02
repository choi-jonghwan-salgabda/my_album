import os
import shutil
from pathlib import Path
from PIL import Image
import piexif  # EXIF 수정을 위한 라이브러리

# --- 설정 ---
# 테스트 데이터가 생성될 디렉토리
SOURCE_DIR = Path("~/SambaData/Backup/FastCamp/data/test_source_photos").expanduser()
# 테스트 데이터의 기반이 될 원본 이미지 파일들
# 이 스크립트와 같은 위치에 두세요.
BASE_IMAGES = [
    # 아래의 원본 경로들은 'quarantine' 디렉토리의 손상되었거나 유효하지 않은 이미지 파일을 가리켜
    # "cannot identify image file" 오류를 발생시켰습니다. 정상적으로 테스트 데이터를 생성하기 위해
    # 유효한 이미지 경로로 교체합니다.
    Path("~/SambaData/Backup/FastCamp/data/quarantine/5467.jpg").expanduser(),
    Path("~/SambaData/Backup/FastCamp/data/quarantine/20150901.jpg").expanduser(),
    Path("~/SambaData/Backup/FastCamp/data/quarantine/이가윤.bmp").expanduser(),
    Path("~/SambaData/Backup/FastCamp/data/quarantine/이가윤.bmp").expanduser(),
    Path("~/SambaData/Backup/FastCamp/data/quarantine/5467.jpg").expanduser(),
    Path("~/SambaData/Backup/FastCamp/data/quarantine/20150901.jpg").expanduser()
    ]

def set_exif_date(image_path: Path, date_str: str):
    """이미지에 EXIF 촬영 날짜를 설정합니다."""
    try:
        exif_dict = piexif.load(str(image_path))
        # 날짜 형식: "YYYY:MM:DD HH:MM:SS"
        exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = date_str.encode('utf-8')
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(image_path))
        print(f"  - EXIF 날짜 설정: '{date_str}' -> {image_path.name}")
    except Exception:
        # piexif.load는 EXIF가 없으면 예외를 발생시키므로 새로 만듭니다.
        exif_dict = {"Exif": {piexif.ExifIFD.DateTimeOriginal: date_str.encode('utf-8')}}
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(image_path))
        print(f"  - 새 EXIF 생성 및 날짜 설정: '{date_str}' -> {image_path.name}")


def remove_exif(image_path: Path):
    """이미지에서 모든 EXIF 데이터를 제거합니다."""
    try:
        piexif.remove(str(image_path))
        print(f"  - EXIF 데이터 제거 -> {image_path.name}")
    except Exception as e:
        print(f"  - EXIF 제거 실패 (이미 없을 수 있음): {image_path.name}: {e}")

def create_test_data():
    """테스트용 디렉토리와 다양한 시나리오의 파일을 생성합니다."""
    if SOURCE_DIR.exists():
        print(f"기존 테스트 디렉토리 삭제: {SOURCE_DIR}")
        shutil.rmtree(SOURCE_DIR)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"테스트 디렉토리 생성: {SOURCE_DIR}")

    # --- 베이스 이미지 검증 및 준비 ---
    print("\n베이스 이미지 검증 및 준비 중...")
    valid_base_images = []
    for i, img_path in enumerate(BASE_IMAGES):
        try:
            # 1. 파일 존재 여부 확인
            if not img_path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없음: {img_path}")

            # 2. 이미지 파일로 열 수 있는지 확인 (verify로 데이터 유효성 검사)
            with Image.open(img_path) as img:
                img.verify()

            # 검증 통과
            print(f"  - 유효한 베이스 이미지 확인: {img_path.name}")
            valid_base_images.append(img_path)

        except (FileNotFoundError, Image.UnidentifiedImageError, IOError) as e:
            print(f"\n경고: 베이스 이미지 '{img_path}'를 사용할 수 없습니다. ({e})")
            # 임시 더미 이미지 생성
            dummy_name = f"dummy_base_{i}.jpg"
            dummy_path = SOURCE_DIR / dummy_name
            try:
                Image.new('RGB', (200, 200), color=('red' if i % 2 == 0 else 'blue')).save(dummy_path)
                print(f"  -> 임시 더미 이미지 생성: {dummy_path}")
                valid_base_images.append(dummy_path)
            except Exception as create_e:
                print(f"  -> 임시 더미 이미지 생성 실패: {create_e}")

    # 테스트를 진행하기에 충분한 이미지가 있는지 확인
    if len(valid_base_images) < 2:
        print("\n치명적 오류: 테스트를 진행하기 위한 유효한 베이스 이미지가 2개 미만입니다. 스크립트를 종료합니다.")
        return

    # --- 시나리오 1: 중복 파일 ---
    print("\n1. 중복 파일 생성 중...")
    shutil.copy(valid_base_images[0], SOURCE_DIR / "original_A.jpg")
    shutil.copy(valid_base_images[0], SOURCE_DIR / "original_A_copy.jpg")
    (SOURCE_DIR / "subdir1").mkdir()
    shutil.copy(valid_base_images[0], SOURCE_DIR / "subdir1" / "original_A_in_subdir.jpg")
    with Image.open(valid_base_images[0]) as img:
        img.save(SOURCE_DIR / "original_A_as_png.png") # 확장자만 다른 중복

    # --- 시나리오 2: 고유 파일 ---
    print("\n2. 고유 파일 생성 중...")
    shutil.copy(valid_base_images[1], SOURCE_DIR / "unique_B.jpg")
    (SOURCE_DIR / "subdir2").mkdir()
    shutil.copy(valid_base_images[1], SOURCE_DIR / "subdir2" / "unique_B (1).jpg") # 이름 패턴 테스트

    # --- 시나리오 3: EXIF 데이터 변형 ---
    print("\n3. EXIF 데이터 변형 파일 생성 중...")
    img_with_exif = SOURCE_DIR / "photo_with_date.jpg"
    shutil.copy(valid_base_images[1], img_with_exif)
    set_exif_date(img_with_exif, "2022:11:25 10:30:00")

    img_no_exif = SOURCE_DIR / "photo_no_exif.jpg"
    shutil.copy(valid_base_images[0], img_no_exif)
    remove_exif(img_no_exif)

    # --- 시나리오 4: 손상 및 예외 파일 ---
    print("\n4. 손상 및 예외 파일 생성 중...")
    (SOURCE_DIR / "empty_file.jpg").touch()
    print("  - 0바이트 파일 생성: empty_file.jpg")

    corrupted_path = SOURCE_DIR / "corrupted.jpg"
    shutil.copy(valid_base_images[0], corrupted_path)
    with open(corrupted_path, "r+b") as f:
        f.truncate(1024)  # 파일을 1KB로 잘라 손상시킴
    print("  - 손상된 파일 생성: corrupted.jpg")

    # --- 시나리오 5: 특수 파일명 ---
    print("\n5. 특수 파일명 파일 생성 중...")
    shutil.copy(valid_base_images[1], SOURCE_DIR / "한글 사진 테스트.jpg")
    print("  - 한글 이름 파일 생성: 한글 사진 테스트.jpg")

    print(f"\n✅ 테스트 데이터 생성이 완료되었습니다. -> '{SOURCE_DIR}'")

if __name__ == "__main__":
    create_test_data()
