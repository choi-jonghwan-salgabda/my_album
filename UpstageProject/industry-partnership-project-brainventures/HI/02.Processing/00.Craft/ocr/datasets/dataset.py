import json
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from collections import OrderedDict

Image.MAX_IMAGE_PIXELS = 108000000
EXIF_ORIENTATION = 274  # Orientation Information: 274


class OCRDataset(Dataset):
    def __init__(self, image_path, annotation_path, transform):
        self.image_path = Path(image_path)
        self.transform = transform

        self.anns = OrderedDict()

        # annotation_path가 없다면, image_path에서 이미지만 불러오기
        if annotation_path is None:
            for ext in ['jpg', 'jpeg', 'png']:
                for file in self.image_path.glob(f'*.{ext}'):
                    if file.suffix.lower() == f'.{ext}':
                        self.anns[file.name] = None
            return

        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

            for filename in annotations['images'].keys():
                # Image file이 경로에 존재하는지 확인
                if (self.image_path / filename).exists():
                    # words 정보를 가지고 있는지 확인
                    if 'words' in annotations['images'][filename]:
                        # Words의 Points 변환
                        gt_words = annotations['images'][filename]['words']
                        polygons = [np.array([np.round(word_data['points'])], dtype=np.int32)
                                    for word_data in gt_words.values()
                                    if len(word_data['points'])]
                        self.anns[filename] = polygons
                    else:
                        self.anns[filename] = None
                else:
                    print(f"Warning: {filename} not found in {self.image_path}. Skipping annotation.")
                    continue

    def __len__(self):
        return len(self.anns.keys())

    def __getitem__(self, idx):
        image_filename = list(self.anns.keys())[idx]
        try:
            image_file = self.image_path / image_filename
            if not image_file.exists():
                raise FileNotFoundError(f"Image file not found: {image_file}")
            image = Image.open(image_file).convert('RGB')
        except FileNotFoundError as e:
            print(f"Error: {e}")
            # 에러 처리 로직 (예: 기본 이미지 반환 또는 예외 다시 발생)
            raise
        except Exception as e:
            print(f"Error opening image {image_filename}: {e}")
            # 다른 예외 처리 (이미지 손상, 메모리 문제 등)
            raise

        # EXIF정보를 확인하여 이미지 회전
        exif = image.getexif()
        if exif:
            if EXIF_ORIENTATION in exif:
                image = OCRDataset.rotate_image(image, exif[EXIF_ORIENTATION])
        org_shape = image.size

        item = OrderedDict(image=image, image_filename=image_filename, shape=org_shape)

        # Words의 Points 불러오기
        polygons = self.anns[image_filename] or None

        if self.transform is None:
            raise ValueError("Transform function is a required value.")

        transformed = self.transform(image=np.array(image), polygons=polygons)
        item.update(image=transformed['image'],
                    polygons=transformed['polygons'],
                    inverse_matrix=transformed['inverse_matrix'],
                    )

        return item

    @staticmethod
    def rotate_image(image, orientation):
        if orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180)
        elif orientation == 4:
            image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            image = image.rotate(-90, expand=True)
        elif orientation == 7:
            image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
        return image
