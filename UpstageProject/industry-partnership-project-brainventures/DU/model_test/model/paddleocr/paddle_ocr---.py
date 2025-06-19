# pip install paddleocr
# pip install paddlepaddle (CPU 버전)
# pip install paddlepaddle-gpu

from paddleocr import PaddleOCR
from io import BytesIO
from PIL import Image
import numpy as np # Import numpy if you choose the numpy array option

# --- Option 1: Pass the actual bytes (Recommended based on your code) ---
image_path = "/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/output/easyocr/annotated_책표지_총류_000001.jpg"
image = Image.open(image_path)
image_bytes_stream = BytesIO()
image.save(image_bytes_stream, format="PNG") # Save to the stream
image_bytes_stream.seek(0)
img_actual_bytes = image_bytes_stream.getvalue() # Get the bytes content

ocr = PaddleOCR(use_angle_cls=True, lang='korean') # Explicitly set use_gpu=False if you installed CPU version or want to force CPU
# Note: The DEBUG/WARNING logs show it's trying to use GPU 0 by default.
# If you don't have a GPU or didn't install paddlepaddle-gpu, explicitly set use_gpu=False.

print("--- Running OCR with Bytes ---")
result = ocr.ocr(img_actual_bytes, cls=True)  # Pass the actual bytes object
print(result)


# --- Option 2: Pass the file path directly (Simpler if reading from file) ---
# image_path = "/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/output/easyocr/annotated_책표지_총류_000001.jpg"
# ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
# print("\n--- Running OCR with File Path ---")
# result = ocr.ocr(image_path, cls=True) # Pass the file path string directly
# print(result)


# --- Option 3: Pass a NumPy array ---
# image_path = "/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/output/easyocr/annotated_책표지_총류_000001.jpg"
# image = Image.open(image_path).convert('RGB') # Ensure RGB format
# image_np = np.array(image)
# ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
# print("\n--- Running OCR with NumPy Array ---")
# result = ocr.ocr(image_np, cls=True) # Pass the numpy array
# print(result)

