# Import necessary libraries
import os
import csv
from PIL import Image
import numpy as np
import cv2 # Using OpenCV for image loading and heatmap processing
import paddle
from paddleocr import PaddleOCR, draw_ocr
from paddleocr.tools.infer.utility import check_gpu, get_image_file_list, check_and_read
from paddleocr.tools.infer.predict_cls import TextClassifier # Import classifier for manual use
from paddleocr.tools.infer.predict_det import TextDetector # Import detector for manual use
from paddleocr.tools.infer.predict_rec import TextRecognizer # Import recognizer for manual use
from paddleocr.ppocr.utils.utility import get_image_file_list
from paddleocr.ppocr.utils.logging import get_logger
import traceback # For detailed error printing
import matplotlib.pyplot as plt # For colormap

logger = get_logger() # Use PaddleOCR's logger

# --- Configuration ---
img_input_folder = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/images/test"
# Define output path for PaddleOCR results CSV
csv_output_path = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/paddleocr/paddleocr_heatmap_test.csv"
# --- NEW: Define output folders for heatmaps ---
heatmap_output_folder = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/paddleocr/heatmaps_raw"
overlay_output_folder = "/data/ephemeral/home/Fastcampus_project/Fastcampus_project/output/paddleocr/heatmaps_overlay"

# --- Grad-CAM Configuration ---
# !!! IMPORTANT: Replace with the actual name of the target layer in the cls_model !!!
# You might need to print(ocr_engine.cls_model) or check PaddleOCR source to find the correct name.
# Common names might involve 'conv', 'features', 'backbone', etc.
GRAD_CAM_TARGET_LAYER_NAME = 'conv2d_5' # Placeholder - VERIFY THIS!

# --- Helper Functions ---
def get_image_files(folder_path):
    """Gets a list of image files from a folder."""
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')
    return [f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(img_extensions)]

def format_coordinates(bbox):
    """Formats bounding box coordinates into a space-separated string."""
    coords = []
    for point in bbox:
        coords.extend([str(int(point[0])), str(int(point[1]))])
    return " ".join(coords)

# --- Grad-CAM Implementation ---
feature_maps = []
gradients = []

def save_feature_map(layer, input, output):
    feature_maps.append(output)

def save_gradient(layer, grad_input, grad_output):
    gradients.append(grad_output[0])

def generate_grad_cam(model, target_layer_name, input_tensor, class_idx):
    """Generates Grad-CAM heatmap for a specific class index."""
    feature_maps.clear()
    gradients.clear()

    # Find the target layer
    target_layer = None
    for name, layer in model.named_sublayers():
        # print(f"Layer name: {name}") # Uncomment to print layer names for debugging
        if name == target_layer_name:
            target_layer = layer
            break

    if target_layer is None:
        logger.error(f"Error: Target layer '{target_layer_name}' not found in the model.")
        return None

    # Register hooks
    forward_hook = target_layer.register_forward_post_hook(save_feature_map)
    backward_hook = target_layer.register_backward_hook(save_gradient)

    try:
        # Forward pass
        model.eval()
        output = model(input_tensor) # [N, num_classes] e.g., [[0.1, 0.9]]

        # Backward pass
        model.clear_gradients()
        target_output = output[0, class_idx] # Get the score for the target class
        target_output.backward(retain_graph=False) # Compute gradients

        if not gradients or not feature_maps:
             logger.error("Error: Gradients or feature maps were not captured.")
             return None

        # Get gradients and feature maps
        guided_gradients = gradients[0] # Should be [N, C, H, W]
        activation_maps = feature_maps[0] # Should be [N, C, H, W]

        # Pool gradients across spatial dimensions (Global Average Pooling)
        weights = paddle.mean(guided_gradients, axis=[2, 3], keepdim=True) # [N, C, 1, 1]

        # Compute weighted sum of feature maps
        cam = paddle.sum(weights * activation_maps, axis=1) # [N, H, W]
        cam = cam.squeeze(0) # Remove batch dim -> [H, W]

        # Apply ReLU
        cam = paddle.nn.functional.relu(cam)

        # Normalize heatmap
        cam = cam - paddle.min(cam)
        cam_max = paddle.max(cam)
        if cam_max == 0: # Avoid division by zero if CAM is all zeros
             logger.warning("Grad-CAM heatmap is all zeros.")
             return cam.numpy() # Return zero heatmap
        cam = cam / cam_max
        cam = cam.numpy()

    except Exception as e:
        logger.error(f"Error during Grad-CAM generation: {e}")
        traceback.print_exc()
        cam = None
    finally:
        # Remove hooks
        forward_hook.remove()
        backward_hook.remove()

    return cam

def visualize_heatmap(original_img_np, heatmap, save_path_raw, save_path_overlay):
    """Saves raw heatmap and overlay."""
    if heatmap is None:
        return

    try:
        # Resize heatmap to original image size
        h, w = original_img_np.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # --- Save Raw Heatmap ---
        # Normalize to 0-255 for saving as grayscale image
        heatmap_normalized = (heatmap_resized * 255).astype(np.uint8)
        cv2.imwrite(save_path_raw, heatmap_normalized)

        # --- Create and Save Overlay ---
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_img_np, 0.6, heatmap_colored, 0.4, 0)
        cv2.imwrite(save_path_overlay, overlay)

    except Exception as e:
        logger.error(f"Error saving heatmap/overlay: {e}")
        traceback.print_exc()

# --- Main Processing Logic ---
print(f"Processing images from: {img_input_folder}")
print(f"Writing CSV output to: {csv_output_path}")
print(f"Saving raw heatmaps to: {heatmap_output_folder}")
print(f"Saving overlay heatmaps to: {overlay_output_folder}")

# Ensure output directories exist
for folder in [os.path.dirname(csv_output_path), heatmap_output_folder, overlay_output_folder]:
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {folder}: {e}")
        exit()

try: # Main try block for file processing
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['filename', 'polygons'])

        image_files = get_image_files(img_input_folder)
        total_files = len(image_files)
        print(f"Found {total_files} image files to process.")

        # Initialize PaddleOCR reader and sub-models
        try:
            # Use default args for simplicity, adjust if needed (e.g., specific model paths)
            args = {'use_gpu': False, 'lang': 'korean', 'use_angle_cls': True} # Keep use_gpu consistent
            ocr_engine = PaddleOCR(**args) # Use default models

            # Manually get sub-models (needed for Grad-CAM on cls)
            text_detector = TextDetector(ocr_engine.args)
            text_classifier = TextClassifier(ocr_engine.args)
            text_recognizer = TextRecognizer(ocr_engine.args)
            print("PaddleOCR Engine and sub-models initialized (lang='korean', use_gpu=False).")

            # --- Verify Target Layer Exists ---
            layer_found = False
            for name, _ in text_classifier.model.named_sublayers():
                if name == GRAD_CAM_TARGET_LAYER_NAME:
                    layer_found = True
                    break
            if not layer_found:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"WARNING: Grad-CAM target layer '{GRAD_CAM_TARGET_LAYER_NAME}' not found in classifier model.")
                print(f"Heatmaps will likely fail. Please check the model structure and update GRAD_CAM_TARGET_LAYER_NAME.")
                print(f"Available layers:")
                for name, _ in text_classifier.model.named_sublayers(): print(f"  - {name}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                 print(f"Grad-CAM target layer '{GRAD_CAM_TARGET_LAYER_NAME}' found in classifier model.")


        except Exception as e:
             print(f"Error initializing PaddleOCR Engine: {e}")
             print(traceback.format_exc())
             print("Please ensure PaddleOCR and paddlepaddle are installed correctly.")
             exit()

        processed_count = 0
        for img_name in image_files:
            img_path = os.path.join(img_input_folder, img_name)
            logger.info(f"Processing image: {img_path}")

            try:
                # --- Load Image using OpenCV (needed for cropping and overlay) ---
                img_cv2, flag = check_and_read(img_path)
                if not flag:
                    logger.warning(f"Cannot read image file (skip) {img_path}")
                    writer.writerow([img_name, "ERROR: Cannot read image"])
                    continue
                if img_cv2 is None:
                     logger.warning(f"Image loaded as None (skip) {img_path}")
                     writer.writerow([img_name, "ERROR: Image loaded as None"])
                     continue

                # --- 1. Run Detection ---
                dt_boxes, det_elapse = text_detector(img_cv2)
                if dt_boxes is None or len(dt_boxes) == 0:
                    logger.info(f"No text detected in {img_name}")
                    writer.writerow([img_name, ""]) # Write empty polygons if no boxes
                    processed_count += 1
                    continue

                # --- 2. Run Classification (with Grad-CAM) & Recognition ---
                img_crop_list = []
                for i in range(len(dt_boxes)):
                    tmp_box = dt_boxes[i].astype(np.int32) # Ensure integer coords
                    img_crop = ocr_engine.get_rotate_crop_image(img_cv2, tmp_box)
                    img_crop_list.append(img_crop)

                # --- Classification + Grad-CAM ---
                cls_res = []
                if text_classifier is not None:
                    img_list_cls, cls_elapse = text_classifier(img_crop_list)
                    logger.debug("cls num  : {}, elapsed : {}".format(
                        len(img_list_cls), cls_elapse))

                    # --- Grad-CAM Generation Loop ---
                    for i, (img_crop_cls, cls_pred) in enumerate(img_list_cls):
                        cls_label, cls_conf = cls_pred
                        cls_res.append({'label': cls_label, 'confidence': cls_conf})

                        try:
                            # Prepare input tensor for Grad-CAM (use the same preprocessing as classifier)
                            # Note: text_classifier expects a list, we process one by one for Grad-CAM
                            cls_input_tensor = text_classifier.process_batch([img_crop_cls])

                            # Get the predicted class index (0 or 1 for angle)
                            predicted_class_idx = np.argmax(text_classifier.model(cls_input_tensor).numpy())

                            # Generate Grad-CAM
                            heatmap = generate_grad_cam(text_classifier.model,
                                                        GRAD_CAM_TARGET_LAYER_NAME,
                                                        cls_input_tensor,
                                                        predicted_class_idx)

                            # Save heatmap and overlay
                            if heatmap is not None:
                                # Use original crop before potential rotation by classifier
                                original_crop_for_vis = img_crop_list[i]
                                # Ensure crop is BGR for cv2 saving
                                if original_crop_for_vis.shape[-1] == 1: # Grayscale
                                     original_crop_for_vis = cv2.cvtColor(original_crop_for_vis, cv2.COLOR_GRAY2BGR)
                                elif original_crop_for_vis.shape[-1] == 4: # RGBA
                                     original_crop_for_vis = cv2.cvtColor(original_crop_for_vis, cv2.COLOR_RGBA2BGR)

                                heatmap_raw_name = f"{os.path.splitext(img_name)[0]}_box{i}_cls{cls_label}_raw.png"
                                overlay_name = f"{os.path.splitext(img_name)[0]}_box{i}_cls{cls_label}_overlay.png"
                                visualize_heatmap(original_crop_for_vis, heatmap,
                                                  os.path.join(heatmap_output_folder, heatmap_raw_name),
                                                  os.path.join(overlay_output_folder, overlay_name))
                        except Exception as grad_cam_e:
                             logger.error(f"Error generating/saving Grad-CAM for {img_name}, box {i}: {grad_cam_e}")
                             traceback.print_exc()

                    # Update img_crop_list based on classification results (potential rotation)
                    img_crop_list = [item[0] for item in img_list_cls] # Use potentially rotated crops for recognition

                # --- Recognition ---
                rec_res, rec_elapse = text_recognizer(img_crop_list)
                logger.debug("rec_res num  : {}, elapsed : {}".format(
                    len(rec_res), rec_elapse))

                # --- Combine Results and Format for CSV ---
                final_results = []
                coords_list = []
                for i in range(len(dt_boxes)):
                    text, score = rec_res[i]
                    # Add classification confidence if available
                    cls_conf = cls_res[i]['confidence'] if cls_res else -1
                    # Combine results: box, (text, rec_score), cls_conf
                    final_results.append([dt_boxes[i], (text, score), cls_conf])
                    # Format coordinates for CSV
                    coords = format_coordinates(dt_boxes[i])
                    coords_list.append(coords)

                # --- Write to CSV ---
                writer.writerow([img_name, '|'.join(coords_list)])
                processed_count += 1
                if processed_count % 10 == 0: # Print progress less often due to slower processing
                    logger.info(f"  Processed {processed_count}/{total_files} images...")

            except FileNotFoundError:
                 logger.error(f"Image file not found during processing: {img_path}")
                 writer.writerow([img_name, "ERROR: File not found"])
            except Exception as e:
                logger.error(f"Error processing image {img_name}: {e}")
                traceback.print_exc()
                writer.writerow([img_name, f"ERROR: {e}"])

    print(f"\nProcessing complete. Processed {processed_count} images.")
    print(f"CSV file saved to {csv_output_path}")
    print(f"Raw heatmaps saved to: {heatmap_output_folder}")
    print(f"Overlay heatmaps saved to: {overlay_output_folder}")

except IOError as e:
    print(f"Error opening or writing to CSV file {csv_output_path}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print(traceback.format_exc())