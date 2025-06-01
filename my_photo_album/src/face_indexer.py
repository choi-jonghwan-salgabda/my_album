import os
import pickle
import numpy as np
import cv2
from pathlib import Path
import face_recognition
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from config_loader import load_config
import logging
import shutil
import tempfile
import gc # Garbage Collector 모듈 추가

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# --- Helper Functions ---

def save_face(image, location, save_path):
    """Saves a cropped face region from an image."""
    top, right, bottom, left = location
    # 좌표 유효성 검사 (이미지 경계를 벗어나지 않도록)
    h, w = image.shape[:2]
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)

    # 크롭 영역이 유효한지 확인
    if top >= bottom or left >= right:
        logging.warning(f"Invalid crop dimensions for {save_path}. Skipping save.")
        return False

    face = image[top:bottom, left:right]
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite는 BGR 형식을 기대하므로 변환
        success = cv2.imwrite(str(save_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        if not success:
             logging.warning(f"cv2.imwrite failed for {save_path}")
             return False
        return True
    except Exception as e:
        logging.error(f"Failed to save cropped face to {save_path}: {e}")
        return False

def save_index(index_file, encodings, paths):
    """Safely saves the current index data (encodings and paths) to a file."""
    logging.debug(f"Attempting to save index with {len(encodings)} faces to {index_file}...")
    temp_file_path = None
    try:
        # Use NamedTemporaryFile for safer saving
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=index_file.parent, suffix=".tmp") as temp_f:
            temp_file_path = Path(temp_f.name)
            pickle.dump({"encodings": encodings, "paths": paths}, temp_f)

        # Atomically replace the old file with the new one
        shutil.move(str(temp_file_path), index_file)
        logging.debug(f"Index saved successfully to: {index_file}")
        return True
    except (pickle.PicklingError, OSError, Exception) as e:
        logging.error(f"Failed to save index file: {e}")
        # Clean up temp file if it exists and saving failed
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except OSError: pass
        return False
    finally:
        # Ensure temp file is removed if rename failed but it still exists
         if temp_file_path and temp_file_path.exists():
             try:
                 temp_file_path.unlink()
             except OSError: pass

# (plot_distribution function - use logging, unchanged from previous example)
def plot_distribution(encodings, output_path):
    """Generates and saves a t-SNE plot of face encodings."""
    if len(encodings) < 2:
        logging.info("Skipping visualization: Need at least 2 encodings.")
        return

    output_path = Path(output_path)
    output_dir = output_path.parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Calculating t-SNE reduction...")
        # Ensure encodings is a NumPy array
        encodings_array = np.array(encodings)
        if encodings_array.ndim == 1: # Handle case with only one encoding after filtering?
             logging.warning("t-SNE requires at least 2 samples. Skipping plot.")
             return
        reduced = TSNE(n_components=2, random_state=42, n_jobs=-1).fit_transform(encodings_array)
        logging.info("Plotting distribution...")
        plt.figure(figsize=(16, 10))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7)
        plt.title("Face Index Distribution (t-SNE)")
        plt.savefig(output_path)
        logging.info(f"📊 Distribution plot saved successfully: {output_path}")
    except ValueError as ve: # Catch specific errors like TSNE needing > 1 sample
         logging.error(f"Error during t-SNE or plotting: {ve}")
    except Exception as e:
        logging.error(f"Failed to generate or save distribution plot to {output_path}: {e}")
    finally:
        plt.close() # Ensure plot is closed to free memory


# --- Main Indexing Function ---
def index_faces(config_path):
    """
    Indexes faces incrementally, saving after each image to minimize memory usage.
    """
    logging.info(f"Loading configuration from: {config_path}")
    try:
        config = load_config(config_path)
        raw_dir = Path(config["data_path"])
        crop_dir = Path(config["cropped_faces_dir"])
        index_file = Path(config["index_output"])
        model = config.get("face_model", "cnn")
        visualization_output = config.get("visualization_output", "static/results/index_distribution.png")
    except KeyError as e:
        logging.error(f"Missing required key in configuration file '{config_path}': {e}")
        return
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return
    except Exception as e:
         logging.error(f"Error loading configuration: {e}")
         return

    logging.info(f"Raw image directory: {raw_dir}")
    logging.info(f"Cropped faces directory: {crop_dir}")
    logging.info(f"Index file: {index_file}")
    logging.info(f"Face detection model: {model}")

    # --- Load existing index OR initialize ---
    face_encodings = []
    face_paths = []
    processed_original_images = set()

    if index_file.exists():
        logging.info(f"Attempting to load existing index file: {index_file}")
        try:
            with open(index_file, "rb") as f:
                data = pickle.load(f)
                face_encodings = data.get("encodings", [])
                face_paths = data.get("paths", [])
                for p in face_paths:
                    original_stem = Path(p).stem.rsplit('_face', 1)[0]
                    processed_original_images.add(original_stem)
                logging.info(f"Successfully loaded {len(face_encodings)} existing face encodings.")
                logging.info(f"Will skip {len(processed_original_images)} already processed original images.")
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, Exception) as e:
            logging.warning(f"Could not load or parse existing index file ({e}). Starting fresh.")
            face_encodings = []
            face_paths = []
            processed_original_images = set()
    else:
        logging.info("No existing index file found. Starting fresh.")

    try:
        crop_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create cropped faces directory {crop_dir}: {e}")
        return

    logging.info("Scanning for image files...")
    try:
        image_files = [p for p in raw_dir.glob("**/*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        total_files = len(image_files)
        logging.info(f"Found {total_files} image files to process.")
    except Exception as e:
        logging.error(f"Error scanning directory {raw_dir}: {e}")
        return

    processed_count = 0
    newly_added_count_session = 0 # Track faces added in this run
    skipped_processed = 0

    for img_path in image_files:
        processed_count += 1
        log_prefix = f"[{processed_count}/{total_files}] {img_path.name}:"

        # --- Resumability Check ---
        if img_path.stem in processed_original_images:
            skipped_processed += 1
            continue

        logging.info(f"{log_prefix} Processing...")

        image = None # Ensure 'image' exists for potential 'del' in finally
        locations = None
        encodings = None
        image_processed_successfully = False

        try:
            image = face_recognition.load_image_file(img_path)
            locations = face_recognition.face_locations(image, model=model)

            if not locations:
                logging.info(f"{log_prefix} ❌ No faces found.")
                processed_original_images.add(img_path.stem) # Mark as processed (no faces)
                # No need to save index if no faces were added
                continue

            encodings = face_recognition.face_encodings(image, known_face_locations=locations)

            if not encodings:
                logging.warning(f"{log_prefix} ⚠️ Found face locations but failed to get encodings.")
                processed_original_images.add(img_path.stem) # Mark as processed (encoding failed)
                continue

            current_image_new_faces = 0
            temp_encodings = []
            temp_paths = []
            for j, (loc, enc) in enumerate(zip(locations, encodings)):
                out_path = crop_dir / f"{img_path.stem}_face{j}{img_path.suffix}"
                if save_face(image, loc, out_path): # Save face first
                    temp_encodings.append(enc)
                    temp_paths.append(str(out_path))
                    current_image_new_faces += 1
                else:
                    logging.warning(f"{log_prefix} Failed to save face {j}, skipping encoding for this face.")

            if current_image_new_faces > 0:
                # Append results for this image to the main lists
                face_encodings.extend(temp_encodings)
                face_paths.extend(temp_paths)
                newly_added_count_session += current_image_new_faces
                logging.info(f"{log_prefix} ✅ Indexed {current_image_new_faces} new face(s).")

                # --- Save index incrementally ---
                if not save_index(index_file, face_encodings, face_paths):
                     logging.error(f"{log_prefix} 🚨 CRITICAL: Failed to save index after processing. Data might be lost on exit.")
                     # Decide how to handle: stop? continue? retry?
                     # For now, we'll log the error and continue, but the index file might be outdated.
                # -----------------------------

                image_processed_successfully = True # Mark success only if faces were added and saved
            else:
                 logging.info(f"{log_prefix} ℹ️ No new faces were successfully saved and indexed for this image.")
                 # Mark as processed even if saving failed for some faces, to avoid retrying indefinitely
                 image_processed_successfully = True


        except MemoryError:
            logging.error(f"\n🚨 Memory Error encountered while processing {img_path.name}. Stopping processing loop.")
            # Attempt to save one last time before breaking (optional, might fail again)
            # save_index(index_file, face_encodings, face_paths)
            break # Exit the loop
        except FileNotFoundError:
             logging.error(f"{log_prefix} ⚠️ File not found (might have been deleted).")
             # Mark as processed to avoid retrying
             image_processed_successfully = True
        except Exception as e:
            logging.error(f"{log_prefix} ⚠️ Error during processing: {e}")
            # Do not mark as processed, will retry next time

        finally:
            # --- Memory Release ---
            if image_processed_successfully:
                 processed_original_images.add(img_path.stem) # Mark as processed *after* successful handling

            del image # Explicitly delete large objects
            del locations
            del encodings
            gc.collect() # Suggest garbage collection
            # --------------------

    logging.info(f"Finished processing loop. Skipped {skipped_processed} already processed images.")
    logging.info(f"Added {newly_added_count_session} new faces during this session.")
    logging.info(f"Total faces in index: {len(face_encodings)}")

    # --- Final Visualization ---
    # Plotting still requires loading all encodings, which uses memory at the end.
    plot_distribution(face_encodings, visualization_output)

    logging.info("Face indexing process finished.")


if __name__ == "__main__":
    import sys
    config_file_arg = sys.argv[1] if len(sys.argv) > 1 else "config/.my_photo_album_config.yaml"
    if os.path.isabs(config_file_arg):
        config_path = config_file_arg
    else:
        # Use script's directory, not current working directory
        script_dir = os.getcwd()
        config_path = os.path.join(script_dir, config_file_arg)

    if not os.path.exists(config_path):
         logging.critical(f"FATAL ERROR: Configuration file '{config_path}' not found.")
         sys.exit(1)

    index_faces(config_path)

