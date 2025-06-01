import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq # Using generic classes, might need adjustment based on Pixtral specifics
from pathlib import Path
import warnings

# Suppress specific warnings if needed (optional)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
# 1. Set the path to your downloaded model files
model_path = Path("/data/ephemeral/home/mistral_models/Pixtral") # Use the correct path where files are downloaded

# 2. Set the path to the image you want to ask about
image_path = "/data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/data_sample/img/책표지_총류_000005.jpg" # <--- *** REPLACE THIS WITH YOUR IMAGE PATH ***

# 3. Define your question for the model
#    Note: The exact prompt format might influence results. Check Pixtral documentation if available.
#    Common formats might involve placeholders like <image> or specific instruction phrasing.
#    Let's try a simple question first.
text_prompt = "give me the text."
# Or, for a more specific question:
# text_prompt = "What text is visible in this image?"
# text_prompt = "Is there a person in this image?"

# --- Model Loading ---
print(f"Loading model and processor from: {model_path}")
try:
    # Use local_files_only=True to ensure it loads from your downloaded path
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, # Use bfloat16 for potentially lower memory usage if supported
        low_cpu_mem_usage=True,     # Helps when loading large models
        local_files_only=True,
        trust_remote_code=True # May be needed depending on the model implementation
    )
    print("Model and processor loaded successfully from local files.")

except Exception as e:
    print(f"Error loading model from local path '{model_path}': {e}")
    print("Ensure the path is correct and contains params.json, consolidated.safetensors, and tekken.json.")
    # Optionally, you could fall back to downloading from the hub here if local fails
    # processor = AutoProcessor.from_pretrained("mistralai/Pixtral-12B-Base-2409")
    # model = AutoModelForVision2Seq.from_pretrained("mistralai/Pixtral-12B-Base-2409", ...)
    exit() # Exit if loading fails

# --- Device Setup (Use GPU if available) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)

# --- Image Loading ---
print(f"Loading image from: {image_path}")
try:
    image = Image.open(image_path).convert("RGB") # Ensure image is in RGB format
except FileNotFoundError:
    print(f"Error: Image file not found at '{image_path}'")
    exit()
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# --- Prepare Inputs ---
print("Preparing inputs for the model...")
try:
    # The processor handles image preprocessing and text tokenization
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print("Inputs prepared.")

except Exception as e:
    print(f"Error processing inputs: {e}")
    exit()

# --- Generate Answer ---
print("Generating answer...")
try:
    with torch.no_grad(): # Inference doesn't require gradient calculation
        # Adjust generation parameters as needed (e.g., max_new_tokens)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512, # Increase or decrease based on expected answer length
            # You might add other parameters like temperature, top_k, top_p for different generation styles
            # temperature=0.7,
            # do_sample=True,
        )

    print("Generation complete.")

except Exception as e:
    print(f"Error during model generation: {e}")
    # If you get Out-of-Memory (OOM) errors, you might need more RAM/VRAM or try techniques
    # like quantization or using smaller models if available.
    exit()

# --- Decode and Print Result ---
print("Decoding the answer...")
try:
    # Decode the generated token IDs back to text
    # skip_special_tokens=True removes tokens like <bos>, <eos>
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print("\n--- Model's Answer ---")
    print(generated_text)
    print("----------------------")

except Exception as e:
    print(f"Error decoding the output: {e}")

