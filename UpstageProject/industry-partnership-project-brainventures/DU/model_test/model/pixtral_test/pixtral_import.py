# /data/ephemeral/home/industry-partnership-project-brainventures/DU/model_test/model/pixtral_test/pixtral_import.py
from huggingface_hub import snapshot_download
from pathlib import Path

mistral_models_path = Path.home().joinpath('mistral_models', 'Pixtral')
mistral_models_path.mkdir(parents=True, exist_ok=True)

print(f"Downloading necessary files for mistralai/Pixtral-12B-Base-2409 to {mistral_models_path}...")

# Explicitly list needed files and force download
snapshot_download(
    repo_id="mistralai/Pixtral-12B-Base-2409",
    allow_patterns=[
        "*.json", # Get all json config files (params.json, tekken.json, processor_config.json, etc.)
        "*.safetensors", # Get the weights
        "README.md", # Get the readme
        ".gitattributes" # Often needed
        # Add other potential file types if needed, e.g., "*.py" if there's custom code
    ],
    local_dir=mistral_models_path,
    local_dir_use_symlinks=False,
    force_download=True # Force it to re-download files even if they seem present
)
print("Download complete.")
