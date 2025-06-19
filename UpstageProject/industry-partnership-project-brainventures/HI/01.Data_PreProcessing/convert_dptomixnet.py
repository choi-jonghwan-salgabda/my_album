import json
import os
import numpy as np
import cv2
from tqdm import tqdm

def convert_json_format(input_json_path, output_json_path, image_dir):
    """Convert custom JSON format to MixNet format"""
    
    # Load the source JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    print(f"Loaded source data with {len(source_data['images'])} images and {len(source_data['annotations'])} annotations")
    
    # Create a mapping from image_id to file_name
    image_id_to_filename = {}
    for image in source_data['images']:
        image_id_to_filename[image['id']] = image['file_name']
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in source_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Create MixNet format data
    mixnet_data = []
    
    for image_id, filename in tqdm(image_id_to_filename.items(), desc="Converting"):
        if image_id not in annotations_by_image:
            print(f"Warning: No annotations for image_id {image_id}")
            continue
        
        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Read image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image: {image_path}")
            continue
            
        height, width = img.shape[:2]
        
        # Collect text polygons
        polygons = []
        texts = []
        
        for ann in annotations_by_image[image_id]:
            # Get polygon points
            if 'polys' in ann:
                # Reshape polygon points to pairs
                polygon = np.array(ann['polys']).reshape(-1, 2).tolist()
                polygons.append(polygon)
                
                # Get text content
                text = ann.get('text', '')
                texts.append(text)
        
        # Skip images without annotations
        if not polygons:
            continue
            
        # Create entry in MixNet format
        mixnet_entry = {
            'image_path': filename,
            'height': height,
            'width': width,
            'polygons': polygons,
            'texts': texts
        }
        
        mixnet_data.append(mixnet_entry)
    
    print(f"Converted {len(mixnet_data)} images to MixNet format")
    
    # Save to new JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(mixnet_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved converted data to {output_json_path}")

if __name__ == "__main__":
    input_json = "/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/jsons/train_poly_pos.json"
    output_json = "/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/jsons/train_mixnet_format.json"
    image_dir = "/data/ephemeral/home/industry-partnership-project-brainventures/data/Fastcampus_project/images/train"
    
    convert_json_format(input_json, output_json, image_dir)