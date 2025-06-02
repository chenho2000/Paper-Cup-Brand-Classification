import requests
import json
import time
from PIL import Image
from pathlib import Path
import random

# Paths
SOURCE_IMAGES_DIR = Path("../images")
SOURCE_LABELS_DIR = Path("../labels")

# Model input size
MODEL_INPUT_SIZE = (640, 640)

def convert_yolo_to_xywh(yolo_bbox, img_width, img_height):
    """Convert YOLO format (x_center, y_center, width, height) to absolute coordinates"""
    x_center, y_center, width, height = yolo_bbox
    
    # Convert to absolute coordinates
    x = x_center * img_width
    y = y_center * img_height
    w = width * img_width
    h = height * img_height
    
    return [x, y, w, h]

def resize_bbox(bbox, original_size, target_size):
    """Resize bounding box from original image size to target size"""
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    
    # Calculate scaling factors
    scale_w = target_w / orig_w
    scale_h = target_h / orig_h
    
    # Resize bbox
    x, y, w, h = bbox
    new_x = x * scale_w
    new_y = y * scale_h
    new_w = w * scale_w
    new_h = h * scale_h
    
    return [new_x, new_y, new_w, new_h]

def read_yolo_label(label_path, img_width, img_height):
    """Read ground truth YOLO format label and un-normalize it, then resize it to match standard image size
    which is 640x640
    """
    ground_truth = []
    
    with open(label_path, 'r') as f:
        for line in f:
            class_id, *bbox = map(float, line.strip().split())
            # Convert YOLO format to absolute coordinates
            bbox = convert_yolo_to_xywh(bbox, img_width, img_height)
            # Resize bbox to match model input size
            bbox = resize_bbox(bbox, (img_width, img_height), MODEL_INPUT_SIZE)
            ground_truth.append({
                "label": int(class_id),
                "bbox": bbox
            })
    
    return ground_truth

def test_monitoring():
    """Test predict and track endpoints and prometheus scrape endpoints by sending requests to the server
    to test real time monitoring.
    """
    base_url = "http://localhost:8000"
    
    # Get list of all image files
    all_image_files = list(SOURCE_IMAGES_DIR.glob("*.jpg"))
    
    sampled_images = set()
    
    models = ["model_0", "model_1"]
    
    while len(sampled_images) < len(all_image_files):
        # Get a random image that hasn't been sampled yet
        remaining_images = [img for img in all_image_files if img not in sampled_images]
        if not remaining_images:
            break
            
        image_path = random.choice(remaining_images)
        sampled_images.add(image_path)
        
        # Get corresponding label file
        label_path = SOURCE_LABELS_DIR / f"{image_path.stem}.txt"

        # Read image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            print(f"Original image size: {img_width}x{img_height}")
            
        # Read ground truth
        ground_truth = read_yolo_label(label_path, img_width, img_height)
        
        # Test each model
        for model in models:
            print(f"\nTesting {model}...")
            
            with open(image_path, 'rb') as f:
                response = requests.post(
                    f"{base_url}/predict",
                    files={"file": f},
                    params={"model": model}
                )
            
            if response.status_code != 200:
                print(f"Prediction failed: {response.text}")
                continue
                
            predictions = response.json()["predictions"]
            
            # Track accuracy
            print("Tracking accuracy and confidence")
            print(f"Predictions: {json.dumps(predictions, indent=2)[:200]}")
            print(f"Ground truth: {json.dumps(ground_truth, indent=2)[:200]}")
        
            request_payload = {
                "model": model,
                "predictions": predictions,
                "ground_truth": ground_truth
            }
            
            accuracy_response = requests.post(
                f"{base_url}/track",
                json=request_payload
            )
            
            if accuracy_response.status_code != 200:
                print(f"Accuracy tracking failed: {accuracy_response.text}")
                continue
                
            print("Accuracy tracked successfully")
        
        time.sleep(1)


if __name__ == "__main__":
    print("Starting...")
    test_monitoring()