"""
Automatically annotate images with cup detection and manual selection for Timmies cups.

This script processes images recursively from a given directory, detecting cups in each image.

When multiple cups are detected, it displays a popup window allowing the user to manually
select the Timmies cup with a mouse click.

The selection is confirmed with the space key, and the annotation process for the current 
image can be saved and exited using the 'q' key.

When no cup is detected, the image will be moved into '_manual' directory for manual annotation.

The annotated images are saved into the '_result' directory.

Usage:
    python annotate_detr.py IMAGE_PATH

Arguments:
    IMAGE_PATH: Path to the directory containing images to be annotated.

Interaction:
    - Left mouse click: Select a cup in the popup window
    - Space key: Confirm the current selection
    - 'q' key: Save the annotation and move to the next image

The script will process all images in the given directory and its subdirectories.
"""

import torch
import requests
import cv2 
import sys

from pathlib import Path
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")

mouse_x, mouse_y = -1, -1

def predict(image):
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    result = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.shape[:2]]), threshold=0.5)[0]
    boxes = []

    # for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        if (label != 41):
            continue

        # boxes.append((box/torch.Tensor(list(image.shape[:2])*2)).tolist())
        boxes.append(box.tolist())
        # print(f"{label} {model.config.id2label[label]}: {score:.2f} {box}")
    return boxes


def convert_to_yolo_format(box, image_height, image_width):
    """
    Convert (top_left_x, top_left_y, bottom_right_x, bottom_right_y) to YOLO format.
    
    Args:
    box (tuple): (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    image_width (int): Width of the image
    image_height (int): Height of the image
    
    Returns:
    tuple: (x_center, y_center, width, height) in YOLO format
    """
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = box
    
    # Calculate width and height of the box
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y
    
    # Calculate center coordinates
    x_center = top_left_x + width / 2
    y_center = top_left_y + height / 2
    
    # Normalize by image dimensions
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    
    return (x_center_norm, y_center_norm, width_norm, height_norm)


def draw_bounding_box(image, box, index):
    height, width = image.shape[:2]
    x_center, y_center, box_width, box_height = box
    
    # Convert YOLO format to OpenCV format
    x = int((x_center - box_width/2) * width)
    y = int((y_center - box_height/2) * height)
    w = int(box_width * width)
    h = int(box_height * height)
    
    # Draw the bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    label = f"{index}"

    # Get the size of the label text
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    # Draw the label background
    cv2.rectangle(image, (x, y - label_height - 5), (x + label_width, y), (0, 255, 0), -1)
    
    # Put the label text
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y


def choose_timmie(image_path, image, boxes):
    global mouse_x, mouse_y
    mouse_x, mouse_y = -1, -1
    timmie_idx = set()
    for i, b in enumerate(boxes):
        b=convert_to_yolo_format(b, *image.shape[:2])
        i_box = draw_bounding_box(image, b, i)
        cv2.namedWindow(f"{image_path}")
        cv2.setMouseCallback(f"{image_path}", mouse_callback)
        cv2.imshow(f"{image_path}", i_box)

    while ((k:=cv2.waitKey(0))!=ord("q")):
        if (k==ord(" ")):
            choice=-1
            for j in range(len(boxes)):
                if (boxes[j][0]<= mouse_x <=boxes[j][2]) and (boxes[j][1]<= mouse_x <=boxes[j][3]):
                    choice = j
            if (choice!=-1):
                print(f"{image_path}: chose box #{choice}: {boxes[choice]}")
                timmie_idx.add(choice)
            else:
                print(f"{image_path}: no box chosen, retry")
    cv2.destroyAllWindows()
    return timmie_idx

def label(img_path : Path, base_path: Path):
    image = cv2.imread(img_path)
    boxes=predict(image)
    timmie_idx = {}
    print(f"labelling {img_path}")

    if (not boxes):
        print(f"no cup detected in {img_path}, moving to _manual")
        (img_path.parent/"_manual").mkdir(parents=True, exist_ok=True)
        img_path.rename(img_path.parent/"_manual"/img_path.name)
        return 

    if (len(boxes)>1):
        timmie_idx = choose_timmie("t", image, boxes)

    result_path=base_path/"_result"
    (result_path/img_path.relative_to(base_path)).parent.mkdir(parents=True, exist_ok=True)

    with open(img_path.with_suffix(".txt").absolute(),"w") as f:
        i_box=image.copy()
        for i, box in enumerate(boxes):
            yolo_box=convert_to_yolo_format(box, *image.shape[:2])
            f.write(f"{999 if len(boxes)==1 or i in timmie_idx else 41} {" ".join(map(str, yolo_box))}\n")
            i_box=draw_bounding_box(i_box, yolo_box, i)

    cv2.imwrite((result_path/img_path.relative_to(base_path)).absolute(), i_box)


assert len(sys.argv)>1
print(f"labeling images in {sys.argv[1]}")
base_path=Path(sys.argv[1])
l_img=list(base_path.glob("**/*.jpg"))

for img in l_img:
    label(img, base_path)