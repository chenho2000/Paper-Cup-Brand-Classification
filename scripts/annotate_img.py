import os
from ultralytics import YOLO

# Set up the model and set paths
model = YOLO("yolo11n.pt")
input_folder = "images"
output_folder = "annotations"
os.makedirs(output_folder, exist_ok=True)

confidence_threshold = 0.5

for image in os.listdir(input_folder):
    if image.endswith(".jpg") or image.endswith(".png") or image.endswith(".jpeg"):
        image_path = os.path.join(input_folder, image)

    # Make inference on the image and store all bounding boxes in boxes
    # boxes is a list of bounding boxes with Normalized [x, y, width, height]
    prediction = model(image_path)
    boxes = prediction[0].boxes

    # Create a label file for the image
    label_path = os.path.join(output_folder, image.rsplit(".", 1)[0] + ".txt")

    with open(label_path, "w") as f:
        for box in boxes:
            confidence = box.conf[0]
            if confidence >= confidence_threshold:
                # normalized bounding box [x_center, y_center, width, height], 
                # take index 0 since its ndarray
                bbox = box.xywhn[0].tolist()
                # class label
                class_label = int(box.cls[0])

                # write in label_path using YOLO txt format
                f.write(f"{class_label} {' '.join(map(str, bbox))}\n")

print("Done")




