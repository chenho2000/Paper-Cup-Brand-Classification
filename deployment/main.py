import random
from datetime import datetime, timedelta
from functools import wraps

import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils.ops import xywh2xyxy

from pathlib import Path
import json

from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST, Histogram, Counter

# For input/output logging for evidently AI
import numpy as np

PREDICTION_LOG_PATH = Path(__file__).parent / "prediction_log.json"


def log_prediction(image, predictions):
    image_np = np.array(image)
    
    # Image stats
    width, height = image.width, image.height
    mean_pixel_value = float(np.mean(image_np))
    std_pixel_value = float(np.std(image_np))
    # Color histogram
    color_hist = {}
    for i, color in enumerate(['r', 'g', 'b']):
        # For each channel, get the histogram of the pixel values using 16 bins and normalize
        hist, _ = np.histogram(image_np[..., i], bins=16, range=(0, 255), density=True)
        color_hist[color] = hist.tolist()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "input_image_stats": {
            "width": width,
            "height": height,
            "mean_pixel_value": mean_pixel_value,
            "std_pixel_value": std_pixel_value,
            "color_histogram": color_hist
        },
        "predictions": predictions
    }
    # Append to JSON file
    if PREDICTION_LOG_PATH.exists():
        with open(PREDICTION_LOG_PATH, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)
    else:
        with open(PREDICTION_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump([entry], f, indent=2)

path_offline_batch_results = Path(__file__).parent / "offline_batch_results.txt"

start_time = datetime.now()
metrics = {
    "total_requests": 0,
    "request_times": [],
    "latencies": [],
    "confidence": []
}

model0 = YOLO("model0.pt")
model1 = YOLO("model1.pt")

default = "model_0"
models = {
    "model_0": {"model": model0, "describe": {
        "model": "model_0",
        "config": {
            "input_size": [640, 640],
            "batch_size": 16,
            "confidence_threshold": 0.25
        },
        "date_registered": "2025-03-15"
    }},
    "model_1": {"model": model1, "describe": {
        "model": "model_1",
        "config": {
            "input_size": [640, 640],
            "batch_size": 16,
            "confidence_threshold": 0.25
        },
        "date_registered": "2025-03-15"
    }}

}


# PROMETHEUS METRICS FOR ACCURACY, CONFIDENCE DISTRIBUTION REQUIREMENT, AND TOTAL PROCESSED

# Accuracy
DETECTION_METRICS = Gauge("model_detection_metrics", "Metrics for object detection", ["model", "class_name", "metric"])

# Confidence scores
CONFIDENCE_HIST = Histogram(
    "model_prediction_confidence",
    "Distribution of YOLO confidence scores from model predictions",
    ["model", "class_name"],
    buckets=[0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
)

# Total processed images
TOTAL_PROCESSED_IMAGES = Counter(
    "model_total_processed_images",
    "Total number of images processed since beginning",
    ["model"]
)


# MODEL METRICS CONFIG

IOU_THRESHOLDS = [0.3, 0.45, 0.6, 0.75]

# EACH MODEL HAS DIFFERENT CONFUSION MATRICES FOR DIFFERENT IOU THRESHOLDS FOR ACCURACY
model_metrics = {
    "predictions": {
        model_name: {
            "confusion_matrices": {
                str(iou): ConfusionMatrix(
                    nc=2,
                    conf=models[model_name]["describe"]["config"]["confidence_threshold"],
                    iou_thres=iou,
                    task="detect"
                ) for iou in IOU_THRESHOLDS
            },
            "total": 0,
            "confidence_scores": {
                "0": {"current": [], "buffer": []},
                "1": {"current": [], "buffer": []}
            }
        } for model_name in models.keys()
    }
}

IMAGE_SIZE = (640, 640)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

app = FastAPI()


def track_request(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        while (len(metrics["request_times"]) > 0) and (
                metrics["request_times"][0] < datetime.now() - timedelta(seconds=60)):
            metrics["request_times"].pop(0)
            if len(metrics["latencies"]) == 0:
                break
        return await func(*args, **kwargs)

    return wrapper


@app.get("/")
@track_request
async def hello():
    return "Hello, World!"


@app.post("/predict")
@app.put("/predict")
@track_request
async def predict_model(image: UploadFile = File(...), model: str = None):
    try:
        global metrics, models, default
        if not model or model not in models.keys():
            model = default
        print("predicting using model", model)
        if not image:
            raise HTTPException(
                status_code=400, detail="No image file provided")
        metrics["request_times"].append(datetime.now())
        metrics["total_requests"] += 1
        start = datetime.now()

        image_pil = Image.open(image.file)
        image_tensor = transform(image_pil).unsqueeze(0)

        output = {
            "predictions": [],
            "model_used": model
        }
        model = models[model]["model"]
        with torch.no_grad():
            pred = model(image_tensor)
        for i in range(len(pred[0].boxes.cls)):
            curr = dict()
            curr["label"] = "timmies" if pred[0].names[int(pred[0].boxes.cls[i])] == "Timmies" else "paper_cup"
            curr["confidence"] = round(float(pred[0].boxes.conf[i]), 2)
            curr["bbox"] = [round(x) for x in pred[0].boxes.xywh[i].tolist()]
            output["predictions"].append(curr)
            metrics["confidence"].append({"time": datetime.now(), "value": curr["confidence"]})
            metrics["confidence"] = metrics["confidence"][-100:]

        latency = (datetime.now() - start).total_seconds()

        metrics["latencies"].append(latency)

        log_prediction(image_pil, output["predictions"])

        return output

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health-status")
@track_request
async def health_status():
    try:
        global start_time
        current_time = datetime.now()
        uptime_duration = current_time - start_time

        days, remainder = divmod(uptime_duration.total_seconds(), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        uptime_str = f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes"

        response = {
            "status": "Healthy",
            "server": "FastAPI",
            "uptime": uptime_str
        }

        return response

    except Exception as e:
        return JSONResponse(content={"Status": "Error", "server": "FastAPI", "uptime": "N/A"}, status_code=500)


@app.get("/management/models")
@track_request
async def get_model():
    try:
        global models
        return {"available_models": list(models.keys())}

    except Exception as e:
        return JSONResponse(content={"available_models": []}, status_code=500)


@app.get("/group-info")
@track_request
async def get_group():
    member = ["Chen, Hongyu, h542chen",
              "Huang, Marcus, m43huang",
              "Lai, Anson, a5lai"]

    random.shuffle(member)

    return {"group": "group12", "members": member}


@app.get("/metrics")
@track_request
async def get_metrics():
    try:
        global metrics, start_time
        request_rate_per_minute = len(metrics["request_times"])

        avg_latency = sum(
            metrics["latencies"]) / len(metrics["latencies"]) if metrics["latencies"] else 0

        response = {
            "request_rate_per_minute": request_rate_per_minute,
            "avg_latency_ms": round(avg_latency * 1000),
            "max_latency_ms": round(max(metrics["latencies"], default=0) * 1000),
            "total_requests": metrics["total_requests"]
        }

        return response
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/metrics_extra")
async def get_metrics_extra():
    try:
        resp={"confidence": metrics["confidence"]}
        if (p:=Path(path_offline_batch_results)).exists():
            with open(p,"r") as f:
                offline_results=json.loads(f.read())
            row_keys = sorted(offline_results.keys(), key=int)
            col_keys = sorted(next(iter(offline_results.values())).keys(), key=int)
            matrix = [
                [offline_results[row][col] for col in col_keys]
                for row in row_keys
            ]
            resp["offline_results"] = {
                "matrix": matrix,
                "row_keys": row_keys,
                "col_keys": col_keys
            }
        return resp
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/management/models/{model}/describe")
@track_request
async def describe_model(model: str):
    global models
    try:
        return models[model]["describe"]
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/management/models/{model}/set-default")
@track_request
async def set_default_model(model: str):
    try:
        global default, models
        if model not in models.keys():
            raise HTTPException(status_code=404, detail="Model not found")
        default = model
        return {
            "success": "true",
            "default_model": model
        }
    except Exception as e:
        return JSONResponse(content={"success": "false", "error": str(e)}, status_code=500)


@app.post("/track")
async def track_accuracy_and_confidence(data: dict):
    """
    Track model accuracy by comparing predictions against ground truth.
    
    Expected request body:
    {
        "model": "model_name",
        "predictions": [{"label": "class_name", "bbox": [x, y, w, h], "confidence": float}],
        "ground_truth": [{"label": int, "bbox": [x, y, w, h]}]
    }
    
    Where:
    - model: The model name(e.g., "model_0")
    - predictions: List of model predictions from the /predict endpoint
      - label: Class name as string
      - bbox: Bounding box in [x, y, width, height] format (absolute coordinates)
      - confidence: Confidence score between 0 and 1
    - ground_truth: List of ground truth annotations
      - label: Class ID as integer
      - bbox: Bounding box in [x, y, width, height] format (absolute coordinates)
    
    Returns success status after updating the model's confusion matrix.
    """
    try:
        model = data.get("model")
        predictions = data.get("predictions", [])
        ground_truth = data.get("ground_truth", [])
        
        if not model or model not in models:
            raise HTTPException(status_code=404, detail="Model not found")
            
        # Convert predictions to required format (x1, y1, x2, y2, conf, class) and append to detections
        # Store confidence scores in model_metrics
        detections = []
        for pred in predictions:
            confidence = pred["confidence"]
            # Convert predictions to xyxy conf class format and append to detections
            box = xywh2xyxy(torch.tensor(pred["bbox"]).view(1, 4)).squeeze()
            class_idx = 1 if pred["label"] == "timmies" else 0
            detection = torch.cat([
                box,
                torch.tensor([confidence]),
                torch.tensor([class_idx])
            ])
            detections.append(detection)

            # Store confidence scores in model_metrics
            model_metrics["predictions"][model]["confidence_scores"][str(class_idx)].append(confidence)

            
        # Convert ground truth boxes to xyxy format
        gt_boxes = []
        gt_cls = []
        for gt in ground_truth:
            box = xywh2xyxy(torch.tensor(gt["bbox"]).view(1, 4)).squeeze()
            gt_boxes.append(box)
            gt_cls.append(gt["label"])
            
        if detections:
            detections = torch.stack(detections)
        else:
            detections = torch.zeros((0, 6))
            
        gt_boxes = torch.stack(gt_boxes) if gt_boxes else torch.zeros((0, 4))
        gt_cls = torch.tensor(gt_cls) if gt_cls else torch.zeros(0)
        
        # Process batch for each IOU threshold
        for iou in IOU_THRESHOLDS:
            model_metrics["predictions"][model]["confusion_matrices"][str(iou)].process_batch(
                detections,
                gt_boxes,
                gt_cls
            )
            print(model_metrics["predictions"][model]["confusion_matrices"][str(iou)].matrix)
        
        # Update total processed images in current run
        model_metrics["predictions"][model]["total"] += 1

        # Increment the all-time counter for total processed images
        TOTAL_PROCESSED_IMAGES.labels(
            model = model
        ).inc()

        return {"success": True}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/prometheus-metrics")
async def prometheus_metrics():
    for model_name, data in model_metrics["predictions"].items():
        if data["total"] > 0:
            # Calculate metrics for each IOU threshold
            for iou in IOU_THRESHOLDS:
                cm = data["confusion_matrices"][str(iou)]
                matrix = cm.matrix
                nc = cm.nc
                
                # Calculate metrics for each class
                for class_id in range(nc):
                    tp = matrix[class_id, class_id]
                    fp = matrix[class_id, :].sum() - tp
                    fn = matrix[:, class_id].sum() - tp
                    
                    if tp + fp + fn > 0:
                        accuracy = tp / (tp + fp + fn)
                        DETECTION_METRICS.labels(
                            model=model_name,
                            class_name=f"{class_id}_iou{iou}",
                            metric='accuracy'
                        ).set(accuracy)
                    
                    if tp + fp > 0:
                        precision = tp / (tp + fp)
                        DETECTION_METRICS.labels(
                            model=model_name,
                            class_name=f"{class_id}_iou{iou}",
                            metric='precision'
                        ).set(precision)
                    
                    if tp + fn > 0:
                        recall = tp / (tp + fn)
                        DETECTION_METRICS.labels(
                            model=model_name,
                            class_name=f"{class_id}_iou{iou}",
                            metric='recall'
                        ).set(recall)
            
            # Update confidence histogram metrics
            for class_id, score_lists in data["confidence_scores"].items():

                scores_to_observe = score_lists["current"]
                score_lists["current"] = score_lists["buffer"]
                score_lists["buffer"] = scores_to_observe
                
                for score in scores_to_observe:
                    CONFIDENCE_HIST.labels(
                        model=model_name,
                        class_name=class_id
                    ).observe(float(score))
                
                scores_to_observe.clear()
            
            # Track total processed images
            DETECTION_METRICS.labels(
                model=model_name,
                class_name = 'total',
                metric='processed_images'
            ).set(data["total"])

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="129.97.250.133", port=6132)
