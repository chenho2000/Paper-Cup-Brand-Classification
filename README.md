# Paper-Cup-Brand-Classification
# Please note that, due to policy, the dataset and models cannot be shared publicly as they are considered course property. Feel free to reach out if you’d like to learn more.
## Overview
A computer vision system for detecting Tim Hortons cups (Timmies) and generic paper cups in images using YOLOv11. The project covers the full ML lifecycle from data collection to deployment.

## Features
- High Accuracy: 99.3% mAP50 on validation set
- Real-time Inference: ~5ms per image on T4 GPU
- Comprehensive Monitoring: Grafana dashboards for metrics tracking
- CI/CD Ready: Dockerized API with health checks

## Model Performance

| Metric     | YOLOv11m | YOLOv11l |
|------------|----------|----------|
| mAP50      | 0.993    | 0.992    |
| Recall     | 0.981    | 0.981    |
| Precision  | 0.989    | 0.985    |

### Prerequisites
- Python 3.8+
- Docker 20+
- NVIDIA GPU (recommended)

## Dataset Acknowledgment

This project uses a paper cup yolo v5 and v8 dataset by **Roboflow**.

- **Dataset**: [paperCupYolo8 Dataset](https://universe.roboflow.com/trash-mroef/papercupyolo8)
               [paperCupYolo5 Dataset](https://universe.roboflow.com/paper-cup-1cyxb/paper-cup-dhxoa)
- **Author**: see above links
- **Publisher**: Roboflow  
- **License**: [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

You can find more information and access the dataset at the two links. Modifications are applied to the data such as augmentation techniques. 

This project uses the **VN Trash Classification Dataset** provided by **The Cong LUONG** on Kaggle.

- **Dataset**: [VN Trash Classification Dataset](https://www.kaggle.com/datasets/mrgetshjtdone/vn-trash-classification)  
- **Author**: The Cong LUONG 
- **Publisher**: Kaggle  
- **License**: MIT License  (License in data folder)

You can find more information and access the dataset at [this link](https://www.kaggle.com/datasets/mrgetshjtdone/vn-trash-classification)

The kaggle dataset and yolo datasets in roboflow is used for training a YOLO model in this project. The views and results presented in this project do not necessarily reflect the views of the dataset author. The data will not be used for research purposes and the task of this project is not necessarily related to the author's work.
