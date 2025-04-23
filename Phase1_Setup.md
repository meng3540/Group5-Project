# 🧠 Phase 1: Jetson Setup & Object Detection Prototype

## 📌 Overview

This project explores the use of an embedded GPU-accelerated platform — the **NVIDIA Jetson Orin Nano** — for **real-time object detection** using a pre-trained **MobileNet SSD model**. The goal is to verify hardware setup, configure the development environment, acquire real-time camera input, and validate object detection functionality as a proof of concept.

---

## 🌟 Problem Statement

Real-time object detection is critical in AI-powered systems like autonomous vehicles, security surveillance, and industrial automation. These tasks are resource-intensive and traditionally run on cloud servers. However, there's a growing need for **low-latency, high-throughput AI inference** directly at the edge.

We aim to:
- Leverage GPU acceleration on a compact embedded system
- Implement a lightweight object detection solution
- Evaluate system performance through real-time visual outputs

---

## ⚙️ Jetson Nano Setup & Configuration

Our setup followed these steps:

1. **Flashed JetPack 6.1** to an SD card using **Balena Etcher**
2. Booted the **Jetson Orin Nano** and performed initial system configuration
3. Enabled remote GUI access via **VNC Server** (vino) and connected from host PC
4. Connected to WiFi and configured SSH access
5. Installed necessary applications:
   - 🔥 **Firefox** (via software center)
   - 🧠 **Codium** (VS Code alternative)
   - 🐍 **Python extension for Codium**

---

## 🧪 Python Environment Test

To confirm our Python setup, we created a `.py` file and imported key ML libraries:

```python
import tensorflow as tf
import keras as kp
```

This confirmed our environment was correctly configured for AI development.

---

## 📸 Camera Input & Object Detection Prototype

We verified the CSI/USB camera by accessing it with OpenCV:

```python
cap = cv2.VideoCapture(0)
```

Next, we cloned a GitHub repository containing the MobileNet SSD object detection script:

```bash
git clone <your_repo_link>
cd MobileNetSSD/
python mobilenet_ssd.py
```

---

## 🤖 Object Detection Model: MobileNet SSD

| Model        | Framework | Input Size | Use Case             |
|--------------|------------|-------------|-----------------------|
| MobileNet SSD | Caffe      | 300×300     | Real-time edge detection |

### Why We Chose It:
- Lightweight and fast → great for embedded platforms
- Compatible with OpenCV’s `cv2.dnn` module
- Pre-trained on common object classes (COCO/VOC-style)
- No need for additional training or fine-tuning

---

## 🧰 Frameworks & Libraries Used

| Type           | Name                     |
|----------------|--------------------------|
| OS             | JetPack 6.1 (Ubuntu 20.04) |
| Programming    | Python 3.x               |
| AI Inference   | OpenCV DNN (`cv2.dnn`)   |
| IDE            | Codium (with Python plugin) |
| Model Format   | Caffe (`.prototxt` + `.caffemodel`) |
| Communication  | SSH, VNC (vino server)   |

---

## 🗄️ Results

We successfully performed real-time object detection using our Jetson setup. Detected objects include:

- 🚆 `train` with confidence 99.9%
- 🢑 `chair` and 💻 `tvmonitor` with confidence >85%

Bounding boxes and labels were rendered live using OpenCV. Screenshots of detection results are saved in the [`camera_test_output/`](./camera_test_output) folder.

---

## 📂 Folder Structure

```
Phase1_Setup/
│
├── README.md
├── camera_test_output/
│   ├── train_detection.png
│   ├── tvmonitor_detection.png
│   └── chair_detection.png
└── mobilenet_ssd.py
```

---

## ✅ Next Steps

- Move forward with **workflow design** and formalize system architecture
- Profile GPU performance using `tegrastats` or NVIDIA Nsight
- Optimize the model using **TensorRT** in later phases

---

> 🔐 *This project lays the foundation for GPU-accelerated object detection on embedded systems. Stay tuned for the full workflow, optimizations, and performance profiling in Phase 2.*
