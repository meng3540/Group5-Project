# 🔄 Phase 2: Workflow Design & Prototype

## 🔄 Overview

In this phase, we document the full AI inferencing workflow implemented on our **Jetson Orin Nano** edge device. The system captures live video input, performs object detection using the **MobileNet SSD** model accelerated through the Jetson GPU, and displays real-time results. This phase also includes a prototype demonstration that validates the core functionality of the system.

---

## 📉 System Block Diagram

```
[Camera Input] ---> [Image Preprocessing] ---> [MobileNet SSD (Caffe)] ---> [GPU Acceleration on Jetson] ---> [Output Frame with Bounding Boxes]
```

---

## 🧭 Workflow Steps

Below is the detailed breakdown of our workflow:

1. **System Initialization**
   - Boot Jetson Orin Nano with JetPack 6.1
   - Connect via VNC or SSH
   - Update packages and install OpenCV, Python3, Git

2. **Environment Setup**
   - Install Codium and Python extension
   - Validate Python with TensorFlow/Keras imports

3. **Camera Configuration**
   - Use OpenCV to access USB/CSI camera
   - Confirm real-time image capture using `cv2.VideoCapture()`

4. **Clone and Configure Model Repository**
   - Clone GitHub repo with MobileNet SSD model
   - Verify `mobilenet_ssd.py` and supporting files (`.prototxt`, `.caffemodel`)

5. **Run Real-Time Inference**
   - Resize input frames to 300x300
   - Normalize image and create input blob
   - Forward blob through DNN using `cv2.dnn.readNetFromCaffe`
   - Parse detections and draw bounding boxes on original frame

6. **Display Output**
   - Use `cv2.imshow()` to visualize annotated frame
   - Capture screenshots for documentation

7. **Performance Monitoring**
   - Monitor GPU and memory usage with `tegrastats`
   - Prepare for later optimizations (e.g., TensorRT)

---

## 🧪 Terminal Commands Used During Workflow

Here are the key terminal commands used to perform setup, execution, and testing:

```bash
# System updates and base installs
sudo apt update && sudo apt upgrade
sudo apt install python3-pip python3-opencv git

# (Optional) Install Codium via snap
sudo snap install codium --classic

# Validate Python libraries
python3
>>> import tensorflow as tf
>>> import keras as kp
>>> exit()

# Clone model repo
git clone <your_repo_link>
cd <repo_folder>

# Run object detection script
python3 mobilenet_ssd.py

# With explicit model arguments
python3 mobilenet_ssd.py --prototxt MobileNetSSD_deploy.prototxt --weights MobileNetSSD_deploy.caffemodel

# Camera test (one-liner)
python3 -c "import cv2; cap=cv2.VideoCapture(0); print('Camera ready' if cap.isOpened() else 'Camera not found')"

# Monitor GPU usage
tegrastats
```

---

## 🚀 Frameworks & Libraries Used

| Component              | Tool/Library                   |
|------------------------|----------------------------------|
| Operating System       | JetPack 6.1 (Ubuntu 20.04)       |
| Programming Language   | Python 3.x                       |
| AI Inference Engine    | OpenCV DNN (with Caffe support)  |
| Pre-trained Model      | MobileNet SSD (`.prototxt`, `.caffemodel`) |
| IDE                    | Codium + Python plugin           |
| Monitoring Tool        | tegrastats / terminal tools      |

---

## 🔗 Resources & References

- [OpenCV DNN Module Docs](https://docs.opencv.org/master/d6/d0f/group__dnn.html)
- [NVIDIA Jetson Getting Started](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
- [MobileNet SSD Caffe Model](https://github.com/chuanqi305/MobileNet-SSD)
- [Jetson Utils Repo](https://github.com/dusty-nv/jetson-utils)
- [Tegrastats for Jetson](https://elinux.org/Jetson/Performance_Monitoring)

---

## 📁 Directory Structure

```
Workflow/
│
├── workflow.md
├── block_diagram.png
├── code/
│   ├── mobilenet_ssd.py
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel (or link)
└── output_samples/
    ├── Chair.jpeg
    ├── TV monitor.jpeg
    ├── Train.jpeg
    └── aeroplane.jpeg
```

---

> ✨ This workflow is modular and can be adapted to integrate advanced features such as model optimization with TensorRT, support for multiple input streams, or deployment in edge-AI pipelines in future development phases.
