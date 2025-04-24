# === MobileNet SSD Object Detection Script ===
# Description: Performs real-time object detection using a Caffe model on Jetson with OpenCV

import numpy as np
import argparse
import cv2


# Argument parser for command-line flexibility
parser = argparse.ArgumentParser(description='Run MobileNet-SSD object detection')
parser.add_argument("--video", help="Path to video file. If empty, the webcam will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                    help="Path to Caffe 'deploy' prototxt file")
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                    help="Path to pre-trained Caffe model")
parser.add_argument("--thr", default=0.2, type=float, help="Confidence threshold")
args = parser.parse_args()

# Class labels for detection
classNames = {
    0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike',
    15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
}

# Capture video source
cap = cv2.VideoCapture(args.video if args.video else 0)

# Load Caffe model
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()

    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args.thr:
            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            heightFactor = frame.shape[0] / 300.0
            widthFactor = frame.shape[1] / 300.0

            xLeftBottom = int(widthFactor * xLeftBottom)
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop = int(widthFactor * xRightTop)
            yRightTop = int(heightFactor * yRightTop)

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))

            if class_id in classNames:
                label = classNames[class_id] + ": " + str(round(confidence * 100, 2)) + "%"
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                              (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                print(label)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:
        break

cap.release()
cv2.destroyAllWindows()
