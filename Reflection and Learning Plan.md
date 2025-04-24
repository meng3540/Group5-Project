#  Phase 3: Reflection & Learning Plan

##  Ahmed Hamza Junaid – Reflection & Learning Plan

###  Reflection
Participating in this project allowed me to engage with embedded GPU computing for the first time and deepened my appreciation for real-time AI on constrained devices. I learned how to properly flash JetPack 6.1 onto the Jetson Orin Nano, access the system remotely via VNC, and install necessary development tools such as Codium and OpenCV. I also explored Python-based object detection pipelines and understood how pre-trained models can be effectively deployed using OpenCV’s `cv2.dnn` module. Being responsible for the initial setup phase taught me how critical hardware configuration and environment preparation is before diving into development. This experience boosted my confidence in deploying AI models beyond typical desktop environments. Additionally, it reinforced my understanding of how edge computing can provide immediate feedback and reduce the dependency on cloud services.

###  Learning Plan
While I was able to get the system running and execute the object detection prototype, I realized there is a deeper layer of performance optimization still to be explored. I aim to improve my knowledge in GPU profiling tools such as Nsight Systems and optimize inference performance using TensorRT. I also plan to expand my understanding of deep learning inference at the hardware level by enrolling in NVIDIA DLI training programs. In the long term, I want to contribute to AI projects in robotics and industrial automation where performance and efficiency are essential. For this, I will study quantization, pruning techniques, and model conversion workflows from frameworks like PyTorch or TensorFlow into deployment-ready formats like ONNX.

---

##  Ameer Ahmad Imwafi – Reflection & Learning Plan

###  Reflection
This project served as a critical hands-on experience that bridged the gap between classroom theory and real-world application. My role focused on camera configuration and ensuring seamless video input acquisition through OpenCV. I became familiar with how to manage input frame rates, normalize video streams, and properly prepare image data for inference with MobileNet SSD. Understanding the inner workings of OpenCV’s DNN module was particularly enlightening, especially how data blobs are created and processed. I also found it eye-opening to observe how quickly inference could be executed on Jetson’s onboard GPU even with limited power draw. The debugging process taught me the value of modular code and clear documentation — especially when dealing with edge hardware where any misconfiguration can significantly impact performance.

###  Learning Plan
To further my skills, I want to explore advanced model deployment workflows using TensorRT for model conversion and inference speed-up. I plan to study the NVIDIA Jetson ecosystem more deeply, including libraries like Jetson-inference and DeepStream SDK. Improving my understanding of ONNX interoperability between different AI frameworks will be key for adapting to real-world pipelines. I also aim to reinforce my skills in CUDA programming and memory management so I can eventually create optimized, hardware-aware solutions. In the near future, I will take advantage of NVIDIA’s free courses, participate in forums like the Jetson community, and experiment with custom object detection models trained in YOLOv8 or EfficientDet.

---

##  Rion Rodrigues – Reflection & Learning Plan

###  Reflection
Being part of this project introduced me to the full lifecycle of deploying an AI system on an embedded device. My main contributions involved testing and validating the detection system, and documenting performance outputs. I gained valuable insight into how system performance can be influenced by the model architecture, input resolution, and device workload. Through testing, I understood the limitations and benefits of edge computing — such as its real-time capabilities, autonomy from network reliance, and potential energy savings. The live feedback loop from the camera to detection output helped me develop a more intuitive grasp of computer vision in action. I also improved my understanding of how inference works step-by-step and how OpenCV integrates pre-trained models using the Caffe format. Collaborating with my team also highlighted the importance of clear task delegation and documentation in technical projects.

###  Learning Plan
Going forward, I plan to expand my knowledge in system-level optimization and AI deployment strategies. Specifically, I want to learn how to optimize AI models for edge hardware using tools like TensorRT and NVIDIA’s DeepStream for multi-stream video analytics. I also want to explore the possibilities of running more complex models, like YOLOv5 or Faster R-CNN, and understand how to downscale them for use on embedded platforms. Participating in online AI competitions like Edge Impulse or AIcrowd will help me gain experience in applying these technologies in realistic problem settings. I will also look into hybrid AI architectures that distribute processing between edge and cloud for maximum scalability, and enroll in micro-courses around embedded AI, CUDA acceleration, and edge deployment.
