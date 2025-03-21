# ROS 2 Ultrafast Lane Detection (UFLD)

This project wraps the **Ultrafast Lane Detection (UFLD)** model in a fully ROS 2 (Humble)-compatible system. It performs **real-time lane detection** from either a video or live webcam stream, and publishes:

- Processed frames (with lane overlays) via `sensor_msgs/msg/Image`
- Detected lane coordinates via `std_msgs/msg/String`

---

## 📦 Origin & Credits

### 🧠 Based on:
- **Original UFLD Paper/Repo by `cfzd`**  
  🔗 https://github.com/cfzd/Ultra-Fast-Lane-Detection  
  > This repo introduced the model architecture, training, and official PyTorch implementation.

- **ONNX Inference Wrapper by `ibaiGorordo`**  
  🔗 https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference  
  > Provided a simplified wrapper for real-time inference. We adapted the model and structure into a ROS 2 node here.

> This ROS 2 version is a full reimplementation and adaptation using the **original PyTorch model**, not ONNX.

---

## 🛠️ Requirements

### ✅ Python Dependencies

Install with:

```bash
pip install -r requirements.txt
this includes:

    torch, torchvision, numpy, opencv-python, Pillow, scipy
```

### ✅ ROS 2 Dependencies

This project is built for ROS 2 Humble. Install required packages:
sudo apt update
sudo apt install \
  ros-humble-rclpy \
  ros-humble-cv-bridge \
  ros-humble-sensor-msgs \
  ros-humble-std-msgs

## 📂 Project Structure
src/
├── ufld_lane_detection/     # Lane detection logic + ROS node
└── webcam_publisher/        # Publishes webcam frames

## ⬇️ Setup Instructions
1. Clone the Repo

git clone https://github.com/ShahazadAbdulla/Abaja-LKA.git
cd Abaja-LKA

2. Download the Pretrained TuSimple Model

### 📥 Model link:
Google Drive - https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view

Place the model at:

src/ufld_lane_detection/ufld_lane_detection/models/tusimple_18.pth

3. Download a Test Video (Optional)

You can use the video from this link to test the system:

🎥 YouTube - Highway Lane Test Video

Download it using yt-dlp:

yt-dlp -f mp4 https://www.youtube.com/watch?v=2CIxM7x-Clc -o ~/Documents/road0.mp4

Update lane_detection_node.py if your path is different:

self.video_path = "/home/shadow0/Documents/road0.mp4"

### ⚙️ Build the ROS Workspace

source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash

### 🚀 Running the Nodes
1. Webcam Publisher (optional)

ros2 run webcam_publisher webcam_publisher_node

Publishes webcam images on /webcam_image.
2. Lane Detection Node

ros2 run ufld_lane_detection lane_detection_node

Processes either video file or webcam, and publishes:

    /lane_detection_output – image with detected lanes
    /lane_coordinates – stringified lane coordinates

### 📸 Switching Between Video and Webcam Input

In lane_detection_node.py, to switch to webcam:

Replace:

self.cap = cv2.VideoCapture(self.video_path)

With:

self.cap = cv2.VideoCapture(0)  # Webcam stream

### 🧪 Topics Published
Topic	Type	Description
/lane_detection_output	sensor_msgs/msg/Image	Processed image with lanes overlaid
/lane_coordinates	std_msgs/msg/String	List of detected lanes and centerline
/webcam_image	sensor_msgs/msg/Image	Raw webcam feed
🚗 Roadmap

We're in the process of integrating a Stanley Controller to close the loop and complete a basic Lane Keeping Assist (LKA) system.
This functionality will be released in a new ROS 2 repo soon.

##📄 License

This project is licensed under the MIT License.

Model, datasets, and core detection logic are licensed under their respective terms from:

    cfzd/Ultra-Fast-Lane-Detection
    ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference
