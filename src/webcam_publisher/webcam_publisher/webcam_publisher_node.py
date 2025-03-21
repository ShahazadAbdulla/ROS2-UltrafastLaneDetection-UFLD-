import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class WebcamPublisherNode(Node):
    def __init__(self):
        super().__init__('webcam_publisher_node')

        # Publisher for webcam frames
        self.publisher_ = self.create_publisher(Image, 'webcam_image', 10)

        # Initialize CvBridge for converting OpenCV images to ROS Image messages
        self.bridge = CvBridge()

        # Open the webcam (device ID 0)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open webcam.")
            rclpy.shutdown()

        # Timer to publish frames at 30 FPS
        self.timer = self.create_timer(1 / 30, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to read frame from webcam.")
            return

        # Convert OpenCV image to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        # Publish the ROS Image message
        self.publisher_.publish(ros_image)
        self.get_logger().info("Published webcam frame.")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
