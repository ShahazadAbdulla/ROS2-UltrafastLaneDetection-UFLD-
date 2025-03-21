import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String  # For publishing lane coordinates
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from ufld_lane_detection.ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector, ModelType

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # ROS 2 Publishers
        self.publisher_frame = self.create_publisher(Image, 'lane_detection_output', 10)
        self.publisher_coords = self.create_publisher(String, 'lane_coordinates', 10)

        # Initialize CvBridge for OpenCV <-> ROS Image conversion
        self.bridge = CvBridge()

        # Parameters for model and video path
        self.model_path = "/home/shadow0/lka_ws/src/ufld_lane_detection/ufld_lane_detection/models/tusimple_18.pth"
        self.video_path = "/home/shadow0/Documents/road0.mp4"

        # Initialize the lane detection model
        self.lane_detector = UltrafastLaneDetector(self.model_path, ModelType.TUSIMPLE, use_gpu=False)
        self.get_logger().info("Lane detection model initialized.")

        # Open the video file
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Error: Could not open video file: {self.video_path}")
            self.cap.release()
            rclpy.shutdown()

    def process_frame(self, frame):
        """
        Detect lanes in a single frame and return the processed frame and lane coordinates.

        Args:
            frame (ndarray): Input video frame.

        Returns:
            tuple: Processed frame with lane detections overlaid and a list of lane coordinates.
        """
        frame_height, frame_width = frame.shape[:2]

        # Detect lanes
        self.lane_detector.detect_lanes(frame)
        lanes_points = self.lane_detector.lanes_points

        # List to hold valid lanes
        valid_lanes = []

        # Filter and process lanes
        for lane_points in lanes_points:
            bottom_half_points = [
                pt for pt in lane_points if pt[1] >= frame_height / 2
            ]
            if len(bottom_half_points) >= 2:  # Minimum number of points required
                valid_lanes.append(bottom_half_points)

        output_frame = frame.copy()
        lane_coordinates = []  # List to store lane coordinates
        if len(valid_lanes) >= 2:
            # Select the two most prominent lanes
            valid_lanes.sort(key=lambda lane: len(lane), reverse=True)
            lane1_points, lane2_points = valid_lanes[:2]

            # Fit polynomials to the lanes
            coeffs1 = self.fit_polynomial(lane1_points, frame_height)
            coeffs2 = self.fit_polynomial(lane2_points, frame_height)

            # Prepare y-values for lane visualization
            y_values = np.linspace(frame_height / 2, frame_height, num=100)
            y_values_flipped = frame_height - y_values

            # Evaluate the polynomials
            x1 = self.evaluate_polynomial(coeffs1, y_values_flipped)
            x2 = self.evaluate_polynomial(coeffs2, y_values_flipped)

            # Compute the centerline
            x_center = (x1 + x2) / 2

            # Add lane coordinates to the list
            lane_coordinates.append({'lane1': list(zip(x1, y_values))})
            lane_coordinates.append({'lane2': list(zip(x2, y_values))})
            lane_coordinates.append({'center': list(zip(x_center, y_values))})

            # Draw the lanes
            self.draw_polyline(output_frame, x1, y_values, color=(0, 0, 255))  # Red
            self.draw_polyline(output_frame, x2, y_values, color=(0, 255, 0))  # Green
            self.draw_polyline(output_frame, x_center, y_values, color=(0, 255, 255))  # Yellow centerline

        return output_frame, lane_coordinates

    def draw_polyline(self, image, x_values, y_values, color, thickness=2):
        """
        Draw a polyline on the image given x and y values.

        Args:
            image (ndarray): Image to draw on.
            x_values (ndarray): x-coordinates.
            y_values (ndarray): y-coordinates.
            color (tuple): Color of the line.
            thickness (int): Thickness of the line.
        """
        for i in range(len(y_values) - 1):
            pt1 = (int(x_values[i]), int(y_values[i]))
            pt2 = (int(x_values[i + 1]), int(y_values[i + 1]))
            cv2.line(image, pt1, pt2, color=color, thickness=thickness)

    def fit_polynomial(self, lane_points, frame_height, degree=2):
        """
        Fit a polynomial to the given lane points.

        Args:
            lane_points (list): List of (x, y) tuples for lane points.
            frame_height (int): Height of the frame.
            degree (int): Degree of the polynomial.

        Returns:
            ndarray: Polynomial coefficients.
        """
        lane_points = np.array(lane_points)
        x = lane_points[:, 0]
        y = lane_points[:, 1]
        y_flipped = frame_height - y
        coeffs = np.polyfit(y_flipped, x, degree)
        return coeffs

    def evaluate_polynomial(self, coeffs, y_values_flipped):
        """
        Evaluate a polynomial at given y-values.

        Args:
            coeffs (ndarray): Polynomial coefficients.
            y_values_flipped (ndarray): Flipped y-values (from bottom to top).

        Returns:
            ndarray: Evaluated x-values.
        """
        return np.polyval(coeffs, y_values_flipped)

    def timer_callback(self):
        """
        Timer callback to process frames from the video.
        """
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("Video processing complete.")
            self.cap.release()
            rclpy.shutdown()
            return

        processed_frame, lane_coordinates = self.process_frame(frame)

        # Convert processed frame to ROS Image message and publish
        ros_image = self.bridge.cv2_to_imgmsg(processed_frame, encoding='bgr8')
        self.publisher_frame.publish(ros_image)

        # Publish lane coordinates as a string message
        coords_msg = String()
        coords_msg.data = str(lane_coordinates)
        self.publisher_coords.publish(coords_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()

    # Use a timer to process video frames at regular intervals
    timer_period = 1 / 30  # Assuming 30 FPS
    node.create_timer(timer_period, node.timer_callback)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.cap.release()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
