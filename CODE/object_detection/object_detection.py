import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int32MultiArray, Float32
import cv2
import numpy as np


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/color/image',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Int32MultiArray, 'my_topic', 10)
        self.bridge = CvBridge()

    def listener_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        height, width, channels = cv_image.shape

        hsv_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_frame, lower_red, upper_red)

        mask = cv2.bitwise_or(mask1, mask2)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area in descending order
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Only draw bounding box for largest contour
        if sorted_contours:
            largest_contour = sorted_contours[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx, cy = x + w/2, y + h/2
            message = f'Width: {width}, Height: {height}, Center X: {cx}, Center Y: {cy}'
            self.get_logger().info(message)

            # Draw bounding box around largest contour
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Publish height, width, and bounding box coordinates
            info_msg = Int32MultiArray()
            info_msg.data = [height, width, x, y, w*h]
            self.publisher_.publish(info_msg)

        # Display image
        cv2.imshow("camera", cv_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

