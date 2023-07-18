import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import os
import csv
import time

class DataCollectionNode(Node):

    def __init__(self):
        super().__init__('data_collection_node')
        self.image_subscriber = self.create_subscription(Image, '/color/preview/image', self.image_callback, 10)
        self.cmd_vel_subscriber = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.current_cmd_vel = Twist()
        self.data_dir = 'data'  # Specify the directory to store the data
        self.csv_file = os.path.join(self.data_dir, 'data.csv')

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        with open(self.csv_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['timestamp', 'image_path', 'linear_x', 'angular_z'])

    def cmd_vel_callback(self, cmd_vel_msg):
        self.current_cmd_vel = cmd_vel_msg

    def image_callback(self, img_msg):
        img = np.array(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        timestamp = time.time()
        image_path = os.path.join(self.data_dir, f"{timestamp:.6f}.png")
        cv2.imwrite(image_path, img)

        with open(self.csv_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([timestamp, image_path, self.current_cmd_vel.linear.x, self.current_cmd_vel.angular.z])

def main(args=None):
    rclpy.init(args=args)
    data_collection_node = DataCollectionNode()
    rclpy.spin(data_collection_node)
    data_collection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

