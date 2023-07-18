import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

class LaneFollower(Node):

    def __init__(self):
        super().__init__('lane_follower')
        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(Image,'/color/preview/image',self.image_callback,10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.model = keras.models.load_model('/home/ani/final_ws/my_model.h5')
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([-0.079766, 0.00000, 0.079766])  # Replace these values with the actual angular_z values used during training

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        preprocessed_image = self.preprocess_image(cv_image)
        angular_z_prob = self.model.predict(preprocessed_image)

        # Convert the predictions to label indices
        angular_z_indices = np.argmax(angular_z_prob, axis=1)

        # Convert the label indices back to original angular_z values
        angular_z = self.label_encoder.inverse_transform(angular_z_indices)

        # Get the first (and only) value in the array
        angular_z = angular_z.item(0)

        self.send_velocity_command(angular_z)

    def preprocess_image(self, image):
        new_width = 224
        new_height = 224
        preprocessed_image = cv2.resize(image, (new_width, new_height))
        preprocessed_image = preprocessed_image.astype('float32') / 255.0
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        return preprocessed_image

    def send_velocity_command(self, angular_z):
        twist = Twist()
        twist.linear.x = 0.03  # Adjust the linear velocity according to your preference
        twist.angular.z = angular_z
        self.cmd_vel_publisher.publish(twist)
        print("Published angular_z:", angular_z)

def main(args=None):
    rclpy.init(args=args)
    lane_follower = LaneFollower()
    rclpy.spin(lane_follower)
    lane_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
