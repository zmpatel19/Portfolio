import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import String


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('subscriber')
        self.subscription = self.create_subscription(Image, '/mycamerahk', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(String, '/blobs_topic', 10) 
        self.pub=self.create_publisher(Image,'/processedvideo',10)
        self.pubo=self.create_publisher(Image,'/originalvideo',10)
        self.br = CvBridge()
        

    def listener_callback(self, data):

        self.get_logger().info('Receiving video frame')
        current_frame = self.br.imgmsg_to_cv2(data)
        original = current_frame
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # range of red blue and green sugested google
        red_lower = np.array([0, 70, 70])
        red_upper = np.array([10, 255, 255])
        lower_red = np.array([170,70,70])
        upper_red = np.array([180,255,255])
        skin_lower = np.array([0, 10, 60])
        skin_upper = np.array([20, 150, 255])
        green_lower = np.array([36, 25, 25])
        green_upper = np.array([86, 255, 255])
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])

        # Thresholding
        red_mask1 = cv2.inRange(hsv_frame, red_lower, red_upper)
        red_mask2 = cv2.inRange(hsv_frame, lower_red, upper_red)
        red_mask3 = cv2.bitwise_or(red_mask1, red_mask2)
        skin_mask = cv2.inRange(hsv_frame, skin_lower, skin_upper)
        red_mask = cv2.bitwise_and(red_mask3, cv2.bitwise_not(skin_mask))
        green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)

        # Find contours 
        min_size = 300
                
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        red_contours = [c for c in red_contours if cv2.contourArea(c) > min_size]
        
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        green_contours = [c for c in green_contours if cv2.contourArea(c) > min_size]
        
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours = [c for c in blue_contours if cv2.contourArea(c) > min_size]

        # Sizes and locations 
        red_blobs = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in red_contours]
        green_blobs = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in green_contours]
        blue_blobs = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in blue_contours]

        # Sort the blobs 
        red_blobs.sort(key=lambda x: x[0], reverse=True)
        green_blobs.sort(key=lambda x: x[0], reverse=True)
        blue_blobs.sort(key=lambda x: x[0], reverse=True)
        
        info_msg = String()
        info_msg.data = f'red_blobs{red_blobs}, green_blobs{green_blobs}, blue_blobs{blue_blobs}'
        self.publisher_.publish(info_msg)
         

        # Publish the list of blobs 
        print("Red Blobs:")
        for blob in red_blobs:
            print(f"Red Blobs: Size: {blob[0]}, Center: ({blob[1][0] + blob[1][2]/2}, {blob[1][1] + blob[1][3]/2})")
            
        
        print("Green Blobs:")
        for blob in green_blobs:
            print(f"Green Blobs: Size: {blob[0]}, Center: ({blob[1][0] + blob[1][2]/2}, {blob[1][1] + blob[1][3]/2})")
        
        print("Blue Blobs:")
        for blob in blue_blobs:
            print(f"Blue Blobs: Size: {blob[0]}, Center: ({blob[1][0] + blob[1][2]/2}, {blob[1][1] + blob[1][3]/2})")           

        for c in red_contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(current_frame, "Red", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for c in green_contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(current_frame, "Green", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for c in blue_contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(current_frame, "Blue", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        
        cv2.imshow("camera", current_frame)
        newmsg=self.br.cv2_to_imgmsg(current_frame)
        self.pub.publish(newmsg)
        oldmsg=self.br.cv2_to_imgmsg(original)
        self.pubo.publish(oldmsg)
        cv2.waitKey(1)

def main(args=None):

    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
