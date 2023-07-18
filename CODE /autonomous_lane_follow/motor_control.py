import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray

class WheelSpeedController(Node):
    def __init__(self):
        super().__init__('wheel_speed_controller')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Int32MultiArray,'my_topic',self.callback,10)
        self.subscription

    def callback(self, msg):
        vel_msg = Twist()
        width = msg.data[0]
        height= msg.data[1] 
        xpos = msg.data[2]
        area = msg.data[4]

        if area > ((width*height)*0.5):
            vel_msg.linear.x = 0.0 
            vel_msg.angular.z = 0.0
        else:
            offset = xpos - (width/2)
            linear_vel = 0.1
            angular_vel = -0.001 * offset
            vel_msg.linear.x = linear_vel
            vel_msg.angular.z = angular_vel
        
        self.get_logger().info('Publishing: %.4f, %.4f, %.4f' % (vel_msg.linear.x, vel_msg.angular.z, msg.data[4]))
        self.publisher_.publish(vel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = WheelSpeedController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



