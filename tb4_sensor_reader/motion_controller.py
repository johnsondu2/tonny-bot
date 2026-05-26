import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class MotionController(Node):

    def __init__(self):
        super().__init__('motion_controller')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer fires every 0.1 seconds (10 Hz control loop)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Track elapsed time
        self.elapsed = 0.0
        self.get_logger().info('Motion controller started')

    def control_loop(self):
        self.elapsed += 0.1
        msg = Twist()

        if self.elapsed < 3.0:
            # Drive forward for 3 seconds
            msg.linear.x = 0.2
            msg.angular.z = 0.0
        elif self.elapsed < 4.6:
            # Rotate for ~1.6 seconds (~90 degrees at 1.0 rad/s)
            msg.linear.x = 0.0
            msg.angular.z = 1.0
        elif self.elapsed < 7.6:
            # Drive forward again
            msg.linear.x = 0.2
            msg.angular.z = 0.0
        elif self.elapsed < 9.2:
            # Rotate again
            msg.linear.x = 0.0
            msg.angular.z = 1.0
        else:
            # Stop
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.get_logger().info('Sequence complete')

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MotionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
