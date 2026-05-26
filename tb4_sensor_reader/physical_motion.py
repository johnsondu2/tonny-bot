import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

NAMESPACE = '/T18'   # ← change to your robot namespace

class PhysicalMotion(Node):
    def __init__(self):
        super().__init__('physical_motion')
        self.publisher = self.create_publisher(
            Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.timer   = self.create_timer(0.1, self.control_loop)
        self.elapsed = 0.0
        self.get_logger().info('Physical motion node started')

    def control_loop(self):
        self.elapsed += 0.1
        msg = Twist()
        if self.elapsed < 3.0:
            msg.linear.x  = 0.15   # forward 3 s
            msg.angular.z = 0.0
        elif self.elapsed < 4.8:
            msg.linear.x  = 0.0    # rotate ~90 deg
            msg.angular.z = 0.87
        elif self.elapsed < 7.8:
            msg.linear.x  = 0.15   # forward again
            msg.angular.z = 0.0
        elif self.elapsed < 9.6:
            msg.linear.x  = 0.0    # rotate ~90 deg
            msg.angular.z = 0.87
        else:
            msg.linear.x  = 0.0    # stop
            msg.angular.z = 0.0
            self.get_logger().info('Sequence complete')
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PhysicalMotion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
