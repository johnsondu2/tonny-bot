import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

NAMESPACE = '/T29'   # ← change to your robot namespace

class PoseReader(Node):
    def __init__(self):
        super().__init__('pose_reader')
        self.current_x = 0.0
        self.current_y = 0.0
        self.create_subscription(
            Odometry,
            f'{NAMESPACE}/odom',
            self.odom_callback,
            10
        )
        self.timer = self.create_timer(1.0, self.report_pose)
        self.get_logger().info('Pose reader started — waiting for odometry...')

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def report_pose(self):
        self.get_logger().info(
            f'Odometry position: x={self.current_x:.3f} m  y={self.current_y:.3f} m'
        )

def main(args=None):
    rclpy.init(args=args)
    node = PoseReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
