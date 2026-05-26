import rclpy
import math                                    # NEW
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class OdomReader(Node):

    def __init__(self):
        super().__init__('odom_reader')
        self.current_x = 0.0
        self.current_y = 0.0

        # Odometry subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        # LiDAR subscriber
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.get_logger().info('Sensor reader started')

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):              # CHANGED - now finds angle too
        valid = [(r, i) for i, r in enumerate(msg.ranges) if r > 0.01]  # CHANGED
        if valid:
            min_range, min_index = min(valid, key=lambda x: x[0])  # CHANGED
            angle_rad = msg.angle_min + min_index * msg.angle_increment  # NEW
            angle_deg = math.degrees(angle_rad)                    # NEW
            if min_range < 0.5:
                self.get_logger().warn(
                    f'Obstacle at {min_range:.2f} m | '
                    f'{angle_deg:.1f} deg | '                      # NEW
                    f'Position x: {self.current_x:.3f}  '
                    f'y: {self.current_y:.3f}')
            else:
                self.get_logger().info(        # CHANGED - now includes angle
                    f'Nearest: {min_range:.2f} m at {angle_deg:.1f} degrees')


def main(args=None):
    rclpy.init(args=args)
    node = OdomReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
