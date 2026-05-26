import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

FORWARD_SPEED = 0.2    # m/s
STOP_DISTANCE = 0.5    # metres — stop if obstacle closer than this


class ReactiveController(Node):

    def __init__(self):
        super().__init__('reactive_controller')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        # Stores the nearest obstacle distance
        self.nearest_obstacle = float('inf')

        # Control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Reactive controller started')

    def scan_callback(self, msg):
        # Update nearest obstacle from LiDAR
        valid = [r for r in msg.ranges if r > msg.range_min]
        self.nearest_obstacle = min(valid) if valid else float('inf')

    def control_loop(self):
        msg = Twist()

        if self.nearest_obstacle > STOP_DISTANCE:
            # Path is clear — drive forward
            msg.linear.x = FORWARD_SPEED
            msg.angular.z = 0.0
            self.get_logger().info(
                f'Driving | nearest: {self.nearest_obstacle:.2f} m')
        else:
            # Obstacle detected — stop
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.get_logger().warn(
                f'STOPPED | obstacle at {self.nearest_obstacle:.2f} m')

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
