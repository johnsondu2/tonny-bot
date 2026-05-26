import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

FORWARD_SPEED  = 0.2   # m/s
TURN_SPEED     = 0.8   # rad/s
AVOID_DISTANCE = 0.6   # metres — start turning if obstacle closer than this


class AvoidController(Node):

    def __init__(self):
        super().__init__('avoid_controller')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.nearest_obstacle = float('inf')
        self.obstacle_angle   = 0.0

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Avoidance controller started')

    def scan_callback(self, msg):
        valid = [
            (r, i) for i, r in enumerate(msg.ranges)
            if r > msg.range_min
        ]
        if valid:
            min_range, min_idx = min(valid, key=lambda x: x[0])
            self.nearest_obstacle = min_range
            self.obstacle_angle = (
                msg.angle_min + min_idx * msg.angle_increment
            )
        else:
            self.nearest_obstacle = float('inf')

    def control_loop(self):
        msg = Twist()

        if self.nearest_obstacle > AVOID_DISTANCE:
            # Clear path — drive forward
            msg.linear.x = FORWARD_SPEED
            msg.angular.z = 0.0
        else:
            # Obstacle too close — turn away from it
            msg.linear.x = 0.0
            # If obstacle is on the left (positive angle), turn right
            # If obstacle is on the right (negative angle), turn left
            if self.obstacle_angle >= 0:
                msg.angular.z = -TURN_SPEED  # turn right
            else:
                msg.angular.z =  TURN_SPEED  # turn left
            self.get_logger().info(
                f'Avoiding | obstacle: {self.nearest_obstacle:.2f} m '
                f'at {self.obstacle_angle:.2f} rad')

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AvoidController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
