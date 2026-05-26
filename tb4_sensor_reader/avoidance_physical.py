import rclpy, math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

NAMESPACE      = '/T18'
FORWARD_SPEED  = 0.15
TURN_SPEED     = 0.8
AVOID_DISTANCE = 0.55
FRONT_ARC_DEG  = 60

class AvoidancePhysical(Node):
    def __init__(self):
        super().__init__('avoidance_physical')
        self.publisher = self.create_publisher(
            Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, 10)
        self.nearest_front = float('inf')
        self.nearest_left  = float('inf')
        self.nearest_right = float('inf')
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Avoidance controller started')

        self.current_x = 0.0
        self.current_y = 0.0
        self.create_subscription(
            Odometry,
            f'{NAMESPACE}/odom',
            self.odom_callback,
            10
        )

    def scan_callback(self, msg):
        inc    = msg.angle_increment
        arc_r  = math.radians(FRONT_ARC_DEG)
        side_r = math.radians(90)
        front_i = int(round(-msg.angle_min / inc))
        half_a  = int(round(arc_r  / inc))
        side_a  = int(round(side_r / inc))
        n = len(msg.ranges)

        def arc_min(lo, hi):
            lo = max(0, lo); hi = min(n - 1, hi)
            vals = [r for r in msg.ranges[lo:hi+1]
                    if msg.range_min < r < msg.range_max]
            return min(vals) if vals else float('inf')

        self.nearest_front = arc_min(front_i - half_a, front_i + half_a)
        self.nearest_left  = arc_min(front_i,          front_i + side_a)
        self.nearest_right = arc_min(front_i - side_a, front_i)

    # BUG 1 FIXED: odom_callback was nested inside scan_callback (wrong indentation)
    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        # BUG 2 FIXED: incomplete assignment — was just `m` instead of the y position
        self.current_y = msg.pose.pose.position.y

    def control_loop(self):
        msg = Twist()
        if self.nearest_front > AVOID_DISTANCE:
            msg.linear.x  = FORWARD_SPEED
            msg.angular.z = 0.0
            self.get_logger().info(
                f'Fwd | front={self.nearest_front:.2f} m')
        else:
            msg.linear.x = 0.0
            if self.nearest_left >= self.nearest_right:
                msg.angular.z =  TURN_SPEED
                self.get_logger().warn('Obstacle — turning LEFT')
            else:
                msg.angular.z = -TURN_SPEED
                self.get_logger().warn('Obstacle — turning RIGHT')
        self.publisher.publish(msg)
        # BUG 3 FIXED: f-string split across lines — must stay on one line
        self.get_logger().info(
            f'Fwd | front={self.nearest_front:.2f} m | pos=({self.current_x:.2f}, {self.current_y:.2f})'
        )

def main(args=None):
    rclpy.init(args=args)
    node = AvoidancePhysical()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
