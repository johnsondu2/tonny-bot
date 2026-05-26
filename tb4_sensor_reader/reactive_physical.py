import rclpy, math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

NAMESPACE     = '/T18'   # ← change to your robot namespace
FORWARD_SPEED = 0.15    # m/s — keep low on physical robot
STOP_DISTANCE = 0.3    # metres — larger than sim due to inertia
FRONT_ARC_DEG = 60      # degrees either side of forward to check

class ReactivePhysical(Node):
    def __init__(self):
        super().__init__('reactive_physical')
        self.publisher = self.create_publisher(
            Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, 10)
        self.nearest_front = float('inf')
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Reactive physical controller started')

    def scan_callback(self, msg):
        arc_rad = math.radians(FRONT_ARC_DEG)
        inc     = msg.angle_increment
        front_i = int(round(-msg.angle_min / inc))
        half_a  = int(round(arc_rad / inc))
        lo = max(0, front_i - half_a)
        hi = min(len(msg.ranges) - 1, front_i + half_a)
        arc = [r for r in msg.ranges[lo:hi+1]
               if msg.range_min < r < msg.range_max]
        self.nearest_front = min(arc) if arc else float('inf')

    def control_loop(self):
        msg = Twist()
        if self.nearest_front > STOP_DISTANCE:
            msg.linear.x = FORWARD_SPEED
            self.get_logger().info(
                f'Driving | nearest front: {self.nearest_front:.2f} m')
        else:
            msg.linear.x = 0.0
            self.get_logger().warn(
                f'STOPPED | obstacle at {self.nearest_front:.2f} m')
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ReactivePhysical()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
