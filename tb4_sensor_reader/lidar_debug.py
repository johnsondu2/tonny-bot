import rclpy, math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

NAMESPACE       = '/T29'
FRONT_ARC_DEG   = 60
FRONT_OFFSET_DEG = -90.0  # adjust until arc points out physical front

class LidarDebug(Node):
    def __init__(self):
        super().__init__('lidar_debug')
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, qos_profile_sensor_data)
        self.get_logger().info('Lidar debug node started')

    def scan_callback(self, msg):
        inc     = msg.angle_increment
        n       = len(msg.ranges)
        offset_i = int(round(math.radians(FRONT_OFFSET_DEG) / inc))
        front_i  = int(round(-msg.angle_min / inc)) + offset_i
        half_a   = int(round(math.radians(FRONT_ARC_DEG / 2) / inc))

        lo = front_i - half_a
        hi = front_i + half_a

        vals = []
        for i in range(lo, hi + 1):
            r = msg.ranges[i % n]
            angle_deg = math.degrees(msg.angle_min + (i % n) * inc)
            if math.isfinite(r) and msg.range_min < r < msg.range_max:
                vals.append((angle_deg, r))

        if not vals:
            self.get_logger().info('No valid readings in front arc')
            return

        min_r = min(vals, key=lambda x: x[1])
        avg_r = sum(r for _, r in vals) / len(vals)

        self.get_logger().info(
            f'closest={min_r[1]:.3f}m at {min_r[0]:.1f}deg  '
            f'avg={avg_r:.3f}m  rays={len(vals)}  '
            f'front_i={front_i}  offset={FRONT_OFFSET_DEG}deg')

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LidarDebug())

if __name__ == '__main__':
    main()