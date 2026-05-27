import rclpy, math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

NAMESPACE = '/T18'   # <-- set to your robot

class LidarProbe(Node):
    def __init__(self):
        super().__init__('lidar_probe')
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan',
                                 self.cb, qos_profile_sensor_data)
        self.printed = False
        self.get_logger().info('Lidar probe started -- waiting for scan...')

    def cb(self, msg):
        if self.printed:
            return
        self.printed = True
        inc = msg.angle_increment
        n   = len(msg.ranges)

        def dist_at_deg(deg):
            # angle in rad, find nearest index, average a few valid points
            ang = math.radians(deg)
            i = int(round((ang - msg.angle_min) / inc))
            i = max(0, min(n - 1, i))
            vals = [r for r in msg.ranges[max(0,i-3):i+4]
                    if msg.range_min < r < msg.range_max]
            return (sum(vals)/len(vals)) if vals else float('inf')

        self.get_logger().info(f'angle_min={msg.angle_min:.3f} inc={inc:.5f} n={n}')
        self.get_logger().info('Distance at each angle (0 = code-forward, +deg = CCW/left):')
        for deg in range(-180, 181, 30):
            d = dist_at_deg(deg)
            bar = '#' * min(40, int(d * 10)) if d != float('inf') else '(inf)'
            self.get_logger().info(f'  {deg:+4d} deg : {d:5.2f} m  {bar}')
        self.get_logger().info('--- Look at which angle has the SMALLEST/LARGEST distance ---')
        self.get_logger().info('--- and compare to where things physically are. ---')

def main():
    rclpy.init()
    node = LidarProbe()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
