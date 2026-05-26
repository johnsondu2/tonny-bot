#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

NAMESPACE = '/T23'


class LidarB2Node(Node):
    def __init__(self):
        super().__init__('lidar_b2_node')

        self.scan_sub = self.create_subscription(
            LaserScan,
            f'{NAMESPACE}/scan',
            self.scan_callback,
            10
        )

        self.get_logger().info(f'B2 LiDAR node started | Subscribed to {NAMESPACE}/scan')
        self.get_logger().info('Printing one line per scan. Press Ctrl+C to stop.')

    def scan_callback(self, msg: LaserScan):
        total_beams = len(msg.ranges)

        valid_ranges = [
            r for r in msg.ranges
            if msg.range_min <= r <= msg.range_max
        ]

        min_valid_range = min(valid_ranges) if valid_ranges else float('inf')
        max_valid_range = max(valid_ranges) if valid_ranges else float('-inf')

        forward_index = 270
        forward_range_raw = msg.ranges[forward_index]

        if msg.range_min <= forward_range_raw <= msg.range_max:
            forward_text = f'{forward_range_raw:.4f} m'
        else:
            forward_text = 'invalid'

        min_text = f'{min_valid_range:.4f} m' if valid_ranges else 'no valid ranges'
        max_text = f'{max_valid_range:.4f} m' if valid_ranges else 'no valid ranges'

        print(
            f'beams={total_beams} | '
            f'min_valid={min_text} | '
            f'max_valid={max_text} | '
            f'forward_index={forward_index} | '
            f'forward_range={forward_text}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarB2Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
