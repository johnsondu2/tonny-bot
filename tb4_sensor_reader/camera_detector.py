import rclpy, cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data
import os
import threading

NAMESPACE = '/T21'
SAVE_DIR  = os.path.expanduser('~/Downloads/map')

RED_LOW1  = np.array([0,   120, 70])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS = 25000

class CameraDetector(Node):
    def __init__(self):
        super().__init__('camera_detector')
        topic = f'{NAMESPACE}/oakd/rgb/image_raw/compressed'
        self.create_subscription(
            CompressedImage, topic, self.image_callback, qos_profile_sensor_data)
        self.get_logger().info('Camera detector started — press ENTER in terminal to save')
        self.latest_frame = None
        self.save_counter = 0

    def image_callback(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2))
        pixels = cv2.countNonZero(mask)
        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]
        cv2.putText(overlay, f'Red pixels: {pixels}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if pixels >= MIN_PIXELS:
            cv2.putText(overlay, 'DETECTED', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        self.latest_frame = overlay
        cv2.imshow('Detection', overlay)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CameraDetector()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print('Press ENTER to save a snapshot. Ctrl+C to quit.')
    try:
        while rclpy.ok():
            input()
            if node.latest_frame is not None:
                node.save_counter += 1
                path = os.path.join(SAVE_DIR, f'snapshot_{node.save_counter}.jpg')
                cv2.imwrite(path, node.latest_frame)
                print(f'Saved {path}')
            else:
                print('No frame yet')
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()