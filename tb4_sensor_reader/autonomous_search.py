import rclpy, cv2, math, csv, os, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data

NAMESPACE        = '/T29'
FORWARD_SPEED    = 0.15
TURN_SPEED       = 0.5
AVOID_DISTANCE   = 0.45
FRONT_ARC_DEG    = 60
FRONT_OFFSET_DEG = -90.0

WAYPOINT_RADIUS  = 0.1
HEADING_TOL      = 0.08
AVOID_FWD_SECS   = 1.0
WAYPOINTS_CSV    = os.path.expanduser('~/Downloads/map/path_waypoints.csv')

# Cube finding
CUBE_PIXEL_THRESHOLD = 2000
CUBE_STOP_PIXELS     = 30000
CUBE_TURN_SPEED      = 0.2
CUBE_FWD_SPEED       = 0.08
SWEEP_DEG            = 180.0

RED_LOW1  = np.array([0,   120, 70])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS = 500000


def load_waypoints(path):
    waypoints = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            waypoints.append((float(row['x_m']), float(row['y_m'])))
    return waypoints


class WaypointNav(Node):
    def __init__(self):
        super().__init__('waypoint_nav')
        self.pub = self.create_publisher(Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, f'{NAMESPACE}/odom',
            self.odom_callback, 10)

        # LiDAR
        self.nearest_front = float('inf')
        self.nearest_left  = float('inf')
        self.nearest_right = float('inf')

        # Odometry
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0

        # Camera
        self.cube_detected     = False
        self.latest_red_pixels = 0

        # Waypoints
        self.waypoints = load_waypoints(WAYPOINTS_CSV)
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints')
        self.wp_index     = 0
        self.search_phase = 'TURNING'

        # Avoidance
        self.avoid_turn_dir  = 1
        self.avoid_fwd_start = None

        # Odometry trail
        self.odom_trail = []

        # Cube finding sub-phases: ALIGN_NEG_Y → SWEEP → ALIGN_CUBE → APPROACH
        self.cf_phase        = 'ALIGN_NEG_Y'
        self.sweep_start_yaw = None
        self.sweep_readings  = []   # list of (yaw, red_pixels)
        self.cube_target_yaw = None
        self.sweep_last_yaw   = None   # yaw at previous tick, for delta accumulation
        self.swept_total      = 0.0    # total radians swept so far

        self.state = 'SEARCHING'
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Waypoint nav started')

    # ── Subscribers ───────────────────────────────────────────────────────────

    def scan_callback(self, msg):
        inc      = msg.angle_increment
        n        = len(msg.ranges)
        offset_i = int(round(math.radians(FRONT_OFFSET_DEG) / inc))
        front_i  = int(round(-msg.angle_min / inc)) + offset_i
        half_a   = int(round(math.radians(FRONT_ARC_DEG / 2) / inc))
        side_a   = int(round(math.radians(90) / inc))

        def arc_min(lo, hi):
            indices = [i % n for i in range(lo, hi + 1)]
            vals = [msg.ranges[i] for i in indices
                    if math.isfinite(msg.ranges[i])
                    and msg.range_min < msg.ranges[i] < msg.range_max]
            return min(vals) if vals else float('inf')

        self.nearest_front = arc_min(front_i - half_a, front_i + half_a)
        self.nearest_left  = arc_min(front_i,          front_i + side_a)
        self.nearest_right = arc_min(front_i - side_a, front_i)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)
        self.odom_trail.append((self.current_x, self.current_y))

    def image_callback(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2))
        self.latest_red_pixels = cv2.countNonZero(mask)
        if self.state not in ('DETECTED', 'DONE', 'CUBE_FINDING'):
            if self.latest_red_pixels >= MIN_PIXELS:
                self.cube_detected = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def stop(self):
        self.pub.publish(Twist())

    def _publish_twist(self, lin, ang):
        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self.pub.publish(cmd)

    def _distance_to_wp(self):
        wx, wy = self.waypoints[self.wp_index]
        return math.sqrt((self.current_x - wx)**2 + (self.current_y - wy)**2)

    def _heading_error(self):
        wx, wy = self.waypoints[self.wp_index]
        dx = wx - self.current_x
        dy = wy - self.current_y
        target = math.atan2(dy, dx)
        err    = target - self.current_yaw
        return math.atan2(math.sin(err), math.cos(err))

    # ── Control loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if self.state == 'DONE':
            return

        # ── SEARCHING ─────────────────────────────────────────────────────
        if self.state == 'SEARCHING':

            if self.cube_detected:
                self.state = 'DETECTED'
                self.stop()
                self.get_logger().info('RED CUBE DETECTED')
                self.get_logger().info(
                    f'Position: x={self.current_x:.3f} y={self.current_y:.3f}')
                return

            if self.nearest_front < AVOID_DISTANCE:
                self.avoid_turn_dir = 1 if self.nearest_left >= self.nearest_right else -1
                self.get_logger().info(
                    f'Obstacle at {self.nearest_front:.2f}m — '
                    f'turning {"LEFT" if self.avoid_turn_dir > 0 else "RIGHT"}')
                self.state = 'AVOIDING'
                return

            if self.search_phase == 'TURNING':
                err = self._heading_error()
                self.get_logger().info(
                    f'[TURNING] wp={self.wp_index+1}/{len(self.waypoints)} '
                    f'err={math.degrees(err):.1f}°')
                if abs(err) < HEADING_TOL:
                    self.stop()
                    self.search_phase = 'DRIVING'
                    self.get_logger().info(f'Aligned — driving to waypoint {self.wp_index+1}')
                else:
                    self._publish_twist(0.0, TURN_SPEED if err > 0 else -TURN_SPEED)

            elif self.search_phase == 'DRIVING':
                dist = self._distance_to_wp()
                self.get_logger().info(
                    f'[DRIVING] wp={self.wp_index+1}/{len(self.waypoints)} dist={dist:.3f}m')
                if dist < WAYPOINT_RADIUS:
                    self.stop()
                    self.wp_index += 1
                    if self.wp_index >= len(self.waypoints):
                        self.state = 'CUBE_FINDING'
                        self.cf_phase = 'ALIGN_NEG_Y'
                        self.get_logger().info('All waypoints reached — CUBE_FINDING')
                    else:
                        self.search_phase = 'TURNING'
                        self.get_logger().info(
                            f'Waypoint reached — turning to waypoint {self.wp_index+1}')
                else:
                    self._publish_twist(FORWARD_SPEED, 0.0)

        # ── AVOIDING ──────────────────────────────────────────────────────
        elif self.state == 'AVOIDING':
            if self.nearest_front >= AVOID_DISTANCE:
                self.avoid_fwd_start = time.time()
                self.state = 'AVOID_FWD'
                self.get_logger().info('Arc clear — forward burst')
            else:
                self._publish_twist(0.0, TURN_SPEED * self.avoid_turn_dir)

        # ── AVOID_FWD ─────────────────────────────────────────────────────
        elif self.state == 'AVOID_FWD':
            if time.time() - self.avoid_fwd_start >= AVOID_FWD_SECS:
                self.stop()
                if self.wp_index < len(self.waypoints) - 1:
                    self.wp_index += 1
                self.search_phase = 'TURNING'
                self.state = 'SEARCHING'
                self.get_logger().info(
                    f'Burst done — re-aligning to waypoint {self.wp_index+1}')
            else:
                self._publish_twist(FORWARD_SPEED, 0.0)

        # ── CUBE_FINDING ──────────────────────────────────────────────────
        elif self.state == 'CUBE_FINDING':

            # Phase 1: turn to face -Y direction (yaw = -pi/2)
            if self.cf_phase == 'ALIGN_NEG_Y':
                target_yaw = -math.pi / 2.0
                err = target_yaw - self.current_yaw
                err = math.atan2(math.sin(err), math.cos(err))
                if abs(err) < HEADING_TOL:
                    self.stop()
                    self.sweep_start_yaw = self.current_yaw
                    self.sweep_last_yaw  = self.current_yaw
                    self.swept_total     = 0.0
                    self.sweep_readings  = []
                    self.cf_phase = 'SWEEP'
                    self.get_logger().info('Aligned to -Y — starting 180° sweep')
                else:
                    self._publish_twist(0.0, CUBE_TURN_SPEED if err > 0 else -CUBE_TURN_SPEED)

            # Phase 2: sweep 180° CW, recording (yaw, red_pixels) each tick
            elif self.cf_phase == 'SWEEP':
                # Accumulate small CW deltas to avoid wraparound jump
                delta = self.sweep_last_yaw - self.current_yaw
                if delta > math.pi:   # wrapped CCW — subtract full circle
                    delta -= 2 * math.pi
                elif delta < -math.pi:  # wrapped CW
                    delta += 2 * math.pi
                self.swept_total    += abs(delta)
                self.sweep_last_yaw  = self.current_yaw
                self.sweep_readings.append((self.current_yaw, self.latest_red_pixels))
                self.get_logger().info(
                    f'[SWEEP] swept={math.degrees(self.swept_total):.1f}° '
                    f'red_px={self.latest_red_pixels}')
                if self.swept_total >= math.radians(SWEEP_DEG):
                    self.stop()
                    above = [(yaw, px) for yaw, px in self.sweep_readings
                             if px >= CUBE_PIXEL_THRESHOLD]
                    if above:
                        mid_idx = len(above) // 2
                        self.cube_target_yaw = above[mid_idx][0]
                        self.get_logger().info(
                            f'Sweep done — cube at yaw={math.degrees(self.cube_target_yaw):.1f}°')
                        self.cf_phase = 'ALIGN_CUBE'
                    else:
                        self.get_logger().warn('No cube detected in sweep — DONE')
                        self.state = 'DONE'
                else:
                    self._publish_twist(0.0, -CUBE_TURN_SPEED)  # CW = negative

            # Phase 3: turn to face cube heading
            elif self.cf_phase == 'ALIGN_CUBE':
                err = self.cube_target_yaw - self.current_yaw
                err = math.atan2(math.sin(err), math.cos(err))
                if abs(err) < HEADING_TOL:
                    self.stop()
                    self.cf_phase = 'APPROACH'
                    self.get_logger().info('Facing cube — approaching')
                else:
                    self._publish_twist(0.0, CUBE_TURN_SPEED if err > 0 else -CUBE_TURN_SPEED)

            # Phase 4: drive forward until red pixels >= CUBE_STOP_PIXELS
            elif self.cf_phase == 'APPROACH':
                self.get_logger().info(f'[APPROACH] red_px={self.latest_red_pixels}')
                if self.latest_red_pixels >= CUBE_STOP_PIXELS:
                    self.stop()
                    self.get_logger().info('Reached cube — DONE')
                    self.state = 'DONE'
                else:
                    self._publish_twist(CUBE_FWD_SPEED, 0.0)

        # ── DETECTED ──────────────────────────────────────────────────────
        elif self.state == 'DETECTED':
            self.stop()
            self.state = 'DONE'

        if self.state == 'DONE':
            self.stop()
            self.get_logger().info('=== DONE ===')


def save_odom_map(trail, map_dir):
    pgm_path  = os.path.join(map_dir, 'map.pgm')
    yaml_path = os.path.join(map_dir, 'map.yaml')
    out_path  = os.path.join(map_dir, 'odom_overlay.png')

    if not os.path.exists(pgm_path) or not os.path.exists(yaml_path):
        print(f'Map files not found in {map_dir} — skipping odom overlay')
        return

    import yaml
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution = meta['resolution']
    origin     = meta['origin']

    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    scale = 8
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    def world_to_px(wx, wy):
        col = int(round((wx - origin[0]) / resolution))
        row = h - int(round((wy - origin[1]) / resolution))
        return col * scale, row * scale

    if len(trail) > 1:
        pts = [world_to_px(x, y) for x, y in trail]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i+1], (0, 200, 255), 2)

    if trail:
        sx, sy = world_to_px(*trail[0])
        ex, ey = world_to_px(*trail[-1])
        cv2.circle(vis, (sx, sy), 7, (60, 60, 255), -1)
        cv2.putText(vis, 'start', (sx+6, sy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1)
        cv2.circle(vis, (ex, ey), 7, (60, 220, 60), -1)
        cv2.putText(vis, 'end', (ex+6, ey-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1)

    cv2.imwrite(out_path, vis)
    print(f'Odom overlay saved to {out_path}  ({len(trail)} points)')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNav()
    try:
        rclpy.spin(node)
    finally:
        save_odom_map(node.odom_trail, os.path.expanduser('~/Downloads/map'))
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()