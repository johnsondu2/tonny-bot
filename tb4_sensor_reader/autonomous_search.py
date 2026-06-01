# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import rclpy, cv2, math, csv, os
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist          # Used to send velocity commands to the robot
from sensor_msgs.msg import LaserScan, CompressedImage  # LiDAR and camera data
from nav_msgs.msg import Odometry            # Robot position/orientation data
from rclpy.qos import qos_profile_sensor_data  # QoS preset suited for high-rate sensor topics

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NAMESPACE        = '/T29'    # Robot's ROS 2 namespace — all topic names are prefixed with this
FORWARD_SPEED    = 0.15      # Linear speed in m/s when driving forward
TURN_SPEED       = 0.5       # Angular speed in rad/s for turns
AVOID_DISTANCE   = 0.45      # Stop and turn if an obstacle is within this many metres (front arc)
FRONT_ARC_DEG    = 60        # Width of the "front danger zone" arc in degrees
FRONT_OFFSET_DEG = -90.0     # Rotational offset to align the LiDAR's front index with the robot's true forward

WAYPOINT_RADIUS  = 0.1       # Distance (m) at which a waypoint is considered "reached"
HEADING_TOL      = 0.08      # Heading error in radians considered "close enough" to the target angle
WAYPOINTS_CSV    = os.path.expanduser('~/Downloads/map/path_waypoints.csv')  # File with the planned search path

# ── Robust avoidance constants (replaces the old fixed AVOID_FWD_SECS) ───────
# Rather than driving for a fixed time, the robot uses two conditions to decide
# when it has genuinely cleared an obstacle:
#
#   Condition 1 — Side arc clears:
#     While passing the obstacle, the LiDAR arc on the obstacle's side will
#     read a short distance. Once the robot passes the edge, that arc opens up
#     past AVOID_SIDE_CLEAR_M. This is the primary "done" signal.
#
#   Condition 2 — Geometric minimum distance:
#     When the obstacle is first detected, the front arc gives its distance.
#     The robot must travel at least that distance plus a fixed buffer before
#     Condition 1 is even checked. This stops the robot from declaring "clear"
#     too early if it happens to be approaching from an angle.
#
#   Condition 3 — Hard cap (fallback):
#     If the side arc never clears (e.g. the "obstacle" is actually a wall that
#     runs the full length of the corridor), the robot gives up after
#     AVOID_MAX_FWD_M metres and re-plans to the next waypoint anyway.
AVOID_SIDE_CLEAR_M = 0.70   # Side arc distance (m) that signals the obstacle edge has been passed
AVOID_GEO_BUFFER_M = 0.25   # Extra metres added to the obstacle distance to form the geometric minimum
AVOID_MAX_FWD_M    = 2.50   # Hard fallback: give up and re-plan after driving this far

# ── Cube detection thresholds ────────────────────────────────────────────────
CUBE_PIXEL_THRESHOLD = 2000   # Minimum red pixels during the sweep to consider the cube "seen" at that angle
CUBE_STOP_PIXELS     = 30000  # During final approach, stop when red pixel count reaches this (cube is close)
CUBE_TURN_SPEED      = 0.2    # Slower turn speed used during the cube-finding sweep and alignment
CUBE_FWD_SPEED       = 0.08   # Slow forward speed during the final approach to the cube
SWEEP_DEG            = 180.0  # Total sweep angle in degrees when scanning for the cube

# ── HSV colour ranges for red detection ─────────────────────────────────────
# Red wraps around the HSV hue axis, so two ranges are needed:
#   Range 1 covers hue 0–10  (red on the low end of the scale)
#   Range 2 covers hue 170–180 (red on the high end of the scale)
RED_LOW1  = np.array([0,   120, 70])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS = 500000  # Pixel count that triggers an immediate cube detection (used in SEARCHING state)


# ─────────────────────────────────────────────────────────────────────────────
# WAYPOINT LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_waypoints(path):
    """
    Reads a CSV file containing the search path waypoints.
    Each row must have 'x_m' and 'y_m' columns (real-world metres).
    Returns a list of (x, y) tuples in order.
    """
    waypoints = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            waypoints.append((float(row['x_m']), float(row['y_m'])))
    return waypoints


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NODE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class WaypointNav(Node):
    def __init__(self):
        super().__init__('waypoint_nav')

        # ── Publishers ────────────────────────────────────────────────────────
        # cmd_vel is the velocity command topic — publishing Twist messages here drives the robot
        self.pub = self.create_publisher(Twist, f'{NAMESPACE}/cmd_vel', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        # LiDAR scan — uses sensor QoS profile to tolerate dropped packets at high rates
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, qos_profile_sensor_data)

        # Compressed camera image — also high rate, so sensor QoS is used
        self.create_subscription(CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback, qos_profile_sensor_data)

        # Odometry — robot's estimated position and orientation from wheel encoders
        self.create_subscription(Odometry, f'{NAMESPACE}/odom',
            self.odom_callback, 10)

        # ── Shared state: LiDAR ───────────────────────────────────────────────
        # These are updated every time a LiDAR scan arrives (scan_callback)
        self.nearest_front = float('inf')  # Closest obstacle directly ahead
        self.nearest_left  = float('inf')  # Closest obstacle to the left (used for turn decisions)
        self.nearest_right = float('inf')  # Closest obstacle to the right

        # ── Shared state: Odometry ────────────────────────────────────────────
        # Updated every time an odometry message arrives (odom_callback)
        self.current_x   = 0.0   # Robot's X position in metres from the origin
        self.current_y   = 0.0   # Robot's Y position in metres from the origin
        self.current_yaw = 0.0   # Robot's heading in radians (0 = facing +X axis)

        # ── Shared state: Camera ─────────────────────────────────────────────
        self.cube_detected     = False  # Set to True once the cube is seen with enough pixels
        self.latest_red_pixels = 0      # Number of red pixels in the most recent camera frame

        # ── Waypoint navigation state ─────────────────────────────────────────
        self.waypoints = load_waypoints(WAYPOINTS_CSV)
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints')
        self.wp_index     = 0          # Index of the current target waypoint
        self.search_phase = 'TURNING'  # Within SEARCHING: either 'TURNING' (aligning) or 'DRIVING' (moving)

        # ── Obstacle avoidance state ──────────────────────────────────────────
        self.avoid_turn_dir      = 1       # +1 = turn left, -1 = turn right (chosen based on which side is more open)
        self.avoid_track_side    = 'right' # Which LiDAR arc to watch during AVOID_PASS ('left' or 'right')
        self.avoid_geo_min_m     = 0.0     # Geometric minimum distance the robot must travel before checking side clear
        self.avoid_pass_start_x  = 0.0     # Robot X when AVOID_PASS began (used to measure distance travelled)
        self.avoid_pass_start_y  = 0.0     # Robot Y when AVOID_PASS began

        # ── Odometry trail (for map overlay at the end) ───────────────────────
        self.odom_trail = []  # List of (x, y) tuples recorded throughout the run

        # ── Cube-finding sub-state machine ────────────────────────────────────
        # After reaching all waypoints, the robot transitions to CUBE_FINDING.
        # This has four sequential phases:
        #   ALIGN_NEG_Y  → rotate to face the -Y direction (toward the far end of the arena)
        #   SWEEP        → slowly rotate 180° CW, recording red pixel counts at each angle
        #   ALIGN_CUBE   → rotate back to the angle where red pixels were highest
        #   APPROACH     → drive slowly forward until the cube fills enough of the frame
        self.cf_phase        = 'ALIGN_NEG_Y'
        self.sweep_start_yaw = None    # Yaw at the moment the sweep began
        self.sweep_readings  = []      # List of (yaw, red_pixels) samples collected during SWEEP
        self.cube_target_yaw = None    # The yaw angle the robot will face after identifying the cube direction
        self.sweep_last_yaw  = None    # Yaw from the previous tick, used to compute incremental rotation delta
        self.swept_total     = 0.0     # Total radians rotated during the sweep (accumulated incrementally)

        # ── Top-level state machine ───────────────────────────────────────────
        # SEARCHING  → following waypoints, watching for the cube
        # AVOIDING   → an obstacle appeared; turning until clear
        # AVOID_FWD  → driving forward briefly after clearing the obstacle
        # CUBE_FINDING → all waypoints done; running the sweep-and-approach sequence
        # DETECTED   → cube seen during SEARCHING; stop immediately
        # DONE       → everything finished; hold position
        self.state = 'SEARCHING'

        # Control loop fires every 0.1 seconds (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Waypoint nav started')

    # ─────────────────────────────────────────────────────────────────────────
    # SUBSCRIBER CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────

    def scan_callback(self, msg):
        """
        Called every time a LiDAR scan arrives.
        Computes the minimum distance in three arcs:
          - front arc (obstacle ahead)
          - left arc (used to decide which way to turn)
          - right arc
        The FRONT_OFFSET_DEG constant adjusts for the fact that index 0 in the
        scan array may not point directly forward on this robot's LiDAR mount.
        """
        inc      = msg.angle_increment                          # Radians between consecutive scan rays
        n        = len(msg.ranges)                              # Total number of rays in the scan
        offset_i = int(round(math.radians(FRONT_OFFSET_DEG) / inc))  # Shift index to correct LiDAR mounting offset
        front_i  = int(round(-msg.angle_min / inc)) + offset_i       # Index of the ray pointing directly forward
        half_a   = int(round(math.radians(FRONT_ARC_DEG / 2) / inc)) # Half-width of the front arc in indices
        side_a   = int(round(math.radians(90) / inc))                 # 90° in indices, used for left/right arcs

        def arc_min(lo, hi):
            """Return the minimum valid range reading between index lo and hi (wraps around using modulo)."""
            indices = [i % n for i in range(lo, hi + 1)]
            vals = [msg.ranges[i] for i in indices
                    if math.isfinite(msg.ranges[i])          # Exclude inf/NaN values
                    and msg.range_min < msg.ranges[i] < msg.range_max]  # Exclude out-of-range readings
            return min(vals) if vals else float('inf')       # Return inf if no valid readings

        self.nearest_front = arc_min(front_i - half_a, front_i + half_a)  # Centre arc
        self.nearest_left  = arc_min(front_i,          front_i + side_a)  # Left 90° arc
        self.nearest_right = arc_min(front_i - side_a, front_i)           # Right 90° arc

    def odom_callback(self, msg):
        """
        Called every time an odometry message arrives.
        Extracts the robot's (x, y) position and yaw (heading) from the message.
        Yaw is derived from the quaternion orientation using the standard formula.
        Also appends the current position to odom_trail for the final map overlay.
        """
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Convert quaternion to yaw (rotation around the vertical Z axis)
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)  # Result in radians: -π to +π

        self.odom_trail.append((self.current_x, self.current_y))

    def image_callback(self, msg):
        """
        Called every time a compressed camera frame arrives.
        Decodes the JPEG image, converts to HSV, and counts red pixels using two masks
        (because red wraps around the hue axis in HSV).
        If the pixel count exceeds MIN_PIXELS while SEARCHING, flags cube_detected = True,
        which the control loop will act on at the next 10 Hz tick.
        Does NOT process images once the cube has been found or the run is over.
        """
        # Decode compressed JPEG bytes into an OpenCV BGR image
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return  # Skip if decoding failed (e.g., corrupted frame)

        # Convert BGR → HSV for colour thresholding
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create a binary mask where red pixels are white (255), everything else is black (0)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),  # Low-end red (hue 0–10)
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2))  # High-end red (hue 170–180)

        self.latest_red_pixels = cv2.countNonZero(mask)  # Count white pixels in the mask

        # Only trigger a detection flag during SEARCHING (not during cube approach or after done)
        if self.state not in ('DETECTED', 'DONE', 'CUBE_FINDING'):
            if self.latest_red_pixels >= MIN_PIXELS:
                self.cube_detected = True  # Control loop will handle this next tick

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def stop(self):
        """Publish a zero-velocity Twist to halt the robot immediately."""
        self.pub.publish(Twist())  # Default Twist has all fields = 0

    def _publish_twist(self, lin, ang):
        """Convenience wrapper: publish a Twist with given linear and angular velocities."""
        cmd = Twist()
        cmd.linear.x  = lin  # Forward/backward speed (m/s)
        cmd.angular.z = ang  # Rotation speed (rad/s); positive = counterclockwise
        self.pub.publish(cmd)

    def _distance_to_wp(self):
        """Return the straight-line distance (metres) from the robot's current position to the active waypoint."""
        wx, wy = self.waypoints[self.wp_index]
        return math.sqrt((self.current_x - wx)**2 + (self.current_y - wy)**2)

    def _heading_error(self):
        """
        Return the signed angular error (radians) between the robot's current yaw
        and the direction it needs to face to head toward the active waypoint.
        Positive = need to turn left (counterclockwise), negative = turn right.
        Wraps to [-π, +π] so the robot always takes the shortest turn.
        """
        wx, wy = self.waypoints[self.wp_index]
        dx = wx - self.current_x
        dy = wy - self.current_y
        target = math.atan2(dy, dx)          # Desired heading angle
        err    = target - self.current_yaw   # Raw error
        return math.atan2(math.sin(err), math.cos(err))  # Wrap to [-π, +π]

    def _advance_waypoint(self):
        """
        Called after the robot has finished passing an obstacle.
        Tries to resume the CURRENT waypoint first — the robot may now have a
        clear path to it from the new position. Only skips to the next waypoint
        if the current one is already within WAYPOINT_RADIUS (i.e. we drove past
        it while avoiding). This is smarter than always skipping, because the
        detour may have actually brought us close to the original target.
        """
        if self._distance_to_wp() < WAYPOINT_RADIUS:
            # We happened to pass through the waypoint's radius during avoidance
            self.wp_index += 1
            self.get_logger().info(
                f'Waypoint {self.wp_index} passed during avoidance — advancing to {self.wp_index+1}')
        else:
            # Current waypoint still ahead — try to reach it from the new position
            self.get_logger().info(
                f'Resuming waypoint {self.wp_index+1} from new position '
                f'({self.current_x:.2f}, {self.current_y:.2f})')
        if self.wp_index >= len(self.waypoints):
            self.state = 'CUBE_FINDING'
            self.cf_phase = 'ALIGN_NEG_Y'
        else:
            self.search_phase = 'TURNING'
            self.state = 'SEARCHING'

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN CONTROL LOOP  (runs at 10 Hz via self.timer)
    # ─────────────────────────────────────────────────────────────────────────

    def control_loop(self):
        if self.state == 'DONE':
            return  # Nothing to do once the run is finished

        # ─────────────────────────────────────────────────────────────────────
        # STATE: SEARCHING
        # The robot follows the loaded waypoints in order. At each waypoint it
        # first turns to face it (TURNING sub-phase) then drives straight to it
        # (DRIVING sub-phase). While doing this it also watches for the cube and
        # for obstacles. If an obstacle appears it switches to AVOIDING.
        # ─────────────────────────────────────────────────────────────────────
        if self.state == 'SEARCHING':

            # ── Cube spotted? ────────────────────────────────────────────────
            if self.cube_detected:
                self.state = 'DETECTED'
                self.stop()
                self.get_logger().info('RED CUBE DETECTED')
                self.get_logger().info(
                    f'Position: x={self.current_x:.3f} y={self.current_y:.3f}')
                return

            # ── Obstacle too close? ──────────────────────────────────────────
            if self.nearest_front < AVOID_DISTANCE:
                # Decide which way to turn: toward the more open side
                self.avoid_turn_dir = 1 if self.nearest_left >= self.nearest_right else -1

                # Record which side the obstacle is on so AVOID_PASS knows which
                # arc to watch. If we turned left, the obstacle was on the right.
                self.avoid_track_side = 'right' if self.avoid_turn_dir == 1 else 'left'

                # Geometric minimum: the robot should travel at least this far
                # before we trust the side-clear signal. nearest_front is how far
                # away the obstacle is right now — adding a buffer accounts for
                # the robot's own width and approach angle.
                raw_dist = self.nearest_front if math.isfinite(self.nearest_front) else AVOID_DISTANCE
                self.avoid_geo_min_m = raw_dist + AVOID_GEO_BUFFER_M

                self.get_logger().info(
                    f'Obstacle at {self.nearest_front:.2f}m — '
                    f'turning {"LEFT" if self.avoid_turn_dir > 0 else "RIGHT"} | '
                    f'geo_min={self.avoid_geo_min_m:.2f}m tracking {self.avoid_track_side} arc')
                self.state = 'AVOIDING'
                return

            # ── TURNING sub-phase: rotate to face the next waypoint ──────────
            if self.search_phase == 'TURNING':
                err = self._heading_error()
                self.get_logger().info(
                    f'[TURNING] wp={self.wp_index+1}/{len(self.waypoints)} '
                    f'err={math.degrees(err):.1f}°')
                if abs(err) < HEADING_TOL:
                    # Heading is close enough — switch to driving
                    self.stop()
                    self.search_phase = 'DRIVING'
                    self.get_logger().info(f'Aligned — driving to waypoint {self.wp_index+1}')
                else:
                    # Keep rotating toward the target heading
                    self._publish_twist(0.0, TURN_SPEED if err > 0 else -TURN_SPEED)

            # ── DRIVING sub-phase: drive straight to the waypoint ────────────
            elif self.search_phase == 'DRIVING':
                dist = self._distance_to_wp()
                self.get_logger().info(
                    f'[DRIVING] wp={self.wp_index+1}/{len(self.waypoints)} dist={dist:.3f}m')
                if dist < WAYPOINT_RADIUS:
                    # Waypoint reached — advance to the next one
                    self.stop()
                    self.wp_index += 1
                    if self.wp_index >= len(self.waypoints):
                        # All waypoints done — enter cube-finding sweep mode
                        self.state = 'CUBE_FINDING'
                        self.cf_phase = 'ALIGN_NEG_Y'
                        self.get_logger().info('All waypoints reached — CUBE_FINDING')
                    else:
                        # More waypoints remain — turn to face the next one
                        self.search_phase = 'TURNING'
                        self.get_logger().info(
                            f'Waypoint reached — turning to waypoint {self.wp_index+1}')
                else:
                    # Not there yet — drive forward
                    self._publish_twist(FORWARD_SPEED, 0.0)

        # ─────────────────────────────────────────────────────────────────────
        # STATE: AVOIDING
        # Spin in place until the front arc is clear, then transition to
        # AVOID_PASS where the robot drives forward using LiDAR to confirm
        # it has genuinely cleared the obstacle's edge.
        # ─────────────────────────────────────────────────────────────────────
        elif self.state == 'AVOIDING':
            if self.nearest_front >= AVOID_DISTANCE:
                # Front is clear — record the current position so AVOID_PASS can
                # measure how far the robot travels from this exact point.
                self.avoid_pass_start_x = self.current_x
                self.avoid_pass_start_y = self.current_y
                self.state = 'AVOID_PASS'
                self.get_logger().info(
                    f'Front clear — entering AVOID_PASS | '
                    f'geo_min={self.avoid_geo_min_m:.2f}m max={AVOID_MAX_FWD_M:.2f}m')
            else:
                # Still blocked — keep turning
                self._publish_twist(0.0, TURN_SPEED * self.avoid_turn_dir)

        # ─────────────────────────────────────────────────────────────────────
        # STATE: AVOID_PASS
        # Drive forward past the obstacle using three exit conditions:
        #
        #   1. Front blocked again → back to AVOIDING (new obstacle or corner)
        #   2. Geometric minimum met AND side arc reads clear → obstacle edge passed
        #   3. Hard distance cap exceeded → give up and re-plan (wall / long obstacle)
        #
        # This replaces the old timed AVOID_FWD and adapts dynamically to the
        # size and distance of whatever the robot is passing.
        # ─────────────────────────────────────────────────────────────────────
        elif self.state == 'AVOID_PASS':
            # Measure how far we've driven since entering this state
            dx = self.current_x - self.avoid_pass_start_x
            dy = self.current_y - self.avoid_pass_start_y
            dist_travelled = math.sqrt(dx**2 + dy**2)

            # Read the tracking arc on the side the obstacle was on
            side_dist = (self.nearest_left if self.avoid_track_side == 'left'
                         else self.nearest_right)

            self.get_logger().info(
                f'[AVOID_PASS] travelled={dist_travelled:.2f}m '
                f'geo_min={self.avoid_geo_min_m:.2f}m '
                f'{self.avoid_track_side}_arc={side_dist:.2f}m')

            # Condition 1: front blocked again — a new obstacle or we've hit a corner
            if self.nearest_front < AVOID_DISTANCE:
                self.get_logger().warn('Front blocked during AVOID_PASS — re-entering AVOIDING')
                self.state = 'AVOIDING'

            # Condition 3: hard cap — the obstacle is a wall or much larger than expected
            elif dist_travelled >= AVOID_MAX_FWD_M:
                self.get_logger().warn(
                    f'AVOID_PASS hard cap reached ({AVOID_MAX_FWD_M:.1f}m) — skipping waypoint')
                self.stop()
                self._advance_waypoint()

            # Condition 2: geometric minimum met AND side arc has opened up
            elif dist_travelled >= self.avoid_geo_min_m and side_dist >= AVOID_SIDE_CLEAR_M:
                self.get_logger().info(
                    f'Obstacle edge cleared (side_arc={side_dist:.2f}m > {AVOID_SIDE_CLEAR_M:.2f}m '
                    f'after {dist_travelled:.2f}m) — resuming search')
                self.stop()
                self._advance_waypoint()

            else:
                # Still passing — keep driving forward
                self._publish_twist(FORWARD_SPEED, 0.0)

        # ─────────────────────────────────────────────────────────────────────
        # STATE: CUBE_FINDING
        # Runs after all waypoints are visited. Uses four sequential sub-phases
        # to precisely locate and approach the red cube:
        #   1. ALIGN_NEG_Y → face -Y (toward far end of arena)
        #   2. SWEEP       → rotate 180° CW, recording red pixel counts per angle
        #   3. ALIGN_CUBE  → turn to face the angle with the most red
        #   4. APPROACH    → drive forward until close enough to the cube
        # ─────────────────────────────────────────────────────────────────────
        elif self.state == 'CUBE_FINDING':

            # ── Sub-phase 1: rotate to face -Y (yaw = -π/2) ──────────────────
            if self.cf_phase == 'ALIGN_NEG_Y':
                target_yaw = -math.pi / 2.0   # -90° = facing the negative Y axis
                err = target_yaw - self.current_yaw
                err = math.atan2(math.sin(err), math.cos(err))  # Wrap to [-π, +π]
                if abs(err) < HEADING_TOL:
                    # Now facing -Y — initialise the sweep
                    self.stop()
                    self.sweep_start_yaw = self.current_yaw
                    self.sweep_last_yaw  = self.current_yaw
                    self.swept_total     = 0.0
                    self.sweep_readings  = []
                    self.cf_phase = 'SWEEP'
                    self.get_logger().info('Aligned to -Y — starting 180° sweep')
                else:
                    self._publish_twist(0.0, CUBE_TURN_SPEED if err > 0 else -CUBE_TURN_SPEED)

            # ── Sub-phase 2: 180° clockwise sweep, sampling red pixels ────────
            elif self.cf_phase == 'SWEEP':
                # Compute how much the robot has rotated since the last tick.
                # We can't just subtract yaw values because they wrap at ±π,
                # so we correct for wrap-around manually.
                delta = self.sweep_last_yaw - self.current_yaw
                if delta > math.pi:    # Wrapped the wrong way — correct by subtracting a full circle
                    delta -= 2 * math.pi
                elif delta < -math.pi: # Opposite wrap — add a full circle
                    delta += 2 * math.pi
                self.swept_total    += abs(delta)   # Accumulate total rotation (always positive)
                self.sweep_last_yaw  = self.current_yaw
                # Record the current yaw and red pixel count as one sample
                self.sweep_readings.append((self.current_yaw, self.latest_red_pixels))
                self.get_logger().info(
                    f'[SWEEP] swept={math.degrees(self.swept_total):.1f}° '
                    f'red_px={self.latest_red_pixels}')

                if self.swept_total >= math.radians(SWEEP_DEG):
                    # 180° complete — find the direction with the most red pixels
                    self.stop()
                    # Filter to only samples where red pixels exceeded the threshold
                    above = [(yaw, px) for yaw, px in self.sweep_readings
                             if px >= CUBE_PIXEL_THRESHOLD]
                    if above:
                        # Pick the middle sample from the detected band to aim at the cube's centre
                        mid_idx = len(above) // 2
                        self.cube_target_yaw = above[mid_idx][0]
                        self.get_logger().info(
                            f'Sweep done — cube at yaw={math.degrees(self.cube_target_yaw):.1f}°')
                        self.cf_phase = 'ALIGN_CUBE'
                    else:
                        # Cube never exceeded threshold during the sweep — give up
                        self.get_logger().warn('No cube detected in sweep — DONE')
                        self.state = 'DONE'
                else:
                    # Still sweeping — rotate clockwise (negative angular velocity)
                    self._publish_twist(0.0, -CUBE_TURN_SPEED)

            # ── Sub-phase 3: rotate to face the identified cube angle ──────────
            elif self.cf_phase == 'ALIGN_CUBE':
                err = self.cube_target_yaw - self.current_yaw
                err = math.atan2(math.sin(err), math.cos(err))  # Wrap to [-π, +π]
                if abs(err) < HEADING_TOL:
                    # Now facing the cube — start approach
                    self.stop()
                    self.cf_phase = 'APPROACH'
                    self.get_logger().info('Facing cube — approaching')
                else:
                    self._publish_twist(0.0, CUBE_TURN_SPEED if err > 0 else -CUBE_TURN_SPEED)

            # ── Sub-phase 4: slowly drive forward until cube fills the frame ──
            elif self.cf_phase == 'APPROACH':
                self.get_logger().info(f'[APPROACH] red_px={self.latest_red_pixels}')
                if self.latest_red_pixels >= CUBE_STOP_PIXELS:
                    # Enough pixels = cube is close — stop and declare DONE
                    self.stop()
                    self.get_logger().info('Reached cube — DONE')
                    self.state = 'DONE'
                else:
                    # Not there yet — keep creeping forward
                    self._publish_twist(CUBE_FWD_SPEED, 0.0)

        # ─────────────────────────────────────────────────────────────────────
        # STATE: DETECTED
        # Cube was seen with enough pixels during SEARCHING (immediate detection).
        # Stop and transition to DONE.
        # ─────────────────────────────────────────────────────────────────────
        elif self.state == 'DETECTED':
            self.stop()
            self.state = 'DONE'

        # ── Ensure robot is stopped when DONE ────────────────────────────────
        if self.state == 'DONE':
            self.stop()
            self.get_logger().info('=== DONE ===')


# ─────────────────────────────────────────────────────────────────────────────
# ODOMETRY MAP OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def save_odom_map(trail, map_dir):
    """
    After the run, draws the robot's odometry trail on top of the SLAM map image
    and saves the result as odom_overlay.png.

    Requires map.pgm (the occupancy grid image) and map.yaml (the metadata file)
    to be present in map_dir. These are produced by Task 6's map-saving step.

    The trail is drawn as an orange line. Start = blue dot, End = green dot.
    """
    pgm_path  = os.path.join(map_dir, 'map.pgm')
    yaml_path = os.path.join(map_dir, 'map.yaml')
    out_path  = os.path.join(map_dir, 'odom_overlay.png')

    if not os.path.exists(pgm_path) or not os.path.exists(yaml_path):
        print(f'Map files not found in {map_dir} — skipping odom overlay')
        return

    # Read the map metadata to convert real-world (x, y) to pixel coordinates
    import yaml
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution = meta['resolution']  # Metres per pixel (e.g. 0.05)
    origin     = meta['origin']      # Real-world (x, y) of the bottom-left pixel corner

    # Load the map as a grayscale image, then convert to BGR so we can draw coloured lines
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    scale = 8  # Upscale factor — makes small maps easier to see
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    def world_to_px(wx, wy):
        """
        Convert a real-world (x, y) position in metres to pixel coordinates
        in the upscaled map image.
        Map images have Y increasing downward but ROS uses Y increasing upward,
        so we flip the row: row = h - row_from_bottom.
        """
        col = int(round((wx - origin[0]) / resolution))
        row = h - int(round((wy - origin[1]) / resolution))
        return col * scale, row * scale  # Scale up to match the resized image

    # Draw the trail as connected line segments
    if len(trail) > 1:
        pts = [world_to_px(x, y) for x, y in trail]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i+1], (0, 200, 255), 2)  # Orange line

    # Draw start (blue) and end (green) markers
    if trail:
        sx, sy = world_to_px(*trail[0])
        ex, ey = world_to_px(*trail[-1])
        cv2.circle(vis, (sx, sy), 7, (60, 60, 255), -1)  # Blue dot = start
        cv2.putText(vis, 'start', (sx+6, sy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1)
        cv2.circle(vis, (ex, ey), 7, (60, 220, 60), -1)  # Green dot = end
        cv2.putText(vis, 'end', (ex+6, ey-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1)

    cv2.imwrite(out_path, vis)
    print(f'Odom overlay saved to {out_path}  ({len(trail)} points)')


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)       # Initialise the ROS 2 Python client library
    node = WaypointNav()        # Create and start the node
    try:
        rclpy.spin(node)        # Hand control to the ROS 2 executor; runs until Ctrl+C
    finally:
        # Always runs on shutdown (even on crash or Ctrl+C):
        save_odom_map(node.odom_trail, os.path.expanduser('~/Downloads/map'))
        cv2.destroyAllWindows() # Close any OpenCV display windows
        node.destroy_node()     # Clean up the ROS 2 node
        rclpy.shutdown()        # Shut down the ROS 2 client library


if __name__ == '__main__':
    main()