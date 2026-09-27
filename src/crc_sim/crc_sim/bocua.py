import math
import signal
import time
import cv2
import numpy as np

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

try:
    from cv_bridge import CvBridge
    HAVE_CV = True
except ImportError:
    HAVE_CV = False


class Starter(Node):

    def __init__(self):
        super().__init__('crc_starter')

        # --- CÁC THAM SỐ ĐIỀU KHIỂN ---
        self.declare_parameter('max_speed', 0.12)
        self.declare_parameter('max_turn', 1.0)
        self.declare_parameter('stop_distance', 0.35)
        self.declare_parameter('rate', 20.0)

        self.max_speed = self.get_parameter('max_speed').value
        self.max_turn = self.get_parameter('max_turn').value
        self.stop_distance = self.get_parameter('stop_distance').value
        rate = self.get_parameter('rate').value

        self.image = None
        self.scan = None
        self.x = self.y = self.yaw = 0.0
        self._last_log = {}

        self.last_error = 0.0
        self.prev_angular_z = 0.0
        self.last_outdoor_cx = None

        # Biến lọc nhiễu Debounce cho Cua gắt
        self.hard_turn_confirm = 0

        # --- CẤU HÌNH DỐC CẦU (RAMP) ---
        self.is_on_ramp = False
        self.ramp_start_time = None
        self.RAMP_DURATION = 3.5

        # --- CẤU HÌNH HẦM (TUNNEL) & CONTINUITY TRACKING ---
        self.is_in_tunnel = False
        self.tunnel_start_time = None
        self.TUNNEL_DURATION = 7.0
        self.last_tunnel_cx = None

        # --- CẤU HÌNH VƯỢT XE (OVERTAKE) ---
        self.overtake_state = "IDLE"  # Trạng thái: IDLE, LANE_CHANGE, PASSING, RETURN_LANE
        self.overtake_start_time = None
        self.TIME_LANE_CHANGE = 1.8  # Thời gian lách sang làn trái
        self.TIME_PASSING = 2.5      # Thời gian giữ thẳng lái vượt qua
        self.TIME_RETURN_LANE = 1.8  # Thời gian trả lái nhập về lại làn

        # --- CẤU HÌNH BIỂN BÁO & ĐÈN GIAO THÔNG ---
        self.SIGN_CONFIRM_NEEDED = 3
        self.stop_sign_confirm = 0
        self.no_highway_confirm = 0

        self.stop_sign_detected_time = None  
        self.is_stopped_for_sign = False     
        self.stop_duration = 3.0             
        self.stop_sign_pending = False
        self.stop_sign_pending_since = 0.0
        self.STOP_SIGN_PENDING_TIMEOUT = 5.0
        self.stop_sign_cooldown_until = 0.0
        self.STOP_SIGN_COOLDOWN = 5.0

        self.is_stopped_for_no_highway = False
        self.no_highway_detected_time = None
        self.no_highway_duration = 3.0
        self.no_highway_pending = False
        self.no_highway_pending_since = 0.0
        self.NO_HIGHWAY_PENDING_TIMEOUT = 5.0
        self.no_highway_cooldown_until = 0.0
        self.NO_HIGHWAY_COOLDOWN = 5.0
        
        self.is_traffic_light_red = False    
        self.no_highway_active = False       

        self.is_light_stopped = False
        self.light_stop_start_time = 0.0
        self.light_stop_duration = 13.0
        self.has_entered_intersection = False

        # Debounce + "chờ tới vạch" cho đèn giao thông (giống logic biển báo)
        self.LIGHT_CONFIRM_NEEDED = 3
        self.light_confirm = 0
        self.light_pending = False
        self.light_pending_since = 0.0
        self.light_pending_state = None
        self.LIGHT_PENDING_TIMEOUT = 7.0

        # --- WATCHDOG THOÁT BẾ TẮC ---
        self.stuck_since = None
        self.stuck_ref_dist = 0.0
        self.STUCK_TIMEOUT = 1.0

        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.bridge = CvBridge() if HAVE_CV else None

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)

        self.create_timer(1.0 / rate, self.tick)

    def on_image(self, msg):
        if self.bridge is None:
            return
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')

    def on_scan(self, msg):
        self.scan = msg

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def drive(self, v, w):
        msg = Twist()
        msg.linear.x = float(max(-self.max_speed, min(self.max_speed, v)))
        msg.angular.z = float(max(-self.max_turn, min(self.max_turn, w)))
        self.pub_cmd.publish(msg)

    def stop(self):
        if rclpy.ok():
            try:
                self.pub_cmd.publish(Twist())
            except Exception:
                pass

    def log_every(self, seconds, text):
        now = time.time()
        if now - self._last_log.get(text[:20], 0.0) >= seconds:
            self._last_log[text[:20]] = now
            self.get_logger().info(text)

    def tick(self):
        try:
            self.control()
        except Exception as e:
            self.get_logger().error(f'control() raised: {e}')
            self.stop()

    def detect_signs(self, img, vis_img):
        stop_seen = False
        no_highway_seen = False
        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Nâng ngưỡng chiều cao/diện tích tối thiểu để chỉ nhận diện khi biển đã ở GẦN
        # (biển ở xa sẽ có blob nhỏ hơn ngưỡng này nên bị loại bỏ, giảm nhiễu)
        MIN_SIGN_HEIGHT = 34
        MIN_SIGN_AREA = 900

        # Nhận diện màu đỏ biển STOP
        lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        cnts, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if MIN_SIGN_AREA < area < 10000:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bh >= MIN_SIGN_HEIGHT and 0.75 <= float(bw)/bh <= 1.3 and by < int(h * 0.6):
                    stop_seen = True
                    cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                    cv2.putText(vis_img, "STOP SIGN", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    break

        # Nhận diện màu xanh biển Cấm Cao Tốc (No Highway)
        lower_blue, upper_blue = np.array([100, 120, 50]), np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        cnts_b, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_b:
            area = cv2.contourArea(c)
            if MIN_SIGN_AREA < area < 10000:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bh >= MIN_SIGN_HEIGHT and 0.7 <= float(bw)/bh <= 1.3 and by < int(h * 0.6):
                    no_highway_seen = True
                    cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (255, 100, 0), 2)
                    cv2.putText(vis_img, "NO HIGHWAY", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
                    break

        return stop_seen, no_highway_seen

    def detect_traffic_light(self, img, vis_img, now):
        h, w, _ = img.shape
        # Thu hẹp ROI xuống nửa dưới của vùng trên (bỏ phần sát chân trời quá xa)
        roi_top = int(h * 0.15)
        roi_bottom = int(h * 0.45)
        upper_roi_hsv = cv2.cvtColor(img[roi_top:roi_bottom, :], cv2.COLOR_BGR2HSV)
        
        # Đỏ, Vàng, Xanh
        mask_red = cv2.inRange(upper_roi_hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
                   cv2.inRange(upper_roi_hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask_yellow = cv2.inRange(upper_roi_hsv, np.array([15, 120, 70]), np.array([35, 255, 255]))
        mask_green = cv2.inRange(upper_roi_hsv, np.array([40, 120, 70]), np.array([90, 255, 255]))

        state = None
        present = False

        # Nâng ngưỡng cao/diện tích tối thiểu để chỉ nhận diện khi đèn đã ở GẦN
        # (đèn ở xa sẽ có blob nhỏ hơn các ngưỡng này nên bị loại bỏ)
        MIN_LIGHT_HEIGHT = 26
        MIN_LIGHT_AREA = 220
        MAX_LIGHT_AREA = 3000

        cnt_r, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_y, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_g, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnt_r:
            bx, by, bw, bh = cv2.boundingRect(c)
            if MIN_LIGHT_AREA < cv2.contourArea(c) < MAX_LIGHT_AREA and bh >= MIN_LIGHT_HEIGHT:
                state, present = 'RED', True
                break
        if not present:
            for c in cnt_y:
                bx, by, bw, bh = cv2.boundingRect(c)
                if MIN_LIGHT_AREA < cv2.contourArea(c) < MAX_LIGHT_AREA and bh >= MIN_LIGHT_HEIGHT:
                    state, present = 'YELLOW', True
                    break
        if not present:
            for c in cnt_g:
                bx, by, bw, bh = cv2.boundingRect(c)
                if MIN_LIGHT_AREA < cv2.contourArea(c) < MAX_LIGHT_AREA and bh >= MIN_LIGHT_HEIGHT:
                    state, present = 'GREEN', True
                    break

        return state, True, present

    def detect_stop_line(self, img, vis_img=None):
        h, w, _ = img.shape
        y0 = int(h * 0.8)
        roi = img[y0:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = roi.shape[0] * roi.shape[1]
        ratio = float(white_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
        
        at_line = ratio > 0.12

        if vis_img is not None:
            # Vẽ khung vùng ROI đang được dùng để phát hiện vạch ngang
            box_color = (0, 255, 0) if at_line else (0, 165, 255)
            cv2.rectangle(vis_img, (0, y0), (w - 1, h - 1), box_color, 2)
            cv2.putText(vis_img, f"STOP-LINE ROI ratio={ratio:.2f} (need>0.12)",
                        (5, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)
            # Tô màu (cyan) đúng những pixel đang được tính là "trắng" trong ROI
            # để thấy trực tiếp vì sao ratio cao/thấp
            mask_bgr = np.zeros_like(vis_img[y0:, :])
            mask_bgr[:, :, 0] = thresh  # kênh Blue
            mask_bgr[:, :, 1] = thresh  # kênh Green -> tổng hợp thành màu cyan
            roi_view = vis_img[y0:, :]
            cv2.addWeighted(mask_bgr, 0.6, roi_view, 1.0, 0, roi_view)
            vis_img[y0:, :] = roi_view

        return at_line, ratio

    def draw_lidar_overlay(self, img, threshold_dist):
        if self.scan is None or not self.scan.ranges:
            return float('inf'), None

        h, w, _ = img.shape
        center_x = w // 2
        center_y = int(h * 0.75)

        scan = self.scan
        min_dist = float('inf')
        closest_point = None

        for angle_deg in range(-15, 16, 2):
            angle_rad = math.radians(angle_deg)
            if scan.angle_min <= angle_rad <= scan.angle_max:
                idx = int((angle_rad - scan.angle_min) / scan.angle_increment)
                if 0 <= idx < len(scan.ranges):
                    r = scan.ranges[idx]
                    if math.isfinite(r) and r > scan.range_min:
                        pt_x = int(center_x + (angle_deg / 15.0) * (w * 0.25))
                        pt_y = int(center_y - (r / 1.5) * (h * 0.4))
                        pt_y = max(20, min(h - 10, pt_y))

                        if r < threshold_dist:
                            cv2.circle(img, (pt_x, pt_y), 6, (0, 0, 255), -1)
                            if r < min_dist:
                                min_dist = r
                                closest_point = (pt_x, pt_y, angle_deg, r)
                        else:
                            cv2.circle(img, (pt_x, pt_y), 3, (0, 255, 0), -1)

                        if r < min_dist:
                            min_dist = r

        if closest_point:
            px, py, ang, dist = closest_point
            cv2.line(img, (center_x, h - 20), (px, py), (0, 0, 255), 2)
            cv2.putText(img, f"OBSTACLE: {dist:.2f}m at {ang}deg", (px - 60, py - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return min_dist, closest_point

    def check_turn_path_clear(self, angular_z, base_half_width=15.0, max_bias_deg=28.0):
        if self.scan is None or not self.scan.ranges:
            return float('inf')

        scan = self.scan
        turn_ratio = max(-1.0, min(1.0, angular_z / 0.35))
        bias = max_bias_deg * turn_ratio
        half = base_half_width + abs(bias) * 0.5

        lo_deg = bias - half
        hi_deg = bias + half

        min_d = float('inf')

        for angle_deg in np.arange(lo_deg, hi_deg + 1.0, 2.0):
            angle_rad = math.radians(angle_deg)
            if scan.angle_min <= angle_rad <= scan.angle_max:
                idx = int((angle_rad - scan.angle_min) / scan.angle_increment)
                if 0 <= idx < len(scan.ranges):
                    r = scan.ranges[idx]
                    if math.isfinite(r) and r > scan.range_min:
                        min_d = min(min_d, r)

        return min_d

    def process_image_mask(self, img):
        h, w, _ = img.shape
        crop_h = int(h * 2 / 3)
        roi = img[crop_h:h, :]

        roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        gray = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)
        roi_enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 110])
        upper_white = np.array([180, 60, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        side_crop = 30
        mask[:, :side_crop] = 0
        mask[:, w - side_crop:] = 0

        return mask, roi

    def largest_blob_ratio(self, mask):
        total = mask.shape[0] * mask.shape[1]
        if total == 0:
            return 0.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest = max(cv2.contourArea(c) for c in contours)
        return float(largest) / float(total)

    def process_tunnel_center_line(self, img):
        h, w, _ = img.shape
        crop_h = int(h * 0.6)
        roi = img[crop_h:h, :]
        roi_h, roi_w, _ = roi.shape

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)

        _, mask = cv2.threshold(enhanced, 175, 255, cv2.THRESH_BINARY)

        margin = int(roi_w * 0.20)
        mask[:, :margin] = 0
        mask[:, roi_w - margin:] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cx = None
        min_dist = float('inf')
        ref_x = self.last_tunnel_cx if (self.last_tunnel_cx is not None) else (roi_w / 2.0)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 35:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / float(bh) if bh > 0 else 99.0

                if aspect_ratio > 2.2:
                    continue

                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    dist = abs(cx - ref_x)

                    if dist < min_dist:
                        min_dist = dist
                        best_cx = cx

        if best_cx is not None:
            self.last_tunnel_cx = best_cx

        return best_cx, mask, roi

    def detect_blue_car_camera(self, img, vis_img=None):
        h, w, _ = img.shape
        roi = img[int(h * 0.3):int(h * 0.8), :] 
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 120, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_blue_car = False
        max_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            total_area = roi.shape[0] * roi.shape[1]
            score = float(area) / float(total_area)
            if score > 0.035:
                has_blue_car = True
                max_score = max(max_score, score)
                if vis_img is not None:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    cv2.rectangle(vis_img, (bx, by + int(h * 0.3)), (bx + bw, by + int(h * 0.3) + bh), (255, 0, 0), 2)
                    cv2.putText(vis_img, "BLUE CAR", (bx, by + int(h * 0.3) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return has_blue_car, max_score

    def control(self):
        if self.image is None:
            self.stop()
            return

        now = time.time()
        vis_img = self.image.copy()
        h, w, _ = self.image.shape

        # ================================================================
        # KIỂM TRA TRẠNG THÁI DỪNG: NẾU ĐANG DỪNG THÌ NGỪNG DETECT HOÀN TOÀN
        # ================================================================
        if self.is_stopped_for_sign:
            elapsed = now - self.stop_sign_detected_time
            if elapsed < self.stop_duration:
                self.stop()
                cv2.putText(vis_img, f"STOP SIGN: {self.stop_duration - elapsed:.1f}s",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Robot Debug View", vis_img)
                cv2.waitKey(1)
                return
            else:
                self.is_stopped_for_sign = False
                self.stop_sign_cooldown_until = now + self.STOP_SIGN_COOLDOWN

        if self.is_stopped_for_no_highway:
            elapsed = now - self.no_highway_detected_time
            if elapsed < self.no_highway_duration:
                self.stop()
                cv2.putText(vis_img, f"NO HIGHWAY: {self.no_highway_duration - elapsed:.1f}s",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                cv2.imshow("Robot Debug View", vis_img)
                cv2.waitKey(1)
                return
            else:
                self.is_stopped_for_no_highway = False
                self.no_highway_cooldown_until = now + self.NO_HIGHWAY_COOLDOWN
                self.get_logger().warn(
                    '[SIGN] NO HIGHWAY: het thoi gian dung mac dinh (3s). Tiep tuc bam line binh thuong.')

        if self.is_light_stopped:
            elapsed = now - getattr(self, 'light_stop_start_time', now)
            duration = getattr(self, 'light_stop_duration', 13.0)

            # Đã dừng vì đèn -> ngừng nhận diện hoàn toàn cho tới khi hết thời gian dừng
            if elapsed < duration:
                self.stop()
                cv2.putText(vis_img, f"TRAFFIC LIGHT BLIND STOP: {duration - elapsed:.1f}s",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Robot Debug View", vis_img)
                cv2.waitKey(1)
                return
            else:
                self.is_light_stopped = False
                self.has_entered_intersection = True
                self.light_confirm = 0
                self.get_logger().info('[LIGHT] Da dem du thoi gian an toan -> Robot tiep tuc di qua giao lo!')

        # ================================================================
        # A. NHẬN DIỆN BIỂN BÁO, ĐÈN GIAO THÔNG, VẠCH NGANG (Khi đang di chuyển)
        # ================================================================
        stop_seen, no_highway_seen = self.detect_signs(self.image, vis_img)
        light_state, light_confirmed, light_present = self.detect_traffic_light(
            self.image, vis_img, now)
        at_stop_line, line_ratio = self.detect_stop_line(self.image, vis_img)

        # Debounce
        self.stop_sign_confirm = (min(self.stop_sign_confirm + 1, 99) if stop_seen
                                   else max(self.stop_sign_confirm - 1, 0))
        self.no_highway_confirm = (min(self.no_highway_confirm + 1, 99) if no_highway_seen
                                    else max(self.no_highway_confirm - 1, 0))
        stop_sign_confirmed = self.stop_sign_confirm >= self.SIGN_CONFIRM_NEEDED
        no_highway_confirmed = self.no_highway_confirm >= self.SIGN_CONFIRM_NEEDED

        # Ghi nhận "đang chờ tới vạch"
        if (stop_sign_confirmed and not self.is_stopped_for_sign
                and now > self.stop_sign_cooldown_until and not self.stop_sign_pending):
            self.stop_sign_pending = True
            self.stop_sign_pending_since = now
            self.get_logger().info('[SIGN] Da xac nhan bien STOP - cho toi vach ngang...')

        if (no_highway_confirmed and not self.is_stopped_for_no_highway
                and now > self.no_highway_cooldown_until and not self.no_highway_pending):
            self.no_highway_pending = True
            self.no_highway_pending_since = now
            self.get_logger().info('[SIGN] Da xac nhan bien NO HIGHWAY - cho toi vach ngang...')

        light_seen_stop_condition = light_present and light_confirmed and (
            light_state == 'RED' or (light_state == 'YELLOW' and not self.has_entered_intersection)
        )
        self.light_confirm = (min(self.light_confirm + 1, 99) if light_seen_stop_condition
                               else max(self.light_confirm - 1, 0))
        light_requires_stop = self.light_confirm >= self.LIGHT_CONFIRM_NEEDED

        if not light_present and not self.is_light_stopped:
            self.has_entered_intersection = False

        if (light_requires_stop and not self.is_light_stopped and not self.light_pending):
            self.light_pending = True
            self.light_pending_since = now
            self.light_pending_state = light_state
            self.get_logger().info(f'[LIGHT] Da xac nhan den {light_state} - cho toi vach ngang...')

        # Nếu đang chờ mà đèn đã chuyển XANH trước khi tới vạch -> huỷ chờ, đi tiếp
        if self.light_pending and light_present and light_confirmed and light_state == 'GREEN':
            self.light_pending = False
            self.light_confirm = 0
            self.get_logger().info('[LIGHT] Da chuyen XANH truoc khi toi vach -> huy cho, tiep tuc di.')

        # ================================================================
        # B. KÍCH HOẠT DỪNG KHI ĐÃ TỚI VẠCH NGANG (pending -> stopping)
        # ================================================================
        if self.stop_sign_pending:
            timed_out = (now - self.stop_sign_pending_since) > self.STOP_SIGN_PENDING_TIMEOUT
            if at_stop_line or timed_out:
                self.is_stopped_for_sign = True
                self.stop_sign_detected_time = now
                self.stop_sign_pending = False
                reason = "toi vach" if at_stop_line else "timeout du phong"
                self.get_logger().info(f'[SIGN] STOP: {reason} -> dung {self.stop_duration:.1f}s')

        if self.no_highway_pending:
            timed_out = (now - self.no_highway_pending_since) > self.NO_HIGHWAY_PENDING_TIMEOUT
            if at_stop_line or timed_out:
                self.is_stopped_for_no_highway = True
                self.no_highway_detected_time = now
                self.no_highway_pending = False
                reason = "toi vach" if at_stop_line else "timeout du phong"
                self.get_logger().info(
                    f'[SIGN] NO HIGHWAY: {reason} -> dung {self.no_highway_duration:.1f}s')

        # Dừng đèn giao thông: chỉ thực sự dừng khi ĐÃ TỚI VẠCH NGANG (hoặc hết thời gian chờ dự phòng)
        if self.light_pending:
            timed_out = (now - self.light_pending_since) > self.LIGHT_PENDING_TIMEOUT
            if at_stop_line or timed_out:
                self.is_light_stopped = True
                self.light_stop_start_time = now
                self.light_stop_duration = 13.0 if self.light_pending_state == 'RED' else 3.0
                self.light_pending = False
                reason = "toi vach" if at_stop_line else "timeout du phong"
                self.get_logger().info(
                    f'[LIGHT] {reason} -> Dung dem nguoc {self.light_stop_duration}s do khuat camera...'
                )

        # Debug: hiển thị bộ đếm confirm & trạng thái chờ để dễ theo dõi vì sao dừng
        cv2.putText(vis_img,
                    f"STOPconf={self.stop_sign_confirm} NHconf={self.no_highway_confirm} "
                    f"LIGHTconf={self.light_confirm}/{self.LIGHT_CONFIRM_NEEDED}",
                    (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        pending_bits = []
        if self.stop_sign_pending:
            pending_bits.append(f"STOP pend {now - self.stop_sign_pending_since:.1f}/{self.STOP_SIGN_PENDING_TIMEOUT:.1f}s")
        if self.no_highway_pending:
            pending_bits.append(f"NH pend {now - self.no_highway_pending_since:.1f}/{self.NO_HIGHWAY_PENDING_TIMEOUT:.1f}s")
        if self.light_pending:
            pending_bits.append(f"LIGHT pend {now - self.light_pending_since:.1f}/{self.LIGHT_PENDING_TIMEOUT:.1f}s")
        if pending_bits:
            cv2.putText(vis_img, " | ".join(pending_bits), (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # ================================================================
        # PHẦN CÒN LẠI: BÁM LINE / DỐC / HẦM / VƯỢT XE / LIDAR
        # ================================================================
        mask_normal, roi_normal = self.process_image_mask(self.image)
        blob_ratio = self.largest_blob_ratio(mask_normal)

        top_brightness = np.mean(self.image[:int(h / 3), :])
        is_ramp_like = (blob_ratio > 0.22)
        is_tunnel_like = (top_brightness < 65.0) and (blob_ratio > 0.18 or self.is_on_ramp)

        if not self.is_on_ramp and not self.is_in_tunnel and is_ramp_like:
            self.is_on_ramp = True
            self.ramp_start_time = now
            self.stuck_since = None
            self.get_logger().info(f'>>> KÍCH HOẠT DỐC CẦU (blob_ratio={blob_ratio:.2f}) <<<')

        if not self.is_in_tunnel and is_tunnel_like:
            self.is_in_tunnel = True
            self.tunnel_start_time = now
            self.stuck_since = None
            self.last_tunnel_cx = None
            self.get_logger().info(f'>>> KÍCH HOẠT VÀO HẦM (top_bright={top_brightness:.1f}) <<<')

        if self.is_on_ramp and (now - self.ramp_start_time > self.RAMP_DURATION):
            self.is_on_ramp = False
            self.get_logger().info('>>> THOÁT DỐC CẦU <<<')

        if self.is_in_tunnel and (now - self.tunnel_start_time > self.TUNNEL_DURATION):
            self.is_in_tunnel = False
            self.last_tunnel_cx = None
            self.last_outdoor_cx = None
            self.last_error = 0.0
            self.prev_angular_z = 0.0
            self.hard_turn_confirm = 0
            self.get_logger().info('>>> THOÁT HẦM: RESET ERROR & TRANSITION OUTDOOR <<<')

        disable_lidar = self.is_on_ramp or self.is_in_tunnel
        has_blue_car, blue_score = self.detect_blue_car_camera(self.image)

        if has_blue_car and self.overtake_state == "IDLE" and not disable_lidar:
            self.overtake_state = "LANE_CHANGE"
            self.overtake_start_time = now
            self.get_logger().info(">>> KÍCH HOẠT CHU TRÌNH VƯỢT XE XANH <<<")

        if self.overtake_state != "IDLE":
            elapsed = now - self.overtake_start_time
            if self.overtake_state == "LANE_CHANGE":
                if elapsed < self.TIME_LANE_CHANGE:
                    self.drive(self.max_speed * 0.8, 0.35)
                    cv2.putText(vis_img, "OVERTAKE: LATCHING LEFT", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    self.overtake_state = "PASSING"
                    self.overtake_start_time = now
            elif self.overtake_state == "PASSING":
                if elapsed < self.TIME_PASSING:
                    self.drive(self.max_speed, 0.0)
                    cv2.putText(vis_img, "OVERTAKE: KEEP STRAIGHT & PASSING", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    self.overtake_state = "RETURN_LANE"
                    self.overtake_start_time = now
            elif self.overtake_state == "RETURN_LANE":
                if elapsed < self.TIME_RETURN_LANE:
                    self.drive(self.max_speed * 0.8, -0.30)
                    cv2.putText(vis_img, "OVERTAKE: RETURNING TO LANE", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                else:
                    self.overtake_state = "IDLE"
                    self.last_outdoor_cx = None
                    self.last_error = 0.0
                    self.get_logger().info(">>> HOÀN THÀNH VƯỢT XE - TRỜ VỀ DÒ LINE <<<")

            cv2.imshow("Robot Debug View", vis_img)
            cv2.waitKey(1)
            return

        if not disable_lidar:
            min_front_dist, closest_info = self.draw_lidar_overlay(vis_img, self.stop_distance)
            will_stop = min_front_dist < self.stop_distance

            if will_stop:
                if self.stuck_since is None:
                    self.stuck_since = now
                    self.stuck_ref_dist = min_front_dist
                stuck_elapsed = now - self.stuck_since
                dist_stable = abs(min_front_dist - self.stuck_ref_dist) < 0.05
            else:
                self.stuck_since = None
                stuck_elapsed = 0.0
                dist_stable = False

            if will_stop and stuck_elapsed > self.STUCK_TIMEOUT and dist_stable:
                self.get_logger().warn(f'>>> STUCK-ESCAPE: Kẹt {stuck_elapsed:.1f}s -> Chuyển TUNNEL MODE <<<')
                self.is_in_tunnel = True
                self.tunnel_start_time = now
                self.stuck_since = None
                will_stop = False
                disable_lidar = True

            if will_stop:
                self.stop()
                self.log_every(1.0, f'Obstacle Stop at {min_front_dist:.2f}m')
                cv2.putText(vis_img, f"OBSTACLE STOP ({min_front_dist:.2f}m)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Robot Debug View", vis_img)
                cv2.waitKey(1)
                return
        else:
            self.stuck_since = None
            cv2.putText(vis_img, "LIDAR IGNORED (RAMP/TUNNEL MODE)", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if self.is_in_tunnel:
            tunnel_cx, tunnel_mask, _ = self.process_tunnel_center_line(self.image)
            if tunnel_cx is not None:
                image_center = w / 2.0
                error = tunnel_cx - image_center
                Kp = 0.0050
                Kd = 0.008
                derivative = error - self.last_error
                self.last_error = error
                angular_z = -float(error * Kp + derivative * Kd)
                angular_z = max(-0.45, min(0.45, angular_z))
                self.drive(self.max_speed * 0.8, angular_z)
                cv2.circle(vis_img, (tunnel_cx, int(h * 0.75)), 8, (255, 0, 255), -1)
                cv2.putText(vis_img, f"TUNNEL TRACKING (cx={tunnel_cx}, err={error:.1f}px)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            else:
                self.drive(self.max_speed * 0.5, -0.1)
                cv2.putText(vis_img, "TUNNEL: SEARCHING CENTER LINE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            M = cv2.moments(mask_normal)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                self.last_outdoor_cx = cx
                image_center = w / 2.0
                raw_error = cx - image_center
                abs_error = abs(raw_error)

                if abs_error >= 60.0:
                    self.hard_turn_confirm = min(self.hard_turn_confirm + 1, 99)
                else:
                    self.hard_turn_confirm = 0
                is_confirmed_hard_turn = self.hard_turn_confirm >= 3

                if is_confirmed_hard_turn:
                    current_speed = self.max_speed * 0.55
                    Kp, Kd = 0.0055, 0.0080
                    max_turn_limit = 0.55
                    error = raw_error
                else:
                    current_speed = self.max_speed
                    Kp, Kd = 0.0035, 0.0060
                    max_turn_limit = 0.35
                    error = 0.0 if abs_error < 15.0 else raw_error

                derivative = error - self.last_error
                self.last_error = error
                raw_angular = -float(error * Kp + derivative * Kd)
                angular_z = max(-max_turn_limit, min(max_turn_limit, raw_angular))

                turn_clear_dist = self.check_turn_path_clear(angular_z)
                if turn_clear_dist < 0.30 and not disable_lidar:
                    current_speed = min(current_speed, self.max_speed * 0.3)
                    angular_z *= 0.6
                    self.log_every(0.5, f'[TURN GUARD] Vật cản hướng rẽ {turn_clear_dist:.2f}m -> Giảm tốc/lái')
                    cv2.putText(vis_img, f"TURN GUARD! dist={turn_clear_dist:.2f}m", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

                self.drive(current_speed, angular_z)
                cv2.circle(vis_img, (cx, int(h * 5 / 6)), 8, (0, 255, 0), -1)
                mode_str = "HARD_TURN" if is_confirmed_hard_turn else ("RAMP" if self.is_on_ramp else "NORMAL")
                cv2.putText(vis_img, f"TRACKING [{mode_str}] (Err={error:.1f}px, v={current_speed:.2f})",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                fallback_w = -0.20 if self.last_error > 0 else 0.20
                self.drive(self.max_speed * 0.6, fallback_w)

        cv2.imshow("Robot Debug View", vis_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = Starter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop()
            time.sleep(0.05)
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()