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
from std_msgs.msg import String

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
        self.hard_turn_confirm = 0

        # --- CẤU HÌNH DỐC CẦU & HẦM ---
        self.is_on_ramp = False
        self.ramp_start_time = None
        self.RAMP_DURATION = 8.0

        # Đóng băng đồng hồ cầu trong lúc robot đang dừng
        self.ramp_pause_start = None
        self.is_in_tunnel = False
        self.tunnel_start_time = None
        self.TUNNEL_DURATION = 8.0
        self.last_tunnel_cx = None

        # --- CẤU HÌNH VƯỢT XE ---
        self.overtake_state = "IDLE"
        self.overtake_start_time = None
        self.TIME_LANE_CHANGE = 1.8
        self.TIME_PASSING = 2.5
        self.TIME_RETURN_LANE = 1.8

        # --- CẤU HÌNH NHÍCH TRÁI SAU DỪNG STOP SIGN ---
        self.post_stop_turn_active = False
        self.post_stop_turn_start = 0.0
        self.POST_STOP_TURN_DURATION = 2.2

        # --- WATCHDOG THOÁT BẾ TẮC ---
        self.stuck_since = None
        self.stuck_ref_dist = 0.0
        self.STUCK_TIMEOUT = 1.0

        # --- QUẢN LÝ DỪNG BIỂN BÁO & ĐÈN GIAO THÔNG ---
        self.current_traffic_signal = "CLEAR"
        self.stop_sign_pending = False
        self.stop_sign_pending_time = 0.0

        self.is_stopped_for_sign = False
        self.sign_stop_start_time = 0.0
        self.STOP_SIGN_DURATION = 3.0
        self.stop_sign_cooldown_until = 0.0
        self.STOP_SIGN_COOLDOWN = 10.0

        self.is_stopped_for_light = False
        self.light_stop_start_time = 0.0
        self.light_stop_duration = 0.0
        self.light_cooldown_until = 0.0
        self.LIGHT_COOLDOWN = 12.0

        self.clahe = cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8, 8))
        self.bridge = CvBridge() if HAVE_CV else None
        # --- PUBLISHERS & SUBSCRIBERS ---
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_status = self.create_publisher(String, '/starter_status', 10)
        self.create_subscription(Image,'/camera/image_raw',self.on_image,qos_profile_sensor_data)
        self.create_subscription(
            LaserScan,
            '/scan',
            self.on_scan,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry,
            '/odom',
            self.on_odom,
            10
        )
        self.create_subscription(
            String,
            '/traffic_signal',
            self.on_traffic_signal,
            10
        )
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            "=== CRC STARTER NODE STARTED "
            "(STOP LINE & RAMP LIDAR COMPLETE) ==="
        )

    # =========================================================================
    # CALLBACK XỬ LÝ TÍN HIỆU TỪ TOPIC
    # =========================================================================
    def on_traffic_signal(self, msg):
        sig = str(msg.data)
        if self.is_on_ramp or self.is_in_tunnel:
            return
        if (
            self.is_stopped_for_sign
            or self.is_stopped_for_light
            or self.post_stop_turn_active):
            return
        now = time.time()
        at_stop_line = "+LINE" in sig
        cooldown_active = (now < getattr(self,'stop_sign_cooldown_until',0.0))
        has_stop_sign_sig = (("SIGN_STOP" in sig) or ("STOP" in sig) or ("NO_HIGHWAY" in sig))
        if has_stop_sign_sig and not cooldown_active:
            if not self.stop_sign_pending:
                self.stop_sign_pending = True
                self.stop_sign_pending_time = now
                self.get_logger().info(
                    ">>> [TRAFFIC] ĐÃ THẤY BIỂN STOP: "
                    "Lưu cờ Pending, chờ chạm Vạch ngang (+LINE)... <<<"
                )
        # Timeout pending 15.0s
        if (
            self.stop_sign_pending
            and (now - self.stop_sign_pending_time > 15.0)):
            self.stop_sign_pending = False
        # --- CHỐNG GHI ĐÈ TÍN HIỆU ---
        if self.stop_sign_pending:
            if at_stop_line:
                self.current_traffic_signal = "SIGN_STOP+LINE"
            else:
                self.current_traffic_signal = "SIGN_STOP (PENDING)"
        else:
            self.current_traffic_signal = sig

        if (
            (self.stop_sign_pending or has_stop_sign_sig)
            and at_stop_line
            and not cooldown_active):
            self.is_stopped_for_sign = True
            self.sign_stop_start_time = now
            self.stop_sign_pending = False

            self.get_logger().info(
                f'>>> [TRAFFIC] CHẠM VẠCH NGANG (+LINE) '
                f'-> KÍCH HOẠT DỪNG {self.STOP_SIGN_DURATION}S '
                f'BIỂN STOP! <<<'
            )
        # 3. KÍCH HOẠT DỪNG ĐÈN GIAO THÔNG
        # RED / YELLOW + VẠCH NGANG
        elif (
            ("RED" in sig or "YELLOW" in sig)
            and at_stop_line):
            if now > getattr(self,'light_cooldown_until',0.0):
                self.is_stopped_for_light = True
                self.light_stop_start_time = now
                self.light_stop_duration = (
                    12.0 if "RED" in sig else 3.0)
                self.get_logger().info(
                    f'>>> [TRAFFIC] PHÁT HIỆN ĐÈN {sig} TẠI VẠCH '
                    f'-> DỪNG {self.light_stop_duration}s! <<<')

    def on_image(self, msg):
        if self.bridge is None:
            return
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        except Exception as e:
            self.get_logger().warn(
                f'Image conversion failed: {e}')

    def on_scan(self, msg):
        self.scan = msg

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.x = p.x
        self.y = p.y

        self.yaw = math.atan2(2.0 * (q.w * q.z+ q.x * q.y),
            1.0 - 2.0 * (q.y * q.y+ q.z * q.z))

    def drive(self, v, w):
        msg = Twist()
        msg.linear.x = float(
            max(-self.max_speed,min(self.max_speed, v)))
        msg.angular.z = float(
            max(-self.max_turn,min(self.max_turn, w)))
        self.pub_cmd.publish(msg)

    def stop(self):
        if rclpy.ok():
            try:
                self.pub_cmd.publish(Twist())
            except Exception:
                pass
    def log_every(self, seconds, text):
        now = time.time()
        if (
            now - self._last_log.get(text[:20],0.0) >= seconds):
            self._last_log[text[:20]] = now
            self.get_logger().info(text)
    def publish_status(self, mode_str):
        status_msg = String()
        status_msg.data = (
            f"MODE:{mode_str}"
            f"|SIGNAL:{self.current_traffic_signal}")
        self.pub_status.publish(status_msg)
    def tick(self):
        try:
            self.control()
        except Exception as e:
            self.get_logger().error(
                f'control() raised: {e}')
            self.stop()
    def draw_lidar_overlay(self, img, threshold_dist):
        if ( self.scan is None or not self.scan.ranges):
            return float('inf'), None
        h, w, _ = img.shape
        center_x = w // 2
        center_y = int(h * 0.75)
        scan = self.scan
        min_dist = float('inf')
        closest_point = None
        for angle_deg in range(-15, 16, 2):
            angle_rad = math.radians(angle_deg)
            if ( scan.angle_min<= angle_rad<= scan.angle_max):
                idx = int(
                    (angle_rad - scan.angle_min)
                    / scan.angle_increment)
                if 0 <= idx < len(scan.ranges):
                    r = scan.ranges[idx]
                    if (math.isfinite(r) and r > scan.range_min):
                        pt_x = int(center_x+ (angle_deg / 15.0) * (w * 0.25))
                        pt_y = int( center_y- (r / 1.5)* (h * 0.4))
                        pt_y = max(20,min(h - 10, pt_y))
                        if r < threshold_dist:
                            cv2.circle(img,(pt_x, pt_y),6,(0, 0, 255),-1)
                            if r < min_dist:
                                min_dist = r
                                closest_point = (pt_x,pt_y,angle_deg,r)
                        else:
                            cv2.circle(img,(pt_x, pt_y),3,(0, 255, 0),-1)
                        if r < min_dist:
                            min_dist = r
        if closest_point:
            px, py, ang, dist = closest_point
            cv2.line(
                img,
                (center_x, h - 20),
                (px, py),
                (0, 0, 255),
                2)
            cv2.putText(
                img,
                f"OBSTACLE: {dist:.2f}m at {ang}deg",
                (px - 60, py - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2)
        return min_dist, closest_point

    def check_turn_path_clear(
        self,
        angular_z,
        base_half_width=15.0,
        max_bias_deg=28.0):
        if (self.scan is None  or not self.scan.ranges):
            return float('inf')
        scan = self.scan
        turn_ratio = max(-1.0,min(1.0, angular_z / 0.35))
        bias = max_bias_deg * turn_ratio
        half = (base_half_width + abs(bias) * 0.5)
        lo_deg = bias - half
        hi_deg = bias + half
        min_d = float('inf')
        for angle_deg in np.arange(
            lo_deg,
            hi_deg + 1.0,
            2.0):
            angle_rad = math.radians(angle_deg)
            if (
                scan.angle_min
                <= angle_rad
                <= scan.angle_max):
                idx = int(
                    (angle_rad - scan.angle_min)
                    / scan.angle_increment
                )
                if 0 <= idx < len(scan.ranges): 
                    r = scan.ranges[idx]
                    if (
                        math.isfinite(r)
                        and r > scan.range_min):
                        min_d = min(min_d, r)
        return min_d

    def process_image_mask(self, img):
        h, w, _ = img.shape
        crop_h = int(h * 2 / 3)
        roi = img[crop_h:h, :]
        roi_blurred = cv2.GaussianBlur(roi,(5, 5), 0)
        gray = cv2.cvtColor(roi_blurred,cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)
        roi_enhanced = cv2.cvtColor(enhanced_gray,cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(roi_enhanced,cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 110])
        upper_white = np.array([180, 60, 255])
        mask = cv2.inRange( hsv,lower_white,upper_white)
        kernel = np.ones((5, 5),np.uint8)
        mask = cv2.erode(
            mask,
            kernel,
            iterations=1
        )
        mask = cv2.dilate(
            mask,
            kernel,
            iterations=2)
        side_crop = 25
        mask[:, :side_crop] = 0
        mask[:, w - side_crop:] = 0
        return mask, roi

    def check_is_ramp(self, mask):
        roi_h, roi_w = mask.shape
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False
        largest_cnt = max(contours,key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        total_area = roi_h * roi_w
        blob_ratio = float(area) / float(total_area + 1e-6)
        bx, by, bw, bh = cv2.boundingRect(largest_cnt)
        aspect_ratio = float(bw) / float( bh + 1e-6)
        is_ramp = (
            (blob_ratio > 0.20)
            and (bh > int(roi_h * 0.35))
            and (aspect_ratio < 2.5)
        )
        return is_ramp

    def largest_blob_ratio(self, mask):
        roi_h, roi_w = mask.shape
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        largest_cnt = max( contours,key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        return float(area) / float(roi_h * roi_w + 1e-6)
    def process_nearest_line(self, mask, img_w):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        ref_x = (
            self.last_outdoor_cx
            if self.last_outdoor_cx is not None
            else (img_w / 2.0)
        )
        best_cx = None
        min_dist = float('inf')
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 45:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10']/ M['m00'])
                    dist = abs(cx - ref_x)
                    if dist < min_dist:
                        min_dist = dist
                        best_cx = cx
        if best_cx is not None:
            self.last_outdoor_cx = best_cx
        return best_cx
    def process_tunnel_center_line(self, img):
        h, w, _ = img.shape
        crop_h = int(h * 0.6)
        roi = img[crop_h:h, :]
        roi_h, roi_w, _ = roi.shape
        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        _, mask = cv2.threshold(enhanced,175,255,cv2.THRESH_BINARY)
        margin = int(roi_w * 0.20)
        mask[:, :margin] = 0
        mask[:, roi_w - margin:] = 0
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        best_cx = None
        min_dist = float('inf')
        ref_x = (
            self.last_tunnel_cx
            if self.last_tunnel_cx is not None
            else (roi_w / 2.0)
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 35:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = (float(bw) / float(bh) if bh > 0else 99.0)
                if aspect_ratio > 2.2:
                    continue
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10']/ M['m00'])
                    dist = abs(cx - ref_x)
                    if dist < min_dist:
                        min_dist = dist
                        best_cx = cx
        if best_cx is not None:
            self.last_tunnel_cx = best_cx
        return best_cx, mask, roi
    def detect_blue_car_camera(self, img):
        h, w, _ = img.shape
        roi = img[int(h * 0.3):int(h * 0.8),:]
        hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 120, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv,lower_blue,upper_blue)
        area = cv2.countNonZero(mask)
        total_area = (roi.shape[0]* roi.shape[1])
        score = float(area) / float(total_area)
        has_blue_car = score > 0.035
        return has_blue_car, score
    def control(self):
        if self.image is None:
            self.stop()
            return

        now = time.time()
        is_any_stop_active = (
            self.is_stopped_for_sign
            or self.is_stopped_for_light
            or self.post_stop_turn_active)
        if self.is_on_ramp:
            if is_any_stop_active:
                if self.ramp_pause_start is None:
                    self.ramp_pause_start = now
            elif self.ramp_pause_start is not None:
                paused_duration = (now - self.ramp_pause_start)
                self.ramp_start_time += paused_duration
                self.ramp_pause_start = None
                self.get_logger().info(
                    f">>> RAMP TIMER RESUME: "
                    f"Bỏ qua {paused_duration:.2f}s "
                    f"robot đã dừng <<<")
        vis_img = self.image.copy()
        h, w, _ = self.image.shape
        sig = str(self.current_traffic_signal)
        # ====================================================================
        # 0. QUẢN LÝ DỪNG PHANH XE
        # ====================================================================

        if self.is_stopped_for_sign:
            elapsed = (now- self.sign_stop_start_time)
            remaining = (self.STOP_SIGN_DURATION- elapsed)
            if remaining > 0:
                self.stop()
                self.publish_status("STOPPED_SIGN")
                cv2.putText(
                    vis_img,
                    f"STOP SIGN: STOPPED ({remaining:.1f}s)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                cv2.imshow("Robot Debug View",vis_img)
                cv2.waitKey(1)
                return
            else:
                self.is_stopped_for_sign = False
                self.stop_sign_cooldown_until = (now+ self.STOP_SIGN_COOLDOWN)
                # KÍCH HOẠT NHÍCH TRÁI
                self.post_stop_turn_active = True
                self.post_stop_turn_start = now
                self.get_logger().info(
                    f">>> HẾT 3S DỪNG BIỂN STOP "
                    f"-> KÍCH HOẠT NHÍCH TRÁI "
                    f"{self.POST_STOP_TURN_DURATION}S "
                    f"ĐỂ RẼ TRÁI <<<"
                )
        if self.is_stopped_for_light:
            elapsed = (now- self.light_stop_start_time)
            remaining = (self.light_stop_duration- elapsed)
            if (remaining > 0 and ("GREEN" not in sig)):
                self.stop()
                self.publish_status("STOPPED_LIGHT")
                cv2.putText(
                    vis_img,
                    f"TRAFFIC LIGHT: STOPPED ({remaining:.1f}s)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                cv2.imshow("Robot Debug View",vis_img)
                cv2.waitKey(1)
                return
            else:
                self.is_stopped_for_light = False
                self.light_cooldown_until = (now+ self.LIGHT_COOLDOWN)
                self.get_logger().info(
                    ">>> HẾT THỜI GIAN DỪNG ĐÈN / "
                    "ĐÈN XANH -> TIẾP TỤC DÒ LINE <<<")
        # --- 0.3 TRẠNG THÁI NHÍCH BẺ LÁI SANG TRÁI ---
        if self.post_stop_turn_active:
            elapsed_turn = (now- self.post_stop_turn_start)
            if (elapsed_turn< self.POST_STOP_TURN_DURATION):
                self.drive(self.max_speed * 0.8,0.35)
                self.publish_status("POST_STOP_LEFT_NUDGE")
                cv2.putText(
                    vis_img,
                    "POST-STOP: NUDGING LEFT "
                    f"({self.POST_STOP_TURN_DURATION - elapsed_turn:.1f}s)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                cv2.imshow("Robot Debug View",vis_img)
                cv2.waitKey(1)
                return
            else:
                self.post_stop_turn_active = False
                self.last_outdoor_cx = None
                self.last_error = 0.0
                self.get_logger().info(
                    ">>> HOÀN THÀNH NHÍCH TRÁI "
                    "-> TRỞ VỀ DÒ LINE GẦN NHẤT <<<"
                )
        # Visual Overlay
        cv2.putText(
            vis_img,
            f"TRAFFIC SIG: {self.current_traffic_signal}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )
        # --------------------------------------------------------------------
        # 1. NHẬN BIẾT DỐC CẦU VÀ HẦM
        # --------------------------------------------------------------------
        mask_normal, roi_normal = (self.process_image_mask(self.image))
        top_brightness = np.mean(self.image[:int(h / 3), :])
        is_moving_normally = not (
            self.is_stopped_for_sign
            or self.is_stopped_for_light
            or self.post_stop_turn_active
        )
        is_ramp_like = (
            self.check_is_ramp(mask_normal)and is_moving_normally)
        is_tunnel_like = (
            top_brightness < 80.0
            and (self.largest_blob_ratio( mask_normal) > 0.15 or self.is_on_ramp))
        # KÍCH HOẠT DỐC CẦU
        if (not self.is_on_ramp and not self.is_in_tunnel and is_ramp_like):
            self.is_on_ramp = True
            self.ramp_start_time = now
            # Reset trạng thái đóng băng
            self.ramp_pause_start = None
            self.stuck_since = None
            self.get_logger().info('>>> KÍCH HOẠT DỐC CẦU (RAMP) ''-> TẮT LIDAR! <<<')
        if (
            self.is_on_ramp
            and is_ramp_like
            and not is_any_stop_active):
            self.ramp_start_time = now
        # KÍCH HOẠT HẦM
        if (not self.is_in_tunnel and is_tunnel_like):
            self.is_in_tunnel = True
            self.tunnel_start_time = now
            self.stuck_since = None
            self.last_tunnel_cx = None
            self.get_logger().info(
                f'>>> KÍCH HOẠT VÀO HẦM '
                f'(top_bright={top_brightness:.1f}) '
                f'-> TẮT LIDAR! <<<')
        # THOÁT DỐC CẦU
        if (self.is_on_ramp and (now - self.ramp_start_time > self.RAMP_DURATION)):
            self.is_on_ramp = False
            self.ramp_pause_start = None
            self.get_logger().info('>>> THOÁT DỐC CẦU ''-> BẬT LẠI LIDAR <<<')
        # THOÁT HẦM
        if (self.is_in_tunnel and (now - self.tunnel_start_time> self.TUNNEL_DURATION)):
            self.is_in_tunnel = False
            self.last_tunnel_cx = None
            self.last_outdoor_cx = None
            self.last_error = 0.0
            self.prev_angular_z = 0.0
            self.hard_turn_confirm = 0
            self.get_logger().info('>>> THOÁT HẦM: RESET ERROR ''& TRANSITION OUTDOOR <<<')
        # --------------------------------------------------------------------
        # 2. XỬ LÝ LIDAR
        # --------------------------------------------------------------------
        disable_lidar = (self.is_on_ramp or self.is_in_tunnel)
        has_blue_car, blue_score = (self.detect_blue_car_camera(self.image))
        # --------------------------------------------------------------------
        # 3. MÁY TRẠNG THÁI VƯỢT XE
        # --------------------------------------------------------------------
        if (has_blue_car
            and self.overtake_state == "IDLE"
            and not disable_lidar):

            self.overtake_state = "LANE_CHANGE"
            self.overtake_start_time = now
            self.get_logger().info(">>> KÍCH HOẠT CHU TRÌNH VƯỢT XE XANH <<<")
        if self.overtake_state != "IDLE":
            self.publish_status(f"OVERTAKE_{self.overtake_state}")
            elapsed = (now- self.overtake_start_time)
            if self.overtake_state == "LANE_CHANGE":
                if elapsed < self.TIME_LANE_CHANGE:
                    self.drive(self.max_speed * 0.8,0.45)
                    cv2.putText(vis_img,"OVERTAKE: LATCHING LEFT",(20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 255),2)
                else:
                    self.overtake_state = "PASSING"
                    self.overtake_start_time = now
            elif self.overtake_state == "PASSING":
                if elapsed < self.TIME_PASSING:
                    self.drive(self.max_speed, 0.0)
                    cv2.putText( vis_img,"OVERTAKE: KEEP STRAIGHT & PASSING",(20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)
                else:
                    self.overtake_state = "RETURN_LANE"
                    self.overtake_start_time = now
            elif self.overtake_state == "RETURN_LANE":
                if elapsed < self.TIME_RETURN_LANE:
                    self.drive(self.max_speed * 0.8,-0.30)
                    cv2.putText( vis_img,"OVERTAKE: RETURNING TO LANE",(20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)
                else:
                    self.overtake_state = "IDLE"
                    self.last_outdoor_cx = None
                    self.last_error = 0.0
                    self.get_logger().info(">>> HOÀN THÀNH VƯỢT XE " "- TRỞ VỀ DÒ LINE <<<")
            cv2.imshow("Robot Debug View",vis_img)
            cv2.waitKey(1)
            return
        # -------------------------------------------------------------------
        # 4. XỬ LÝ DỪNG VẬT CẢN (LIDAR) & WATCHDOG
        # --------------------------------------------------------------------
        if not disable_lidar:
            min_front_dist, closest_info = (self.draw_lidar_overlay(vis_img,self.stop_distance))
            will_stop = (min_front_dist < self.stop_distance)
            if will_stop:
                if self.stuck_since is None:
                    self.stuck_since = now
                    self.stuck_ref_dist = (min_front_dist)
                stuck_elapsed = (now- self.stuck_since)
                dist_stable = (abs(min_front_dist- self.stuck_ref_dist) < 0.05)
            else:
                self.stuck_since = None
                stuck_elapsed = 0.0
                dist_stable = False
            if (will_stop and stuck_elapsed > self.STUCK_TIMEOUT and dist_stable):
                self.get_logger().warn(f'>>> STUCK-ESCAPE: Kẹt 'f'{stuck_elapsed:.1f}s 'f'-> Chuyển TUNNEL MODE <<<')
                self.is_in_tunnel = True
                self.tunnel_start_time = now
                self.stuck_since = None
                will_stop = False
                disable_lidar = True
            if will_stop:
                self.stop()
                self.publish_status("STOP_LIDAR_OBSTACLE")
                self.log_every(1.0,f'Obstacle Stop at {min_front_dist:.2f}m')
                cv2.putText(vis_img,f"OBSTACLE STOP ({min_front_dist:.2f}m)",(20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),2)
                cv2.imshow( "Robot Debug View",vis_img)
                cv2.waitKey(1)
                return
        else:
            self.stuck_since = None
            cv2.putText(vis_img,"LIDAR IGNORED (RAMP/TUNNEL MODE)",(20, h - 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 255),1)
        # --------------------------------------------------------------------
        # 5. ĐIỀU KHIỂN BÁM LINE GẦN NHẤT
        # --------------------------------------------------------------------
        if self.is_in_tunnel:
            self.publish_status("TUNNEL")
            tunnel_cx, tunnel_mask, _ = (self.process_tunnel_center_line(self.image))
            if tunnel_cx is not None:
                image_center = w / 2.0
                error = (tunnel_cx - image_center)
                Kp = 0.0050
                Kd = 0.008
                derivative = ( error- self.last_error)
                self.last_error = error
                angular_z = -float(error * Kp + derivative * Kd)
                angular_z = max(-0.45,min(0.45, angular_z))
                self.drive(self.max_speed * 0.8,angular_z)
                cv2.circle(vis_img,(tunnel_cx,int(h * 0.75)),8,(255, 0, 255),-1)
                cv2.putText(vis_img,f"TUNNEL TRACKING "f"(cx={tunnel_cx}, "f"err={error:.1f}px)",(20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.6,
                    (255, 0, 255),2)
            else:
                self.drive(self.max_speed * 0.5,0.0)
                cv2.putText(
                    vis_img,
                    "TUNNEL: SEARCHING CENTER LINE (0.0)",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )
        else:
            cx = self.process_nearest_line(mask_normal,w)
            if cx is not None:
                image_center = w / 2.0
                raw_error = (cx- image_center)
                abs_error = abs(raw_error)
                if abs_error >= 60.0:
                    self.hard_turn_confirm = min(
                        self.hard_turn_confirm + 1,99)
                else:
                    self.hard_turn_confirm = 0
                is_confirmed_hard_turn = (self.hard_turn_confirm >= 3)
                if is_confirmed_hard_turn:
                    current_speed = (self.max_speed * 0.55)
                    Kp = 0.0055
                    Kd = 0.0080
                    max_turn_limit = 0.55
                    error = raw_error
                else:
                    current_speed = (self.max_speed)
                    Kp = 0.0035
                    Kd = 0.0060
                    max_turn_limit = 0.35
                    if abs_error < 15.0:
                        error = 0.0
                    else:
                        error = raw_error
                derivative = (error- self.last_error)
                self.last_error = error
                raw_angular = -float(error * Kp+ derivative * Kd)
                angular_z = max(-max_turn_limit,
                    min(max_turn_limit,raw_angular))
                turn_clear_dist = (self.check_turn_path_clear(angular_z))
                if (turn_clear_dist < 0.30 and not disable_lidar):
                    current_speed = min(current_speed,self.max_speed * 0.3)
                    angular_z *= 0.6
                    self.log_every(
                        0.5,
                        f'[TURN GUARD] '
                        f'Vật cản hướng rẽ '
                        f'{turn_clear_dist:.2f}m '
                        f'-> Giảm tốc/lái'
                    )

                    cv2.putText(
                        vis_img,
                        f"TURN GUARD! "
                        f"dist={turn_clear_dist:.2f}m",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 128, 255),
                        2
                    )
                self.drive(current_speed,angular_z)
                cv2.circle(
                    vis_img,
                    (cx,int(h * 5 / 6)),8,(0, 255, 0),-1)
                mode_str = (
                    "HARD_TURN"
                    if is_confirmed_hard_turn
                    else (
                        "RAMP"
                        if self.is_on_ramp
                        else "NORMAL"))
                self.publish_status(f"NEAREST_{mode_str}")
                cv2.putText(
                    vis_img,
                    f"TRACKING NEAREST LINE "
                    f"(cx={cx}, "
                    f"Err={error:.1f}px)",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            else:
                self.publish_status("SEARCHING_LINE")
                self.drive(self.max_speed * 0.5,0.0)
                cv2.putText(
                    vis_img,
                    "FALLBACK: SEARCHING NEAREST LINE (0.0)",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )
        cv2.imshow("Robot Debug View",vis_img)
        cv2.waitKey(1)
def main(args=None):
    rclpy.init(args=args)
    node = Starter()
    try:
        rclpy.spin(node)
    except (
        KeyboardInterrupt,
        ExternalShutdownException):
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
