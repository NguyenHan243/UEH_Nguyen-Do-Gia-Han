#!/usr/bin/env python3
"""
traffic_perception_test.py
---------------------------------------------------------------
Node nhận diện thị giác (Traffic Perception Node) - TẦM NHÌN XA TỐI ƯU:
  - Nhận diện Biển báo STOP từ xa (khi xe đi giữa làn đường, diện tích biển >= 120px).
  - Soi chữ "STOP" trắng chính xác từ khoảng cách xa.
  - Publish tín hiệu chuẩn 'SIGN_STOP' để start.py lưu cờ Pending chờ vạch ngang.
---------------------------------------------------------------
"""

import math
import time

import cv2
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from cv_bridge import CvBridge
    HAVE_CV = True
except ImportError:
    HAVE_CV = False


class TrafficPerceptionNode(Node):

    def __init__(self):
        super().__init__('traffic_perception_node')

        self.image = None
        self.bridge = CvBridge() if HAVE_CV else None

        self.last_light_state = None
        self.last_light_time = 0.0
        self.red_stop_start_time = None

        self.stop_sign_timer = 0.0
        self.stop_sign_cooldown_until = 0.0

        self.sub_image = self.create_subscription(
            Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data
        )
        self.pub_signal = self.create_publisher(String, '/traffic_signal', 10)

        self.create_timer(1.0 / 20.0, self.tick)
        self.get_logger().info('=== TRAFFIC PERCEPTION NODE (FAR DISTANCE DETECTION) ===')

    def on_image(self, msg):
        if self.bridge is None:
            return
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')

    def tick(self):
        if self.image is None:
            return
        try:
            self.process_frame()
        except Exception as e:
            self.get_logger().error(f'process_frame() raised: {e}')

    # ===================================================================
    # SOI CHỮ TRẮNG "STOP" TỪ KHỎANG CÁCH XA (HẠ NGƯỠNG AREA >= 3px)
    # -- GIỮ NGUYÊN 100% NHƯ BẢN GỐC, vì hàm này vẫn đang được dùng trong
    #    detect_traffic_lights() để loại trừ chữ STOP khỏi bị hiểu nhầm là
    #    đèn đỏ - không liên quan tới lỗi biển cảnh báo bị nhận nhầm STOP --
    # ===================================================================
    def has_white_stop_text(self, img, box):
        x, y, w, h = box
        pad = 2
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_white = [c for c in white_contours if cv2.contourArea(c) >= 3]
        return len(valid_white) >= 1

    # ===================================================================
    # [HÀM MỚI - THÊM ĐỂ SỬA LỖI NHẬN NHẦM BIỂN] ĐO TỈ LỆ % DIỆN TÍCH
    # ĐỎ/TRẮNG CỦA BIỂN. Dùng TỈ LỆ (không phải số pixel tuyệt đối) nên
    # không đổi theo khoảng cách -> vẫn detect được từ xa như yêu cầu gốc,
    # nhưng phân biệt được STOP (khối đỏ đặc, red~0.7-0.9) với biển cảnh
    # báo tam giác (red~0.48-0.58, white~0.38-0.42) - số liệu đo trực tiếp
    # từ ảnh thật bạn cung cấp.
    # ===================================================================
    def color_area_ratios(self, img, cnt):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        total = cv2.countNonZero(mask)
        if total < 12:
            return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        region = hsv[mask == 255]
        h_ch, s_ch, v_ch = region[:, 0], region[:, 1], region[:, 2]
        red = np.count_nonzero(((h_ch < 10) | (h_ch > 170)) & (s_ch > 80) & (v_ch > 60))
        white = np.count_nonzero((s_ch < 60) & (v_ch > 150))
        return {'red': red / float(total), 'white': white / float(total)}

    # ===================================================================
    # PHẦN 1: NHẬN DIỆN ĐÈN GIAO THÔNG
    # ===================================================================
    def detect_traffic_lights(self, img, vis_img):
        results = []
        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        r1 = cv2.inRange(hsv, (0, 70, 130), (12, 255, 255))
        r2 = cv2.inRange(hsv, (165, 70, 130), (180, 255, 255))
        red_mask = cv2.bitwise_or(r1, r2)
        yellow_mask = cv2.inRange(hsv, (13, 50, 130), (35, 255, 255))
        green_mask = cv2.inRange(hsv, (35, 25, 130), (95, 255, 255))

        color_masks = [('RED', red_mask), ('YELLOW', yellow_mask), ('GREEN', green_mask)]

        for color_name, mask in color_masks:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)

                if cy < 15 or cx < 15 or cx > (w - 15):
                    continue

                area = cv2.contourArea(cnt)
                if radius < 7.0 or area < 50:
                    continue

                peri = cv2.arcLength(cnt, True)
                circularity = 4 * math.pi * area / (peri * peri + 1e-6)
                if circularity < 0.55:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)

                pad_x, pad_y = int(bw * 1.5), int(bh * 2.5)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)

                roi_gray = gray[y1:y2, x1:x2]
                if roi_gray.size == 0:
                    continue

                dark_pixels = np.sum(roi_gray < 85)
                dark_ratio = dark_pixels / float(roi_gray.size)

                if color_name == 'RED' and self.has_white_stop_text(img, (x, y, bw, bh)):
                    continue

                if dark_ratio > 0.28:
                    box_w, box_h = int(bw * 2.4), int(bh * 4.2)
                    box_x = max(0, int(cx - box_w / 2))

                    if color_name == 'RED':
                        box_y = max(0, int(cy - box_h * 0.2))
                    elif color_name == 'YELLOW':
                        box_y = max(0, int(cy - box_h * 0.5))
                    else:
                        box_y = max(0, int(cy - box_h * 0.8))

                    draw_color = {'RED': (0, 0, 255), 'YELLOW': (0, 255, 255), 'GREEN': (0, 255, 0)}[color_name]

                    cv2.rectangle(vis_img, (box_x, box_y), (box_x + box_w, box_y + box_h), draw_color, 2)
                    cv2.putText(vis_img, f"LIGHT: {color_name}", (box_x, max(0, box_y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

                    results.append({'state': color_name, 'box': (box_x, box_y, box_w, box_h)})

        return results

    # ===================================================================
    # PHẦN 2: NHẬN DIỆN BIỂN BÁO (NHẬN TỪ XA KHI XE ĐI GIỮA LÀN)
    # ===================================================================
    def classify_shape(self, cnt):
        peri = cv2.arcLength(cnt, True)
        if peri < 18:
            return None
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        n = len(approx)

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(bh + 1e-6)
        if aspect < 0.5 or aspect > 2.0:
            return None

        area = cv2.contourArea(cnt)
        circularity = 4 * math.pi * area / (peri * peri + 1e-6)

        # [SỬA - CHỈ 2 NHÁNH NÀY, giữ nguyên thứ tự ưu tiên gốc OCTAGON->
        # TRIANGLE->SQUARE->CIRCLE] Thêm điều kiện circularity để loại trừ
        # tam giác bị nhiễu cạnh ở xa (n lệch thành 6/7) khỏi bị nhận nhầm
        # OCTAGON. Đo thực tế: STOP~0.93, tam giác cảnh báo chỉ ~0.56-0.58.
        if (n == 8 or n == 7 or n == 6) and circularity > 0.80:
            return 'OCTAGON'
        elif n == 3 or (n in (5, 6, 7) and circularity <= 0.72):
            return 'TRIANGLE'
        elif n == 4:
            return 'SQUARE'
        elif n >= 9:
            if circularity > 0.60:
                return 'CIRCLE'
        return None

    def classify_sign_color(self, img, cnt):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        pixels = hsv[mask == 255]
        if len(pixels) < 10:
            return 'UNKNOWN'

        sat_ok = pixels[pixels[:, 1] > 60]
        if len(sat_ok) < 10:
            return 'WHITE/BLACK'

        mean_hue = float(np.median(sat_ok[:, 0]))
        if mean_hue < 10 or mean_hue > 170:
            return 'COLOR_RED'
        elif 15 <= mean_hue < 35:
            return 'COLOR_YELLOW'
        elif 90 <= mean_hue < 130:
            return 'COLOR_BLUE'
        elif 40 <= mean_hue < 85:
            return 'COLOR_GREEN'
        return 'OTHER'

    SIGN_LABELS = {
        ('OCTAGON', 'COLOR_RED'): 'SIGN_STOP',
        ('CIRCLE', 'COLOR_RED'): 'PROHIBITION_SIGN',
        ('CIRCLE', 'COLOR_BLUE'): 'MANDATORY_SIGN',
        ('TRIANGLE', 'COLOR_RED'): 'WARNING_SIGN',
        ('TRIANGLE', 'COLOR_YELLOW'): 'WARNING_SIGN',
    }

    def detect_signs(self, img, vis_img, light_boxes):
        results = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 140)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # HẠ NGƯỠNG DIỆN TÍCH XUỐNG 120px ĐỂ NHẬN DIỆN BIỂN STOP TỪ XA
            if area < 120:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 10 or bh < 10:
                continue

            cx, cy = x + bw // 2, y + bh // 2
            inside_light = any(lx <= cx <= lx + lbw and ly <= cy <= ly + lbh for lx, ly, lbw, lbh in light_boxes)
            if inside_light:
                continue

            shape = self.classify_shape(cnt)
            if shape is None:
                continue

            # [SỬA] Thay logic cũ (SIGN_LABELS theo hue trung vị + đếm pixel
            # trắng tuyệt đối) bằng kiểm tra TỈ LỆ diện tích đỏ/trắng, vì đây
            # là nguồn gốc khiến biển cảnh báo (tam giác đỏ-trắng) bị gán
            # nhầm thành SIGN_STOP. classify_sign_color/SIGN_LABELS cho các
            # shape khác (CIRCLE, SQUARE...) vẫn giữ nguyên bên dưới.
            if shape in ('OCTAGON', 'TRIANGLE'):
                ratios = self.color_area_ratios(img, cnt)
                if ratios is None:
                    continue
                if shape == 'OCTAGON' and ratios['red'] > 0.60 and ratios['white'] < 0.32:
                    label = 'SIGN_STOP'
                    color = 'COLOR_RED'
                elif shape == 'TRIANGLE' and ratios['white'] > 0.25 and 0.30 < ratios['red'] < 0.65:
                    label = 'WARNING_SIGN'
                    color = 'COLOR_RED'
                else:
                    continue
            else:
                color = self.classify_sign_color(img, cnt)
                label = self.SIGN_LABELS.get((shape, color), 'WARNING_SIGN')

            cv2.rectangle(vis_img, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
            cv2.putText(vis_img, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

            results.append({'shape': shape, 'color': color, 'label': label, 'box': (x, y, bw, bh)})

        return results

    def detect_stop_line(self, img, vis_img):
        h, w, _ = img.shape
        roi_top = int(h * 0.78)
        roi_margin = int(w * 0.05)
        roi = img[roi_top:, roi_margin:w - roi_margin]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 175])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)

        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        thresh = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_horiz)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 350:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            abs_x = x + roi_margin
            abs_y = y + roi_top
            aspect_ratio = bw / float(bh + 1e-6)

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
                angle += 90.0

            while angle > 90: angle -= 180
            while angle < -90: angle += 180

            if bw > int(w * 0.35) and aspect_ratio > 3.0 and abs(angle) < 12.0:
                cv2.rectangle(vis_img, (abs_x, abs_y), (abs_x + bw, abs_y + bh), (255, 0, 255), 2)
                cv2.putText(vis_img, "STOP LINE OK", (abs_x, abs_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                return True, float(bw / w)

        return False, 0.0

    def process_frame(self):
        now = time.time()
        vis_img = self.image.copy()

        lights = self.detect_traffic_lights(self.image, vis_img)
        light_boxes = [l['box'] for l in lights]
        signs = self.detect_signs(self.image, vis_img, light_boxes)
        has_line, line_ratio = self.detect_stop_line(self.image, vis_img)

        if lights:
            self.last_light_state = lights[0]['state']
            self.last_light_time = now

        current_signal = "CLEAR"

        if lights:
            current_signal = lights[0]['state']
            self.red_stop_start_time = None
        elif has_line and self.last_light_state == 'RED' and (now - self.last_light_time) < 5.0:
            if self.red_stop_start_time is None:
                self.red_stop_start_time = now

            elapsed_red = now - self.red_stop_start_time

            if elapsed_red <= 4.5:
                current_signal = "RED"
                cv2.putText(vis_img, f"MEMORY: HOLD RED ({4.5 - elapsed_red:.1f}s)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                current_signal = "GREEN"
                self.last_light_state = 'GREEN'
                self.red_stop_start_time = None
        elif signs:
            current_signal = signs[0]['label']

        if has_line:
            current_signal += "+LINE"

        msg = String()
        msg.data = current_signal
        self.pub_signal.publish(msg)

        cv2.putText(vis_img, f"PUB SIGNAL: [{current_signal}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Traffic Perception NODE", vis_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficPerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()