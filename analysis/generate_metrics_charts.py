import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_pid_tracking_error(df, output_dir):
    """Biểu đồ dao động sai số tâm làn (Error Tracking)"""
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['lane_error_px'], label='Cross-Track Error (px)', color='blue', alpha=0.7)
    plt.axhline(60, color='red', linestyle='--', label='Ngưỡng Cua Gắt (Hard Turn Threshold)')
    plt.axhline(-60, color='red', linestyle='--')
    plt.title('Biểu Đồ Lệch Tâm Theo Thời Gian Thực (Line Tracking Error)')
    plt.xlabel('Thời gian (s)')
    plt.ylabel('Sai số tâm (px)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pid_tracking_error.png'))
    plt.close()

def plot_speed_vs_steering(df, output_dir):
    """Biểu đồ phản ứng của vận tốc và góc đánh lái khi qua cua/gặp vật cản"""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(df['timestamp'], df['speed'], 'g-', label='Vận tốc (m/s)')
    ax2.plot(df['timestamp'], df['angular_z'], 'm-', label='Góc bẻ lái (rad/s)')
    ax1.set_xlabel('Thời gian (s)')
    ax1.set_ylabel('Vận tốc (m/s)', color='g')
    ax2.set_ylabel('Góc bẻ lái (rad/s)', color='m')
    plt.title('Phản Hồi Vận Tốc và Góc Lái Khi Nhận Diện Cua Gắt/Vật Cản')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_steering_response.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Xuất biểu đồ phân tích dữ liệu điều khiển robot.')
    parser.add_argument('--data', type=str, default='control_logs.csv', help='Đường dẫn tới file CSV dữ liệu log')
    parser.add_argument('--output', type=str, default='charts', help='Thư mục lưu ảnh đầu ra')
    args = parser.parse_args()

    # Tự động tạo thư mục đầu ra nếu chưa có
    os.makedirs(args.output, exist_ok=True)

    if not os.path.exists(args.data):
        print(f"Lỗi: Không tìm thấy file dữ liệu '{args.data}'. Vui lòng kiểm tra lại đường dẫn!")
    else:
        df = pd.read_csv(args.data)
        plot_pid_tracking_error(df, args.output)
        plot_speed_vs_steering(df, args.output)
        print(f"--> Đã xuất thành công các biểu đồ vào thư mục: {args.output}/")