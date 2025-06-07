import cv2
from ultralytics import YOLO
import time

# 加载 TensorRT 模型
model = YOLO("yolov8n.engine")

# 打开摄像头（你也可以设置为 1 或 其他）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

# 设置分辨率（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ 开始实时检测，按 'q' 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 读取帧失败")
        break

    start = time.time()

    # 推理
    results = model(frame, verbose=False)

    # 可视化检测框
    annotated_frame = results[0].plot()

    end = time.time()
    fps = 1 / (end - start)

    # 在图像上显示 FPS
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8-TensorRT Real-time", annotated_frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

