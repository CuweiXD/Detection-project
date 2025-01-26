import cv2
from ultralytics import YOLO
import time
import torch
import numpy as np

#檢查 GPU 是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#載入預訓練模型並將其移至 GPU（如果可用）
model_path = 'C:/Users/user/Videos/best_model.pt'
model = YOLO(model_path).to(device)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

output_path = 'C:/Users/user/Videos/output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:

    start_time = time.time()
    ret, frame = cap.read()

    if not ret:
        print("攝影機無法讀取影像")
        break

    # 使用 YOLO 模型進行預測
    results = model.predict(source=frame, save=False, save_txt=False, imgsz=416, conf=0.5, verbose=False, device=device)

    # 繪製標記
    annotated_frame = results[0].plot()

    

    for result in results[0].boxes:
        
        x1,y1,x2,y2 = map(int, result.xyxy[0])

        CenterX = int((x1+x2)/2)
        CenterY = int((y1+y2)/2)

        cv2.circle(annotated_frame,(CenterX,CenterY),5,(0,0,255),-1) 


    end_time = time.time()
    fps_text = 1 / (end_time - start_time)

    # 在影像上顯示 FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps_text:.2f}",
        (10, 30),                           # 文字的位置 (左上角)
        cv2.FONT_HERSHEY_SIMPLEX,           # 字型
        1,                                  # 字體大小
        (0, 255, 0),                        # 字體顏色 (BGR) 綠色
        2,                                  # 線條粗細
        cv2.LINE_AA                         # 抗鋸齒
    )

    # 保存處理過的影像到影片檔案
    out.write(annotated_frame)

    # 顯示影像
    cv2.imshow('YOLO Detection', annotated_frame)

    #print(f"FPS: {fps_text:.2f}")

    if cv2.waitKey(1) == 27:

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"保存於: {output_path}")
        break