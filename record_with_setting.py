import cv2
import datetime
import mediapipe as mp

# 啟動攝影機
cap = cv2.VideoCapture(0)

# 設定錄影格式
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30.0
# frame_size = (int(cap.get(3)), int(cap.get(4)))
frame_size = (1920, 1080)

# 設定人物偵測器
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 設定錄影狀態
recording = False
last_detection_time = 0
out = None

# 新增一個勾選框
show_bounding_box = False

while True:
    # 讀取攝影機畫面
    ret, frame = cap.read()

    #偵測人物
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 檢查是否偵測到人物
    if results.pose_landmarks:
        # 設定錄影狀態
        recording = True
        last_detection_time = datetime.datetime.now().timestamp()
        if out is None:
            # 設定錄影檔案名稱
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out = cv2.VideoWriter(filename + '.avi', fourcc, fps, frame_size)

        # 繪製人物框線
        if show_bounding_box:
            landmarks = results.pose_landmarks.landmark
            x_min = min(landmark.x for landmark in landmarks)
            y_min = min(landmark.y for landmark in landmarks)
            x_max = max(landmark.x for landmark in landmarks)
            y_max = max(landmark.y for landmark in landmarks)
            cv2.rectangle(frame, (int(x_min * frame.shape[1]), int(y_min * frame.shape[0])), 
                        (int(x_max * frame.shape[1]), int(y_max * frame.shape[0])), (0, 255, 0), 2)
    else:
        # 檢查是否超過30秒沒有偵測到人物
        if recording and datetime.datetime.now().timestamp() - last_detection_time > 30:
            recording = False
            if out is not None:
                out.release()
                out = None

    # 顯示錄影狀態
    if recording:
        cv2.putText(frame, "Recording", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Not Recording", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 顯示時間
    cv2.putText(frame, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 顯示勾選框
    cv2.putText(frame, "顯示人物框線: " + str(show_bounding_box), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 顯示畫面
    cv2.imshow('frame', frame)

    # 錄影
    if recording and out is not None:
        out.write(frame)

    # 等待按鍵
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        show_bounding_box = not show_bounding_box

# 關閉攝影機和錄影
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()