import cv2
import datetime
import mediapipe as mp
import os
import configparser

import functions

# 讀取 INI 設定檔
config = configparser.ConfigParser()
config.read('config.ini')

# 檢查是否為第一次開啟
if not config.has_section('Settings'):
    config.add_section('Settings')
    config.set('Settings', 'first_run', 'True')
    config.set('Settings', 'available_resolutions', '')
    config.set('Settings', 'preferred_resolution', '')

    if os.path.exists('initializing.jpg'):
        img = cv2.imread('initializing.jpg')
        if img is not None:
            cv2.namedWindow('Initializing...')
            cv2.imshow('Initializing...', img)
            

        else:
            print("Error reading image file")
    else:
        print("Image file not found")

    # 初始化可用解析度
    available_resolutions = functions.list_camera_resolutions()
    cv2.destroyAllWindows()
    # 儲存 INI 設定檔
    preferred_resolution = (640, 480)  # 預設解析度
    config.set('Settings', 'available_resolutions', str(available_resolutions))
    config.set('Settings', 'preferred_resolution', str(available_resolutions[0]))

    # 創建一個視窗告知使用者正在初始化


    # 儲存 INI 設定檔
    with open('config.ini', 'w') as f:
        config.write(f) 

# 讀取可用解析度和偏好解析度從 INI 檔案
available_resolutions = eval(config.get('Settings', 'available_resolutions'))
preferred_resolution = eval(config.get('Settings', 'preferred_resolution'))

# 設定攝影機捕捉物件以偏好解析度
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_resolution[1])

capture_folder = 'Capture'
if not os.path.exists(capture_folder):
    os.makedirs(capture_folder)

# 設定錄影格式
frame_size = preferred_resolution
fi = (preferred_resolution[0], preferred_resolution[1])
resolution_index = available_resolutions.index(fi)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = 30.0

# 設定人物偵測器
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 設定錄影狀態
recording = False
last_detection_time = 0
out = None
show_bounding_box = False
detection_mode = False

while True:
    # 讀取攝影機畫面
    ret, frame = cap.read()
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #偵測人物


    if detection_mode:
        #偵測人物
    
        # 檢查是否偵測到人物
        if results.pose_landmarks:
            # 設定錄影狀態
            recording = True
            last_detection_time = datetime.datetime.now().timestamp()
            if out is None:
                # 設定錄影檔案名稱
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                out = cv2.VideoWriter(os.path.join(capture_folder, filename + '.mp4'), fourcc, fps, frame_size)

        else:
            # 檢查是否超過30秒沒有偵測到人物
            if recording and datetime.datetime.now().timestamp() - last_detection_time > 30:
                recording = False
                if out is not None:
                    out.release()
                    out = None
    else:
        # 顯示待機模式
        cv2.putText(frame, "Standby Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType = cv2.LINE_AA)
        recording = False
        if out is not None:
            out.release()
            out = None

    # 繪製人物框線
    if show_bounding_box:
        landmarks = results.pose_landmarks.landmark
        x_min = min(landmark.x for landmark in landmarks)
        y_min = min(landmark.y for landmark in landmarks)
        x_max = max(landmark.x for landmark in landmarks)
        y_max = max(landmark.y for landmark in landmarks)
        cv2.rectangle(frame, (int(x_min * frame.shape[1]), int(y_min * frame.shape[0])), 
                    (int(x_max * frame.shape[1]), int(y_max * frame.shape[0])), (0, 255, 0), 2)


    # 顯示錄影狀態
    if recording and detection_mode:
        cv2.putText(frame, "Recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType = cv2.LINE_AA)
    elif detection_mode and not recording:
        cv2.putText(frame, "Not Recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType = cv2.LINE_AA)

    # 顯示時間
    cv2.putText(frame, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (frame.shape[1] - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType = cv2.LINE_AA)

    # 顯示Show Bounding Box
    cv2.putText(frame, "Show Bounding Box: " + str(show_bounding_box), (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType = cv2.LINE_AA)

    # 顯示畫質、幀率
    cv2.putText(frame, str(frame_size[0]) + 'x' +str(frame_size[1]) + '  30fps', (frame.shape[1] - 175, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType = cv2.LINE_AA)

    # 顯示畫面
    cv2.imshow('Motion Detection CCTV', frame)

    # 錄影
    if recording and out is not None:
        out.write(frame)

    # 等待按鍵
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        show_bounding_box = not show_bounding_box
    elif key == ord('r'):
        detection_mode = not detection_mode
    elif key == ord('c'):
        cap.release()
        resolution_index = (resolution_index + 1) % len(available_resolutions)
        frame_size = available_resolutions[resolution_index]
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

        # 更新偏好解析度在 INI 檔案
        config.set('Settings', 'preferred_resolution', f"({frame_size[0]}, {frame_size[1]})")
        with open('config.ini', 'w') as f:
            config.write(f)

# 關閉攝影機和錄影
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()