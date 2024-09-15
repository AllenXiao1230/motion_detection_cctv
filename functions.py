import cv2

def list_camera_resolutions():
    # 打開鏡頭，index 0 表示第一個鏡頭
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("無法開啟鏡頭")
        return

    # 嘗試常見的 16:9 解析度
    resolutions_16_9 = [
        (640, 360),    # 360p
        (1280, 720),   # HD 720p
        (1920, 1080),  # Full HD 1080p

    ]
    
    supported_resolutions = []

    for width, height in resolutions_16_9:
        # 直接設置並檢查解析度
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 立即獲取實際設置的解析度
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # 如果設置成功，則記錄支持的解析度
        if actual_width == width and actual_height == height:
            supported_resolutions.append((int(actual_width), int(actual_height)))

    cap.release()

    if supported_resolutions:
            print(supported_resolutions)
            return supported_resolutions
    else:
        print("無可用 16:9 解析度")
        return "error"

