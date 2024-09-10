import cv2
import numpy as np

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# 設定方法
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# 人臉偵測設定
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)


# 執行人臉偵測
with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)               # 讀取攝影鏡頭
    while True:
        ret, frame = cap.read()             # 讀取影片的每一幀
        w = frame.shape[1]                  # 畫面寬度
        h = frame.shape[0]                  # 畫面高度
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_landmarker_result = landmarker.detect(mp_image)

        face_landmarks_list = face_landmarker_result.face_landmarks
        annotated_image = np.copy(frame)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
          face_landmarks = face_landmarks_list[idx]

          # Draw the face landmarks.
          face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
          ])

          solutions.drawing_utils.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks_proto,
              connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp.solutions.drawing_styles
              .get_default_face_mesh_tesselation_style())
          solutions.drawing_utils.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks_proto,
              connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp.solutions.drawing_styles
              .get_default_face_mesh_contours_style())
          solutions.drawing_utils.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks_proto,
              connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())


        #print(face_landmarker_result)
        if not ret:
            print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
            break
        cv2.imshow('oxxostudio', annotated_image)     # 如果讀取成功，顯示該幀的畫面
        if cv2.waitKey(10) == ord('q'):     # 每一毫秒更新一次，直到按下 q 結束
            break
    cap.release()                           # 所有作業都完成後，釋放資源
    cv2.destroyAllWindows()                 # 結束所有視窗
    