import cv2
cap = cv2.VideoCapture(0)
# Set the resolution and codec
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create a VideoWriter object
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280, 720))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame as needed
    # ...

    # Write the frame to the output video
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()