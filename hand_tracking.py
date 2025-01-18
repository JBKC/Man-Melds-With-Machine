'''
Main hand tracking script acting as mouse control
Originally designed to run on Raspberry Pi + stream output to laptop
Run from Terminal
'''

import cv2
import mediapipe as mp

# intialise mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Get hand locations
    if results.multi_hand_landmarks:
        h, w, _ = frame.shape  # Get the dimensions of the frame

        INDEX_ID = 8                # use tip of index finger as reference point for movement

        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_info.classification[0].label  # left vs right hand

            loc = hand_landmarks.landmark[INDEX_ID]

            # Convert normalized coordinates to pixel coordinates
            x_pixel = int(loc.x * w)
            y_pixel = int(loc.y * h)

            # flip x-coordinates and labels to handle mirroring
            x_pixel = w - x_pixel
            if hand_label == "Left":
                hand_label = "Right"
            else:
                hand_label = "Left"

            print(f"{hand_label}: x={x_pixel}, y={y_pixel}")

    # Display the frame (optional)
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()