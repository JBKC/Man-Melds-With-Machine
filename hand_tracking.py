'''
Main hand tracking script acting as mouse control
Originally designed to run on Raspberry Pi + stream output to laptop
Run from Terminal
'''

import cv2
import mediapipe as mp
import serial
import math

# initialize serial communication
serial_port = serial.Serial('/dev/ttyGS0', 9600, timeout=1)

# initialise mediapipe
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

def dist(lm1, lm2, w, h):
    '''
    Calculate Euclidian distance between 2 landmarks
    '''
    dx = (lm1.x - lm2.x) * w
    dy = (lm1.y - lm2.y) * h
    return math.sqrt(dx ** 2 + dy ** 2)

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

        INDEX_ID = 8  # use tip of index finger as reference point for movement
        THUMB_ID = 4  # use tip of thumb & little finger to register mouse clicks
        LITTLE_ID = 20
        THRESH = 20   # distance threshold to register click

        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_info.classification[0].label  # left vs right hand

            loc = hand_landmarks.landmark[INDEX_ID]

            # normalise coordinates between (0,0) and (1,1)
            # flip both axes
            x_loc = 1.0 - loc.x
            y_loc = 1.0 - loc.y          # flip y axis

            # handle mirroring
            if hand_label == "Left":
                hand_label = "R"
            else:
                hand_label = "L"

            # print(f"{hand_label}: x={x_loc:.2f}, y={y_loc:.2f}")

            # detect mouse clicks
            click = dist(hand_landmarks.landmark[THUMB_ID], hand_landmarks.landmark[LITTLE_ID], w, h)
            if click < THRESH:
                serial_port.write(b"click\n")

            data = f"{hand_label},{x_loc:.2f},{y_loc:.2f}\n"
            serial_port.write(data.encode())


    # Display the frame
    # cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()