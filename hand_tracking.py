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

        # control commands
        MOVE_ID = 9   # experiment with reference point of movement
        THUMB_TIP = 4
        INDEX_TIP = 8
        THUMB_J = 3   # joint of thumb as reference for clicks

        # closed fist commands
        MIDDLE_TIP = 12
        RING_TIP = 16
        LITTLE_TIP = 20
        WRIST = 0

        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_info.classification[0].label  # left vs right hand

            loc = hand_landmarks.landmark[MOVE_ID]

            # normalise coordinates between (0,0) and (1,1)
            # flip both axes
            x_loc = 1.0 - loc.x
            y_loc = 1.0 - loc.y

            # handle mirroring
            if hand_label == "Left":
                hand_label = "R"
            else:
                hand_label = "L"

            # print(f"{hand_label}: x={x_loc:.2f}, y={y_loc:.2f}")

            # click = touch tips of thumb and index finger
            THRESH = dist(hand_landmarks.landmark[THUMB_TIP], hand_landmarks.landmark[THUMB_J], w, h)    # distance threshold to register click
            click = dist(hand_landmarks.landmark[THUMB_TIP], hand_landmarks.landmark[INDEX_TIP], w, h)
            if THRESH > click:
                serial_port.write(b"click\n")

            # exit code = close fist
            HAND_SIZE = dist(hand_landmarks.landmark[WRIST], hand_landmarks.landmark[MOVE_ID], w, h)
            if (
                    HAND_SIZE >
                    dist(hand_landmarks.landmark[WRIST], hand_landmarks.landmark[INDEX_TIP], w, h) and
                    HAND_SIZE >
                    dist(hand_landmarks.landmark[WRIST], hand_landmarks.landmark[MIDDLE_TIP], w, h) and
                    HAND_SIZE >
                    dist(hand_landmarks.landmark[WRIST], hand_landmarks.landmark[RING_TIP], w, h) and
                    HAND_SIZE >
                    dist(hand_landmarks.landmark[WRIST], hand_landmarks.landmark[LITTLE_TIP], w, h)
            ):
                serial_port.write(b"stop\n")

            data = f"{hand_label},{x_loc:.4f},{y_loc:.4f}\n"
            serial_port.write(data.encode())

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    # Display the (mirrored) frame
    mirror = cv2.flip(frame, 1)
    cv2.imshow("Hand Tracking", mirror)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()