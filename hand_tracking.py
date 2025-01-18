'''
Main hand tracking script
Originally designed to run on Raspberry Pi + stream output to laptop
'''

import cv2
import mediapipe as mp

# Initialize Mediapipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure hand detection
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Open the camera feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB (Mediapipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Draw bounding boxes and get hand locations
    if results.multi_hand_landmarks:
        h, w, _ = frame.shape  # Get the dimensions of the frame

        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_info.classification[0].label  # "Left" or "Right"

            # Get bounding box coordinates
            x_min, x_max = 1, 0
            y_min, y_max = 1, 0

            for lm in hand_landmarks.landmark:
                x_min = min(x_min, lm.x)
                x_max = max(x_max, lm.x)
                y_min = min(y_min, lm.y)
                y_max = max(y_max, lm.y)

            # Convert normalized coordinates to pixel coordinates
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)

            # Flip x-coordinates to handle mirroring
            x_min, x_max = w - x_max, w - x_min

            # Adjust hand label for mirrored view
            if hand_label == "Left":
                hand_label = "Right"
            else:
                hand_label = "Left"

            # Print the hand's center location
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            print(f"{hand_label}: x={x_center}, y={y_center}")

    # Display the frame (optional)
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()