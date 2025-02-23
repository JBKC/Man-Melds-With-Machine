"""
Simplified version of hand_tracking.py for testing Mediapipe landmarking & responsiveness.
"""

import cv2
import mediapipe as mp
import math
from config import FRAME_SIZE, FPS


# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize camera
# cap = cv2.VideoCapture(0)               # built-in camera
cap = cv2.VideoCapture(1)               # iPhone camera

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE['width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE['height'])
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # low latency

# Open in fullscreen
window_name = "Hand Tracking"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def dist(lm1, lm2, w, h):
    """
    Calculate Euclidean distance between 2 landmarks
    """
    dx = (lm1.x - lm2.x) * w
    dy = (lm1.y - lm2.y) * h
    return math.sqrt(dx ** 2 + dy ** 2)


def main():
    """
    Main loop for hand tracking
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with Mediapipe
        results = hands.process(rgb_frame)

        # Draw landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        mirror = cv2.flip(frame, 1)
        cv2.imshow(window_name, mirror)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
