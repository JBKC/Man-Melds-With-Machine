'''
Main script called from Docker container
'''

import cv2
import mediapipe as mp
import time
import socket

def main():
    print("started")

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    # Socket setup for forwarding the video
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", 8080))  # Bind to all interfaces on port 8080
    server_socket.listen(1)

    print("Waiting for client to connect...")
    client_socket, addr = server_socket.accept()
    print(f"Client connected: {addr}")

    # Forward the webcam feed
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with MediaPipe Hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw hands landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Send frame to client
        _, encoded_frame = cv2.imencode(".jpg", frame)
        client_socket.sendall(encoded_frame.tobytes())

    # Cleanup
    cap.release()
    client_socket.close()
    server_socket.close()
    print("Finished")



if __name__ == '__main__':
    main()


