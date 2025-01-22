'''
Main hand tracking script acting as mouse control
Originally designed to run on Raspberry Pi + stream output to laptop
Run from Terminal
'''

import cv2
import mediapipe as mp
import serial
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor
import struct

# initialize serial communication
serial_port = serial.Serial('/dev/ttyGS0', 115200, timeout=1)

# initialise mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

HAND_LANDMARKS = {
    'MOVE_ID': 9,                   # reference point of movement (base of third finger)
    'THUMB_TIP': 4,
    'INDEX_TIP': 8,
    'THUMB_J': 3,                   # reference for cliks (joint of thumb)
    'MIDDLE_TIP': 12,
    'RING_TIP': 16,
    'LITTLE_TIP': 20,
    'WRIST': 0,
}
FRAME_SIZE = {'width': 480, 'height': 270}

# intialise camera & optimise
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use V4L2 backend explicitly
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE['width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE['height'])
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                 # low latency
# open in fullscreen
# window_name = "Hand Tracking"
# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

previous_data = None
executor = ThreadPoolExecutor()


def dist(lm1, lm2, w, h):
    '''
    Calculate Euclidian distance between 2 landmarks
    '''
    dx = (lm1.x - lm2.x) * w
    dy = (lm1.y - lm2.y) * h
    return math.sqrt(dx ** 2 + dy ** 2)

async def process_frame(frame_queue, result_queue):
    """Process each frame to track hand"""

    while True:
        frame = await frame_queue.get()
        if frame is None:
            break

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get landmarks via mediapipe and append to Asyncio queue
        results = hands.process(rgb_frame)
        await result_queue.put(results)

async def send_data(result_queue):
    """Send data over serial communication."""

    global previous_data        # simple check to stop duplication of data for efficiency

    while True:
        results = await result_queue.get()          # retrieve landmarks from Asyncio queue
        if results is None:
            break

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):

                ## CASE 0 -> sending realtime hand movement coordinates
                # get and mirror hand labels (due to mirrored screen)
                hand_label = 'R' if hand_info.classification[0].label == "Left" else 'L'

                # reference point for hand movement
                loc = hand_landmarks.landmark[HAND_LANDMARKS['MOVE_ID']]
                # normalise coords and flip axes
                x_loc, y_loc = 1.0 - loc.x, 1.0 - loc.y

                # scale floats to integers for efficient sending over serial
                x_loc = int(x_loc * 1000)
                y_loc = int(y_loc * 1000)

                # print(f"{hand_label}: x={x_loc}, y={y_loc}")

                # binary encode the data for sending over serial with no padding (6 bytes = 1 char + 2 ints + newline)
                data = struct.pack('=c2H', hand_label.encode(), x_loc, y_loc) + b'\n'

                # avoid sending duplicate data
                if data != previous_data:
                    serial_port.write(data)
                    print(data)
                    previous_data = data

                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                ## CASE 1 -> click detected (click = touch tips of thumb and index finger)
                # set distance threshold to register click
                THRESH = dist(
                    hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                    hand_landmarks.landmark[HAND_LANDMARKS['THUMB_J']],
                    FRAME_SIZE['width'], FRAME_SIZE['height'])
                click = dist(
                    hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                    hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                    FRAME_SIZE['width'], FRAME_SIZE['height'])

                if THRESH > click:
                    # send 1 byte
                    serial_port.write(b'C\n')

                ## CASE 2 -> exit (exit = close fist)
                HAND_SIZE = dist(
                    hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                    hand_landmarks.landmark[HAND_LANDMARKS['MOVE_ID']],
                    FRAME_SIZE['width'], FRAME_SIZE['height'])
                if (
                    HAND_SIZE >
                    dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']], hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                         FRAME_SIZE['width'], FRAME_SIZE['height']) and
                    HAND_SIZE >
                    dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']], hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                         FRAME_SIZE['width'], FRAME_SIZE['height']) and
                    HAND_SIZE >
                    dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']], hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                         FRAME_SIZE['width'], FRAME_SIZE['height']) and
                    HAND_SIZE >
                    dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']], hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                         FRAME_SIZE['width'], FRAME_SIZE['height'])
                ):
                    # send 1 byte
                    serial_port.write(b'E\n')

async def main():
    """Main event loop."""

    frame_queue = asyncio.Queue()
    result_queue = asyncio.Queue()

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    # create and immediately run tasks
    async with asyncio.TaskGroup() as tg:
        tg.create_task(process_frame(frame_queue, result_queue))
        tg.create_task(send_data(result_queue))

        while cap.isOpened():
            # reading frames is blocking - send to run on separate thread
            ret, frame = await asyncio.get_event_loop().run_in_executor(executor, cap.read)
            if not ret:
                print("Failed to grab frame")
                break

            # attach frame to queue for processing
            await frame_queue.put(frame)

            # Display the frame
            # mirror = cv2.flip(frame, 1)
            # cv2.imshow("Hand Tracking", mirror)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # stop processes
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())