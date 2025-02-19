'''
Hand tracking script acting with full keyboard + mouse controls
Adds drag & speech functionality
'''

import cv2
import mediapipe as mp
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor
import struct
import time
from config import HAND_LANDMARKS, FRAME_SIZE

# define RUN_MODE
RUN_MODE = "serial" if __name__ == "__main__" else "async"
print(f"Running in {RUN_MODE} mode")

# initialise mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    # max_num_hands=2,              # increase to 2 when zoom functionality working
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use V4L2 backend explicitly
# optimise camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE['width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE['height'])
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # low latency

### optional: open camera in fullscreen
# window_name = "Hand Tracking"
# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

executor = ThreadPoolExecutor()

## failsafe for drag mode
# if mediapipe drops / glitches, it can exit drag abruptly, and this can be jarring
# it can often mistake drag for other gestures where fingers are closed
# therefore need a cooldown on the drag gesture to make sure it has been exited properly before performing a new action
last_drag = 0
cooldown = 1.0          # seconds

def dist(lm1, lm2, w, h):
    """Calculate Euclidian distance between 2 landmarks"""

    dx = (lm1.x - lm2.x) * w
    dy = (lm1.y - lm2.y) * h
    return math.sqrt(dx ** 2 + dy ** 2)

async def process_frame(frame_queue, landmark_queue):
    """Process each camera frame to track hand movements"""

    # calculate real FPS
    frame_count = 0
    start_time = time.time()

    while True:
        frame = await frame_queue.get()
        if frame is None:
            break

        # Increment frame count
        frame_count += 1

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Calculate FPS every second
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            frame_count = 0  # Reset frame count
            start_time = time.time()  # Reset start time

        # start_process = time.time()  # Start timing frame processing

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get landmarks via mediapipe and append to Asyncio queue
        results = hands.process(rgb_frame)

        # end_process = time.time()  # End timing frame processing
        # print(f"Time to process frame: {end_process - start_process:.6f} seconds")

        await landmark_queue.put(results)

async def send_data(landmark_queue, data_queue, serial_port):
    """
    RUN_MODE = serial: sends data packets over serial to be read by control_machine.py
    RUN_MODE = async: appends data packets to queues to be read by control_machine.py
    """

    global last_drag

    while True:
        results = await landmark_queue.get()  # retrieve landmarks from Asyncio queue
        if results is None:
            break

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):

                ### determine what mode you're in

                # get and mirror hand labels (due to mirrored screen)
                hand_label = 'R' if hand_info.classification[0].label == "Left" else 'L'

                # calculate hand size (used for some commands)
                HAND_SIZE = dist(
                    hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                    hand_landmarks.landmark[HAND_LANDMARKS['MOVE_ID']],
                    FRAME_SIZE['width'], FRAME_SIZE['height'])
                # edited hand size for when base of middle finger is not visible
                FIST_SIZE = dist(
                    hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                    hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_J']],
                    FRAME_SIZE['width'], FRAME_SIZE['height'])

                ### CASE 1: scrolling mode = index and middle finger extended, ring and little closed
                if (
                        FIST_SIZE/2 <
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height']) and
                        FIST_SIZE/2 <
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height']) and
                        FIST_SIZE >
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height']) and
                        FIST_SIZE >
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height']) and
                        hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']].x -
                        hand_landmarks.landmark[HAND_LANDMARKS['THUMB_J']].x > 0
                ):

                    current_time = time.time()

                    # if perceived to be a scroll BUT are currently dragging - very likely to be a 'misread' drag
                    if current_time - last_drag > cooldown:
                        last_drag = current_time
                        loc = hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']]
                        # normalise coords and flip axes
                        x_loc, y_loc = 1.0 - loc.x, 1.0 - loc.y
                        # scale floats to integers for efficient sending over serial
                        x_loc = int(x_loc * 1000)
                        y_loc = int(y_loc * 1000)

                        # 6 bytes = 1 char (D for drag) + 2 int (x,y location of cursor) + newline
                        data = struct.pack('=c2H', b'D', x_loc, y_loc) + b'\n'
                        if RUN_MODE == "serial":
                            serial_port.write(data)
                        else:
                            await data_queue.put(data)
                        # print(data)


                    else:
                        # if there wasn't a drag recently, it's a scroll

                        # reference for scroll movement = tip of index finger
                        scroll_loc = hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']]
                        # reference for scroll anchor (reference point moves with hand so everything is relative)
                        anchor_loc = hand_landmarks.landmark[HAND_LANDMARKS['INDEX_J']]

                        # normalise coord and flip axis
                        scroll_loc = 1.0 - scroll_loc.y
                        anchor_loc = 1.0 - anchor_loc.y
                        # scale float to integer for efficient sending over serial
                        scroll_loc = int(scroll_loc * 1000)
                        anchor_loc = int(anchor_loc * 1000)

                        # binary encode the data for sending over serial with no padding
                        # 6 bytes = 1 char (S for scrolling) + 2 int (scroll and move y-locations) + newline
                        data = struct.pack('=c2H', b'S', scroll_loc, anchor_loc) + b'\n'

                        # transmit data depending on mode
                        if RUN_MODE == "serial":
                            serial_port.write(data)
                        else:
                            await data_queue.put(data)
                        # print(data)

                ### CASE 3: drag mode - all fingers closed into fist (replaces exit code)
                elif (
                        FIST_SIZE >
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height']) and
                        FIST_SIZE >
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height']) and
                        FIST_SIZE >
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height']) and
                        FIST_SIZE >
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height']) and
                        # final condition to ensure fist is achieved - thumb doubled back on itself along x-axis
                        hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']].x -
                        hand_landmarks.landmark[HAND_LANDMARKS['THUMB_J']].x < 0
                ):

                    # reset drag timer
                    current_time = time.time()
                    last_drag = current_time

                    # reference point for hand movement - thumb_tip easier to see when closed fist
                    loc = hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']]
                    # normalise coords and flip axes
                    x_loc, y_loc = 1.0 - loc.x, 1.0 - loc.y
                    # scale floats to integers for efficient sending over serial
                    x_loc = int(x_loc * 1000)
                    y_loc = int(y_loc * 1000)

                    # 6 bytes = 1 char (D for drag) + 2 int (x,y location of cursor) + newline
                    data = struct.pack('=c2H', b'D', x_loc, y_loc) + b'\n'
                    if RUN_MODE == "serial":
                        serial_port.write(data)
                    else:
                        await data_queue.put(data)
                    # print(data)

                ### CASE 4: speech mode = devil horns (middle and ring fingers closed)
                elif (
                        FIST_SIZE <
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height']) and
                        FIST_SIZE >
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height']) and
                        FIST_SIZE >
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height']) and
                        FIST_SIZE <
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height'])
                ):

                    # 2 bytes = 1 char (V for voice) + newline
                    if RUN_MODE == "serial":
                        serial_port.write(b'V\n')
                    else:
                        serial_port.write(b'V\n')

                ### CASE 5: zoom mode = 2 hands in view (not currently functional)
                elif num_hands == 2:

                    # temporary dictionary to store landmarks
                    hand_landmarks_dict = {}

                    # Collect landmarks in the loop
                    for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_label = hand_info.classification[0].label[0]
                        hand_landmarks_dict[hand_label] = hand_landmarks

                    # Process only if both hands are detected
                    if 'R' in hand_landmarks_dict and 'L' in hand_landmarks_dict:
                        right_lms = hand_landmarks_dict['R']
                        left_lms = hand_landmarks_dict['L']

                        right_loc = right_lms.landmark[HAND_LANDMARKS['MOVE_ID']]
                        left_loc = left_lms.landmark[HAND_LANDMARKS['MOVE_ID']]

                        # print(f"Right Loc - x: {right_loc.x}, y: {right_loc.y}")
                        # print(f"Left Loc - x: {left_loc.x}, y: {left_loc.y}")

                        # invert + scale
                        x_left = int(left_loc.x * 1000)
                        x_right = int(right_loc.x * 1000)
                        y_left = int(left_loc.y * 1000)
                        y_right = int(right_loc.y * 1000)

                        # Euclidian distance between points
                        distance = int(((x_left - x_right) ** 2 + (y_left - y_right) ** 2) ** 0.5)
                        # print(distance)

                        # 4 bytes = 1 char (Z for zoom) + 1 int (distance) + newline
                        data = struct.pack('=cH', b'Z', distance) + b'\n'
                        if RUN_MODE == "serial":
                            serial_port.write(data)
                        else:
                            await data_queue.put(data)

                ### CASE 2: cursor mode = open palm
                else:
                    ## CASE 2.0 -> no commands: send realtime hand position
                    # (this is the primary way to exit a drag)

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
                    if RUN_MODE == "serial":
                        serial_port.write(data)
                    else:
                        await data_queue.put(data)

                    # print(data)

                    ## CASE 2.1 -> click detected (click = touch tips of thumb and index finger)
                    # set distance threshold to register click
                    THRESH = dist(
                        hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                        hand_landmarks.landmark[HAND_LANDMARKS['THUMB_J']],
                        FRAME_SIZE['width'], FRAME_SIZE['height'])
                    click = dist(
                        hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                        hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                        FRAME_SIZE['width'], FRAME_SIZE['height'])

                    if (THRESH > click and
                            # HAND_SIZE <
                            # dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                            #      hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                            #      FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height'])
                    ):

                        # drag failsafe
                        current_time = time.time()

                        if current_time - last_drag > cooldown:
                            last_drag = current_time
                            loc = hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']]
                            # normalise coords and flip axes
                            x_loc, y_loc = 1.0 - loc.x, 1.0 - loc.y
                            # scale floats to integers for efficient sending over serial
                            x_loc = int(x_loc * 1000)
                            y_loc = int(y_loc * 1000)

                            # 6 bytes = 1 char (D for drag) + 2 int (x,y location of cursor) + newline
                            data = struct.pack('=c2H', b'D', x_loc, y_loc) + b'\n'
                            if RUN_MODE == "serial":
                                serial_port.write(data)
                            else:
                                await data_queue.put(data)
                            # print(data)
                        else:
                            # send 1 byte
                            if RUN_MODE == "serial":
                                serial_port.write(b'C\n')
                            else:
                                await data_queue.put(b'C\n')

                    ## CASE 2.2 -> exit (= close fist)
                    # if (
                    #         HAND_SIZE / 2 >
                    #         dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                    #              hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                    #              FRAME_SIZE['width'], FRAME_SIZE['height']) and
                    #         HAND_SIZE / 2 >
                    #         dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                    #              hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                    #              FRAME_SIZE['width'], FRAME_SIZE['height']) and
                    #         HAND_SIZE >
                    #         dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                    #              hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                    #              FRAME_SIZE['width'], FRAME_SIZE['height']) and
                    #         HAND_SIZE >
                    #         dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                    #              hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                    #              FRAME_SIZE['width'], FRAME_SIZE['height'])
                    # ):
                    #     # send 1 byte
                    #     if RUN_MODE == "serial":
                    #         serial_port.write(b'E\n')
                    #     else:
                    #         await data_queue.put(b'E\n')

                    ## CASE 2.3 -> change tab forward
                    tabf = dist(
                        hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                        hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                        FRAME_SIZE['width'], FRAME_SIZE['height'])

                    if (THRESH > tabf and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            # HAND_SIZE <
                            # dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                            #      hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                            #      FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height'])
                    ):
                        # drag failsafe
                        current_time = time.time()

                        if current_time - last_drag > cooldown:
                            last_drag = current_time
                            loc = hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']]
                            # normalise coords and flip axes
                            x_loc, y_loc = 1.0 - loc.x, 1.0 - loc.y
                            # scale floats to integers for efficient sending over serial
                            x_loc = int(x_loc * 1000)
                            y_loc = int(y_loc * 1000)

                            # 6 bytes = 1 char (D for drag) + 2 int (x,y location of cursor) + newline
                            data = struct.pack('=c2H', b'D', x_loc, y_loc) + b'\n'
                            if RUN_MODE == "serial":
                                serial_port.write(data)
                            else:
                                await data_queue.put(data)
                            # print(data)

                        else:
                            # send 1 byte
                            if RUN_MODE == "serial":
                                serial_port.write(b'F\n')
                            else:
                                await data_queue.put(b'F\n')

                    ## CASE 2.4 -> change tab backward
                    tabb = dist(
                        hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                        hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                        FRAME_SIZE['width'], FRAME_SIZE['height'])

                    if (THRESH > tabb and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            # HAND_SIZE <
                            # dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                            #      hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                            #      FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height'])
                    ):
                        # drag failsafe
                        current_time = time.time()

                        if current_time - last_drag > cooldown:
                            last_drag = current_time
                            loc = hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']]
                            # normalise coords and flip axes
                            x_loc, y_loc = 1.0 - loc.x, 1.0 - loc.y
                            # scale floats to integers for efficient sending over serial
                            x_loc = int(x_loc * 1000)
                            y_loc = int(y_loc * 1000)

                            # 6 bytes = 1 char (D for drag) + 2 int (x,y location of cursor) + newline
                            data = struct.pack('=c2H', b'D', x_loc, y_loc) + b'\n'
                            if RUN_MODE == "serial":
                                serial_port.write(data)
                            else:
                                await data_queue.put(data)
                            # print(data)

                        else:
                            # send 1 byte
                            if RUN_MODE == "serial":
                                serial_port.write(b'B\n')
                            else:
                                await data_queue.put(b'B\n')

                    ## CASE 2.5 -> mission control
                    tabm = dist(
                        hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                        hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                        FRAME_SIZE['width'], FRAME_SIZE['height'])

                    if (THRESH > tabm and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height'])
                            # HAND_SIZE <
                            # dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                            #      hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                            #      FRAME_SIZE['width'], FRAME_SIZE['height'])
                    ):
                        # drag failsafe
                        current_time = time.time()

                        if current_time - last_drag > cooldown:
                            last_drag = current_time
                            loc = hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']]
                            # normalise coords and flip axes
                            x_loc, y_loc = 1.0 - loc.x, 1.0 - loc.y
                            # scale floats to integers for efficient sending over serial
                            x_loc = int(x_loc * 1000)
                            y_loc = int(y_loc * 1000)

                            # 6 bytes = 1 char (D for drag) + 2 int (x,y location of cursor) + newline
                            data = struct.pack('=c2H', b'D', x_loc, y_loc) + b'\n'
                            if RUN_MODE == "serial":
                                serial_port.write(data)
                            else:
                                await data_queue.put(data)
                            # print(data)

                        else:
                            # send 1 byte
                            if RUN_MODE == "serial":
                                serial_port.write(b'M\n')
                            else:
                                await data_queue.put(b'M\n')

async def main(data_queue=None):
    """Main event loop"""

    # initialize serial communication conditionally
    if RUN_MODE == "serial":
        import serial
        ## EDIT PORT based to your camera input
        serial_port = serial.Serial('/dev/ttyGS0', 115200, timeout=1)
    else:
        serial_port = None

    frame_queue = asyncio.Queue()               # stores camera frames
    landmark_queue = asyncio.Queue()            # stores landmarks within the frames

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    # create and immediately run tasks
    async with asyncio.TaskGroup() as tg:
        tg.create_task(process_frame(frame_queue, landmark_queue))
        tg.create_task(send_data(landmark_queue, data_queue, serial_port))

        while cap.isOpened():
            # send to run on separate thread (reading frames is blocking process)
            ret, frame = await asyncio.get_event_loop().run_in_executor(executor, cap.read)
            if not ret:
                print("Failed to grab frame")
                break

            # attach frame to queue for processing
            await frame_queue.put(frame)

            ### optional: display the frame
            # mirror = cv2.flip(frame, 1)
            # cv2.imshow("Hand Tracking", mirror)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # stop processes
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())