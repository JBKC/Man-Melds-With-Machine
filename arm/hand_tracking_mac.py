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
from config import HAND_LANDMARKS, FRAME_SIZE, FPS

# initialise mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# initialize camera
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE['width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE['height'])
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # low latency

executor = ThreadPoolExecutor()

# differentiate between click and drag modes
clicking = False
first_click = 0
last_drag = 0                   # running record of last drag action to ensure continuity
prev_side = 0                   # tracking forward/back browser commands
drag_threshold = 0.15           # duration for click to be held in order to qualify as a drag
drag_cooldown = 0.5             # ensure drag isn't exited abruptly

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
            # print(f"FPS: {fps:.2f}")
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

async def send_data(landmark_queue, data_queue):

    global clicking, first_click, last_drag, prev_side

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

                # calculate hand size (used for tap commands)
                HAND_SIZE = dist(
                    hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                    hand_landmarks.landmark[HAND_LANDMARKS['MOVE_ID']],
                    FRAME_SIZE['width'], FRAME_SIZE['height'])
                # edited hand size for when base of middle finger is not visible
                FIST_SIZE = dist(
                    hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                    hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_J']],
                    FRAME_SIZE['width'], FRAME_SIZE['height'])

                # x positions
                thumb_x = hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']].x
                index_x = hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']].x
                middle_x = hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']].x
                little_x = hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']].x

                ### CASE 1: scrolling mode = index and middle finger extended, ring and little closed + upright
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
                             FRAME_SIZE['width'], FRAME_SIZE['height'])

                        # check for scrolling mode - x position of index + middle in-between thumb and little finger (hand-invariant)
                        and
                        min(thumb_x, little_x) < index_x < max(thumb_x, little_x) and
                        min(thumb_x, little_x) < middle_x < max(thumb_x, little_x)

                ):

                    # exit drag mode
                    clicking = False

                    # reference for scroll movement = tip of index finger
                    scroll_loc = hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']]
                    # reference for scroll anchor (reference point moves with hand so everything is relative)
                    anchor_loc = hand_landmarks.landmark[HAND_LANDMARKS['INDEX_J']]

                    # normalise coord and flip axis
                    scroll_loc = 1.0 - scroll_loc.y
                    anchor_loc = 1.0 - anchor_loc.y
                    # scale float to integer for efficient sending
                    scroll_loc = int(scroll_loc * 1000)
                    anchor_loc = int(anchor_loc * 1000)

                    # binary encode the data for sending with no padding
                    # 6 bytes = 1 char (S for scrolling) + 2 int (scroll and move y-locations) + newline
                    data = struct.pack('=c2H', b'S', scroll_loc, anchor_loc) + b'\n'
                    await data_queue.put(data)
                    # print(data)

                ### CASE 1.1: forward/back browser page mode = index and middle finger extended to the side, other fingers closed
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
                             FRAME_SIZE['width'], FRAME_SIZE['height'])

                        # x position of index + middle should be to one side of thumb and little finger (hand-invariant)
                        and not
                        min(thumb_x, little_x) < index_x < max(thumb_x, little_x) and not
                        min(thumb_x, little_x) < middle_x < max(thumb_x, little_x)
                ):

                    # exit drag mode
                    clicking = False

                    # we are in forward/back mode
                    # track direction of the finger swipe relative to the thumb
                    current_side = 1 if index_x < thumb_x and middle_x < thumb_x else -1  # 1 = right, -1 = left (x coordinates inverted in mediapipe)

                    # detect when the fingers switch sides
                    if prev_side != 0 and current_side != prev_side:
                        if current_side == 1:
                            # rightward movement = forward swipe
                            await data_queue.put(b'Y\n')
                        else:
                            # leftward movement = back swipe
                            await data_queue.put(b'Z\n')

                    # Update the last known side
                    prev_side = current_side





                ### CASE 2: speech mode = devil horns (middle and ring fingers closed)
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

                    # exit drag mode
                    clicking = False

                    # 2 bytes = 1 char (V for voice) + newline
                    await data_queue.put(b'V\n')

                ### CASE 2.1: browser typing mode = little finger raised
                # used to snap to the search bar at the top of a browser
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
                        FIST_SIZE <
                        dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                             hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                             FRAME_SIZE['width'], FRAME_SIZE['height'])

                ):

                    # exit drag mode
                    clicking = False

                    # 2 bytes = 1 char (T for type) + newline
                    await data_queue.put(b'T\n')


                ### CASE 3: cursor mode = open palm & tap commands
                else:

                    # establish live hand location
                    loc = hand_landmarks.landmark[HAND_LANDMARKS['MOVE_ID']]
                    # normalise coords and flip axes
                    x_loc, y_loc = 1.0 - loc.x, 1.0 - loc.y
                    # scale floats to integers for efficient sending
                    x_loc = int(x_loc * 1000)
                    y_loc = int(y_loc * 1000)

                    # print(f"{hand_label}: x={x_loc}, y={y_loc}")

                    ## CASE 3.0 -> click detected (click = touch tips of thumb and index finger)
                    ## CASE 3.1 -> drag = click + hold (pinch)

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

                        if not clicking:
                            # first click detected - send data to queue
                            await data_queue.put(b'C\n')
                            # set time of this click
                            first_click = time.time()
                            clicking = True

                        elif clicking:
                            # already clicked recently
                            current_time = time.time()
                            if current_time - first_click < drag_threshold:
                                # if click was below time threshold, still send data (will be rejected by control_mac.py due to cooldown)
                                await data_queue.put(b'C\n')
                            else:
                                # otherwise click has been sustained a while - go into drag mode
                                last_drag = time.time()
                                # 6 bytes = 1 char (D for drag) + 2 int (x,y location of cursor) + newline
                                data = struct.pack('=c2H', b'D', x_loc, y_loc) + b'\n'
                                await data_queue.put(data)

                    ## CASE 3.2 -> no click command: send realtime hand position ("default" mode)
                    else:

                        # first make sure current drag is not exited accidentally
                        current_time = time.time()
                        if current_time - last_drag < drag_cooldown:
                            # still dragging
                            data = struct.pack('=c2H', b'D', x_loc, y_loc) + b'\n'
                            await data_queue.put(data)

                        else:
                            # exit drag mode
                            clicking = False
                            # binary encode the data for sending with no padding (6 bytes = 1 char + 2 ints + newline)
                            data = struct.pack('=c2H', hand_label.encode(), x_loc, y_loc) + b'\n'
                            await data_queue.put(data)


                    ## CASE 3.3 -> exit (= close fist)
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

                    ## CASE 3.4 -> change tab forward
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

                        # drag check
                        current_time = time.time()
                        if current_time - last_drag < drag_cooldown:
                            # still dragging
                            data = struct.pack('=c2H', b'D', x_loc, y_loc) + b'\n'
                            await data_queue.put(data)

                        else:
                            # exit drag mode
                            clicking = False
                            await data_queue.put(b'F\n')

                    ## CASE 3.5 -> change tab backward
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
                        # drag check
                        current_time = time.time()
                        if current_time - last_drag < drag_cooldown:
                            # still dragging
                            data = struct.pack('=c2H', b'D', x_loc, y_loc) + b'\n'
                            await data_queue.put(data)

                        else:
                            clicking = False
                            await data_queue.put(b'B\n')

                    ## CASE 3.6 -> mission control
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

                        # drag check
                        current_time = time.time()
                        if current_time - last_drag < drag_cooldown:
                            # still dragging
                            data = struct.pack('=c2H', b'D', x_loc, y_loc) + b'\n'
                            await data_queue.put(data)

                        else:
                            clicking = False
                            await data_queue.put(b'M\n')


async def main(data_queue=None):
    """Main event loop"""

    frame_queue = asyncio.Queue()               # stores camera frames
    landmark_queue = asyncio.Queue()            # stores landmarks within the frames

    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    # create and immediately run tasks
    async with asyncio.TaskGroup() as tg:
        tg.create_task(process_frame(frame_queue, landmark_queue))
        tg.create_task(send_data(landmark_queue, data_queue))

        while cap.isOpened():
            # send to run on separate thread (reading frames is blocking process)
            ret, frame = await asyncio.get_event_loop().run_in_executor(executor, cap.read)
            if not ret:
                print("Failed to grab frame")
                break

            # attach frame to queue for processing
            await frame_queue.put(frame)


            ### optional: display the frame
            ### avoid printing landmarks synchronously as commands will not be as accurate
            # mirror = cv2.flip(frame, 1)
            # cv2.imshow("Hand Tracking", mirror)
            ####

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # stop processes
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())