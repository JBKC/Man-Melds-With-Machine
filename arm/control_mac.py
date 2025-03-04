'''
Translates data hand data into mouse and keyboard actions
3 categories: cursor movement, scroll, commands
'''

from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
import pyautogui
from screeninfo import get_monitors
from collections import deque
import asyncio
import time
import struct
from config import PARAMS

# Initialize
mouse = MouseController()
pykeyboard = KeyboardController()
monitors = get_monitors()
primary_monitor = monitors[0]
SCREEN_WIDTH = primary_monitor.width
SCREEN_HEIGHT = primary_monitor.height

# Smoothing buffers for moving average
buffer_size = 3
x_buffer = deque(maxlen=buffer_size)
y_buffer = deque(maxlen=buffer_size)

# initialise mouse clicks / position
last_click = 0
cooldown = 0.5          # seconds
browser_cooldown = 1.0
scroll_anchor = None
drag_mode = False
voice_mode = False

########
class StopException(Exception):
    """Custom exception to signal a graceful shutdown."""
    pass

def map_to_screen(loc):
    """
    Map integer coordinates received (in range 0->1000) to screen coordinates
    Includes zoom to avoid edge effects
    """
    def zoom(value, screen_size):
        if value < 200:
            return 0  # Minimum screen coordinate
        elif value > 800:
            return screen_size  # Maximum screen coordinate
        else:
            # interpolate
            return int(((value - 200) / 600) * screen_size)

    screen_x = int(zoom(loc[0], SCREEN_WIDTH))
    screen_y = int(zoom(loc[1], SCREEN_HEIGHT))
    return [screen_x, screen_y]

def velocity_scale(cur, tar, GAIN=PARAMS['GAIN'], DAMP=PARAMS['DAMP'], SENSITIVITY=PARAMS['SENSITIVITY'], MIN_STEP=1):
    """
    Adjust speed of cursor based on distance between current and target position by calculating a scaling factor
    :param cur: current [x,y] coords of cursor
    :param tar: target [x,y] coords of cursor
    :param GAIN: higher GAIN = bigger step size, meaning faster cursor movement
    :param DAMP: damping multiplier at small distances - higher DAMP = smaller step sizes = less static jitter
    :param SENSITIVITY: damping limit - damping applied when distance < SENSITIVITY
    :param MIN_STEP: Stops division by zero
    """

    # calculate Euclidian distance between current and target locations of the hand
    distance = ((tar[0] - cur[0]) ** 2 + (tar[1] - cur[1]) ** 2) ** 0.5

    # apply damping to limit step size when distances are very small (reduce static jitter)
    damping = max(1, SENSITIVITY / max(distance, 1e-6))
    damping *= DAMP if damping > 1 else 1

    # calculate scaling factor for cursor steps
    if distance < SENSITIVITY:
        # drastically reduce GAIN for small movements
        scaling_factor = MIN_STEP + (distance * damping)
    else:
        scaling_factor = MIN_STEP + (distance / GAIN) * damping

    # print(int(distance), int(scaling_factor))

    # calculate cursor step sizes
    dx = (tar[0] - cur[0]) / scaling_factor
    dy = (tar[1] - cur[1]) / scaling_factor

    # calculate new (intermediate) positions
    new = [cur[0] + dx, cur[1] + dy]

    # interpolate movement for large distances only
    if distance > SENSITIVITY:
        interpolate(cur, new)

    else:
        mouse.position = (new[0], new[1])

    # return values to loop
    return new

def lerp(start, end, factor):
    """Linear interpolation between start (current) and end (target) points"""
    return start + (end - start) * factor

def interpolate(start, end, steps=PARAMS['STEPS'], delay=PARAMS['DELAY']):
    """
    Interpolates between current and target positions to fill the visual gaps of the cursor
    More steps = smoother mouse cursor but more perceived lag
    """
    for i in range(1, steps + 1):
        # Interpolate between start and end positions
        interp_x = lerp(start[0], end[0], i / steps)
        interp_y = lerp(start[1], end[1], i / steps)

        # Move the mouse to the interpolated position
        mouse.position = (int(interp_x), int(interp_y))

        # Small delay to ensure smooth visual movement
        time.sleep(delay)


async def process_data(data_queue, cur):
    """Process data and perform cursor actions"""

    global last_click, scroll_anchor, drag_mode, voice_mode

    while True:

        # Get the next packet from the queue
        data = await data_queue.get()
        # remove newline
        data = data[:-1]
        print(data)

        # Read movement packets
        if len(data) == 5:  # Movement and scroll packets: 1 char + 2 unsigned integers

            try:
                start_processing = time.time()  # Start timing data processing

                # scrolling mode detected
                if data.startswith(b'S'):
                    print("SCROLL MODE")

                    # exit if currently in drag mode
                    if drag_mode:
                        mouse.release(Button.left)
                        drag_mode = False

                    # if currently in voice mode, exit by entering the text
                    if voice_mode:
                        pykeyboard.release(Key.alt)
                        time.sleep(1.0)
                        pykeyboard.press(Key.enter)
                        pykeyboard.release(Key.enter)
                        voice_mode = False

                    # Unpack binary data
                    _, scroll_loc, anchor_loc = struct.unpack('=c2H', data)
                    # Flip y-axis
                    scroll_loc = 1000 - int(scroll_loc)
                    anchor_loc = 1000 - int(anchor_loc)

                    # set scroll anchor (relative to hand position)
                    if scroll_anchor is None:
                        scroll_anchor = anchor_loc

                    scroll_y = int((scroll_anchor - scroll_loc) / PARAMS['SCROLL'])

                    mouse.scroll(dx=0, dy=scroll_y)

                # drag mode detected
                elif data.startswith(b'D'):
                    print("DRAG MODE")

                    scroll_anchor = None
                    # enter whatever text was input
                    if voice_mode:
                        pykeyboard.release(Key.alt)
                        time.sleep(1.0)
                        pykeyboard.press(Key.enter)
                        pykeyboard.release(Key.enter)
                        voice_mode = False

                    _, x_loc, y_loc = struct.unpack('=c2H', data)
                    loc = [int(x_loc), 1000 - int(y_loc)]
                    # print(loc)

                    if not drag_mode:
                        # start drag - keep mouse pressed down
                        mouse.press(Button.left)
                        drag_mode = True

                    else:
                        # if already in drag mode, start moving the cursor
                        tar = map_to_screen(loc)
                        cur = velocity_scale(cur, tar)

                        # cur = map_to_screen(loc)
                        # mouse.position = (cur[0], cur[1])


                # default mode - cursor movement mode
                else:
                    scroll_anchor = None

                    if drag_mode:
                        mouse.release(Button.left)
                        drag_mode = False

                    if voice_mode:
                        pykeyboard.release(Key.alt)
                        # enter whatever text was input
                        time.sleep(1.0)
                        pykeyboard.press(Key.enter)
                        pykeyboard.release(Key.enter)
                        voice_mode = False

                    # Unpack binary data (1 char + 2 unsigned integers)
                    hand_label, x_loc, y_loc = struct.unpack('=c2H', data)

                    # Flip y-axis
                    loc = [int(x_loc), 1000 - int(y_loc)]

                    # Convert to screen coordinates
                    tar = map_to_screen(loc)
                    # Velocity scaling and move cursor
                    cur = velocity_scale(cur, tar)

                    ## no velocity scaling option (raw hand coordS)
                    # cur = map_to_screen(loc)
                    # mouse.position = (cur[0], cur[1])

                    # print(f"{hand_label.decode()}: x={int(cur[0])}, y={int(cur[1])}")

            except Exception as e:
                print(f"Error processing movement data: {e}")

        # Read command packets
        elif len(data) == 1:  # Command packet: 1 byte
            command = data

            if command == b'V':      # voice command
                print("VOICE MODE")
                scroll_anchor = None
                if drag_mode:
                    mouse.release(Button.left)
                    drag_mode = False

                if not voice_mode:
                    # activate voice mode
                    pykeyboard.press(Key.alt)
                    voice_mode = True
                else:
                    # if voice mode already activated, do nothing (user is speaking)
                    pass

            if command == b'T':  # browser type command
                current_time = time.time()
                if current_time - last_click > cooldown:
                    with pykeyboard.pressed(Key.cmd):
                        pykeyboard.press('l')
                        pykeyboard.release('l')
                    print("TYPE")
                    last_click = current_time
                else:
                    print("Double click blocked")

            if command == b'C':  # Click command
                current_time = time.time()

                # stop any key hangovers
                pykeyboard.release(Key.ctrl)
                pykeyboard.release(Key.cmd)

                if current_time - last_click > cooldown:
                    mouse.press(Button.left)
                    mouse.release(Button.left)
                    print("CLICK")
                    last_click = current_time
                else:
                    print("Double click blocked")

            if command == b'X':  # browser mode (prevents other actions from occurring)
                print("BROWSER MODE")

                # exit if currently in drag mode
                if drag_mode:
                    mouse.release(Button.left)
                    drag_mode = False

                # if currently in voice mode, exit by entering the text
                if voice_mode:
                    pykeyboard.release(Key.alt)
                    time.sleep(1.0)
                    pykeyboard.press(Key.enter)
                    pykeyboard.release(Key.enter)
                    voice_mode = False
                continue

            if command == b'Y':  # forward page
                current_time = time.time()
                if current_time - last_click > browser_cooldown:
                    with pykeyboard.pressed(Key.cmd):
                        pykeyboard.press(Key.right)
                        pykeyboard.release(Key.right)
                    print("FORWARD")
                    last_click = current_time
                else:
                    print("Double page next blocked")

            if command == b'Z':  # back page
                current_time = time.time()
                if current_time - last_click > browser_cooldown:
                    with pykeyboard.pressed(Key.cmd):
                        pykeyboard.press(Key.left)
                        pykeyboard.release(Key.left)
                    print("BACK")
                    last_click = current_time
                else:
                    print("Double page back blocked")

            if command == b'F':  # next tab
                current_time = time.time()
                if current_time - last_click > cooldown:
                    with pykeyboard.pressed(Key.ctrl):
                        pykeyboard.press(Key.tab)
                        pykeyboard.release(Key.tab)
                    print("NEXT TAB")
                    last_click = current_time
                else:
                    print("Double tab back blocked")

            if command == b'B':  # previous tab
                current_time = time.time()
                if current_time - last_click > cooldown:
                    with pykeyboard.pressed(Key.ctrl):
                        with pykeyboard.pressed(Key.shift):
                            pykeyboard.press(Key.tab)
                            pykeyboard.release(Key.tab)
                    print("PREVIOUS TAB")
                    last_click = current_time
                else:
                    print("Double tab back blocked")

            if command == b'N':  # new tab
                current_time = time.time()
                if current_time - last_click > cooldown:
                    with pykeyboard.pressed(Key.cmd):
                        pykeyboard.press('t')
                        pykeyboard.release('t')
                    print("NEW TAB")
                    last_click = current_time
                else:
                    print("Double tab back blocked")

            if command == b'M':  # mission control
                current_time = time.time()
                if current_time - last_click > cooldown:
                    # with pykeyboard.pressed(Key.ctrl):
                    #     pykeyboard.press(Key.up)
                    #     pykeyboard.release(Key.up)
                    pyautogui.keyDown("ctrl")
                    pyautogui.keyDown("up")
                    pyautogui.keyUp("up")
                    pyautogui.keyUp("ctrl")
                    print("MISSION CONTROL")
                    last_click = current_time
                else:
                    print("Double mission control blocked")


        end_processing = time.time()  # End timing data processing
        # print(f"Time to process data packet: {end_processing - start_processing:.6f} seconds")


async def main(data_queue=None):
    """Main event loop"""

    print("Listening for data from Hand Tracking script...")

    # set initial cur_x, cur_y (current location of cursor)
    cur = [0,0]

    # process received data
    try:
        await process_data(data_queue, cur)

    except StopException:
        # if "stop" received, shut down program gracefully
        print("PROGRAM ENDED")


if __name__ == "__main__":
    asyncio.run(main())