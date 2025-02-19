'''
Translates data from serial into mouse and keyboard actions
3 categories: cursor movement, scroll, commands
Call script directly only when processing camera feed on a separate machine (e.g. Raspberry Pi)
'''

from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
import keyboard
import pyautogui
from screeninfo import get_monitors
from collections import deque
import asyncio
import time
import sys
import struct
from AppKit import NSScreen
from config import PARAMS

# define RUN_MODE
RUN_MODE = "serial" if __name__ == "__main__" else "async"

# Initialize
mouse = MouseController()
pykeyboard = KeyboardController()

def get_screen_bounds():
    """Returns a list of screen bounds [(x_start, y_start, width, height)] for macOS"""
    from AppKit import NSScreen
    screens = NSScreen.screens()
    return [(int(s.frame().origin.x), int(s.frame().origin.y), int(s.frame().size.width), int(s.frame().size.height)) for s in screens]


def map_to_screen(loc):
    """
    Maps (0-1000) hand tracking input to the full multi-monitor setup.
    """
    screens = get_screen_bounds()

    # Determine total virtual screen space
    min_x = min(screen[0] for screen in screens)  # Leftmost x-coordinate
    min_y = min(screen[1] for screen in screens)  # Topmost y-coordinate
    max_x = max(screen[0] + screen[2] for screen in screens)  # Rightmost boundary
    max_y = max(screen[1] + screen[3] for screen in screens)  # Bottommost boundary

    virtual_width = max_x - min_x
    virtual_height = max_y - min_y

    def zoom(value, screen_size):
        """Scale 0-1000 input range to screen size."""
        # if value < 100:
        #     return 0
        # elif value > 900:
        #     return screen_size
        # else:
        #     return int(((value - 100) / 800) * screen_size)

        return int(value/1000*screen_size)

    # Normalize hand input to full multi-monitor space
    screen_x = min_x + zoom(loc[0], virtual_width)
    screen_y = min_y + zoom(loc[1], virtual_height)

    return [screen_x, screen_y]

# Smoothing buffers for moving average
buffer_size = 3
x_buffer = deque(maxlen=buffer_size)
y_buffer = deque(maxlen=buffer_size)

# initialise mouse clicks / position
last_click = 0
cooldown = 0.5          # seconds
scroll_anchor = None
zoom_anchor = None
zoom_mode = False
drag_mode = False
voice_mode = False

# from Quartz.CoreGraphics import (
#     CGEventCreateKeyboardEvent,
#     CGEventCreateScrollWheelEvent,
#     CGEventPost,
#     kCGEventScrollWheel,
#     kCGSessionEventTap,
#     kCGEventFlagMaskCommand,
#     kCGEventKeyDown,
#     kCGEventKeyUp
# )
#
# def press_cmd():
#     """Press down the Command key."""
#     cmd_keycode = 0x37  # 0x37 is the keycode for Cmd (Command key)
#     event = CGEventCreateKeyboardEvent(None, cmd_keycode, True)  # True for key press
#     CGEventPost(kCGSessionEventTap, event)
#
# def release_cmd():
#     """Release the Command key."""
#     cmd_keycode = 0x37  # 0x37 is the keycode for Cmd
#     event = CGEventCreateKeyboardEvent(None, cmd_keycode, False)  # False for key release
#     CGEventPost(kCGSessionEventTap, event)
#
# def scroll_event(dy):
#     """Send a smooth scroll event with the Command key held down."""
#     scroll_event = CGEventCreateScrollWheelEvent(None, 0, 1, dy)  # Scroll on 1 axis, dy is the scroll amount
#     CGEventPost(kCGSessionEventTap, scroll_event)  # Post the scroll event



########
class StopException(Exception):
    """Custom exception to signal a graceful shutdown."""
    pass



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

async def read_serial(serial_reader, data_queue):
    """
    Read data asynchronously from the serial port
    Protocol =
    2 bytes for command (1 char + newline)
    6 bytes for scroll (1 char + 2 int + newline)
    6 bytes for cursor movement (1 char + 2 int + newline)
    """
    buffer = b''  # store packets as they come in

    while True:
        start_read = time.time()

        try:
            # read 1 byte of data and immediately append to buffer
            chunk = await serial_reader.read(1)
            if not chunk:
                continue  # Skip if no data is received

            buffer += chunk

            # Process complete packets (ending with newline)
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)  # Split at the first newline
                if len(line) > 0:
                    await data_queue.put(line)  # Queue the complete packet

                    end_read = time.time()
                    # print(f"Time to process 1 packet: {end_read - start_read:.6f} seconds")

        except Exception as e:
            print(f"Error reading serial data: {e}")
            break


async def process_data(data_queue, cur):
    """Process serial data and perform cursor actions"""

    global last_click, scroll_anchor, zoom_anchor, zoom_mode, drag_mode, voice_mode

    while True:

        # Get the next packet from the queue
        data = await data_queue.get()
        # print(data)

        # Read movement packets
        if len(data) == 5:  # Movement and scroll packets: 1 char + 2 unsigned integers

            try:
                start_processing = time.time()  # Start timing data processing

                # scrolling mode detected
                if data.startswith(b'S'):
                    print("SCROLL MODE")

                    zoom_mode = False
                    zoom_anchor = None
                    if zoom_mode == True:
                        release_cmd()  # Release Cmd
                        zoom_mode = False
                    if drag_mode == True:
                        mouse.release(Button.left)
                        drag_mode = False
                    if voice_mode == True:
                        pykeyboard.release(Key.alt)
                        # enter whatever text was input
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
                    zoom_anchor = None
                    if zoom_mode == True:
                        release_cmd()  # Release Cmd
                        zoom_mode = False
                    if voice_mode == True:
                        pykeyboard.release(Key.alt)
                        # enter whatever text was input
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
                        cur = map_to_screen(loc)
                        mouse.position = (cur[0], cur[1])

                # cursor movement mode (default)
                else:

                    scroll_anchor = None
                    zoom_anchor = None
                    if zoom_mode == True:
                        release_cmd()  # Release Cmd
                        zoom_mode = False
                    if drag_mode == True:
                        mouse.release(Button.left)
                        drag_mode = False
                    if voice_mode == True:
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
                    # velocity_start = time.time()  # Start timing velocity scaling
                    cur = velocity_scale(cur, tar)
                    # velocity_end = time.time()  # End timing velocity scaling
                    # print(f"Time for velocity scaling and cursor movement: {velocity_end - velocity_start:.6f} seconds")

                    ## no velocity scaling option
                    # cur = map_to_screen(loc)
                    # mouse.position = (cur[0], cur[1])

                    # print(f"{hand_label.decode()}: x={int(cur[0])}, y={int(cur[1])}")

            except Exception as e:
                print(f"Error processing movement data: {e}")

        # zoom packets
        ## NOTE - zoom functionality not yet working - placeholder
        elif len(data) == 3:

            try:
                # zoom mode detected
                if data.startswith(b'Z'):
                    print("ZOOM MODE")

                    if drag_mode == True:
                        mouse.release(Button.left)
                        drag_mode = False
                    scroll_anchor = None
                    if voice_mode == True:
                        pykeyboard.release(Key.alt)
                        # enter whatever text was input
                        time.sleep(1.0)
                        pykeyboard.press(Key.enter)
                        pykeyboard.release(Key.enter)
                        voice_mode = False

                    # Unpack binary data
                    _, distance = struct.unpack('=cH', data)
                    # print(distance)

                    # set zoom anchor (initial distance)
                    if zoom_anchor is None:
                        zoom_anchor = distance

                    if not zoom_mode:
                        # start zoom - keep cmd pressed down
                        press_cmd()  # Press down Cmd
                        zoom_mode = True
                    else:
                        # Update position during dragging
                        zoom = int((zoom_anchor - distance) / PARAMS['SCROLL'])
                        scroll_event(zoom)  # Send smooth scroll event
                        # mouse.scroll(dx=0, dy=zoom)
                        print(zoom)

            except Exception as e:
                print(f"Error processing zoom data: {e}")

        # Read command packets
        elif len(data) == 1:  # Command packet: 1 byte
            command = data

            if command == b'V':      # voice command
                print("VOICE MODE")

                scroll_anchor = None
                zoom_anchor = None
                if zoom_mode == True:
                    release_cmd()  # Release Cmd
                    zoom_mode = False
                if drag_mode == True:
                    mouse.release(Button.left)
                    drag_mode = False

                if not voice_mode:
                    # activate voice mode
                    pykeyboard.press(Key.alt)
                    voice_mode = True

                else:
                    # if voice mode already activated with gesture active, do nothing (user is speaking)
                    pass

            if command == b'C':  # Click command
                current_time = time.time()
                if current_time - last_click > cooldown:
                    mouse.click(Button.left)
                    print("CLICK")
                    last_click = current_time
                else:
                    print("Double click blocked")
            elif command == b'E':  # Exit command
                raise StopException()
            if command == b'F':  # next tab
                current_time = time.time()
                if current_time - last_click > cooldown:
                    with pykeyboard.pressed(Key.ctrl):
                        pykeyboard.press(Key.tab)
                        pykeyboard.release(Key.tab)
                    print("NEXT TAB")
                    last_click = current_time
                else:
                    print("Double tab forward blocked")
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
            if command == b'M':  # mission control
                current_time = time.time()
                if current_time - last_click > cooldown:
                    pyautogui.keyDown("ctrl")
                    pyautogui.press("up")
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

    # set initial cur_x, cur_y
    cur = [0,0]

    # get data_queue from hand_tracking script if in async mode
    if RUN_MODE == "async" and data_queue is not None:
        try:
            await process_data(data_queue, cur)

        except StopException:
            # if "stop" received, shut down program gracefully
            print("PROGRAM ENDED")

    # initialize data_queue if in serial mode
    elif RUN_MODE == "serial":
        import serial_asyncio
        # serial_port = await serial_asyncio.open_serial_connection(url='/dev/tty.usbmodem14101', baudrate=115200)
        serial_port = await serial_asyncio.open_serial_connection(url='/dev/tty.usbmodem14201', baudrate=115200)
        reader, writer = serial_port

        data_queue = asyncio.Queue()

        try:
            # create and run tasks for reading and processing data
            async with asyncio.TaskGroup() as tg:
                tg.create_task(read_serial(reader, data_queue))
                tg.create_task(process_data(data_queue, cur))

        except StopException:
            # if "stop" received, shut down program gracefully
            print("PROGRAM ENDED")
    else:
        print("Invalid RUN_MODE or missing data_queue")


if __name__ == "__main__":
    asyncio.run(main())