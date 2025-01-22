import serial
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
from collections import deque
import serial_asyncio
import asyncio
import time
import sys

# Initialize
mouse = Controller()
monitors = get_monitors()
primary_monitor = monitors[0]
screen_width = primary_monitor.width
screen_height = primary_monitor.height

# Smoothing buffers for moving average
buffer_size = 3
x_buffer = deque(maxlen=buffer_size)
y_buffer = deque(maxlen=buffer_size)

# initialise mouse clicks
last_click = 0
cooldown = 0.5          # seconds

########
class StopException(Exception):
    """Custom exception to signal a graceful shutdown."""
    pass

def map_to_screen(x, y):
    """
    Map normalized coordinates (0,1) to screen coordinates
    Includes zoom to avoid edge effects
    """
    def zoom(value, screen_size):
        if value < 0.2:
            return 0  # Minimum screen coordinate
        elif value > 0.8:
            return screen_size  # Maximum screen coordinate
        else:
            # interpolate
            return int(((value - 0.2) / 0.6) * screen_size)

    screen_x = zoom(x, screen_width)
    screen_y = zoom(y, screen_height)
    return screen_x, screen_y

def velocity_scale(cur_x, cur_y, tar_x, tar_y, GAIN=1000, DAMP=50, SENSITIVITY=5, MIN_STEP=1):
    """
    Adjust speed of cursor based on distance between current and target position by calculating a scaling factor
    :param cur_x, cur_y: current coords of cursor
    :param tar_x, tar_y: target coords of cursor
    :param GAIN: higher GAIN = bigger step size, meaning faster cursor movement
    :param DAMP: damping multiplier at small distances - higher DAMP = smaller step sizes = less static jitter
    :param SENSITIVITY: damping limit - damping applied when distance < SENSITIVITY
    :param MIN_STEP: Stops division by zero
    """

    # calculate Euclidian distance between current and target locations
    distance = ((tar_x - cur_x) ** 2 + (tar_y - cur_y) ** 2) ** 0.5

    # apply damping to limit step size when distances are very small (reduce static jitter)
    damping = max(1, SENSITIVITY / max(distance, 1e-6))
    damping *= DAMP if damping > 1 else 1

    # calculate scaling factor for cursor steps
    scaling_factor = MIN_STEP + (distance / GAIN) * damping
    # print(damping, distance, scaling_factor)

    # calculate cursor step sizes
    dx = (tar_x - cur_x) / scaling_factor
    dy = (tar_y - cur_y) / scaling_factor

    # calculate new (intermediate) positions
    new_x = cur_x + dx
    new_y = cur_y + dy

    # interpolate movement for large distances only
    if distance > SENSITIVITY:
        interpolate(cur_x, cur_y, new_x, new_y)
        # mouse.position = (new_x, new_y)
    else:
        mouse.position = (new_x, new_y)

    # must return values to loop
    return new_x, new_y


def lerp(start, end, factor):
    '''Linear interpolation between start (current) and end (target) points'''
    return start + (end - start) * factor

def interpolate(start_x, start_y, end_x, end_y, steps=20, delay=0.005):
    """Interpolates between current and target positions to fill the visual gaps of the cursor"""
    for i in range(1, steps + 1):
        # Interpolate between start and end positions
        interp_x = lerp(start_x, end_x, i / steps)
        interp_y = lerp(start_y, end_y, i / steps)

        # Move the mouse to the interpolated position
        mouse.position = (int(interp_x), int(interp_y))

        # Small delay to ensure smooth visual movement
        time.sleep(delay)

async def read_serial(serial_reader, data_queue):
    """Read data asynchronously from the serial port."""
    buffer = b''  # To handle partial reads

    while True:
        try:
            # Read a chunk of data
            chunk = await serial_reader.read(32)  # Read up to 32 bytes or adjust as needed
            buffer += chunk

            # Process complete lines
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                data = line.decode().strip()
                await data_queue.put(data)

        except Exception as e:
            print(f"Error reading serial data: {e}")
            break

async def process_data(data_queue, cur_x, cur_y):
    """Process serial data and perform cursor actions"""
    global last_click

    while True:
        data = await data_queue.get()

        if "," in data:
            # read input x,y coords
            hand_label, x_loc, y_loc = data.split(',')
            x_loc, y_loc = float(x_loc), 1.0 - float(y_loc)             # flip y axis

            # convert to screen coordinates
            tar_x, tar_y = map_to_screen(x_loc, y_loc)

            # velocity scaling to create smooth movement & move cursor
            cur_x, cur_y = velocity_scale(cur_x, cur_y, tar_x, tar_y)

            print(f"{hand_label}: x={int(cur_x)}, y={int(cur_y)}")

        elif data == "click":
            current_time = time.time()
            if current_time - last_click > cooldown:
                mouse.click(Button.left)
                print("CLICK")
                last_click = current_time
            else:
                print("Double click blocked")

        elif data == "stop":
            raise StopException()

async def main():
    """Main event loop."""

    print("Listening for data from Raspberry Pi...")

    # set initial cur_x, cur_y
    cur_x, cur_y = 0, 0
    data_queue = asyncio.Queue() # for appending received data ready to be processed + translated into cursor action

    # get asynchronous access to serial port
    serial_port = await serial_asyncio.open_serial_connection(
        url='/dev/tty.usbmodem14101', baudrate=115200
    )

    try:
        # create and run tasks for reading and processing data
        async with asyncio.TaskGroup() as tg:
            tg.create_task(read_serial(serial_port[0], data_queue))
            tg.create_task(process_data(data_queue, cur_x, cur_y))

    except StopException:
        # if "stop" received, shut down program gracefully
        print("PROGRAM ENDED")


if __name__ == "__main__":
    asyncio.run(main())