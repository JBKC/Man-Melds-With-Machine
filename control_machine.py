import serial
import pyautogui
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
from collections import deque
import time

# Initialize
serial_port = serial.Serial('/dev/tty.usbmodem14101', 9600, timeout=1)
mouse = Controller()

monitors = get_monitors()
primary_monitor = monitors[0]
screen_width = primary_monitor.width
screen_height = primary_monitor.height

# Smoothing buffers for moving average
buffer_size = 3
x_buffer = deque(maxlen=buffer_size)
y_buffer = deque(maxlen=buffer_size)

########

def velocity_scale(cur_x, cur_y, tar_x, tar_y, GAIN=5, DAMP=1000, SENSITIVITY=10, SCALING=50, MIN_STEP=1, MAX_STEP=20):
    """
    Adjust speed of cursor based on distance between current and target position by calculating a scaling factor
    :param cur_x, cur_y: current coords of cursor
    :param tar_x, tar_y: target coords of cursor
    :param GAIN: higher GAIN = bigger step size, meaning faster cursor movement
    :param DAMP: damping multiplier at small distances - higher DAMP = less static jitter
    :param SENSITIVITY: damping limit - damping applied when distance < SENSITIVITY
    :param SCALING: scales down distances for application of reasonable GAIN / damping values
    :param MAX_STEP: Euclidian distance of maximum step size permitted
    """

    # calculate Euclidian distance between current and target locations
    distance = ((tar_x - cur_x) ** 2 + (tar_y - cur_y) ** 2) ** 0.5

    # apply damping to limit step size when distances are very small (reduce static jitter)
    damping = max(1, SENSITIVITY / max(distance, 1e-6))
    damping *= DAMP if damping > 1 else 1

    # calculate scaling factor for cursor steps
    scaling_factor = MIN_STEP + (distance / SCALING) / GAIN / damping
    scaling_factor = min(scaling_factor, MAX_STEP)

    # print(distance)
    # print(scaling_factor)

    # calculate cursor step sizes
    dx = (tar_x - cur_x) / scaling_factor
    dy = (tar_y - cur_y) / scaling_factor

    print(dx)

    return cur_x + dx, cur_y + dy

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

def lerp(start, end, factor):
    '''Linear interpolation between start (current) and end (target) points'''
    return start + (end - start) * factor

def moving_average(x, y):
    '''Moving average of last N points where N = buffer_size'''
    x_buffer.append(x)
    y_buffer.append(y)
    return sum(x_buffer) / len(x_buffer), sum(y_buffer) / len(y_buffer)


def interpolate(start_x, start_y, end_x, end_y, steps=10, delay=0.005):
    """Interpolates between current and target positions to fill the visual gaps of the cursor"""
    for i in range(1, steps + 1):
        # Interpolate between start and end positions
        interp_x = lerp(start_x, end_x, i / steps)
        interp_y = lerp(start_y, end_y, i / steps)

        # Move the mouse to the interpolated position
        mouse.position = (int(interp_x), int(interp_y))

        # Small delay to ensure smooth visual movement
        time.sleep(delay)


# initialise current cursor position
cur_x, cur_y = 0, 0

# initialise mouse clicks
last_click = 0
cooldown = 0.2          # seconds

print("Listening for data from Raspberry Pi...")

while True:
    try:
        # Read data from the serial port
        data = serial_port.readline().decode().strip()

        if "," in data:
            # read input x,y coords
            hand_label, x_loc, y_loc = data.split(',')
            x_loc, y_loc = float(x_loc), 1.0 - float(y_loc)             # flip y axis

            # convert to screen coordinates
            tar_x, tar_y = map_to_screen(x_loc, y_loc)

            # velocity scaling to create smooth movement
            cur_x, cur_y = velocity_scale(cur_x, cur_y, tar_x, tar_y)

            # interpolate cursor position to create smooth visuals during movement
            # interpolate(cur_x, cur_y, tar_x, tar_y)

            # update current position
            # cur_x, cur_y = new_x, new_y

            mouse.position = (cur_x, cur_y)
            # print(f"{hand_label}: x={int(cur_x)}, y={int(cur_y)}")

        elif data == "click":
            current_time = time.time()
            if current_time - last_click > cooldown:
                mouse.click(Button.left)
                print("CLICK")
                last_click = current_time
            else:
                print("Double click blocked")

    except Exception as e:
        print(f"Error: {e}")
        break
