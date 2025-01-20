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

# Smoothing buffers (optional)
buffer_size = 3
x_buffer = deque(maxlen=buffer_size)
y_buffer = deque(maxlen=buffer_size)

def lerp(start, end, factor):
    '''Linear interpolation between start (current) and end (target) points'''
    return start + (end - start) * factor

def moving_average(x, y):
    '''Moving average of last N points where N = buffer_size'''
    x_buffer.append(x)
    y_buffer.append(y)
    return sum(x_buffer) / len(x_buffer), sum(y_buffer) / len(y_buffer)

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

def velocity_scale(cur_x, cur_y, tar_x, tar_y, GAIN=5, scaling=50, min_speed=1, max_speed=20, DAMPING=0):
    """
    Adjust movement speed based on the distance to the target
    :param GAIN: multiplier on the speed of movement (high = faster)
    :param FRICTION:
    :param scaling: scale down speed as a function of distance
    """
    # Calculate Euclidian distance to the target
    distance = ((tar_x - cur_x) ** 2 + (tar_y - cur_y) ** 2) ** 0.5

    # apply damping to small movements
    # small distance = high damping = high speed = small step sizes
    damping = max(1, DAMPING / max(distance, 1e-6))

    # infer speed of cursor
    speed = min_speed + (distance / scaling) / GAIN / damping
    speed = min(speed, max_speed)  # Clamp to max_speed
    # print(f'speed: {speed}, distance: {distance}')

    # Update current position with scaled movement
    # creates smaller step sizes at fast speed (for smooth movement) and larger step sizes at slow speed (for precision)
    dx = (tar_x - cur_x) / speed
    dy = (tar_y - cur_y) / speed

    return cur_x + dx, cur_y + dy

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
            # read (x,y) coords
            hand_label, x_loc, y_loc = data.split(',')
            x_loc, y_loc = float(x_loc), 1.0 - float(y_loc)             # flip y axis

            # convert to screen coordinates
            tar_x, tar_y = map_to_screen(x_loc, y_loc)

            # velocity smoothing to create smooth movement
            new_x, new_y = velocity_scale(cur_x, cur_y, tar_x, tar_y)

            # interpolate cursor position to create smooth visuals during movement
            interpolate(cur_x, cur_y, new_x, new_y, steps=5, delay=0.005)

            # update current position
            cur_x, cur_y = new_x, new_y

            mouse.position = (cur_x, cur_y)
            print(f"{hand_label}: x={int(cur_x)}, y={int(cur_y)}")

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
