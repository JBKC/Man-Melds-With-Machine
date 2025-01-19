import serial
import pyautogui
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
from collections import deque

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
    """Map normalized coordinates [0,1] to screen coordinates."""
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)
    return screen_x, screen_y

def velocity_scale(cur_x, cur_y, tar_x, tar_y, min_speed=1, max_speed=20):
    """Adjust movement speed based on the distance to the target"""
    # Calculate Euclidian distance to the target
    distance = ((tar_x - cur_x) ** 2 + (tar_y - cur_y) ** 2) ** 0.5
    print(distance)

    # Determine scaling factor (speed increases with distance)
    speed = min_speed + (distance / 50)
    speed = min(speed, max_speed)  # Clamp to max_speed

    # Update current position with scaled movement
    dx = (tar_x - cur_x) / speed
    dy = (tar_y - cur_y) / speed

    return cur_x + dx, cur_y + dy

# initialise current cursor position
cur_x, cur_y = 0, 0

print("Listening for data from Raspberry Pi...")

while True:
    try:
        # Read data from the serial port
        data = serial_port.readline().decode().strip()

        if "," in data:
            # read (x,y) coords
            hand_label, x_loc, y_loc = data.split(',')
            tar_x, tar_y = float(x_loc), 1.0 - float(y_loc)             # flip y axis

            # smooth movement with velocity scaling
            cur_x, cur_y = velocity_scale(cur_x, cur_y, tar_x, tar_y)

            # convert to screen coordinates
            screen_x, screen_y = map_to_screen(cur_x, cur_y)
            mouse.position = (screen_x, screen_y)
            print(f"{hand_label}: x={int(screen_x)}, y={int(screen_y)}")

        elif data == "click":
            mouse.click(Button.left)
            print("CLICK")

    except Exception as e:
        print(f"Error: {e}")
        break
