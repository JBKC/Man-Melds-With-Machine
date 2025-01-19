import serial
import pyautogui
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
from collections import deque

# Initialize serial port
serial_port = serial.Serial('/dev/tty.usbmodem14101', 9600, timeout=1)

# Initialize mouse controller
mouse = Controller()

# Get screen dimensions
monitors = get_monitors()
primary_monitor = monitors[0]
screen_width = primary_monitor.width
screen_height = primary_monitor.height

# Smoothing buffers
buffer_size = 5
x_buffer = deque(maxlen=buffer_size)
y_buffer = deque(maxlen=buffer_size)

def smooth_coordinates(x, y):
    x_buffer.append(x)
    y_buffer.append(y)
    return sum(x_buffer) / len(x_buffer), sum(y_buffer) / len(y_buffer)

def map_to_screen(x, y):
    """Map normalized coordinates [0,1] to screen coordinates."""
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)
    return smooth_coordinates(screen_x, screen_y)

print("Listening for data from Raspberry Pi...")

while True:
    try:
        # Read data from the serial port
        data = serial_port.readline().decode().strip()
        if "," in data:
            hand_label, x_loc, y_loc = data.split(',')
            x_loc, y_loc = float(x_loc), 1.0 - float(y_loc)             # flip y axis

            # Map to screen and smooth
            screen_x, screen_y = map_to_screen(x_loc, y_loc)

            # Debug output
            print(f"{hand_label}: x={screen_x}, y={screen_y}")

            # Move the mouse
            mouse.position = (screen_x, screen_y)

        elif data == "click":
            mouse.click(Button.left)
            print("Mouse click triggered!")

    except Exception as e:
        print(f"Error: {e}")
        break
