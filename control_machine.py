'''
Collects data produced by hand_tracking.py
'''

import serial
from pynput.mouse import Controller

# Initialize serial port
serial_port = serial.Serial('/dev/tty.usbmodem14101', 9600, timeout=1)  # Replace with your serial port

# Initialize mouse controller
mouse = Controller()

# Screen dimensions (adjust to your screen resolution)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

def map_to_screen(x, y):
    """Map normalized coordinates [0,1] to screen coordinates."""
    return int(x * SCREEN_WIDTH), int(y * SCREEN_HEIGHT)

print("Listening for data from Raspberry Pi...")

while True:
    try:
        # Read data from the serial port
        data = serial_port.readline().decode().strip()
        if data:
            print(data)
            # # Parse the x and y coordinates
            # x_loc, y_loc = map(float, data.split(','))
            # screen_x, screen_y = map_to_screen(x_loc, y_loc)
            #
            # # Move the mouse to the mapped screen position
            # mouse.position = (screen_x, screen_y)
    except Exception as e:
        print(f"Error: {e}")
        break
