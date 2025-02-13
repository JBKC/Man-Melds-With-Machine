
# set subjective parameters for UX
PARAMS = {
    'GAIN': 5000,               # higher gain = faster cursor movement
    'DAMP': 50,                 # higher damping = less jitter when holding your hand still
    'SENSITIVITY': 10,          # higher sensitivity = expands region on the screen where your hand is considered to be held still
    'STEPS': 20,                # higher steps = smoother cursor movement (too much and it will become slow)
    'DELAY': 0.0001             # higher delay = longer time between calculating each step of the cursor position. avoid high steps and high delay
}

# mediapipe landmarks
HAND_LANDMARKS = {
    'MOVE_ID': 9,  # reference point of movement (base of third finger)
    'THUMB_TIP': 4,
    'INDEX_TIP': 8,
    'THUMB_J': 3,  # reference for cliks (joint of thumb)
    'MIDDLE_TIP': 12,
    'RING_TIP': 16,
    'LITTLE_TIP': 20,
    'WRIST': 0,
}

# define virtual frame size in pixels (480x270 is a good tradeoff between resolution and processing speed)
FRAME_SIZE = {'width': 480, 'height': 270}

# initialise serial connections (only used when running the 2 scripts from separate machines)
SERIAL = {
    'camera_port': serial.Serial('/dev/ttyGS0', 115200, timeout=1),
    'serial_port': serial_asyncio.open_serial_connection(
        url='/dev/tty.usbmodem14101', baudrate=115200
    )
}