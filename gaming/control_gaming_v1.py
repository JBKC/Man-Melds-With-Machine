'''
Translates hand data into mouse and keyboard actions, with WebSocket server for browser games
Focusses on FPS controls
'''

from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
from screeninfo import get_monitors
from collections import deque
import asyncio
import time
import struct
import json
import websockets
from websockets.server import serve

from arm.config import GAMING_PARAMS
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

# Initialise mouse clicks / position
active = None           # trigger to start all actions
last_click = 0
cooldown = 0.5          # seconds
scroll_anchor = None
drag_mode = False

# WebSocket server settings
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8765
browser_clients = set()  # Track connected browser clients

########
class StopException(Exception):
    """Custom exception to signal a graceful shutdown."""
    pass

def map_to_screen(loc):
    """Map integer coordinates (0->1000) to screen coordinates."""
    def zoom(value, screen_size):
        if value < 200:
            return 0
        elif value > 800:
            return screen_size
        else:
            return int(((value - 200) / 600) * screen_size)
    return [int(zoom(loc[0], SCREEN_WIDTH)), int(zoom(loc[1], SCREEN_HEIGHT))]

async def broadcast_to_browser(event_type, data):
    """Broadcast event data to all connected browser clients."""
    if browser_clients:
        message = json.dumps({"type": event_type, "data": data})
        await asyncio.gather(*(client.send(message) for client in browser_clients))

async def websocket_handler(websocket, path):
    """Handle WebSocket connections from the browser."""
    browser_clients.add(websocket)
    print(f"Browser connected: {len(browser_clients)} clients")
    try:
        await websocket.wait_closed()
    finally:
        browser_clients.remove(websocket)
        print(f"Browser disconnected: {len(browser_clients)} clients")

async def process_data(data_queue, cur):
    """Process data and perform cursor actions, sending movement/clicks to browser."""
    global active, last_click, scroll_anchor, drag_mode

    first_movement = True  # track if this is the first movement packet
    post_calibration = False  # track if this is the first packet post-calibration

    while True:
        data = await data_queue.get()
        data = data[:-1]  # Remove newline
        # print(data)

        # use scroll gesture to trigger motion (calibration in the centre)
        if not active and data == b'S':
            print("ACTIVATED")
            active = True

        elif active:

            if len(data) == 5:  # Movement packets
                try:
                    start_processing = time.time()

                    if data.startswith(b'D'):  # Drag mode == automatic firing
                        print("AUTOMATIC FIRE")
                        scroll_anchor = None

                        if not drag_mode:
                            mouse.press(Button.left)
                            drag_mode = True
                        else:
                            _, x_loc, y_loc = struct.unpack('=c2H', data)
                            loc = [int(x_loc), 1000 - int(y_loc)]
                            tar = map_to_screen(loc)
                            x_delta = (tar[0] - cur[0]) * GAMING_PARAMS['SENSITIVITY']
                            y_delta = (tar[1] - cur[1]) * GAMING_PARAMS['SENSITIVITY']
                            cur = tar
                            await broadcast_to_browser("mousemove", {"movementX": x_delta, "movementY": y_delta})


                    else:  # Camera movement (right analog stick equivalent)
                        scroll_anchor = None
                        if drag_mode:
                            mouse.release(Button.left)
                            drag_mode = False

                        _, x_loc, y_loc = struct.unpack('=c2H', data)
                        loc = [int(x_loc), 1000 - int(y_loc)]

                        if first_movement:
                            # if first movement packet received, centre on middle of screen
                            cur = [SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2]  # Center of the screen
                            first_movement = False
                            print(f"Initial cursor position centered at: {cur}")
                            continue  # Skip processing this packet to avoid snapping

                        if post_calibration:
                            # if just finished recalibrating, move cursor to location of hand on screen
                            cur = loc
                            post_calibration = False
                            print(f"Initial cursor position centered at: {cur}")
                            continue  # Skip processing this packet to avoid snapping

                        tar = map_to_screen(loc)
                        x_delta = (tar[0] - cur[0]) * 3
                        y_delta = (tar[1] - cur[1]) * 3
                        print(x_delta, y_delta)
                        cur = tar
                        await broadcast_to_browser("mousemove", {"movementX": x_delta, "movementY": y_delta})

                except Exception as e:
                    print(f"Error processing movement data: {e}")

            elif len(data) == 1:  # Command packets
                command = data
                current_time = time.time()

                if command == b'S':     # scroll CALIBRATION (move hand without cursor moving - like lifting your mouse)
                    print("CALIBRATING")
                    post_calibration = True

                    if drag_mode:
                        mouse.release(Button.left)
                        drag_mode = False

                elif command == b'C':  # Fire weapon
                    if current_time - last_click > cooldown:
                        mouse.press(Button.left)
                        mouse.release(Button.left)
                        # await broadcast_to_browser("click", {"button": "left"})
                        print("FIRE")
                        last_click = current_time
                    else:
                        print("Double click blocked")

                elif command == b'M':
                    # use mission control command (thumb and pinky tap) to exit code
                    raise StopException()

        elif not active:
            pass

async def start_websocket_server():
    """Start the WebSocket server for browser clients."""
    server = await serve(websocket_handler, WEBSOCKET_HOST, WEBSOCKET_PORT)
    print(f"WebSocket server started on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    await server.wait_closed()

async def main(data_queue=None):
    """Main event loop with WebSocket server."""
    print("Listening for data from Hand Tracking script...")
    cur = [0, 0]

    async with asyncio.TaskGroup() as tg:
        tg.create_task(start_websocket_server())
        tg.create_task(process_data(data_queue, cur))

if __name__ == "__main__":
    asyncio.run(main())