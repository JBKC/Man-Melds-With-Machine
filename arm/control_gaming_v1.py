'''
Translates hand data into mouse and keyboard actions, with WebSocket server for browser games
'''

from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
import pyautogui
from screeninfo import get_monitors
from collections import deque
import asyncio
import time
import struct
import json
import websockets
from websockets.server import serve
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
last_click = 0
cooldown = 0.5          # seconds
scroll_anchor = None
zoom_anchor = None
zoom_mode = False
drag_mode = False
voice_mode = False

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
    global last_click, scroll_anchor, zoom_anchor, zoom_mode, drag_mode, voice_mode

    while True:
        data = await data_queue.get()
        data = data[:-1]  # Remove newline
        # print(f"Received: {data}")

        if len(data) == 5:  # Movement packets
            try:
                start_processing = time.time()

                if data.startswith(b'S'):  # Scrolling mode
                    print("SCROLL MODE")
                    zoom_mode = False
                    zoom_anchor = None
                    if drag_mode:
                        mouse.release(Button.left)
                        drag_mode = False
                    if voice_mode:
                        pykeyboard.release(Key.alt)
                        time.sleep(1.0)
                        pykeyboard.press(Key.enter)
                        pykeyboard.release(Key.enter)
                        voice_mode = False

                    _, scroll_loc, anchor_loc = struct.unpack('=c2H', data)
                    scroll_loc = 1000 - int(scroll_loc)
                    anchor_loc = 1000 - int(anchor_loc)
                    if scroll_anchor is None:
                        scroll_anchor = anchor_loc
                    scroll_y = int((scroll_anchor - scroll_loc) / PARAMS['SCROLL'])
                    mouse.scroll(dx=0, dy=scroll_y)

                elif data.startswith(b'D'):  # Drag mode
                    print("DRAG MODE")
                    scroll_anchor = None
                    zoom_anchor = None
                    if voice_mode:
                        pykeyboard.release(Key.alt)
                        time.sleep(1.0)
                        pykeyboard.press(Key.enter)
                        pykeyboard.release(Key.enter)
                        voice_mode = False

                    _, x_loc, y_loc = struct.unpack('=c2H', data)
                    loc = [int(x_loc), 1000 - int(y_loc)]
                    if not drag_mode:
                        mouse.press(Button.left)
                        drag_mode = True
                    else:
                        cur = map_to_screen(loc)
                        mouse.position = (cur[0], cur[1])

                else:  # Cursor movement mode (send to browser)
                    scroll_anchor = None
                    zoom_anchor = None
                    if drag_mode:
                        mouse.release(Button.left)
                        drag_mode = False
                    if voice_mode:
                        pykeyboard.release(Key.alt)
                        time.sleep(1.0)
                        pykeyboard.press(Key.enter)
                        pykeyboard.release(Key.enter)
                        voice_mode = False

                    _, x_loc, y_loc = struct.unpack('=c2H', data)
                    loc = [int(x_loc), 1000 - int(y_loc)]
                    tar = map_to_screen(loc)
                    x_delta = tar[0] - cur[0]
                    y_delta = tar[1] - cur[1]
                    cur = tar
                    await broadcast_to_browser("mousemove", {"movementX": x_delta, "movementY": y_delta})

            except Exception as e:
                print(f"Error processing movement data: {e}")

        elif len(data) == 3:  # Zoom packets (placeholder)
            if data.startswith(b'Z'):
                print("ZOOM MODE")
                if drag_mode:
                    mouse.release(Button.left)
                    drag_mode = False
                scroll_anchor = None
                if voice_mode:
                    pykeyboard.release(Key.alt)
                    time.sleep(1.0)
                    pykeyboard.press(Key.enter)
                    pykeyboard.release(Key.enter)
                    voice_mode = False
                _, distance = struct.unpack('=cH', data)
                if zoom_anchor is None:
                    zoom_anchor = distance
                # Placeholder zoom logic

        elif len(data) == 1:  # Command packets
            command = data
            current_time = time.time()

            if command == b'V':
                print("VOICE MODE")
                scroll_anchor = None
                zoom_anchor = None
                if drag_mode:
                    mouse.release(Button.left)
                    drag_mode = False
                if not voice_mode:
                    pykeyboard.press(Key.alt)
                    voice_mode = True

            elif command == b'C':  # Click command (send to browser)
                if current_time - last_click > cooldown:
                    await broadcast_to_browser("click", {"button": "left"})
                    print("CLICK sent to browser")
                    last_click = current_time
                else:
                    print("Double click blocked")

            elif command == b'E':
                raise StopException()

            elif command == b'F':
                if current_time - last_click > cooldown:
                    with pykeyboard.pressed(Key.ctrl):
                        pykeyboard.press(Key.tab)
                        pykeyboard.release(Key.tab)
                    print("NEXT TAB")
                    last_click = current_time
                else:
                    print("Double tab forward blocked")

            elif command == b'B':
                if current_time - last_click > cooldown:
                    with pykeyboard.pressed(Key.ctrl):
                        with pykeyboard.pressed(Key.shift):
                            pykeyboard.press(Key.tab)
                            pykeyboard.release(Key.tab)
                    print("PREVIOUS TAB")
                    last_click = current_time
                else:
                    print("Double tab back blocked")

            elif command == b'M':
                if current_time - last_click > cooldown:
                    pyautogui.keyDown("ctrl")
                    pyautogui.press("up")
                    pyautogui.keyUp("ctrl")
                    print("MISSION CONTROL")
                    last_click = current_time
                else:
                    print("Double mission control blocked")

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