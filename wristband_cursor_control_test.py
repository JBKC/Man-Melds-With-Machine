'''
Uses inputs from wristband to control cursor movements + clicks
'''

import torch
import numpy as np
import serial
import asyncio
from pynput.mouse import Controller, Button
import time

# Create a mouse controller object
mouse = Controller()

# Move the cursor to a specific position (x, y)
mouse.position = (500, 300)
print(f"Moved to {mouse.position}")

# Wait for 1 second
time.sleep(1)

# Move the cursor relatively
mouse.move(100, 50)
print(f"New position: {mouse.position}")

# Perform a left click
mouse.click(Button.left, 1)

# Perform a double click
mouse.click(Button.left, 2)

# Perform a right click
mouse.click(Button.right, 1)

# Scroll up
mouse.scroll(0, 2)

# Scroll down
mouse.scroll(0, -2)
import collections

async def producer(ser):
    """
    Parses streaming data and appends to a buffer (sliding window).
    """

    print(f'Streaming data....')

    while True:
        if ser.in_waiting > 0:
            packet = ser.readline().decode('utf-8').strip()
            parts = packet.split(',')
            timestamp = float(parts[0])
            accel = tuple((float(parts[1]), float(parts[2]), float(parts[3])))
            print(parts)

            await asyncio.sleep(0)

async def main():
    '''
    Runs main asynchronous tasks
    '''

    serial_port = '/dev/tty.ESP32-Classic-ESP32SPP'
    baud_rate = 115200

    # Check Bluetooth connection
    print(f"Attempting to connect to Bluetooth device at {serial_port}...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print("Successfully connected to the ESP32 via Bluetooth!")
    except serial.SerialException as e:
        print(f"Failed to connect to the Bluetooth device: {e}")
        return

    # extract & process data concurrently
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(producer(ser))

    # run tasks
    await asyncio.gather(task1)


if __name__ == '__main__':
    asyncio.run(main())
