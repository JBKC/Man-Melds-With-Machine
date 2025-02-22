'''
Uses inputs from wristband to control cursor movements + clicks
'''

import torch
import numpy as np
import serial
import asyncio
from pynput.mouse import Controller, Button
import time
import collections



async def producer(ser, buffer):

    print(f'Streaming data....')

    while True:
        try:
            packet = ser.readline().decode('utf-8').strip()
            parts = packet.split(',')
            timestamp = float(parts[0])
            acc = tuple((float(parts[1]), float(parts[2]), float(parts[3])))

            buffer.append(acc)

        except Exception as e:
            print(f"Error reading data: {e}")

        await asyncio.sleep(0)

async def consumer(buffer, mouse):
    '''
    Simple mouse movements
    '''

    ### NEXT - add speed changes depending on angle

    while True:
        x,y,z = buffer[-1]
        print(y)
        if x > 0.2:
            mouse.move(dx=1, dy=0)
        if x < -0.2:
            mouse.move(dx=-1, dy=0)
        if y > 0.2:
            mouse.move(dx=0, dy=1)
        if y < -0.2:
            mouse.move(dx=0, dy=-1)

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

    buffer = collections.deque(maxlen=100)                    # buffer for storing accelerometer inputs

    # Create a mouse controller object
    mouse = Controller()

    # extract & process data concurrently
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(producer(ser, buffer))
        task2 = tg.create_task(consumer(buffer, mouse))

    # run tasks
    await asyncio.gather(task1, task2)


if __name__ == '__main__':
    asyncio.run(main())
