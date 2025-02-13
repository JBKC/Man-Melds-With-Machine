"""
Script for running both the camera tracking and machine controller scripts asynchronously
Call this script when running the app from a single machine - ie RUN_MODE = async
"""

import asyncio
import hand_tracking_v2, control_machine


async def run_scripts():
    """Simultaneously call 2 scripts"""

    # Create shared queue for async communication:
    # stores bytes of data that gets passed from hand_tracking_v2.py to control_machine.py
    data_queue = asyncio.Queue()

    await asyncio.gather(
        hand_tracking_v2.main(data_queue),
        control_machine.main(data_queue)
    )

if __name__ == "__main__":
    asyncio.run(run_scripts())