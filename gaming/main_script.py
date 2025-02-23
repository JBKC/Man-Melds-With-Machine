"""
Script for running both the camera tracking and machine controller scripts asynchronously
"""

import asyncio
import tracking_gaming as hand_tracking
import control_gaming_v1 as control_mac


async def run_scripts():
    """Simultaneously call 2 scripts"""

    # Create shared queue for async communication:
    # stores bytes of data that gets passed from hand_tracking to control_mac
    data_queue = asyncio.Queue()


    await asyncio.gather(
        hand_tracking.main(data_queue),
        control_mac.main(data_queue)
    )

if __name__ == "__main__":
    asyncio.run(run_scripts())