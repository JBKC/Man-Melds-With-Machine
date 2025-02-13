import asyncio
import hand_tracking_v2, control_machine

async def run_scripts():
    """Simultaneously call 2 scripts (producer and consumer)"""
    await asyncio.gather(
        hand_tracking_v2.main(),
        control_machine.main()
    )

if __name__ == "__main__":
    asyncio.run(run_scripts())