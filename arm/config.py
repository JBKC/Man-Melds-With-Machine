
# set subjective parameters for UX
PARAMS = {
    'GAIN': 5000,               # higher gain = faster cursor movement
    'DAMP': 50,                 # higher damping = less jitter when holding your hand still
    'SENSITIVITY': 3,          # higher sensitivity = expands region on the screen where your hand is considered to be held still
    'STEPS': 50,                # higher steps = smoother cursor movement (too much and it will become slow)
    'DELAY': 0.0001,            # higher delay = longer time between calculating each step of the cursor position. avoid high steps and high delay
    'SCROLL': 10                # higher scroll = slower / "heavier" scrolling
}

# mediapipe landmarks
HAND_LANDMARKS = {
    'MOVE_ID': 9,       # reference point of movement (base of third finger)
    'THUMB_TIP': 4,
    'INDEX_TIP': 8,
    'INDEX_J': 6,       # anchor for scrolling (first joint on index finger)
    'THUMB_J': 3,       # reference for clicks (joint of thumb)
    'MIDDLE_TIP': 12,
    'MIDDLE_J': 10,     # reference for fist (first joint on middle finger)
    'RING_TIP': 16,
    'LITTLE_TIP': 20,
    'WRIST': 0,
}

# define virtual frame size in pixels (480x270 is a good tradeoff between resolution and processing speed)
# FRAME_SIZE = {'width': 480, 'height': 270}
# FRAME_SIZE = {'width': 960, 'height': 540}
FRAME_SIZE = {'width': 1920, 'height': 1080}


# FPS of virtual camera
FPS = 60