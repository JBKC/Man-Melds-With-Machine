
params = {
    'batch_size': 256,

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
