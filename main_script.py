'''
Main script called from Docker container
'''

import mediapipe as mp
from pynput.mouse import Controller, Button

def main():
    print("started")

    # initialise
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    keyboard = Controller()

    print("finished")


if __name__ == '__main__':
    main()


