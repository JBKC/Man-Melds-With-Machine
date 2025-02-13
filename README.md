# Man-Melds-With-Machine
Control your Mac using hand gestures, recorded by camera and interpreted by computer vision 

## Commands and what they do
1. Open Palm = 

# Always-On-Health-Monitor
Wrist wearable health monitor that takes sensor inputs and gives various real-time readings on health, fatigue state etc.

## STEP 1
Get good supervised HR estimate by training on the PPG-Dalia dataset, by recreating KID-PPG architecture (2024) - https://arxiv.org/abs/2405.09559

## STEP 2
Add in activity detection as in PPG Dalia (walking, cycling, etc)

## STEP 3
Use trained model to run realtime inference using Adafruit PPG and accelerometer sensors

## STEP 4
Model optimisation to run locally on a smartwatch devices, using Raspberry Pi or similar
