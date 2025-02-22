# Man-Melds-With-Machine
Control your Mac in real-time using hand gestures, recorded by camera and interpreted by computer vision 

High-level how it works = run code, raise one hand in front of camera, camera reads hand gesture, gesture gets interpreted into mouse / keyboard action, Mac is controlled 

See demo here: https://x.com/jbelevate/status/1888661396189290976

## Commands and what they do
1. Move cursor = open palm (reference point for movement is base of third finger)
2. Scroll = keep index and middle finger raised; curl ring and little finger into fist (reference point for scroll is tip of index finger)
3. Left mouse click = tap tip of thumb and tip of index finger
4. Tab backwards = tap tip of thumb and tip of middle finger
5. Tab forwards = tap tip of thumb and tip of ring finger
6. Mission control = tap tip of thumb and tip of little finger
7. Exit program = make a fist with all fingers

## Install it
### 1. Clone this repository:
   ```
   git clone https://github.com/JBKC/Man-Melds-With-Machine.git
   ```

### 2. Create and activate virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

### 3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Run it
Program revolves around 2 scripts - `hand_tracking_v2.py` and `control_machine.py`.
First script reads camera frames and sends data to second script, which translates data into mouse / keyboard actions.
Use `config.py` to change paramaters including how the UX feels

### 1. If operating from a single machine
Run
```
main_script.py
```
This will automatically call both scripts simultaneously - just raise your hand to begin controlling the screen

### 2. If operating off 2 machines (e.g. Mac and Raspberry Pi)
Run 
```
hand_tracking_v2.py
```
on the Pi, and
```
control_machine.py
```
on the Mac. Then raise hand to start moving the cursor around



