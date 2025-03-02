### Installing and Running

1. Create a Python environment: <br>
   `python3 -m venv venv`

2. Activate the environment: <br>
   `source ./venv/bin/activate`

3. Install the required dependencies: <br>
   `pip install -r ./requirements.txt`

4. If running on a Linux distro: <br>

   - For X11 sessions: <br>
     `export QT_QPA_PLATFORM=xcb` <br>
     **_Note:_** If xcb is missing dependencies, install `libxcb-xinerama0` via `apt` or `dnf`.<br>
   - For Wayland sessions:<br>
     `export QT_QPA_PLATFORM=wayland`

5. Run the application: <br>
   `python3 ./main.py`

Link to this repository: https://github.com/Soof4/DIP-Project
