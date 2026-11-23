# Camera Diatonic Harmonica

Play a C-major diatonic harmonica with your webcam in your terminal. Each extended finger is counted and mapped to a harmonica hole:

- **Fingertip ABOVE the visible baseline** → plays that hole's **higher** note (blow/high)
- **Fingertip BELOW the visible baseline** → plays that hole's **deeper** note (draw/low)
- **Curled finger** → muted (supports single notes or multi-finger chords)

On first run, the app downloads FluidR3 GM harmonica `.mp3` samples from [gleitz/midi-js-soundfonts](https://github.com/gleitz/midi-js-soundfonts) into `sounds/` (with a built-in free-reed `.wav` synthesizer fallback if offline).

## Setup & Run on CachyOS (Arch Linux)

1. Ensure camera and PipeWire/SDL audio packages are available:
   ```bash
   sudo pacman -S --needed python python-virtualenv v4l-utils sdl2_mixer pipewire-pulse
   ```
2. Unzip and enter the `harmonica` folder:
   ```bash
   cd harmonica
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   *(If `mediapipe` has no wheel for Python 3.13+ on your system, install Python 3.12 via `pyenv` or `uv` and create the venv with `python3.12 -m venv .venv`.)*
3. Run the app:
   ```bash
   python main.py
   ```

## Controls

- **`W` / `S`** (or **Up / Down** arrows): Move the visible baseline up or down
- **`M`**: Toggle between **5-Finger Mode** (1 hand, Holes 4–8) and **10-Finger Mode** (2 hands, Holes 1–10)
- **`Q` / `ESC`**: Quit
