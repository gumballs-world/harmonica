"""Configuration and C-major diatonic harmonica mappings."""

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SOUNDS_DIR = BASE_DIR / "sounds"

# Pre-rendered FluidR3 GM Harmonica soundfont samples (CC-BY-3.0, gleitz/midi-js-soundfonts)
SOUNDFONT_BASE_URL = (
    "https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM/harmonica-mp3"
)

# Default normalized vertical position of the visible baseline (0.0 = top, 1.0 = bottom)
DEFAULT_BASELINE_Y = 0.50
BASELINE_STEP = 0.03

# Audio settings
SAMPLE_RATE = 44100
DEFAULT_DURATION_SEC = 1.6
FADE_OUT_MS = 120
MAX_MIXER_CHANNELS = 16


@dataclass(frozen=True)
class HarmonicaHole:
    hole: int
    finger_label: str
    high_note: str  # Played when fingertip is ABOVE the baseline
    low_note: str   # Played when fingertip is BELOW the baseline


# Note frequencies (Hz) for synthesis fallback & HUD display
NOTE_FREQUENCIES: dict[str, float] = {
    "C4": 261.63,
    "D4": 293.66,
    "E4": 329.63,
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00,
    "B4": 493.88,
    "C5": 523.25,
    "D5": 587.33,
    "E5": 659.25,
    "F5": 698.46,
    "G5": 783.99,
    "A5": 880.00,
    "B5": 987.77,
    "C6": 1046.50,
    "D6": 1174.66,
    "E6": 1318.51,
    "F6": 1396.91,
    "G6": 1567.98,
    "A6": 1760.00,
    "B6": 1975.53,
    "C7": 2093.00,
}

# 5-Finger (1-Hand) Mode: Middle octave of a 10-hole C diatonic harmonica (Holes 4-8)
# Each pair uses that hole's diatonic notes ordered so ABOVE = higher pitch, BELOW = deeper pitch.
FIVE_FINGER_MAP: tuple[HarmonicaHole, ...] = (
    HarmonicaHole(hole=4, finger_label="Thumb", high_note="D5", low_note="C5"),
    HarmonicaHole(hole=5, finger_label="Index", high_note="F5", low_note="E5"),
    HarmonicaHole(hole=6, finger_label="Middle", high_note="A5", low_note="G5"),
    HarmonicaHole(hole=7, finger_label="Ring", high_note="C6", low_note="B5"),
    HarmonicaHole(hole=8, finger_label="Pinky", high_note="E6", low_note="D6"),
)

# 10-Finger (2-Hand) Mode: Full 10-hole C major diatonic harmonica
# Left hand (Pinky -> Thumb) = Holes 1-5, Right hand (Thumb -> Pinky) = Holes 6-10
TEN_FINGER_MAP: tuple[HarmonicaHole, ...] = (
    HarmonicaHole(hole=1, finger_label="L Pinky", high_note="D4", low_note="C4"),
    HarmonicaHole(hole=2, finger_label="L Ring", high_note="G4", low_note="E4"),
    HarmonicaHole(hole=3, finger_label="L Middle", high_note="B4", low_note="G4"),
    HarmonicaHole(hole=4, finger_label="L Index", high_note="D5", low_note="C5"),
    HarmonicaHole(hole=5, finger_label="L Thumb", high_note="F5", low_note="E5"),
    HarmonicaHole(hole=6, finger_label="R Thumb", high_note="A5", low_note="G5"),
    HarmonicaHole(hole=7, finger_label="R Index", high_note="C6", low_note="B5"),
    HarmonicaHole(hole=8, finger_label="R Middle", high_note="E6", low_note="D6"),
    HarmonicaHole(hole=9, finger_label="R Ring", high_note="G6", low_note="F6"),
    HarmonicaHole(hole=10, finger_label="R Pinky", high_note="C7", low_note="A6"),
)

# BGR colors for OpenCV HUD
COLOR_BASELINE = (0, 230, 255)
COLOR_HIGH = (80, 235, 120)     # Green for above-line (higher note)
COLOR_LOW = (60, 155, 255)      # Amber/orange for below-line (deeper note)
COLOR_MUTED = (140, 140, 140)   # Gray for curled/inactive finger
COLOR_TEXT = (245, 245, 245)
COLOR_PANEL_BG = (24, 24, 28)
