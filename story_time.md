# Harmonica — story_time.md: The Python Files, Line by Line

This document is written so you can read it and then write the code yourself, without help.
It walks the Python files of the project: `audio_manager.py`, `config.py`, `hand_tracker.py`,
and `main.py`.

---

## 1. `config.py` — configuration and mappings, line by line

The file has 90 lines.

### L1: `"""Configuration and C-major diatonic harmonica mappings."""`
Module docstring (documentation of the module).

### L3: `from dataclasses import dataclass`
Import `dataclass` — used for `HarmonicaHole` (L25–30).

### L4: `from pathlib import Path`
Import `Path`.

### L6: `BASE_DIR = Path(__file__).resolve().parent`
Directory containing `config.py` (the harmonica project root).

### L7: `SOUNDS_DIR = BASE_DIR / "sounds"`
Directory where note audio samples live.

### L10–12: `SOUNDFONT_BASE_URL = (...)`
Base URL of the FluidR3 GM harmonica soundfont MP3 samples hosted on GitHub
(`gleitz.github.io/midi-js-soundfonts`, CC-BY-3.0 licensed, `gleitz/midi-js-soundfonts`).

### L15: `DEFAULT_BASELINE_Y = 0.50`
Default vertical position of the visible baseline: 0.0 = top of the frame, 1.0 = bottom
(normalized).

### L16: `BASELINE_STEP = 0.03`
Step size when the baseline moves up/down.

### L19: `SAMPLE_RATE = 44100`
Audio sample rate in Hz.

### L20: `DEFAULT_DURATION_SEC = 1.6`
Duration of a note sample in seconds.

### L21: `FADE_OUT_MS = 120`
(Unused in the current version — no other reference to it in the code.)

### L22: `MAX_MIXER_CHANNELS = 16`
(Unused in the current version — no other reference.)

### L25–30:
```
@dataclass(frozen=True)
class HarmonicaHole:
    hole: int
    finger_label: str
    high_note: str  # Played when fingertip is ABOVE the baseline
    low_note: str   # Played when fingertip is BELOW the baseline
```
A frozen (immutable) dataclass describing one harmonica hole: its number, the finger label, and
the note played when the fingertip is above vs. below the baseline.

### L34–57: `NOTE_FREQUENCIES: dict[str, float]`
Dictionary mapping note names (`"C4"` … `"C7"`) to their frequencies in Hz — used for the
synthesis fallback and HUD display.

### L61–67: `FIVE_FINGER_MAP: tuple[HarmonicaHole, ...]`
5-finger mode mapping: the middle octave of a 10-hole C diatonic harmonica (holes 4–8):
Thumb→hole 4, Index→hole 5, Middle→hole 6, Ring→hole 7, Pinky→hole 8. Each hole has a high_note
(fingertip above the baseline) and a low_note (below).

### L71–82: `TEN_FINGER_MAP: tuple[HarmonicaHole, ...]`
10-finger mode mapping: all 10 holes. Left hand (Pinky→Thumb) = holes 1–5, right hand
(Thumb→Pinky) = holes 6–10.

### L85–90: Color constants (BGR tuples for OpenCV)
`COLOR_BASELINE` (cyan-ish, for the baseline line), `COLOR_HIGH` (green — fingertip above the
line), `COLOR_LOW` (orange/amber — below the line), `COLOR_MUTED` (gray — curled finger),
`COLOR_TEXT` (white), `COLOR_PANEL_BG` (dark panel background).

---

## 2. `audio_manager.py` — audio playback, line by line

The file has 194 lines.

### L1: `"""Downloads online harmonica note samples and manages polyphonic playback via sounddevice."""`
Module docstring.

### L3: `from __future__ import annotations`
Allows newer type syntax (e.g., `list[str]`, `str | None`) even on older Pythons.

### L5–10: Imports
- L5: `shutil` — used to find `ffmpeg` (L125).
- L6: `subprocess` — used to run `ffmpeg` (L126–141).
- L7: `threading` — `Lock` for the active-notes dict.
- L8: `wave` — read/write WAV files.
- L9: `Path`.
- L10: `from typing import Dict`.
- L12: `numpy as np`.
- L13: `requests` — HTTP download of MP3 samples.
- L14: `sounddevice as sd` — audio playback via PortAudio.
- L16–24: imports from `config` — `DEFAULT_DURATION_SEC`, `FIVE_FINGER_MAP`, `NOTE_FREQUENCIES`,
  `SAMPLE_RATE`, `SOUNDFONT_BASE_URL`, `SOUNDS_DIR`, `TEN_FINGER_MAP`.

### L27–32: `def _all_required_notes() -> list[str]:`
Collect the unique set of notes needed: for every hole in `FIVE_FINGER_MAP + TEN_FINGER_MAP`,
add both `high_note` and `low_note`; return the sorted list.

### L35–67: `def _synthesize_harmonica_wav(note: str, out_path: Path) -> Path:`
Fallback free-reed harmonica synthesizer (used when offline or `ffmpeg` is unavailable):
- L37: `freq = NOTE_FREQUENCIES[note]` — the note's frequency in Hz.
- L38–39: `n_samples = int(SAMPLE_RATE * DEFAULT_DURATION_SEC)`; `t = np.linspace(0.0, DEFAULT_DURATION_SEC, n_samples, endpoint=False)` — time vector.
- L41–42: vibrato: `vibrato = 1.0 + 0.003 * np.sin(2.0 * np.pi * 5.2 * t)`; `phase = 2.0 * np.pi * freq * np.cumsum(vibrato) / SAMPLE_RATE` — a wobble of ±0.3% at 5.2 Hz, accumulated into the phase.
- L44–50: `signal = 0.55 * np.sin(phase) + 0.25 * np.sin(2.0 * phase) + 0.16 * np.sin(3.0 * phase) + 0.08 * np.sin(4.0 * phase) + 0.05 * np.sin(5.0 * phase)` — a harmonic stack of the fundamental plus harmonics 2–5 (the "hollow" harmonica timbre).
- L52–56: envelope — `attack_len = int(0.03 * SAMPLE_RATE)`; `release_len = int(0.08 * SAMPLE_RATE)`; `env = np.ones(n_samples)`; ramp up `env[:attack_len]` over 30 ms; ramp down `env[-release_len:]` over 80 ms.
- L58–59: `audio = np.clip(signal * env * 0.45, -1.0, 1.0)`; `pcm = (audio * 32767).astype(np.int16)` — normalize, clip, convert to 16-bit PCM.
- L61–65: `with wave.open(str(out_path), "wb") as wf:` — mono (`setnchannels(1)`), 16-bit (`setsampwidth(2)`), `setframerate(SAMPLE_RATE)`, `writeframes(pcm.tobytes())`.
- L67: `return out_path`.

### L70–85: `def _load_wav_float32(path: Path) -> np.ndarray:`
Read a WAV file and return mono float32 data in [-1, 1):
- L71–74: `wave.open(..., "rb")`, read channels/sampwidth/frames, `frames = wf.readframes(wf.getnframes())`.
- L76–81: dtype conversion — 2-byte samples → `np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0`; 4-byte → `int32 / 2147483648.0`; otherwise → `uint8 / 128.0 - 1.0`.
- L83–84: if multi-channel, `data.reshape(-1, n_channels).mean(axis=1)` — downmix to mono by averaging.
- L85: `return data`.

### L88–194: `class AudioManager:` — polyphonic playback
#### L88–89
`"""Caches online harmonica samples and mixes active finger notes in real time."""` — class docstring.

#### L91–108: `def __init__(self, sounds_dir: Path = SOUNDS_DIR) -> None:`
- L92–93: store `sounds_dir`; `mkdir(parents=True, exist_ok=True)` to create it.
- L95–98: `self.samples: Dict[str, np.ndarray] = {}` (cached note buffers), `self._lock = threading.Lock()`, `self._active: Dict[int, list] = {}` — slot_id → `[note_name, sample_cursor]`.
- L99: `self._prepare_notes()` — load samples at startup.
- L101–108: `self._stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=1024, callback=self._audio_callback)`; `self._stream.start()` — open and start the audio output stream (PortAudio). The callback is invoked by the audio thread with `frames` per block.

#### L110–145: `def _download_or_fallback(self, note: str) -> Path:`
Get the WAV for `note`, downloading/converting if needed:
- L111–113: if `sounds_dir / f"{note}.wav"` exists and is non-empty → return it.
- L115–123: else, if `f"{note}.mp3"` is missing/empty, download it from `f"{SOUNDFONT_BASE_URL}/{note}.mp3"` with `requests.get(url, timeout=8)`; if status 200 and content > 512 bytes, write the MP3. (Errors are swallowed — offline fallback below.)
- L125–143: if the MP3 exists and `ffmpeg` is available, convert to WAV: `subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3_path), "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav_path)], capture_output=True)`; if it succeeded and the WAV exists → return it.
- L145: `return _synthesize_harmonica_wav(note, wav_path)` — synthesize a fallback sample.

#### L147–153: `def _prepare_notes(self) -> None:`
- L148: `notes = _all_required_notes()` — list of unique notes.
- L149: print `f"[Audio] Preparing {len(notes)} harmonica note samples..."`.
- L150–152: for each note, download/convert and store `self.samples[note] = _load_wav_float32(wav_path)`.
- L153: print `f"[Audio] Ready ({len(self.samples)} notes loaded)."`.

#### L155–174: `def _audio_callback(self, outdata, frames, time_info, status) -> None:`
Called by PortAudio to fill `outdata` (shape (frames, 1)) with the mix:
- L156: `mix = np.zeros(frames, dtype=np.float32)`.
- L157–158: `with self._lock: active_items = list(self._active.values())` — snapshot of active `[note, cursor]` pairs.
- L160–170: for each active `state`:
  - L162: `buf = self.samples.get(note)`; L163–164: if missing or empty, skip.
  - L167–169: `idx = (np.arange(frames) + cursor) % n`; `mix += buf[idx]` — sample the buffer starting at `cursor`, looping with modulo.
  - L170: `state[1] = (cursor + frames) % n` — advance the cursor.
- L172–174: if any active, `mix /= max(1.0, len(active_items) * 0.65)` — divide by active_count × 0.65 to keep levels in check; `outdata[:, 0] = np.clip(mix, -0.95, 0.95)`.

#### L176–185: `def update_finger_note(self, slot_id: int, note: str | None) -> None:`
Start, switch, or stop the note assigned to a finger slot:
- L178–182: `with self._lock: current = self._active.get(slot_id); if note is None: self._active.pop(slot_id, None); return` — `None` stops/removes the slot; if the slot already plays this note, return early.
- L185: `self._active[slot_id] = [note, 0]` — new note, cursor 0.

#### L187–189: `def stop_all(self) -> None:`
- L188–189: `with self._lock: self._active.clear()` — all slots silenced.

#### L191–194: `def close(self) -> None:`
- L192: `self.stop_all()`
- L193–194: `self._stream.stop(); self._stream.close()`.

---

## 3. `hand_tracker.py` — MediaPipe hand tracking, line by line

The file has 178 lines.

### L16–20: `MODEL_URL` / `MODEL_PATH`
`MODEL_URL` — URL of `hand_landmarker.task` (latest) from the MediaPipe model registry;
`MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"` — stored in
the `models/` folder of the project.

### L23–30: `HAND_CONNECTIONS: tuple[tuple[int, int], ...]`
The standard 21-point MediaPipe hand skeleton connections: Thumb (0–1, 1–2, 2–3, 3–4; 0–5, 5–6,
6–7, 7–8; 5–9, 9–10, 10–11, 11–12; 9–13, 13–14, 14–15, 15–16; 13–17, 17–18, 18–19, 19–20), and
Palm base (0–17).

### L33–38:
```
@dataclass
class FingerState:
    slot_id: int          # 0..4 in 5-finger mode, 0..9 in 10-finger mode
    tip_px: tuple[int, int]
    is_extended: bool     # True when finger is counted/active, False when curled
    is_above_line: bool   # True when fingertip is above the visible baseline
```
A frozen dataclass describing the state of one finger slot.

### L41–42: `def _dist(a, b) -> float: return math.hypot(a.x - b.x, a.y - b.y)`
2D Euclidean distance between two `(x, y)` points (used on MediaPipe landmarks).

### L45–62: `def _is_finger_extended(landmarks, finger_idx: int) -> bool:`
Orientation-invariant extension check so fingers pointing DOWN below the baseline are still
counted as extended (docstring at L46–50).
- L51: `wrist = landmarks[0]`.
- L52–56 (thumb): `return _dist(thumb_tip, pinky_mcp) > _dist(thumb_ip, pinky_mcp) * 1.12` — the
  thumb is "extended" if the tip is farther from the pinky base than the IP joint is
  (×1.12 factor).
- L58–59: `tip_ids = (4, 8, 12, 16, 20)` (tip landmark per finger, 0=Thumb);
  `pip_ids = (3, 6, 10, 14, 18)` (PIP joint per finger).
- L60–62: `tip = landmarks[tip_ids[finger_idx]]; pip = landmarks[pip_ids[finger_idx]]`;
  `return _dist(tip, wrist) > _dist(pip, wrist) * 1.15` — extended if the tip is farther from the
  wrist than the PIP is (×1.15).

### L65–75: `def _ensure_model(model_path: Path = MODEL_PATH) -> Path:`
- L66–67: if the file exists and `st_size > 100_000` → return it (already downloaded).
- L69: `model_path.parent.mkdir(parents=True, exist_ok=True)`.
- L70–74: print `[Tracker] Downloading MediaPipe hand_landmarker.task model...`;
  `resp = requests.get(MODEL_URL, timeout=30)`; `resp.raise_for_status()`;
  `model_path.write_bytes(resp.content)`; print `[Tracker] Model downloaded.`.
- L75: `return model_path`.

### L78–83: `def _draw_hand_skeleton(frame_bgr, landmarks, w: int, h: int) -> None:`
- L79: `pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]` — normalized → pixels.
- L80–81: for each `(a, b)` in `HAND_CONNECTIONS`: `cv2.line(frame_bgr, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)` — gray lines.
- L82–83: for each point: `cv2.circle(frame_bgr, (x, y), 3, (255, 255, 255), -1)` — white filled circles.

### L86–178: `class HandTracker:` — detect hands and map to finger slots
#### L86–87
`"""Detects hands via MediaPipe Tasks HandLandmarker and maps fingers to slots."""` — class docstring.

#### L89: `TIP_INDICES = (4, 8, 12, 16, 20)` — class attribute: MediaPipe tip landmark IDs.

#### L91–103: `def __init__(self, max_hands: int = 2) -> None:`
- L92: `model_path = _ensure_model()`.
- L93–100: `options = mp_vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(model_path)), running_mode=mp_vision.RunningMode.VIDEO, num_hands=max_hands, min_hand_detection_confidence=0.60, min_hand_presence_confidence=0.60, min_tracking_confidence=0.55)`.
- L101: `self._landmarker = mp_vision.HandLandmarker.create_from_options(options)`.
- L102: `self._start_ms = int(time.monotonic() * 1000)` — time base for frame timestamps.
- L103: `self._last_ts_ms = -1` — last timestamp sent to the landmarker.

#### L105–175: `def process(self, frame_bgr, baseline_y_norm: float, ten_finger_mode: bool) -> tuple[list[FingerState], int]:`
Process a mirrored BGR frame; returns `(finger_states, total_extended_count)`.
- L112: `h, w, _ = frame_bgr.shape`.
- L113: `rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)`.
- L114: `mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)`.
- L116–119: `ts_ms = int(time.monotonic() * 1000) - self._start_ms`; if `ts_ms <= self._last_ts_ms: ts_ms = self._last_ts_ms + 1`; `self._last_ts_ms = ts_ms` — enforce strictly increasing timestamps (VIDEO mode needs them).
- L121: `results = self._landmarker.detect_for_video(mp_image, ts_ms)`.
- L123–125: `states: list[FingerState] = []`; if no hands detected: `return states, 0`.
- L127–134: `detected` list: for each `lms` in `results.hand_landmarks`: draw skeleton via `_draw_hand_skeleton`; `wrist_x = lms[0].x`; default label `"Right"`; if `results.handedness` has an entry, use `results.handedness[idx][0].category_name`; append `(wrist_x, label, lms)`.
- L136: `detected.sort(key=lambda item: item[0])` — sort by wrist x-coordinate.
- L138–151 (5-finger mode): take `lms` of the first detected hand (`detected[0]`); for each finger index `f_idx` (0–4) and its `tip_id` in `TIP_INDICES`: `tip = lms[tip_id]`; `extended = _is_finger_extended(lms, f_idx)`; `above = tip.y < baseline_y_norm`; append `FingerState(slot_id=f_idx, tip_px=(int(tip.x * w), int(tip.y * h)), is_extended=extended, is_above_line=above)`.
- L152–172 (10-finger mode): for each of up to two hands (`detected[:2]`): `hand_pos` 0 → `order = [4, 3, 2, 1, 0]` (pinky→thumb, → slots 0–4) with `slot_offset = 0`; hand_pos 1 → `order = [0, 1, 2, 3, 4]` (thumb→pinky, → slots 5–9) with `slot_offset = 5`. For each `local_idx, f_idx` in `enumerate(order)`: `tip = lms[self.TIP_INDICES[f_idx]]`; `extended = _is_finger_extended(lms, f_idx)`; `above = tip.y < baseline_y_norm`; append `FingerState(slot_id=slot_offset + local_idx, ...)`.
- L174–175: `extended_count = sum(1 for s in states if s.is_extended)`; `return states, extended_count`.

#### L177–178: `def close(self) -> None: self._landmarker.close()`

---

## 4. `main.py` — camera loop and HUD, line by line

The file has 189 lines.

### L25–30: `def _open_camera(index: int = 0) -> cv2.VideoCapture:`
- L26–27: on Linux, `cap = cv2.VideoCapture(index, cv2.CAP_V4L2)`; if opened, `return cap`.
- L30: `return cv2.VideoCapture(index)` — fallback without V4L2.

### L33–91: `def _draw_hud(frame, baseline_y_norm, ten_finger_mode, extended_count, active_labels) -> None:`
- L41: `line_y = int(baseline_y_norm * h)`.
- L44: `cv2.line(frame, (0, line_y), (w, line_y), COLOR_BASELINE, 3)` — horizontal baseline line across the frame.
- L47–66: zone labels: "BLOW / HIGHER NOTE  ^" above the line in `COLOR_HIGH`; "DRAW / DEEPER NOTE  v" below it in `COLOR_LOW`.
- L69–81: top status bar (`COLOR_PANEL_BG` rectangle): `mode_name` ("10-Finger (2 Hands: Holes 1-10)" or "5-Finger (1 Hand: Holes 4-8)"); `notes_str = ", ".join(active_labels) if active_labels else "None (extend a finger)"`; text like `f"Mode: {mode_name}  |  Fingers: {extended_count}  |  Playing: {notes_str}"`.
- L82–91: controls line: `[W/S or Up/Down] Move Baseline  |  [M] Switch 5/10-Finger Mode  |  [Q] Quit`.

### L94–185: `def main() -> None:`
- L95: `audio = AudioManager()` — starts audio stream.
- L96: `tracker = HandTracker(max_hands=2)`.
- L97: `cap = _open_camera(0)`.
- L99–104: `if not cap.isOpened(): audio.close(); tracker.close(); raise RuntimeError("Could not open webcam. Check camera permissions or /dev/video0.")`.
- L106–107: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)`.
- L109–110: `baseline_y_norm = DEFAULT_BASELINE_Y` (0.5); `ten_finger_mode = False`.
- L112: `print("[Harmonica] Camera started. Press 'Q' or 'ESC' to quit.")`.
- L113–117: `try: while True: ok, frame = cap.read(); if not ok: break`.
- L119–123: `frame = cv2.flip(frame, 1)`; `finger_map = TEN_FINGER_MAP if ten_finger_mode else FIVE_FINGER_MAP`; `states, extended_count = tracker.process(frame, baseline_y_norm, ten_finger_mode)`.
- L125–155: for each finger state:
  - L129–130: skip if `st.slot_id >= len(finger_map)`; `seen_slots.add(st.slot_id)`; `hole_cfg = finger_map[st.slot_id]`.
  - L134–139: if extended: `note = hole_cfg.high_note if st.is_above_line else hole_cfg.low_note`; `color = COLOR_HIGH if st.is_above_line else COLOR_LOW`; `tag = f"H{hole_cfg.hole}:{note}"`; append to `active_labels`; `audio.update_finger_note(st.slot_id, note)`.
  - L140–143: else (curl/not extended): `color = COLOR_MUTED`; `tag = f"H{hole_cfg.hole}:--"`; `audio.update_finger_note(st.slot_id, None)`.
  - L145–155: draw the finger dot on the frame: `cv2.circle(frame, st.tip_px, 11, color, -1)`; `cv2.putText(frame, tag, (st.tip_px[0] + 12, st.tip_px[1] + 5), FONT_HERSHEY_SIMPLEX, 0.55, color, 2, LINE_AA)`.
- L157–160: silence slots no longer seen: `for slot_id in range(len(finger_map)): if slot_id not in seen_slots: audio.update_finger_note(slot_id, None)`.
- L162–168: `_draw_hud(frame, baseline_y_norm, ten_finger_mode, extended_count, active_labels)`.
- L170–171: `cv2.imshow("Diatonic Harmonica (Camera)", frame); key = cv2.waitKey(1) & 0xFF`.
- L172–180: keys: `q` or `27` (Esc) → break; `w` or `82` (Up) → `baseline_y_norm = max(0.15, baseline_y_norm - BASELINE_STEP)`; `s` or `84` (Down) → `baseline_y_norm = min(0.85, baseline_y_norm + BASELINE_STEP)`; `m` → `audio.stop_all(); ten_finger_mode = not ten_finger_mode`.
- L181–185: `finally:` — `cap.release(); cv2.destroyAllWindows(); tracker.close(); audio.close()`.
- L188–189: `if __name__ == "__main__": main()`.
