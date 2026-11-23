"""Downloads online harmonica note samples and manages polyphonic playback via sounddevice."""

from __future__ import annotations

import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Dict

import numpy as np
import requests
import sounddevice as sd

from config import (
    DEFAULT_DURATION_SEC,
    FIVE_FINGER_MAP,
    NOTE_FREQUENCIES,
    SAMPLE_RATE,
    SOUNDFONT_BASE_URL,
    SOUNDS_DIR,
    TEN_FINGER_MAP,
)


def _all_required_notes() -> list[str]:
    notes = set()
    for item in (*FIVE_FINGER_MAP, *TEN_FINGER_MAP):
        notes.add(item.high_note)
        notes.add(item.low_note)
    return sorted(notes)


def _synthesize_harmonica_wav(note: str, out_path: Path) -> Path:
    """Fallback free-reed harmonic synthesizer if offline or ffmpeg is unavailable."""
    freq = NOTE_FREQUENCIES[note]
    n_samples = int(SAMPLE_RATE * DEFAULT_DURATION_SEC)
    t = np.linspace(0.0, DEFAULT_DURATION_SEC, n_samples, endpoint=False)

    vibrato = 1.0 + 0.003 * np.sin(2.0 * np.pi * 5.2 * t)
    phase = 2.0 * np.pi * freq * np.cumsum(vibrato) / SAMPLE_RATE

    signal = (
        0.55 * np.sin(phase)
        + 0.25 * np.sin(2.0 * phase)
        + 0.16 * np.sin(3.0 * phase)
        + 0.08 * np.sin(4.0 * phase)
        + 0.05 * np.sin(5.0 * phase)
    )

    attack_len = int(0.03 * SAMPLE_RATE)
    release_len = int(0.08 * SAMPLE_RATE)
    env = np.ones(n_samples, dtype=np.float64)
    env[:attack_len] = np.linspace(0.0, 1.0, attack_len)
    env[-release_len:] = np.linspace(1.0, 0.0, release_len)

    audio = np.clip(signal * env * 0.45, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return out_path


def _load_wav_float32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data


class AudioManager:
    """Caches online harmonica samples and mixes active finger notes in real time."""

    def __init__(self, sounds_dir: Path = SOUNDS_DIR) -> None:
        self.sounds_dir = sounds_dir
        self.sounds_dir.mkdir(parents=True, exist_ok=True)

        self.samples: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        # slot_id -> [note_name, sample_cursor]
        self._active: Dict[int, list] = {}
        self._prepare_notes()

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _download_or_fallback(self, note: str) -> Path:
        wav_path = self.sounds_dir / f"{note}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 0:
            return wav_path

        mp3_path = self.sounds_dir / f"{note}.mp3"
        if not (mp3_path.exists() and mp3_path.stat().st_size > 0):
            url = f"{SOUNDFONT_BASE_URL}/{note}.mp3"
            try:
                resp = requests.get(url, timeout=8)
                if resp.status_code == 200 and len(resp.content) > 512:
                    mp3_path.write_bytes(resp.content)
            except requests.RequestException:
                pass

        if mp3_path.exists() and shutil.which("ffmpeg"):
            res = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(mp3_path),
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "1",
                    str(wav_path),
                ],
                capture_output=True,
            )
            if res.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0:
                return wav_path

        return _synthesize_harmonica_wav(note, wav_path)

    def _prepare_notes(self) -> None:
        notes = _all_required_notes()
        print(f"[Audio] Preparing {len(notes)} harmonica note samples...")
        for note in notes:
            wav_path = self._download_or_fallback(note)
            self.samples[note] = _load_wav_float32(wav_path)
        print(f"[Audio] Ready ({len(self.samples)} notes loaded).")

    def _audio_callback(self, outdata, frames, time_info, status) -> None:
        mix = np.zeros(frames, dtype=np.float32)
        with self._lock:
            active_items = list(self._active.values())

        for state in active_items:
            note, cursor = state
            buf = self.samples.get(note)
            if buf is None or len(buf) == 0:
                continue

            # Loop smoothly through the sustain region of the sample
            n = len(buf)
            idx = (np.arange(frames) + cursor) % n
            mix += buf[idx]
            state[1] = (cursor + frames) % n

        if active_items:
            mix /= max(1.0, len(active_items) * 0.65)
        outdata[:, 0] = np.clip(mix, -0.95, 0.95)

    def update_finger_note(self, slot_id: int, note: str | None) -> None:
        """Starts, switches, or stops the note assigned to a finger slot."""
        with self._lock:
            current = self._active.get(slot_id)
            if note is None:
                self._active.pop(slot_id, None)
                return
            if current is not None and current[0] == note:
                return
            self._active[slot_id] = [note, 0]

    def stop_all(self) -> None:
        with self._lock:
            self._active.clear()

    def close(self) -> None:
        self.stop_all()
        self._stream.stop()
        self._stream.close()
