"""MediaPipe Tasks HandLandmarker tracker with baseline-relative pitch state."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import requests
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

# Standard 21-point MediaPipe hand skeleton connections
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
)


@dataclass
class FingerState:
    slot_id: int          # 0..4 in 5-finger mode, 0..9 in 10-finger mode
    tip_px: tuple[int, int]
    is_extended: bool     # True when finger is counted/active, False when curled
    is_above_line: bool   # True when fingertip is above the visible baseline


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _is_finger_extended(landmarks, finger_idx: int) -> bool:
    """
    Orientation-invariant extension check so fingers pointing DOWN below the
    baseline are still counted as extended (not mistaken for curled fingers).
    finger_idx: 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky
    """
    wrist = landmarks[0]
    if finger_idx == 0:
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        pinky_mcp = landmarks[17]
        return _dist(thumb_tip, pinky_mcp) > _dist(thumb_ip, pinky_mcp) * 1.12

    tip_ids = (4, 8, 12, 16, 20)
    pip_ids = (3, 6, 10, 14, 18)
    tip = landmarks[tip_ids[finger_idx]]
    pip = landmarks[pip_ids[finger_idx]]
    return _dist(tip, wrist) > _dist(pip, wrist) * 1.15


def _ensure_model(model_path: Path = MODEL_PATH) -> Path:
    if model_path.exists() and model_path.stat().st_size > 100_000:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print("[Tracker] Downloading MediaPipe hand_landmarker.task model...")
    resp = requests.get(MODEL_URL, timeout=30)
    resp.raise_for_status()
    model_path.write_bytes(resp.content)
    print("[Tracker] Model downloaded.")
    return model_path


def _draw_hand_skeleton(frame_bgr, landmarks, w: int, h: int) -> None:
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame_bgr, (x, y), 3, (255, 255, 255), -1)


class HandTracker:
    """Detects hands via MediaPipe Tasks HandLandmarker and maps fingers to slots."""

    TIP_INDICES = (4, 8, 12, 16, 20)

    def __init__(self, max_hands: int = 2) -> None:
        model_path = _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.55,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._start_ms = int(time.monotonic() * 1000)
        self._last_ts_ms = -1

    def process(
        self, frame_bgr, baseline_y_norm: float, ten_finger_mode: bool
    ) -> tuple[list[FingerState], int]:
        """
        Processes a mirrored BGR frame.
        Returns (finger_states, total_extended_count).
        """
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(time.monotonic() * 1000) - self._start_ms
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        results = self._landmarker.detect_for_video(mp_image, ts_ms)

        states: list[FingerState] = []
        if not results.hand_landmarks:
            return states, 0

        detected = []
        for idx, lms in enumerate(results.hand_landmarks):
            _draw_hand_skeleton(frame_bgr, lms, w, h)
            wrist_x = lms[0].x
            label = "Right"
            if results.handedness and idx < len(results.handedness):
                label = results.handedness[idx][0].category_name
            detected.append((wrist_x, label, lms))

        detected.sort(key=lambda item: item[0])

        if not ten_finger_mode:
            _, _, lms = detected[0]
            for f_idx, tip_id in enumerate(self.TIP_INDICES):
                tip = lms[tip_id]
                extended = _is_finger_extended(lms, f_idx)
                above = tip.y < baseline_y_norm
                states.append(
                    FingerState(
                        slot_id=f_idx,
                        tip_px=(int(tip.x * w), int(tip.y * h)),
                        is_extended=extended,
                        is_above_line=above,
                    )
                )
        else:
            for hand_pos, (_, _, lms) in enumerate(detected[:2]):
                if hand_pos == 0:
                    order = [4, 3, 2, 1, 0]  # Pinky..Thumb -> Holes 1..5
                    slot_offset = 0
                else:
                    order = [0, 1, 2, 3, 4]  # Thumb..Pinky -> Holes 6..10
                    slot_offset = 5

                for local_idx, f_idx in enumerate(order):
                    tip = lms[self.TIP_INDICES[f_idx]]
                    extended = _is_finger_extended(lms, f_idx)
                    above = tip.y < baseline_y_norm
                    states.append(
                        FingerState(
                            slot_id=slot_offset + local_idx,
                            tip_px=(int(tip.x * w), int(tip.y * h)),
                            is_extended=extended,
                            is_above_line=above,
                        )
                    )

        extended_count = sum(1 for s in states if s.is_extended)
        return states, extended_count

    def close(self) -> None:
        self._landmarker.close()
