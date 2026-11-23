"""Camera-controlled Diatonic Harmonica in Python."""

from __future__ import annotations

import platform

import cv2

from audio_manager import AudioManager
from config import (
    BASELINE_STEP,
    COLOR_BASELINE,
    COLOR_HIGH,
    COLOR_LOW,
    COLOR_MUTED,
    COLOR_PANEL_BG,
    COLOR_TEXT,
    DEFAULT_BASELINE_Y,
    FIVE_FINGER_MAP,
    TEN_FINGER_MAP,
)
from hand_tracker import HandTracker


def _open_camera(index: int = 0) -> cv2.VideoCapture:
    if platform.system() == "Linux":
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(index)


def _draw_hud(
    frame,
    baseline_y_norm: float,
    ten_finger_mode: bool,
    extended_count: int,
    active_labels: list[str],
) -> None:
    h, w, _ = frame.shape
    line_y = int(baseline_y_norm * h)

    # Visible horizontal baseline across the camera feed
    cv2.line(frame, (0, line_y), (w, line_y), COLOR_BASELINE, 3)

    # Zone labels above and below the baseline
    cv2.putText(
        frame,
        "BLOW / HIGHER NOTE  ^",
        (16, max(28, line_y - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_HIGH,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "DRAW / DEEPER NOTE  v",
        (16, min(h - 14, line_y + 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_LOW,
        2,
        cv2.LINE_AA,
    )

    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 54), COLOR_PANEL_BG, -1)
    mode_name = "10-Finger (2 Hands: Holes 1-10)" if ten_finger_mode else "5-Finger (1 Hand: Holes 4-8)"
    notes_str = ", ".join(active_labels) if active_labels else "None (extend a finger)"
    cv2.putText(
        frame,
        f"Mode: {mode_name}  |  Fingers: {extended_count}  |  Playing: {notes_str}",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Controls: [W/S or Up/Down] Move Baseline  |  [M] Switch 5/10-Finger Mode  |  [Q] Quit",
        (14, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (190, 190, 200),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    audio = AudioManager()
    tracker = HandTracker(max_hands=2)
    cap = _open_camera(0)

    if not cap.isOpened():
        audio.close()
        tracker.close()
        raise RuntimeError(
            "Could not open webcam. Check camera permissions or /dev/video0."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    baseline_y_norm = DEFAULT_BASELINE_Y
    ten_finger_mode = False

    print("[Harmonica] Camera started. Press 'Q' or 'ESC' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            finger_map = TEN_FINGER_MAP if ten_finger_mode else FIVE_FINGER_MAP
            states, extended_count = tracker.process(
                frame, baseline_y_norm, ten_finger_mode
            )

            seen_slots: set[int] = set()
            active_labels: list[str] = []

            for st in states:
                if st.slot_id >= len(finger_map):
                    continue
                seen_slots.add(st.slot_id)
                hole_cfg = finger_map[st.slot_id]

                if st.is_extended:
                    note = hole_cfg.high_note if st.is_above_line else hole_cfg.low_note
                    color = COLOR_HIGH if st.is_above_line else COLOR_LOW
                    tag = f"H{hole_cfg.hole}:{note}"
                    active_labels.append(tag)
                    audio.update_finger_note(st.slot_id, note)
                else:
                    color = COLOR_MUTED
                    tag = f"H{hole_cfg.hole}:--"
                    audio.update_finger_note(st.slot_id, None)

                cv2.circle(frame, st.tip_px, 11, color, -1)
                cv2.putText(
                    frame,
                    tag,
                    (st.tip_px[0] + 12, st.tip_px[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Silence any slots whose hand/finger is no longer visible
            for slot_id in range(len(finger_map)):
                if slot_id not in seen_slots:
                    audio.update_finger_note(slot_id, None)

            _draw_hud(
                frame,
                baseline_y_norm,
                ten_finger_mode,
                extended_count,
                active_labels,
            )

            cv2.imshow("Diatonic Harmonica (Camera)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("w"), 82):  # W or Up arrow
                baseline_y_norm = max(0.15, baseline_y_norm - BASELINE_STEP)
            elif key in (ord("s"), 84):  # S or Down arrow
                baseline_y_norm = min(0.85, baseline_y_norm + BASELINE_STEP)
            elif key == ord("m"):
                audio.stop_all()
                ten_finger_mode = not ten_finger_mode
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        audio.close()


if __name__ == "__main__":
    main()
