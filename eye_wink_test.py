#!/usr/bin/env python3
"""
Eye-mouse  |  Phase 1: camera picker + wink detection test bench
=================================================================
No cursor control yet. This only answers: "can my webcam reliably tell a
deliberate wink from a normal blink?"

Run:
    python eye_wink_test.py              # first camera
    python eye_wink_test.py --list       # list cameras and exit
    python eye_wink_test.py --camera 1   # pick camera by index
    python eye_wink_test.py --log run.csv

Keys (click the video window first):
    c        next camera            b   baseline calibration (do this first)
    t        start / restart trials m   metric: blendshape <-> EAR
    s        swap L/R eyes          r   reset counters
    + / -    close threshold        a / z   asymmetry margin  up / down
    [ / ]    min wink time (ms)     q or Esc   quit
The video is mirrored, so "L eye" is YOUR left eye, on the left of the image.
"""
import argparse
import csv
import os
import random
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import winsound
except ImportError:  # not on Windows
    winsound = None

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")
BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW = "Eye-mouse Phase 1"

# FaceMesh landmark indices (user's left / right eye, six points for EAR)
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]
L_IRIS, R_IRIS = (473, 474), (468, 469)  # (center, edge point)


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
def list_cameras():
    """Return [(index, name)]. Uses pygrabber for real names on Windows."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        if names:
            return list(enumerate(names))
    except Exception:
        pass
    found = []
    for i in range(6):
        cap = cv2.VideoCapture(i, BACKEND)
        if cap.isOpened():
            found.append((i, f"Camera {i}"))
        cap.release()
    return found


def open_camera(index, width=1280, height=720):
    cap = cv2.VideoCapture(index, BACKEND)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


# --------------------------------------------------------------------------- #
# Face landmarks
# --------------------------------------------------------------------------- #
def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    print("Downloading face landmark model (one time, a few MB)...")
    tmp = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, MODEL_PATH)
    except Exception as exc:
        print(f"Could not download the model: {exc}\nDownload it manually from:\n"
              f"  {MODEL_URL}\nand save it next to this script as face_landmarker.task")
        sys.exit(1)


class Tracker:
    def __init__(self):
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = 0

    def process(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        res = self.lm.detect_for_video(img, ts)
        if not res.face_landmarks:
            return None
        pts = np.array([[p.x, p.y] for p in res.face_landmarks[0]], np.float32)
        bs = {}
        if res.face_blendshapes:
            bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
        return pts, bs


def ear(pts, idx, w, h):
    p = pts[idx] * np.array([w, h], np.float32)
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return float(v / (2 * np.linalg.norm(p[0] - p[3]) + 1e-6))


def raw_closure(metric, pts, bs, w, h, swap, ear_open=0.30):
    """Return (left, right) eye closure in 0..1 (0 open, 1 closed), plus debug."""
    el, er = ear(pts, L_EYE, w, h), ear(pts, R_EYE, w, h)
    bl, br = bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)
    if swap:
        bl, br = br, bl
    if metric == "blend":
        l, r = bl, br
    else:
        l = float(np.clip((1 - el / ear_open) / 0.75, 0, 1))
        r = float(np.clip((1 - er / ear_open) / 0.75, 0, 1))
    return l, r, (el, er, bl, br)


# --------------------------------------------------------------------------- #
# Baseline calibration (eyelids sit lower when looking down / sideways)
# --------------------------------------------------------------------------- #
class BaselineRun:
    STAGES = [("Look at the CENTER of your main screen", 2.5),
              ("Look at the LEFT (laptop) screen", 2.5),
              ("Look DOWN at the bottom of the screens", 2.5)]
    SETTLE = 0.8

    def __init__(self):
        self.t0 = time.monotonic()
        self.samples = []

    def status(self):
        el, acc = time.monotonic() - self.t0, 0.0
        for text, dur in self.STAGES:
            if el < acc + dur:
                return text, acc + dur - el, el - acc
            acc += dur
        return None

    def add(self, l, r):
        st = self.status()
        if st and st[2] >= self.SETTLE:
            self.samples.append((l, r))

    def result(self):
        if len(self.samples) < 10:
            return 0.0, 0.0
        a = np.array(self.samples)
        return float(np.percentile(a[:, 0], 90)), float(np.percentile(a[:, 1], 90))


def adjust(v, base):
    return float(np.clip((v - base) / max(1e-3, 1 - base), 0, 1))


# --------------------------------------------------------------------------- #
# Wink detection
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    close_thr: float = 0.55   # adjusted closure needed to count an eye as closed
    asym_min: float = 0.30    # winking eye must be this much more closed than the other
    min_ms: float = 220       # hold time before a wink fires
    long_ms: float = 700      # both-eyes-closed hold (future: double click / drag)
    grace_ms: float = 70      # ignore dropouts shorter than this


class WinkDetector:
    def __init__(self, p):
        self.p = p
        self.reset()

    def reset(self):
        self.cand, self.start, self.last, self.fired = None, 0.0, 0.0, False

    def classify(self, aL, aR):
        p = self.p
        lc, rc = aL >= p.close_thr, aR >= p.close_thr
        if lc and aL - aR >= p.asym_min:
            return "L"
        if rc and aR - aL >= p.asym_min:
            return "R"
        if lc and rc:
            return "B"
        return None

    def _end(self, events):
        dur = self.last - self.start
        if not self.fired and self.cand is not None:
            if self.cand == "B":
                events.append(("BLINK", dur))
            elif dur >= 0.1:
                events.append(("REJECT_" + self.cand, dur))  # too short to count
        self.cand, self.fired = None, False

    def update(self, t, aL, aR):
        p, events = self.p, []
        c = self.classify(aL, aR)
        if c == self.cand:
            if c is not None:
                self.last = t
        elif c is None:
            if self.cand is not None and t - self.last > p.grace_ms / 1000:
                self._end(events)
        else:
            if self.cand is not None:
                self._end(events)
            self.cand, self.start, self.last, self.fired = c, t, t, False
        if self.cand is not None and not self.fired:
            held = t - self.start
            if self.cand in ("L", "R") and held >= p.min_ms / 1000:
                events.append(("WINK_" + self.cand, held))
                self.fired = True
            elif self.cand == "B" and held >= p.long_ms / 1000:
                events.append(("LONG_CLOSE", held))
                self.fired = True
        return events


# --------------------------------------------------------------------------- #
# Guided trials: measures real hit / false-trigger rates
# --------------------------------------------------------------------------- #
class Trials:
    READY, ACT = 1.0, 3.0
    EXPECT = {"WINK LEFT": "WINK_L", "WINK RIGHT": "WINK_R", "BLINK NORMALLY": None}

    def __init__(self, n_each=5):
        self.seq = list(self.EXPECT) * n_each
        random.shuffle(self.seq)
        self.i, self.t0 = 0, time.monotonic()
        self.cur, self.results, self.stray = [], [], 0
        self.summary = None

    def update(self, now, fired):
        """fired = list of wink-type event names this frame. Returns (phase, label, secs_left)."""
        if self.summary:
            return "done", "", 0.0
        el = now - self.t0
        label = self.seq[self.i]
        if el < self.READY:
            self.stray += len(fired)
            return "ready", label, self.READY - el
        if el < self.READY + self.ACT:
            self.cur += fired
            return "go", label, self.READY + self.ACT - el
        want = self.EXPECT[label]
        ok = (not self.cur) if want is None else (self.cur == [want])
        self.results.append((label, ok))
        self.i, self.t0, self.cur = self.i + 1, now, []
        if self.i >= len(self.seq):
            lines = []
            for lab in self.EXPECT:
                r = [ok for l, ok in self.results if l == lab]
                lines.append(f"{lab}: {sum(r)}/{len(r)} correct")
            lines.append(f"Stray triggers between trials: {self.stray}")
            self.summary = lines
            print("\n=== TRIAL RESULTS ===\n" + "\n".join(lines) + "\n")
            return "done", "", 0.0
        return "ready", self.seq[self.i], self.READY


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def put(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, val, thr, label):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    col = (60, 60, 230) if val >= thr else (80, 200, 80)
    cv2.rectangle(img, (x, y), (x + int(w * val), y + h), col, -1)
    tx = x + int(w * thr)
    cv2.line(img, (tx, y - 3), (tx, y + h + 3), (255, 255, 255), 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    put(img, f"{label} {val:.2f}", (x + w + 8, y + h - 3))


def draw_eyes(img, pts):
    h, w = img.shape[:2]
    for idx in (L_EYE, R_EYE):
        poly = np.array([[(1 - pts[i][0]) * w, pts[i][1] * h] for i in idx], np.int32)
        cv2.polylines(img, [poly], True, (255, 220, 0), 1, cv2.LINE_AA)
    for c, e in (L_IRIS, R_IRIS):
        cx, cy = (1 - pts[c][0]) * w, pts[c][1] * h
        ex, ey = (1 - pts[e][0]) * w, pts[e][1] * h
        cv2.circle(img, (int(cx), int(cy)), max(2, int(np.hypot(cx - ex, cy - ey))),
                   (0, 255, 255), 1, cv2.LINE_AA)


def beep(freq):
    if winsound:
        threading.Thread(target=winsound.Beep, args=(freq, 70), daemon=True).start()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list cameras and exit")
    ap.add_argument("--camera", type=int, default=None, help="camera index")
    ap.add_argument("--log", default=None, help="write per-frame CSV here")
    ap.add_argument("--no-beep", action="store_true")
    args = ap.parse_args()

    cams = list_cameras()
    if args.list:
        for i, name in cams:
            print(f"[{i}] {name}")
        return
    if not cams:
        print("No cameras found.")
        return
    cam_ids = [i for i, _ in cams]
    names = dict(cams)
    cur = args.camera if args.camera in cam_ids else cam_ids[0]

    ensure_model()
    tracker = Tracker()
    cap = open_camera(cur)
    if cap is None:
        print(f"Could not open camera {cur}.")
        return

    params = Params()
    det = WinkDetector(params)
    metric, swap = "blend", False
    base = (0.0, 0.0)
    baseline, trials = None, None
    stats = dict.fromkeys(["BLINK", "WINK_L", "WINK_R", "LONG_CLOSE", "REJECT_L", "REJECT_R"], 0)
    log_lines = deque(maxlen=8)
    flash = ("", 0.0)
    fps, prev_t, fail = 0.0, time.monotonic(), 0

    logf = writer = None
    if args.log:
        logf = open(args.log, "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["t", "metric", "raw_l", "raw_r", "adj_l", "adj_r", "event"])

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(__doc__)

    while True:
        ok, frame = cap.read()
        now = time.monotonic()
        if not ok:
            fail += 1
            if fail > 60:
                print("Camera stopped delivering frames.")
                break
            time.sleep(0.01)
            continue
        fail = 0
        fps = 0.9 * fps + 0.1 / max(1e-6, now - prev_t)
        prev_t = now

        res = tracker.process(frame)
        disp = cv2.flip(frame, 1)
        if disp.shape[1] > 960:
            disp = cv2.resize(disp, (960, int(disp.shape[0] * 960 / disp.shape[1])))
        H, W = frame.shape[:2]

        aL = aR = 0.0
        dbg = None
        events = []
        if res is None:
            det.reset()
            put(disp, "NO FACE DETECTED", (20, 90), 0.8, (60, 60, 255), 2)
        else:
            pts, bs = res
            draw_eyes(disp, pts)
            rl, rr, dbg = raw_closure(metric, pts, bs, W, H, swap)
            if baseline is not None:
                baseline.add(rl, rr)
                if baseline.status() is None:
                    base = baseline.result()
                    print(f"Baseline done: L={base[0]:.2f} R={base[1]:.2f}")
                    baseline = None
                det.reset()
            else:
                aL, aR = adjust(rl, base[0]), adjust(rr, base[1])
                events = det.update(now, aL, aR)
            if writer:
                writer.writerow([f"{now:.3f}", metric, f"{rl:.3f}", f"{rr:.3f}",
                                 f"{aL:.3f}", f"{aR:.3f}", ";".join(e[0] for e in events)])

        fired = []
        for name, dur in events:
            stats[name] += 1
            log_lines.appendleft(f"{time.strftime('%H:%M:%S')}  {name}  {dur * 1000:.0f} ms")
            if name.startswith("WINK") or name == "LONG_CLOSE":
                fired.append(name)
                flash = (name, now)
                if not args.no_beep:
                    beep({"WINK_L": 800, "WINK_R": 1200}.get(name, 500))

        # ---------------- HUD ----------------
        put(disp, f"{names.get(cur, cur)}  [{cur}]   {fps:4.1f} fps   "
                  f"metric={metric}  swap={'ON' if swap else 'off'}", (10, 22))
        put(disp, f"close>={params.close_thr:.2f}  asym>={params.asym_min:.2f}  "
                  f"hold>={params.min_ms:.0f}ms   baseline L={base[0]:.2f} R={base[1]:.2f}",
            (10, 44), 0.5, (200, 255, 200))
        bar(disp, 10, 60, 220, 16, aL, params.close_thr, "L eye")
        bar(disp, 10, 84, 220, 16, aR, params.close_thr, "R eye")
        if dbg:
            put(disp, f"EAR L/R {dbg[0]:.2f}/{dbg[1]:.2f}   blend L/R {dbg[2]:.2f}/{dbg[3]:.2f}",
                (10, 118), 0.45, (180, 180, 180))
        put(disp, "  ".join(f"{k}:{v}" for k, v in stats.items()),
            (10, disp.shape[0] - 12), 0.45, (255, 255, 180))
        for n, line in enumerate(log_lines):
            put(disp, line, (disp.shape[1] - 250, 24 + 20 * n), 0.5, (200, 220, 255))
        if flash[0] and now - flash[1] < 0.6:
            put(disp, flash[0].replace("_", " "), (disp.shape[1] // 2 - 110, 70),
                1.2, (0, 255, 0), 3)

        if baseline is not None:
            st = baseline.status()
            if st:
                put(disp, f"BASELINE: {st[0]}  ({st[1]:.1f}s)  - eyes OPEN, blink normally",
                    (10, disp.shape[0] // 2), 0.7, (0, 200, 255), 2)
        elif trials is not None:
            phase, label, left = trials.update(now, fired)
            y = disp.shape[0] // 2
            if phase == "ready":
                put(disp, f"Trial {trials.i + 1}/{len(trials.seq)}  get ready: {label}",
                    (10, y), 0.8, (0, 200, 255), 2)
            elif phase == "go":
                put(disp, f"{label}  ({left:.1f}s)", (10, y), 1.1, (0, 255, 255), 3)
            else:
                for n, line in enumerate(trials.summary):
                    put(disp, line, (10, y + 32 * n), 0.8, (0, 255, 0), 2)
                put(disp, "press t to run again", (10, y + 32 * len(trials.summary) + 10), 0.6)
        else:
            put(disp, "b=baseline  t=trials  c=camera  m=metric  s=swap  q=quit",
                (10, disp.shape[0] - 34), 0.5, (200, 200, 200))

        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("b"):
            baseline, trials = BaselineRun(), None
        elif key == ord("t"):
            trials, baseline = Trials(), None
            det.reset()
        elif key == ord("c") and len(cam_ids) > 1:
            cur = cam_ids[(cam_ids.index(cur) + 1) % len(cam_ids)]
            new = open_camera(cur)
            if new is not None:
                cap.release()
                cap = new
                det.reset()
            else:
                print(f"Could not open camera {cur}")
                cur = cam_ids[(cam_ids.index(cur) - 1) % len(cam_ids)]
        elif key == ord("m"):
            metric = "ear" if metric == "blend" else "blend"
            base = (0.0, 0.0)
            det.reset()
        elif key == ord("s"):
            swap = not swap
        elif key == ord("r"):
            stats = dict.fromkeys(stats, 0)
        elif key in (ord("+"), ord("=")):
            params.close_thr = min(0.95, params.close_thr + 0.05)
        elif key == ord("-"):
            params.close_thr = max(0.10, params.close_thr - 0.05)
        elif key == ord("a"):
            params.asym_min = min(0.90, params.asym_min + 0.05)
        elif key == ord("z"):
            params.asym_min = max(0.05, params.asym_min - 0.05)
        elif key == ord("]"):
            params.min_ms += 20
        elif key == ord("["):
            params.min_ms = max(60, params.min_ms - 20)

    cap.release()
    cv2.destroyAllWindows()
    if logf:
        logf.close()


if __name__ == "__main__":
    main()
