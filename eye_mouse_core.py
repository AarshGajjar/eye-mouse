"""
EyeMouse core (no UI): camera thread, gaze model, click engine, cursor driver, UI snapping.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field, asdict, fields

import numpy as np

try:
    import cv2
except ImportError:  # only needed at runtime
    cv2 = None

IS_WIN = sys.platform == "win32"
try:
    import winsound
except ImportError:
    winsound = None

APP_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "EyeMouse")
os.makedirs(APP_DIR, exist_ok=True)
CFG_PATH = os.path.join(APP_DIR, "config.json")
MODEL_JSON = os.path.join(APP_DIR, "gaze_model.json")
TASK_PATH = os.path.join(APP_DIR, "face_landmarker.task")
TASK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")
BACKEND = cv2.CAP_DSHOW if (cv2 is not None and IS_WIN) else (cv2.CAP_ANY if cv2 else 0)

VK_CONTROL, VK_MENU, VK_P = 0x11, 0x12, 0x50   # pause hotkey = Ctrl+Alt+P


# =========================================================================== #
# Config
# =========================================================================== #
@dataclass
class Config:
    camera: int = 0
    metric: str = "blend"          # "blend" | "ear"
    swap_eyes: bool = False
    # wink detection (close_thr lowered from the Phase 1 default of 0.55)
    close_thr: float = 0.40
    asym_min: float = 0.25
    min_ms: float = 220.0
    long_ms: float = 700.0
    grace_ms: float = 70.0
    baselines: dict = field(default_factory=lambda: {"blend": [0.0, 0.0], "ear": [0.0, 0.0]})
    # engage
    engage_mode: str = "hold"      # "hold" | "toggle" | "always"
    engage_vk: int = 0x77          # F8
    engage_name: str = "F8"
    # cursor
    smooth: float = 1.5            # One-Euro min cutoff (Hz): lower = steadier, laggier
    beta: float = 0.01
    snap: bool = True
    snap_radius: int = 110
    mouse_yield_s: float = 1.2
    clicks: bool = True
    beep: bool = True
    show_dot: bool = False
    preview: bool = True
    calib_passes: int = 2

    @classmethod
    def load(cls):
        cfg = cls()
        try:
            with open(CFG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            names = {f.name for f in fields(cls)}
            for k, v in data.items():
                if k in names:
                    setattr(cfg, k, v)
        except Exception:
            pass
        return cfg

    def save(self):
        try:
            with open(CFG_PATH, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
        except Exception:
            pass


# =========================================================================== #
# Win32 layer (stubs on other platforms so the logic stays testable)
# =========================================================================== #
if IS_WIN:
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    user32.GetAsyncKeyState.restype = ctypes.c_short
    ULONG_PTR = ctypes.c_size_t

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    class _MONITORINFOEXW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                    ("szDevice", wintypes.WCHAR * 32)]

    _MonEnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                                      ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    user32.EnumDisplayMonitors.argtypes = [wintypes.HDC, ctypes.c_void_p, _MonEnumProc, wintypes.LPARAM]
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]


def get_monitors():
    """Physical-pixel monitor rectangles, sorted left-to-right. Call after Qt has set DPI awareness."""
    if not IS_WIN:
        return [dict(name="DISPLAY1", left=0, top=0, right=1920, bottom=1080, primary=True)]
    mons = []

    def cb(hm, hdc, lprc, lp):
        info = _MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        user32.GetMonitorInfoW(hm, ctypes.byref(info))
        r = info.rcMonitor
        mons.append(dict(name=info.szDevice, left=r.left, top=r.top, right=r.right,
                         bottom=r.bottom, primary=bool(info.dwFlags & 1)))
        return True

    proc = _MonEnumProc(cb)
    user32.EnumDisplayMonitors(None, None, proc, 0)
    mons.sort(key=lambda m: (m["left"], m["top"]))
    return mons


class WinIO:
    """Real mouse / keyboard access. Swap for a fake in tests."""

    def set_cursor(self, x, y):
        if IS_WIN:
            user32.SetCursorPos(int(round(x)), int(round(y)))

    def get_cursor(self):
        if not IS_WIN:
            return (0, 0)
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return (pt.x, pt.y)

    def click(self, button):
        if not IS_WIN:
            return
        down, up = (0x0002, 0x0004) if button == "left" else (0x0008, 0x0010)
        arr = (_INPUT * 2)()
        for i, fl in enumerate((down, up)):
            arr[i].type = 0
            arr[i].u.mi = _MOUSEINPUT(0, 0, 0, fl, 0, 0)
        user32.SendInput(2, arr, ctypes.sizeof(_INPUT))

    def key_down(self, vk):
        return bool(IS_WIN and (user32.GetAsyncKeyState(int(vk)) & 0x8000))


def beep(freq):
    if winsound:
        threading.Thread(target=winsound.Beep, args=(freq, 60), daemon=True).start()


# =========================================================================== #
# Cameras + MediaPipe
# =========================================================================== #
def list_cameras():
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
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def ensure_task_file():
    if os.path.exists(TASK_PATH) and os.path.getsize(TASK_PATH) > 1_000_000:
        return
    tmp = TASK_PATH + ".part"
    urllib.request.urlretrieve(TASK_URL, tmp)
    os.replace(tmp, TASK_PATH)


class Tracker:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=TASK_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = 0

    def process(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        res = self.lm.detect_for_video(img, ts)
        if not res.face_landmarks:
            return None
        pts = np.array([[p.x, p.y] for p in res.face_landmarks[0]], np.float32)
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        mat = None
        if getattr(res, "facial_transformation_matrixes", None):
            mat = np.array(res.facial_transformation_matrixes[0], np.float64)
        return pts, bs, mat


# =========================================================================== #
# Eye closure + wink detection (from Phase 1)
# =========================================================================== #
L_EYE = [362, 385, 387, 263, 373, 380]   # user's left eye: inner=362, outer=263
R_EYE = [33, 160, 158, 133, 153, 144]    # user's right eye: outer=33, inner=133
L_IRIS_C, R_IRIS_C = 473, 468


def _ear(pts, idx, w, h):
    p = pts[idx] * np.array([w, h], np.float32)
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return float(v / (2 * np.linalg.norm(p[0] - p[3]) + 1e-6))


def raw_closures(pts, bs, w, h, swap, ear_open=0.30):
    """Both metrics at once: {'blend': (l, r), 'ear': (l, r)}, 0 = open, 1 = closed."""
    el, er = _ear(pts, L_EYE, w, h), _ear(pts, R_EYE, w, h)
    bl, br = bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)
    if swap:
        bl, br = br, bl
    ear_l = float(np.clip((1 - el / ear_open) / 0.75, 0, 1))
    ear_r = float(np.clip((1 - er / ear_open) / 0.75, 0, 1))
    return {"blend": (float(bl), float(br)), "ear": (ear_l, ear_r)}


def adjust(v, base):
    return float(np.clip((v - base) / max(1e-3, 1 - base), 0, 1))


class WinkDetector:
    """cfg supplies close_thr, asym_min, min_ms, long_ms, grace_ms."""

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
                events.append(("REJECT_" + self.cand, dur))
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


# =========================================================================== #
# Gaze features + model
# =========================================================================== #
N_FEAT = 14  # uL vL uR vR | hL hR vbL vbR (blendshape gaze) | yaw pitch roll tx ty tz


def gaze_features(pts, bs, mat, w, h):
    """Head-pose-aware gaze feature vector (length N_FEAT)."""
    def eye(inner, outer, iris):
        a = pts[inner] * (w, h)
        b = pts[outer] * (w, h)
        c = pts[iris] * (w, h)
        e = b - a
        ln = float(np.linalg.norm(e)) + 1e-6
        ex = e / ln
        ey = np.array([-ex[1], ex[0]])
        if ey[1] < 0:            # same "down" direction for both eyes
            ey = -ey
        return float(np.dot(c - a, ex) / ln), float(np.dot(c - (a + b) / 2, ey) / ln)

    uL, vL = eye(362, 263, L_IRIS_C)
    uR, vR = eye(133, 33, R_IRIS_C)
    g = lambda k: float(bs.get(k, 0.0))
    hL = g("eyeLookOutLeft") - g("eyeLookInLeft")
    hR = g("eyeLookOutRight") - g("eyeLookInRight")
    vbL = g("eyeLookUpLeft") - g("eyeLookDownLeft")
    vbR = g("eyeLookUpRight") - g("eyeLookDownRight")
    yaw = pitch = roll = tx = ty = tz = 0.0
    if mat is not None:
        R = mat[:3, :3]
        R = R / (np.linalg.norm(R[:, 0]) + 1e-9)
        yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])))
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        tx, ty, tz = (float(v) for v in mat[:3, 3])
    return np.array([uL, vL, uR, vR, hL, hR, vbL, vbR, yaw, pitch, roll, tx, ty, tz], np.float64)


def expand(X):
    """Linear features + quadratic terms of the four core gaze signals."""
    X = np.atleast_2d(X)
    core = np.stack([(X[:, 0] - X[:, 2]) / 2, (X[:, 1] + X[:, 3]) / 2,
                     (X[:, 4] - X[:, 5]) / 2, (X[:, 6] + X[:, 7]) / 2], axis=1)
    quad = [core[:, i] * core[:, j] for i in range(4) for j in range(i, 4)]
    return np.hstack([X, np.stack(quad, axis=1)])


def _ridge(Xe, Y, alpha):
    mu, sd = Xe.mean(0), Xe.std(0) + 1e-6
    Z = (Xe - mu) / sd
    ym = Y.mean(0)
    n = len(Z)
    W = np.linalg.solve(Z.T @ Z + alpha * n * np.eye(Z.shape[1]), Z.T @ (Y - ym))
    return dict(mu=mu, sd=sd, W=W, ym=ym, alpha=alpha)


def _ridge_predict(m, Xe):
    return ((Xe - m["mu"]) / m["sd"]) @ m["W"] + m["ym"]


ALPHAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


class GazeModel:
    """Stage 1: which monitor. Stage 2: (x, y) as a 0..1 fraction of that monitor."""

    def __init__(self, monitors):
        self.monitors = [dict(name=m["name"], w=m["right"] - m["left"], h=m["bottom"] - m["top"])
                         for m in monitors]
        self.clf = None
        self.reg = {}
        self.report = []

    # ---- fitting ----
    def fit(self, samples):
        """samples: list of (mon_idx, point_id, feature_vector)."""
        mon = np.array([s[0] for s in samples])
        pid = np.array([s[1] for s in samples])
        X = np.array([s[2] for s in samples])
        Y = np.array([s[3] for s in samples])          # fractions
        self.report = []
        for mi, m in enumerate(self.monitors):
            sel = mon == mi
            if sel.sum() < 30:
                raise ValueError(f"Not enough usable samples for monitor {mi + 1} "
                                 f"({int(sel.sum())}). Was your face visible?")
            Xe, Ym, P = expand(X[sel]), Y[sel], pid[sel]
            best = None
            for alpha in ALPHAS:
                errs = []
                for p in np.unique(P):
                    tr, te = P != p, P == p
                    if tr.sum() < 20:
                        continue
                    mdl = _ridge(Xe[tr], Ym[tr], alpha)
                    d = (_ridge_predict(mdl, Xe[te]) - Ym[te]) * (m["w"], m["h"])
                    errs.append(np.hypot(d[:, 0], d[:, 1]).mean())
                score = float(np.mean(errs)) if errs else 1e9
                if best is None or score < best[0]:
                    best = (score, alpha)
            self.reg[mi] = _ridge(Xe, Ym, best[1])
            self.reg[mi]["cv_px"] = best[0]
            self.report.append(f"Screen {mi + 1}: typical error ~{best[0]:.0f} px "
                               f"(leave-one-dot-out test)")
        if len(self.monitors) > 1:
            from sklearn.linear_model import LogisticRegression
            mu, sd = X.mean(0), X.std(0) + 1e-6
            lr = LogisticRegression(C=1.0, max_iter=1000).fit((X - mu) / sd, mon)
            acc = float((lr.predict((X - mu) / sd) == mon).mean())
            self.clf = dict(mu=mu, sd=sd, coef=lr.coef_, b=lr.intercept_, classes=lr.classes_)
            self.report.append(f"Screen detection accuracy: {acc * 100:.0f}%")
        return self

    # ---- inference ----
    def monitor_probs(self, x):
        if self.clf is None:
            return None
        c = self.clf
        z = ((x - c["mu"]) / c["sd"]) @ c["coef"].T + c["b"]
        if z.shape[-1] == 1:
            p1 = 1 / (1 + math.exp(-float(z[0])))
            p = np.array([1 - p1, p1])
        else:
            e = np.exp(z - z.max())
            p = e / e.sum()
        full = np.zeros(len(self.monitors))
        full[np.asarray(c["classes"], int)] = p
        return full

    def xy_fraction(self, mi, x):
        f = _ridge_predict(self.reg[mi], expand(x))[0]
        return float(np.clip(f[0], 0, 1)), float(np.clip(f[1], 0, 1))

    # ---- persistence ----
    def to_json(self):
        def enc(d):
            return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        return dict(monitors=self.monitors, report=self.report,
                    clf=enc(self.clf) if self.clf else None,
                    reg={str(k): enc(v) for k, v in self.reg.items()})

    @classmethod
    def from_json(cls, d):
        obj = cls.__new__(cls)
        obj.monitors, obj.report = d["monitors"], d.get("report", [])
        dec = lambda e: {k: (np.array(v) if isinstance(v, list) else v) for k, v in e.items()}
        obj.clf = dec(d["clf"]) if d.get("clf") else None
        obj.reg = {int(k): dec(v) for k, v in d["reg"].items()}
        return obj

    def matches(self, monitors):
        cur = [(m["name"], m["right"] - m["left"], m["bottom"] - m["top"]) for m in monitors]
        return cur == [(m["name"], m["w"], m["h"]) for m in self.monitors]

    def save(self):
        with open(MODEL_JSON, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls):
        try:
            with open(MODEL_JSON, "r", encoding="utf-8") as f:
                return cls.from_json(json.load(f))
        except Exception:
            return None


class MonitorSelector:
    """Sticky monitor choice: switch only when the other screen wins clearly for a moment."""

    def __init__(self, n):
        self.n, self.cur, self.p = n, 0, None
        self.cand, self.cand_t = None, 0.0

    def update(self, t, probs):
        if probs is None:
            return self.cur, False
        self.p = probs if self.p is None else 0.6 * self.p + 0.4 * probs
        best = int(np.argmax(self.p))
        if best == self.cur or self.p[best] < 0.62:
            self.cand = None
            return self.cur, False
        if self.cand != best:
            self.cand, self.cand_t = best, t
        elif t - self.cand_t >= 0.15:
            self.cur, self.cand = best, None
            return self.cur, True
        return self.cur, False


class OneEuro:
    def __init__(self, min_cutoff=1.5, beta=0.01, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _a(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-3))

    def __call__(self, t, v):
        v = np.asarray(v, np.float64)
        if self.x is None:
            self.x, self.dx, self.t = v, np.zeros_like(v), t
            return v
        dt = max(t - self.t, 1e-3)
        d = (v - self.x) / dt
        self.dx = self.dx + self._a(self.d_cutoff, dt) * (d - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + self._a(cutoff, dt) * (v - self.x)
        self.t = t
        return self.x


# =========================================================================== #
# UI-Automation snapping
# =========================================================================== #
CLICKABLE = {"ButtonControl", "HyperlinkControl", "MenuItemControl", "TabItemControl",
             "ListItemControl", "CheckBoxControl", "RadioButtonControl", "ComboBoxControl",
             "SplitButtonControl", "TreeItemControl", "EditControl", "DataItemControl",
             "HeaderItemControl"}


def pick_target(gx, gy, cands, radius, prev_key=None):
    """cands: [(key, l, t, r, b)]. Returns (key, x, y) or None. Sticky towards prev_key."""
    scored = []
    for key, l, t, r, b in cands:
        cx, cy = min(max(gx, l), r), min(max(gy, t), b)
        d = math.hypot(gx - cx, gy - cy)
        if d > radius:
            continue
        s = d + 0.15 * math.hypot(gx - (l + r) / 2, gy - (t + b) / 2)
        scored.append((s, key, l, t, r, b))
    if not scored:
        return None
    scored.sort(key=lambda z: z[0])
    best = scored[0]
    for z in scored:
        if z[1] == prev_key and z[0] <= best[0] * 1.35 + 8:
            best = z
            break
    _, key, l, t, r, b = best
    x = min(max(gx, l + 6), r - 6) if (r - l) > 140 else (l + r) / 2
    y = min(max(gy, t + 4), b - 4) if (b - t) > 60 else (t + b) / 2
    return key, x, y


def collect_candidates(auto, gx, gy, radius, own_pid):
    """Sample UIA elements on a small disc around the gaze point."""
    pts = [(gx, gy)]
    for rr, n in ((0.4 * radius, 6), (0.8 * radius, 10)):
        pts += [(gx + rr * math.cos(2 * math.pi * k / n), gy + rr * math.sin(2 * math.pi * k / n))
                for k in range(n)]
    cands = {}
    for x, y in pts:
        if any(l <= x <= r and t <= y <= b and (r - l) * (b - t) < 6000
               for (l, t, r, b) in cands):
            continue
        try:
            ctl = auto.ControlFromPoint(int(x), int(y))
            found = None
            for _ in range(4):
                if ctl is None:
                    break
                if ctl.ControlTypeName in CLICKABLE:
                    found = ctl
                    break
                ctl = ctl.GetParentControl()
            if found is None or found.ProcessId == own_pid or found.IsOffscreen:
                continue
            rc = found.BoundingRectangle
            l, t, r, b = rc.left, rc.top, rc.right, rc.bottom
            w, h = r - l, b - t
            if w < 4 or h < 4 or h > 300 or w * h > 400_000:
                continue
            cands[(l, t, r, b)] = (l, t, r, b, (found.Name or found.ControlTypeName)[:40])
        except Exception:
            continue
    return list(cands.values())


class SnapWorker(threading.Thread):
    """Background thread: finds clickable UI elements near a fixation and publishes a snap target."""

    def __init__(self, engine):
        super().__init__(daemon=True, name="snap")
        self.eng = engine
        self.stop_flag = False
        self.prev_key = None
        self.last_q = None

    def run(self):
        try:
            import uiautomation as auto
        except Exception as exc:
            self.eng.status["snap_error"] = f"uiautomation not available: {exc}"
            return
        try:
            init = getattr(auto, "UIAutomationInitializerInThread", None)
            ctx = init() if init else None
            if ctx is not None:
                ctx.__enter__()
        except Exception:
            ctx = None
        fails = 0
        while not self.stop_flag:
            time.sleep(0.04)
            q = self.eng.snap_query
            if not (self.eng.cfg.snap and q and time.monotonic() - q[2] < 0.35):
                continue
            if self.last_q and math.hypot(q[0] - self.last_q[0], q[1] - self.last_q[1]) < 20 \
                    and time.monotonic() - self.last_q[2] < 1.0:
                continue
            try:
                R = self.eng.cfg.snap_radius
                cands = collect_candidates(auto, q[0], q[1], R, os.getpid())
                keyed = [((c[0], c[1], c[2], c[3]), c[0], c[1], c[2], c[3]) for c in cands]
                pick = pick_target(q[0], q[1], keyed, R, self.prev_key)
                self.last_q = (q[0], q[1], time.monotonic())
                self.prev_key = pick[0] if pick else None
                names = {(c[0], c[1], c[2], c[3]): c[4] for c in cands}
                self.eng.snap_result = None if pick is None else dict(
                    x=pick[1], y=pick[2], gaze=(q[0], q[1]), t=time.monotonic(),
                    name=names.get(pick[0], ""))
                fails = 0
            except Exception as exc:
                fails += 1
                self.eng.status["snap_error"] = f"{type(exc).__name__}: {exc}"
                if fails > 8:
                    break


# =========================================================================== #
# Cursor driver: freeze-before-blink, mouse yielding, clicks
# =========================================================================== #
class CursorDriver:
    LOOKBACK = 0.12     # cursor position restored from this long before eyelid closure starts
    CLICK_GAP = 0.35

    def __init__(self, cfg, io):
        self.cfg, self.io = cfg, io
        self.hist = deque(maxlen=40)
        self.frozen = False
        self.frozen_pos = None
        self.frozen_since = 0.0
        self.open_since = None
        self.last_set = None
        self.yield_until = 0.0
        self.last_click = 0.0
        self.was_engaged = False
        self.actions = []

    def _move(self, pos):
        self.io.set_cursor(*pos)
        self.last_set = (pos[0], pos[1])

    def _pos_before(self, t):
        pick = None
        for ht, x, y in self.hist:
            if ht <= t:
                pick = (x, y)
        return pick

    def update(self, now, aL, aR, events, target, engaged):
        """target = smoothed/snapped gaze pixel (or None). Returns list of action strings."""
        cfg, acts = self.cfg, []
        a = max(aL, aR)
        freeze_thr = 0.5 * cfg.close_thr
        cur = self.io.get_cursor()
        if self.last_set and math.hypot(cur[0] - self.last_set[0], cur[1] - self.last_set[1]) > 10:
            self.yield_until = now + cfg.mouse_yield_s      # the physical mouse moved: yield
            self.last_set = None
            self.frozen = False
            acts.append("YIELD")
        if not engaged or now < self.yield_until:
            self.frozen = False
            if not engaged:
                self.last_set = None
            self.was_engaged = engaged
            return acts
        if not self.was_engaged:
            self.hist.clear()
            self.was_engaged = True

        if target is not None:
            self.hist.append((now, target[0], target[1]))

        if not self.frozen:
            if a >= freeze_thr:
                pos = self._pos_before(now - self.LOOKBACK) or target or self.last_set
                if pos is not None:
                    self.frozen, self.frozen_pos = True, pos
                    self.frozen_since, self.open_since = now, None
                    self._move(pos)
            elif target is not None:
                self._move(target)
        else:
            if a < freeze_thr * 0.7:
                if self.open_since is None:
                    self.open_since = now
                elif now - self.open_since > 0.15:
                    self.frozen = False
                    acts.append("UNFREEZE")
            else:
                self.open_since = None
            if now - self.frozen_since > 2.5:
                self.frozen = False

        for name, _ in events:
            if name in ("WINK_L", "WINK_R") and cfg.clicks and now - self.last_click > self.CLICK_GAP:
                pos = self.frozen_pos if self.frozen else (target or self.last_set)
                if pos is None:
                    continue
                self._move(pos)
                self.io.click("left" if name == "WINK_L" else "right")
                self.last_click = now
                acts.append("CLICK_L" if name == "WINK_L" else "CLICK_R")
        return acts


# =========================================================================== #
# Engine thread
# =========================================================================== #
class Engine(threading.Thread):
    def __init__(self, cfg, monitors, io=None):
        super().__init__(daemon=True, name="engine")
        self.cfg, self.monitors = cfg, monitors
        self.io = io or WinIO()
        self.model = None
        self.sel = MonitorSelector(len(monitors))
        self.filt = OneEuro(cfg.smooth, cfg.beta)
        self.det = WinkDetector(cfg)
        self.driver = CursorDriver(cfg, self.io)
        self.stop_flag = False
        self.paused = False
        self.suspended = False          # True while the calibration wizard runs
        self.toggle_on = False
        self.cam_request = None
        self.status = dict(fps=0.0, face=False, aL=0.0, aR=0.0, engaged=False, paused=False,
                           mon=0, gaze=None, msg="starting", frozen=False, snap="", error="")
        self.log = deque(maxlen=7)
        self.preview = None
        self.snap_query = None
        self.snap_result = None
        self.snapper = None
        # calibration sample exchange
        self.calib_tag = None
        self.calib_samples, self.calib_raw = [], {"blend": [], "ear": []}
        self._lock = threading.Lock()
        self._fix = deque(maxlen=12)
        self._prev = dict(pause=False, engage=False)
        self._snap_active = False
        self._was_engaged = False
        self._pv_n = 0

    # ---- calibration API (called from the UI thread) ----
    def set_model(self, model):
        self.model = model
        self.sel = MonitorSelector(len(self.monitors))
        self.filt.reset()

    def pop_calibration(self):
        with self._lock:
            s, r = self.calib_samples, self.calib_raw
            self.calib_samples, self.calib_raw = [], {"blend": [], "ear": []}
        return s, r

    def request_camera(self, idx):
        self.cam_request = idx

    def start_snapping(self):
        if self.snapper is None:
            self.snapper = SnapWorker(self)
            self.snapper.start()

    # ---- helpers ----
    def _engaged(self):
        cfg, io = self.cfg, self.io
        pause_now = io.key_down(VK_CONTROL) and io.key_down(VK_MENU) and io.key_down(VK_P)
        if pause_now and not self._prev["pause"]:
            self.paused = not self.paused
            self.log.appendleft("PAUSED" if self.paused else "RESUMED")
        self._prev["pause"] = pause_now
        if self.paused or self.suspended or self.model is None:
            return False
        if cfg.engage_mode == "always":
            return True
        down = io.key_down(cfg.engage_vk)
        if cfg.engage_mode == "hold":
            return down
        if down and not self._prev["engage"]:
            self.toggle_on = not self.toggle_on
        self._prev["engage"] = down
        return self.toggle_on

    def _fixating(self, now, pos):
        self._fix.append((now, pos[0], pos[1]))
        recent = [(x, y) for t, x, y in self._fix if now - t <= 0.3]
        if len(recent) < 4 or now - self._fix[0][0] < 0.12:
            return False
        a = np.array(recent)
        return float(np.linalg.norm(a - a.mean(0), axis=1).max()) < 45

    def _estimate(self, now, feat):
        """Feature vector -> filtered (and possibly snapped) virtual-desktop pixel."""
        m = self.model
        mi, switched = self.sel.update(now, m.monitor_probs(feat))
        fx, fy = m.xy_fraction(mi, feat)
        mon = self.monitors[mi]
        px = mon["left"] + fx * (mon["right"] - mon["left"] - 1)
        py = mon["top"] + fy * (mon["bottom"] - mon["top"] - 1)
        if switched:
            self.filt.reset()
        self.filt.min_cutoff, self.filt.beta = self.cfg.smooth, self.cfg.beta
        sx, sy = self.filt(now, (px, py))
        return mi, float(sx), float(sy)

    def _apply_snap(self, now, sx, sy, fixating):
        cfg = self.cfg
        if not cfg.snap:
            return (sx, sy), ""
        R = cfg.snap_radius
        if fixating:
            self.snap_query = (sx, sy, now)
        res = self.snap_result
        if res is not None and time.monotonic() - res["t"] < 1.5:
            near = math.hypot(sx - res["x"], sy - res["y"]) <= R * (1.3 if self._snap_active else 1.0)
            fresh = math.hypot(sx - res["gaze"][0], sy - res["gaze"][1]) <= R * 0.6
            if near and fresh and (fixating or self._snap_active):
                self._snap_active = True
                return (res["x"], res["y"]), res["name"]
        self._snap_active = False
        return (sx, sy), ""

    # ---- main loop ----
    def run(self):
        st = self.status
        try:
            ensure_task_file()
            tracker = Tracker()
        except Exception as exc:
            st["error"] = f"Could not start face tracker: {exc}"
            return
        cap, fail, prev_t = None, 0, time.monotonic()
        while not self.stop_flag:
            if self.cam_request is not None or cap is None:
                idx = self.cfg.camera if self.cam_request is None else self.cam_request
                self.cam_request = None
                new = open_camera(idx)
                if new is None:
                    st["error"] = f"Could not open camera {idx}"
                    time.sleep(0.5)
                    continue
                if cap is not None:
                    cap.release()
                cap, st["error"] = new, ""
                self.cfg.camera = idx
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok:
                fail += 1
                if fail > 90:
                    st["error"] = "Camera stopped delivering frames"
                    cap.release()
                    cap, fail = None, 0
                time.sleep(0.01)
                continue
            fail = 0
            st["fps"] = 0.9 * st["fps"] + 0.1 / max(1e-6, now - prev_t)
            prev_t = now
            try:
                self._frame(tracker, frame, now)
            except Exception as exc:                      # never let one bad frame kill the tool
                st["error"] = f"{type(exc).__name__}: {exc}"
        if cap is not None:
            cap.release()

    def _frame(self, tracker, frame, now):
        cfg, st = self.cfg, self.status
        engaged = self._engaged()
        st.update(paused=self.paused, engaged=engaged)
        if engaged != self._was_engaged:
            self._was_engaged = engaged
            self.filt.reset()
            self._fix.clear()
        want_preview = cfg.preview
        if self.paused and not want_preview and self.calib_tag is None:
            time.sleep(0.03)
            st.update(face=False, gaze=None)
            return

        H, W = frame.shape[:2]
        res = tracker.process(frame)
        if res is None:
            self.det.reset()
            st.update(face=False, gaze=None, aL=0.0, aR=0.0)
            self.driver.update(now, 0.0, 0.0, [], None, False)   # release freeze state
            self._preview(frame, None, 0, 0, want_preview)
            return
        pts, bs, mat = res
        raws = raw_closures(pts, bs, W, H, cfg.swap_eyes)
        rl, rr = raws[cfg.metric]
        b = cfg.baselines.get(cfg.metric, [0.0, 0.0])
        aL, aR = adjust(rl, b[0]), adjust(rr, b[1])
        events = self.det.update(now, aL, aR)
        feat = gaze_features(pts, bs, mat, W, H)

        if self.calib_tag is not None:
            with self._lock:
                for k in ("blend", "ear"):
                    self.calib_raw[k].append(raws[k])
                if raws["blend"][0] < 0.45 and raws["blend"][1] < 0.45 and raws["ear"][0] < 0.6:
                    self.calib_samples.append((self.calib_tag, feat))

        for name, dur in events:
            if name.startswith("WINK") or name == "LONG_CLOSE":
                self.log.appendleft(f"{time.strftime('%H:%M:%S')} {name} {dur * 1000:.0f}ms")

        gaze_px, target, snap_name = None, None, ""
        closing = max(aL, aR) >= 0.5 * cfg.close_thr
        if self.model is not None and not closing:
            mi, sx, sy = self._estimate(now, feat)
            st["mon"] = mi
            gaze_px = (sx, sy)
            fixating = self._fixating(now, gaze_px)
            target, snap_name = self._apply_snap(now, sx, sy, fixating and engaged and not self.driver.frozen)
        elif self.status.get("gaze"):
            gaze_px = self.status["gaze"]

        acts = []
        if not self.suspended:
            acts = self.driver.update(now, aL, aR, events, target, engaged)
        for a in acts:
            if a.startswith("CLICK"):
                self.log.appendleft(f"{time.strftime('%H:%M:%S')} {a}")
                if cfg.beep:
                    beep(800 if a == "CLICK_L" else 1200)
        st.update(face=True, aL=aL, aR=aR, gaze=gaze_px, frozen=self.driver.frozen, snap=snap_name)
        self._preview(frame, pts, aL, aR, want_preview)

    def _preview(self, frame, pts, aL, aR, on):
        if not on or cv2 is None:
            self.preview = None
            return
        self._pv_n += 1
        if self._pv_n % 2:
            return
        img = cv2.flip(cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1]))), 1)
        h, w = img.shape[:2]
        if pts is not None:
            for idx in (L_EYE, R_EYE):
                poly = np.array([[(1 - pts[i][0]) * w, pts[i][1] * h] for i in idx], np.int32)
                cv2.polylines(img, [poly], True, (255, 220, 0), 1, cv2.LINE_AA)
        thr = self.cfg.close_thr
        for k, (label, v) in enumerate((("L", aL), ("R", aR))):
            y = h - 44 + 20 * k
            cv2.rectangle(img, (40, y), (240, y + 14), (40, 40, 40), -1)
            col = (60, 60, 230) if v >= thr else (80, 200, 80)
            cv2.rectangle(img, (40, y), (40 + int(200 * v), y + 14), col, -1)
            tx = 40 + int(200 * thr)
            cv2.line(img, (tx, y - 2), (tx, y + 16), (255, 255, 255), 2)
            cv2.putText(img, label, (18, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self.preview = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =========================================================================== #
# Calibration helpers (shared by the UI)
# =========================================================================== #
CALIB_POINTS = [(0.08, 0.08), (0.5, 0.08), (0.92, 0.08),
                (0.29, 0.29), (0.71, 0.29),
                (0.08, 0.5), (0.5, 0.5), (0.92, 0.5),
                (0.29, 0.71), (0.71, 0.71),
                (0.08, 0.92), (0.5, 0.92), (0.92, 0.92)]


def fit_from_calibration(engine, monitors):
    """Turn collected samples into a GazeModel + baselines. Returns (model, report_lines)."""
    samples, raws = engine.pop_calibration()
    rows = []
    for (mi, pid), feat in samples:
        fx, fy = CALIB_POINTS[pid % len(CALIB_POINTS)]
        rows.append((mi, pid, feat, (fx, fy)))
    model = GazeModel(monitors).fit(rows)
    lines = list(model.report)
    for k in ("blend", "ear"):
        if len(raws[k]) >= 30:
            a = np.array(raws[k])
            engine.cfg.baselines[k] = [float(np.percentile(a[:, 0], 90)), float(np.percentile(a[:, 1], 90))]
    b = engine.cfg.baselines.get(engine.cfg.metric, [0, 0])
    lines.append(f"Open-eye baseline: L {b[0]:.2f}  R {b[1]:.2f}")
    return model, lines
