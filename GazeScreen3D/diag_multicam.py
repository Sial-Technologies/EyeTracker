"""
Standalone multi-camera contamination diagnostic (no Tk, no CameraReader).

Point every camera at a scene you cannot confuse (lens covered, lamp, patterned
target). Saved JPEGs are the ground truth; the "looks_like" column flags frames
whose thumbnail is closer to another camera's baseline than to its own.

Usage:
  cd GazeScreen3D
  python diag_multicam.py --probe
  python diag_multicam.py --indices 2 3 1 --backend dshow
  python diag_multicam.py --indices 2 3 1 --backend msmf
  python diag_multicam.py --indices 2 3 1 --backend dshow --fourcc MJPG
  python diag_multicam.py --indices 2 3 1 --backend dshow --size 320x240
  python diag_multicam.py --indices 1 3 2 --backend dshow
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(ROOT, "diag_out")

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
BLACK_MEAN = 2.0
BASELINE_FRAMES = 10
THUMB_SIZE = (16, 12)
# Another camera's baseline must beat our own by this factor before we flag it,
# otherwise ordinary scene motion would produce false positives.
LOOKS_LIKE_MARGIN = 0.6


def decode_fourcc(value):
    code = int(value)
    if code <= 0:
        return "?"
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return chars if chars.isprintable() else f"0x{code:08x}"


def thumbnail(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


class Log:
    """Print to console and mirror into the run folder."""

    def __init__(self, path):
        self._f = open(path, "w", encoding="utf-8")
        self._lock = threading.Lock()

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            print(line, flush=True)
            self._f.write(line + "\n")
            self._f.flush()

    def close(self):
        self._f.close()


def open_capture(index, backend, fourcc, size, fps):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    # FOURCC must precede size on DirectShow or the driver keeps its default format.
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def describe_capture(cap):
    try:
        backend_name = cap.getBackendName()
    except cv2.error:
        backend_name = "?"
    return (
        f"backend={backend_name} fourcc={decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))} "
        f"size={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"fps={cap.get(cv2.CAP_PROP_FPS):.1f}"
    )


class CamStream:
    """One VideoCapture + one reader thread; no reconnect, no fallback."""

    def __init__(self, index, cap):
        self.index = index
        self.cap = cap
        self.ok = 0
        self.fail = 0
        self.black = 0
        self.latest = None
        self.baseline = None
        self._baseline_acc = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                with self._lock:
                    self.fail += 1
                time.sleep(0.01)
                continue
            owned = frame.copy()
            mean = float(owned.mean())
            with self._lock:
                self.ok += 1
                if mean < BLACK_MEAN:
                    self.black += 1
                self.latest = (owned, mean)
                if self.baseline is None and mean >= BLACK_MEAN:
                    self._baseline_acc.append(thumbnail(owned))
                    if len(self._baseline_acc) >= BASELINE_FRAMES:
                        self.baseline = np.mean(self._baseline_acc, axis=0)
                        self._baseline_acc = []

    def snapshot(self):
        with self._lock:
            frame, mean = self.latest if self.latest is not None else (None, None)
            return {
                "ok": self.ok,
                "fail": self.fail,
                "black": self.black,
                "frame": None if frame is None else frame.copy(),
                "mean": mean,
            }

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cap.release()


def classify(frame, streams, self_index):
    """Which camera's baseline does this frame most resemble?"""
    if frame is None:
        return "-"
    thumb = thumbnail(frame)
    dists = {
        s.index: float(np.abs(thumb - s.baseline).mean())
        for s in streams
        if s.baseline is not None
    }
    if self_index not in dists:
        return "no-baseline"
    own = dists[self_index]
    best_idx = min(dists, key=dists.get)
    if best_idx != self_index and dists[best_idx] < own * LOOKS_LIKE_MARGIN:
        return f"cam{best_idx}!! (d={dists[best_idx]:.1f} own={own:.1f})"
    return f"self (d={own:.1f})"


def run_streams(args, log, run_dir):
    backend = BACKENDS[args.backend]
    streams = []
    t0 = time.perf_counter()
    totals = {}

    def tick(label):
        elapsed = time.perf_counter() - t0
        log(f"--- t={elapsed:5.1f}s {label}")
        for s in streams:
            snap = s.snapshot()
            prev_ok, prev_fail = totals.get(s.index, (0, 0))
            totals[s.index] = (snap["ok"], snap["fail"])
            mean = "-" if snap["mean"] is None else f"{snap['mean']:6.1f}"
            log(
                f"cam{s.index}: +ok={snap['ok'] - prev_ok:3d} +fail={snap['fail'] - prev_fail:3d} "
                f"black_total={snap['black']:4d} mean={mean} looks_like={classify(snap['frame'], streams, s.index)}"
            )
            if snap["frame"] is not None:
                path = os.path.join(run_dir, f"cam{s.index}_t{elapsed:05.1f}.jpg")
                cv2.imwrite(path, snap["frame"])

    try:
        for n, index in enumerate(args.indices):
            if n > 0:
                # Keep earlier cameras streaming during the delay, like the setup GUI does.
                end = time.perf_counter() + args.open_delay
                while time.perf_counter() < end:
                    time.sleep(min(1.0, end - time.perf_counter()))
                    tick(f"before opening cam{index}")
            log(f"opening cam{index} ({args.backend}) ...")
            t_open = time.perf_counter()
            cap = open_capture(index, backend, args.fourcc, args.size, args.fps)
            if cap is None:
                log(f"cam{index}: FAILED to open with {args.backend} (no fallback)")
                continue
            log(f"cam{index}: opened in {time.perf_counter() - t_open:.2f}s  {describe_capture(cap)}")
            stream = CamStream(index, cap)
            streams.append(stream)
            stream.start()

        end = time.perf_counter() + args.seconds
        while time.perf_counter() < end:
            time.sleep(1.0)
            tick("all opened")
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        log("=== summary")
        for s in streams:
            snap = s.snapshot()
            log(f"cam{s.index}: ok={snap['ok']} fail={snap['fail']} black={snap['black']}")
            s.stop()


def run_probe(args, log, run_dir):
    """Open each index alone on each backend: reveals DSHOW vs MSMF index mapping."""
    names = [args.backend] if args.backend else list(BACKENDS)
    for name in names:
        log(f"=== probe {name}")
        for index in range(args.max_index):
            cap = open_capture(index, BACKENDS[name], args.fourcc, args.size, args.fps)
            if cap is None:
                log(f"{name} idx{index}: not opened")
                continue
            frame = None
            for _ in range(15):
                ret, f = cap.read()
                if ret and f is not None:
                    frame = f
            info = describe_capture(cap)
            cap.release()
            if frame is None:
                log(f"{name} idx{index}: opened but no frame  {info}")
                continue
            path = os.path.join(run_dir, f"probe_{name}_idx{index}.jpg")
            cv2.imwrite(path, frame)
            log(f"{name} idx{index}: mean={float(frame.mean()):6.1f}  {info}  -> {os.path.basename(path)}")
            time.sleep(0.3)


def parse_size(text):
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected WxH, got {text!r}") from exc


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", action="store_true", help="open each index alone and save one frame per backend")
    p.add_argument("--max-index", type=int, default=8, help="probe: highest index to try (exclusive)")
    p.add_argument("--indices", type=int, nargs="+", default=[2, 3, 1], help="open order")
    p.add_argument("--backend", choices=sorted(BACKENDS), default=None, help="default: dshow (probe: both)")
    p.add_argument("--fourcc", default=None, help="e.g. MJPG, YUY2")
    p.add_argument("--size", type=parse_size, default=(640, 480), help="WxH, e.g. 640x480")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seconds", type=float, default=20.0, help="run time after all cameras open")
    p.add_argument("--open-delay", type=float, default=3.0, help="seconds between opens")
    args = p.parse_args(argv)
    if args.fourcc is not None and len(args.fourcc) != 4:
        p.error("--fourcc must be exactly 4 characters")
    if not args.probe and args.backend is None:
        args.backend = "dshow"
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.probe:
        tag = f"probe_{args.backend or 'both'}"
    else:
        order = "-".join(str(i) for i in args.indices)
        tag = f"{args.backend}_{args.fourcc or 'default'}_{args.size[0]}x{args.size[1]}_order{order}"
    run_dir = os.path.join(OUT_ROOT, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    log = Log(os.path.join(run_dir, "log.txt"))
    log(f"opencv {cv2.__version__}  args={vars(args)}")
    log(f"output: {run_dir}")
    try:
        if args.probe:
            run_probe(args, log, run_dir)
        else:
            run_streams(args, log, run_dir)
    finally:
        log.close()


if __name__ == "__main__":
    main()
