"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        WATER DEPTH DETECTION — YOLOv8 + Homography + OCR Validation        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Pipeline:                                                                   ║
║  Frame → YOLO Seg → Largest Water Contour → Waterline →                     ║
║  Smoothing → Homography → Real-world Depth → OCR Validation → HUD           ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALLATION  (run once):

    pip install ultralytics opencv-python numpy easyocr

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN:

    # Video file
    python water_depth_detector.py --source path/to/flood.mp4 --model best.pt

    # Webcam (index 0)
    python water_depth_detector.py --source 0 --model best.pt

    # Enable OCR gauge validation
    python water_depth_detector.py --source flood.mp4 --model best.pt --ocr

    # Reuse a saved calibration (skip interactive step)
    python water_depth_detector.py --source flood.mp4 --model best.pt \
                                   --calib calib.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW CALIBRATION WORKS:

    1.  A frame appears — CLICK exactly 4 reference points on it.
        Choose points with KNOWN real-world positions (e.g. gauge markings,
        road edge markers, poles).  Points must NOT be collinear.

    2.  After 4 clicks the terminal prompts for real-world coordinates.
        Enter X,Y in metres for each point (e.g.  0,0  or  1.5,2.0).

    3.  Press ENTER (or SPACE) in the calibration window to confirm.
        Press  r  to reset and re-click at any time.

    4.  Calibration is auto-saved to  calib.json  for future runs.

KEY BINDINGS during detection:
    q  — quit
    r  — redo calibration
    p  — pause / unpause
    s  — save current frame as PNG

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL TUNING CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SMOOTH_BUF_SIZE  = 12      # rolling-median buffer size (frames)
SPIKE_THRESHOLD  = 0.40    # max allowed depth jump between frames (metres)
OCR_DIFF_THRESH  = 0.15    # OCR vs homography mismatch warning threshold (m)
MORPH_KERNEL_SZ  = 7       # kernel for morphological mask clean-up
MIN_CONTOUR_AREA = 200     # ignore contours smaller than this (px²) inside ROI
OCR_EVERY_N      = 15      # run OCR once every N frames (saves CPU)
CALIB_FILE_DEF   = "calib.json"
ROI_HALF_WIDTH   = 30      # pixels left/right of gauge centre for waterline ROI
COLLINEAR_EPS    = 1e-3    # epsilon added to X when points are nearly collinear
MIN_WORLD_Y_SPAN = 0.3     # warn if calibration Y range is too small (metres)

# Colours (BGR)
COL_WATERLINE = (0, 255, 255)    # cyan
COL_MASK      = (255, 100, 0)    # orange-blue water tint
COL_CALIB_PT  = (0, 255, 0)      # green calibration dots
COL_TEXT      = (255, 255, 255)  # white
COL_WARN      = (0, 0, 255)      # red
MASK_ALPHA    = 0.35             # opacity of water overlay


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Water depth detector — YOLOv8 + homography + OCR"
    )
    p.add_argument("--source", default="0",
                   help="Video file path or webcam index (default: 0)")
    p.add_argument("--model",  default="best.pt",
                   help="YOLOv8 segmentation model (default: best.pt)")
    p.add_argument("--ocr",    action="store_true",
                   help="Enable EasyOCR gauge validation")
    p.add_argument("--calib",  default=CALIB_FILE_DEF,
                   help=f"Calibration JSON file (default: {CALIB_FILE_DEF})")
    p.add_argument("--conf",   type=float, default=0.25,
                   help="YOLO confidence threshold (default: 0.25)")
    p.add_argument("--gauge-x", dest="gauge_x", type=int, default=None,
                   help="Pixel X-coordinate of gauge centre for ROI waterline "
                        "detection. If omitted, inferred from calibration points.")
    p.add_argument("--roi-width", dest="roi_width", type=int,
                   default=ROI_HALF_WIDTH,
                   help=f"Half-width of gauge ROI in pixels (default: {ROI_HALF_WIDTH})")
    p.add_argument("--debug", action="store_true",
                   help="Print per-frame wl_y / world_y debug values")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_yolo_model(model_path: str):
    """Load a YOLOv8 segmentation model from disk."""
    if not Path(model_path).exists():
        sys.exit(f"[ERROR] Model file not found: {model_path}")
    try:
        from ultralytics import YOLO  # lazy import
        mdl = YOLO(model_path)
        print(f"[OK] Model loaded: {model_path}")
        return mdl
    except Exception as exc:
        sys.exit(f"[ERROR] Cannot load model: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
#  OCR  (optional)
# ══════════════════════════════════════════════════════════════════════════════

def init_ocr():
    """Attempt to load EasyOCR; return reader or None on failure."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("[OK] EasyOCR ready (CPU)")
        return reader
    except ImportError:
        print("[WARN] easyocr not installed — OCR disabled.  pip install easyocr")
        return None
    except Exception as exc:
        print(f"[WARN] OCR init failed: {exc}  — OCR disabled")
        return None


def read_gauge_ocr(reader, frame: np.ndarray,
                   gauge_x: "int | None" = None,
                   roi_half: int = ROI_HALF_WIDTH * 6) -> "float | None":
    """
    VALIDATION ONLY — OCR result never overrides homography depth.

    Crops to a vertical strip around the gauge (if gauge_x is known) to
    reduce false positives from background numbers, then returns the first
    plausible depth reading in [0.05, 5.0] m with confidence > 0.6.
    Returns None on any failure.
    """
    if reader is None:
        return None
    try:
        h, w = frame.shape[:2]
        # Crop to gauge column if possible (reduces false OCR hits)
        if gauge_x is not None:
            x1 = max(0, gauge_x - roi_half)
            x2 = min(w, gauge_x + roi_half)
            roi = frame[:, x1:x2]
        else:
            roi = frame

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Light contrast enhancement helps digit recognition
        gray = cv2.equalizeHist(gray)
        hits = reader.readtext(gray, allowlist="0123456789.", detail=1)
        for (_, text, conf) in hits:
            text = text.strip().replace(",", ".")
            if conf > 0.6:                     # tighter confidence gate
                try:
                    val = float(text)
                    if 0.05 <= val <= 5.0:     # realistic flood gauge range
                        return val
                except ValueError:
                    pass
    except Exception as exc:
        print(f"[WARN] OCR read error: {exc}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def segment_frame(model, frame: np.ndarray, conf: float) -> list:
    """
    Run YOLOv8 segmentation on `frame`.

    Returns:
        List of binary uint8 mask arrays (same H×W as frame, values 0 or 255).
        Empty list when no masks are detected.
    """
    results = model(frame, conf=conf, verbose=False)[0]

    if results.masks is None or len(results.masks) == 0:
        return []

    h, w     = frame.shape[:2]
    out_masks = []

    # results.masks.data : Tensor [N, mask_h, mask_w], float in [0, 1]
    raw = results.masks.data.cpu().numpy()   # shape (N, mh, mw)
    for mask in raw:
        # up-sample to frame resolution
        resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        binary  = (resized > 0.5).astype(np.uint8) * 255
        out_masks.append(binary)

    return out_masks


# ══════════════════════════════════════════════════════════════════════════════
#  WATER MASK → ROI-BASED WATERLINE DETECTION  (fixes constant wl_y bug)
# ══════════════════════════════════════════════════════════════════════════════

def merge_masks(masks: list, frame_shape: tuple) -> np.ndarray:
    """
    Merge all segmentation masks into one binary image (0 or 255).
    Applies morphological open+close to remove noise and fill holes.
    """
    h, w   = frame_shape[:2]
    merged = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        merged = cv2.bitwise_or(merged, m)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SZ, MORPH_KERNEL_SZ)
    )
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN,  kernel)
    return merged


def detect_waterline_in_roi(
    masks:      list,
    frame_shape: tuple,
    gauge_x:    int,
    roi_half:   int = ROI_HALF_WIDTH,
) -> "tuple[int | None, tuple | None, np.ndarray | None]":
    """
    Detect the waterline y-coordinate by restricting analysis to a narrow
    vertical ROI centred on the flood gauge.

    Why ROI?
        • Bridge edges and reflections span the full image width.
        • The actual gauge is a thin vertical strip — restricting to it
          discards those horizontal false-positive structures.

    Algorithm:
        1. Merge all masks and clean with morphology.
        2. Zero out everything outside [gauge_x ± roi_half].
        3. In that column, find the topmost water pixel (min y where mask==255).
           This is the real waterline, not a bridge ledge.
        4. As a fallback, also find the largest contour inside the ROI.

    Returns:
        (wl_y, roi_rect, full_merged_mask)
        wl_y     — waterline pixel row, or None
        roi_rect — (x1, y1, x2, y2) of the ROI box for visualisation
        merged   — full-frame merged mask for overlay drawing
    """
    if not masks:
        return None, None, None

    h, w   = frame_shape[:2]
    merged = merge_masks(masks, frame_shape)

    # ── Clamp ROI to frame boundaries ───────────────────────────────────────
    x1 = max(0, gauge_x - roi_half)
    x2 = min(w, gauge_x + roi_half)
    roi_rect = (x1, 0, x2, h)

    # ── Extract ROI column from mask ─────────────────────────────────────────
    roi_mask = np.zeros_like(merged)
    roi_mask[:, x1:x2] = merged[:, x1:x2]

    # ── Method 1: topmost water pixel in the ROI column ──────────────────────
    water_rows = np.where(roi_mask[:, x1:x2].any(axis=1))[0]
    if len(water_rows) > 0:
        wl_y = int(water_rows.min())
        return wl_y, roi_rect, merged

    # ── Method 2 (fallback): largest contour inside ROI ──────────────────────
    cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, roi_rect, merged

    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < MIN_CONTOUR_AREA:
        return None, roi_rect, merged

    pts  = best.reshape(-1, 2)
    wl_y = int(pts[:, 1].min())
    return wl_y, roi_rect, merged


def infer_gauge_x(calib_pixel_pts: list, frame_width: int) -> int:
    """
    Infer the gauge X pixel from calibration points.
    Uses the median X of all calibration points (they should all be on/near
    the gauge).  Falls back to frame centre if no calibration yet.
    """
    if calib_pixel_pts:
        xs = [pt[0] for pt in calib_pixel_pts]
        return int(np.median(xs))
    return frame_width // 2


# ══════════════════════════════════════════════════════════════════════════════
#  SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class Smoother:
    """
    Rolling median for the waterline pixel and spike filter for depth.
    """

    def __init__(self, buf_size: int = SMOOTH_BUF_SIZE):
        self._wl_buf:    deque          = deque(maxlen=buf_size)
        self._last_depth: "float | None" = None

    def smooth_waterline(self, y_px: int) -> int:
        """Add raw waterline reading, return smoothed median."""
        self._wl_buf.append(y_px)
        return int(np.median(self._wl_buf))

    def filter_depth(self, depth: float) -> float:
        """Reject spikes; return last stable depth on spike."""
        if self._last_depth is not None:
            if abs(depth - self._last_depth) > SPIKE_THRESHOLD:
                print(
                    f"\n[SPIKE] depth={depth:.3f}m "
                    f"jumped {abs(depth-self._last_depth):.3f}m — ignored"
                )
                return self._last_depth
        self._last_depth = depth
        return depth


# ══════════════════════════════════════════════════════════════════════════════
#  HOMOGRAPHY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def pixel_to_world(H: np.ndarray, px: float, py: float) -> "tuple[float,float]":
    """Transform a single pixel point to real-world coords using homography."""
    pt    = np.array([[[px, py]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)
    return float(world[0][0][0]), float(world[0][0][1])


def waterline_depth(
    H:           np.ndarray,
    wl_y:        int,
    gauge_x:     int,
    ref_world_y: float = 0.0,
    debug:       bool  = False,
) -> float:
    """
    Convert waterline pixel-row `wl_y` to a real-world depth (metres).

    Samples at gauge_x (the gauge column) for accuracy — NOT the frame centre,
    which could project onto a completely different world plane.

    Depth = |world_Y_at_waterline  −  reference_Y (ground/zero mark)|

    Debug prints show raw world_y so you can verify scaling is correct.
    """
    world_x, world_y = pixel_to_world(H, float(gauge_x), float(wl_y))
    depth = abs(world_y - ref_world_y)

    if debug:
        print(f"\n  [DBG] wl_y={wl_y}px  gauge_x={gauge_x}px  "
              f"→  world=({world_x:.4f}, {world_y:.4f})m  "
              f"ref_y={ref_world_y:.4f}m  depth={depth:.4f}m")

    return round(depth, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

class Calibrator:
    """
    GUI-based 4-point calibration:
      1. User clicks 4 pixel points in the image.
      2. Terminal prompts for corresponding real-world (X, Y) in metres.
      3. Homography is computed and stored.
      4. State is persisted to a JSON file.
    """

    _WIN = "CALIBRATION  |  Click 4 points  |  ENTER=confirm  r=reset  q=quit"

    def __init__(self) -> None:
        self.pixel_pts: list            = []
        self.world_pts: list            = []
        self.H:         "np.ndarray | None" = None

    # ── mouse callback ──────────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pixel_pts) < 4:
                self.pixel_pts.append((x, y))
                print(f"  → Point {len(self.pixel_pts)} registered at pixel ({x}, {y})")
            else:
                print("  [INFO] Already have 4 points. Press r to reset.")

    # ── draw helpers ────────────────────────────────────────────────────────

    def _annotate(self, base: np.ndarray) -> np.ndarray:
        vis = base.copy()
        for i, (px, py) in enumerate(self.pixel_pts):
            px, py = int(px), int(py)
            cv2.circle(vis, (px, py), 9, COL_CALIB_PT, -1)
            cv2.circle(vis, (px, py), 11, (0, 0, 0), 2)
            cv2.putText(vis, f"P{i+1}", (px + 13, py - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(vis, f"P{i+1}", (px + 13, py - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_CALIB_PT, 2)

        # status strip at bottom
        n  = len(self.pixel_pts)
        bg = (30, 30, 30)
        cv2.rectangle(vis, (0, vis.shape[0] - 36),
                      (vis.shape[1], vis.shape[0]), bg, -1)
        cv2.putText(
            vis,
            f"Selected: {n}/4   |   ENTER or SPACE = confirm   r = reset",
            (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_TEXT, 1,
        )
        return vis

    # ── terminal prompt for real-world coords ────────────────────────────────

    def _prompt_world(self) -> bool:
        """
        Prompt the user for real-world (X, Y) in metres for each pixel point.
        Returns True on success, False if user wants to cancel and re-click.
        """
        print()
        print("─" * 62)
        print("  Enter real-world coordinates for each point (metres).")
        print("  Format:  X,Y   (e.g.  0,0   or   1.5,2.3)")
        print("  Blank input → cancel and re-click points.")
        print("─" * 62)

        world = []
        for i, (px, py) in enumerate(self.pixel_pts):
            while True:
                raw = input(
                    f"  Point {i+1}  pixel=({px},{py})  →  real-world X,Y (m): "
                ).strip()
                if raw == "":
                    print("  [CANCEL] Returning to point selection.\n")
                    return False
                try:
                    parts = raw.replace(";", ",").split(",")
                    X, Y  = float(parts[0].strip()), float(parts[1].strip())
                    world.append((X, Y))
                    print(f"             Accepted: ({X:.3f}, {Y:.3f}) m")
                    break
                except (ValueError, IndexError):
                    print("  [ERROR] Bad format — try again  (e.g.  1.2,0.5)")

        self.world_pts = world
        return True

    # ── compute H ───────────────────────────────────────────────────────────

    def _fit_homography(self) -> bool:
        src = np.array(self.pixel_pts, dtype=np.float32)
        dst = np.array(self.world_pts, dtype=np.float32)

        # ── Guard: warn if real-world Y range is very small ─────────────────
        y_vals = [pt[1] for pt in self.world_pts]
        y_span = max(y_vals) - min(y_vals)
        if y_span < MIN_WORLD_Y_SPAN:
            print(
                f"\n  [WARN] Real-world Y range is only {y_span:.3f} m "
                f"(< {MIN_WORLD_Y_SPAN} m).\n"
                f"         Depth resolution will be very coarse.\n"
                f"         Use calibration points that span at least "
                f"{MIN_WORLD_Y_SPAN} m vertically."
            )

        # ── Guard: detect near-collinear pixel points ────────────────────────
        # Collinear points make the homography degenerate. Fix: nudge X coords.
        xs = src[:, 0]
        if (xs.max() - xs.min()) < 5.0:
            print(
                "\n  [WARN] Calibration pixel points are nearly vertically "
                "aligned (X spread < 5 px).\n"
                "         Adding small epsilon variation to X to avoid "
                "degenerate homography."
            )
            eps = np.array([-COLLINEAR_EPS * 2, COLLINEAR_EPS * 2,
                             -COLLINEAR_EPS,     COLLINEAR_EPS],
                           dtype=np.float32)
            src[:, 0] += eps[:len(src)]

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            print(
                "[ERROR] Homography computation failed!\n"
                "        Ensure points are non-collinear and spread apart.\n"
                "        Try clicking points at the four corners of a "
                "rectangular reference (e.g. a ruler face)."
            )
            return False
        self.H = H
        inliers = int(mask.sum()) if mask is not None else "N/A"
        print(f"[OK] Homography computed. Inliers: {inliers}/4  "
              f"(world Y span: {y_span:.3f} m)")

        # ── Sanity-check: verify calibration points round-trip correctly ─────
        print("  [CHECK] Pixel → World round-trip for calibration points:")
        for i, ((px, py), (wx, wy)) in enumerate(
                zip(self.pixel_pts, self.world_pts)):
            cx, cy = pixel_to_world(self.H, float(px), float(py))
            err = ((cx - wx) ** 2 + (cy - wy) ** 2) ** 0.5
            print(f"    P{i+1}: pixel=({px},{py})  "
                  f"expected=({wx:.3f},{wy:.3f})  "
                  f"got=({cx:.3f},{cy:.3f})  err={err:.4f}m")
        return True

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            "pixel_pts": self.pixel_pts,
            "world_pts": self.world_pts,
            "H": self.H.tolist() if self.H is not None else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Calibration saved → {path}")

    def load(self, path: str) -> bool:
        if not Path(path).exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.pixel_pts = [tuple(pt) for pt in data["pixel_pts"]]
            self.world_pts = [tuple(pt) for pt in data["world_pts"]]
            self.H = np.array(data["H"], dtype=np.float64) if data["H"] else None
            if self.H is None:
                return False
            print(f"[OK] Calibration loaded from {path}")
            return True
        except Exception as exc:
            print(f"[WARN] Could not read calibration file: {exc}")
            return False

    # ── main interactive UI ──────────────────────────────────────────────────

    def run_interactive(self, frame: np.ndarray) -> None:
        """
        Block until the user successfully calibrates.
        Sets self.H, self.pixel_pts, self.world_pts on completion.
        """
        # Reset state
        self.pixel_pts = []
        self.world_pts = []
        self.H         = None

        cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self._WIN,
            min(frame.shape[1], 1280),
            min(frame.shape[0], 720),
        )
        cv2.setMouseCallback(self._WIN, self._on_mouse)

        print()
        print("═" * 62)
        print("  CALIBRATION MODE")
        print("  Click 4 reference points on the image (use gauge marks,")
        print("  poles, or any fixed objects with known real positions).")
        print("  Points must NOT all lie on the same straight line.")
        print("  ENTER / SPACE = confirm    r = reset    q = quit")
        print("═" * 62)
        print()

        while True:
            vis = self._annotate(frame)
            cv2.imshow(self._WIN, vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                cv2.destroyWindow(self._WIN)
                sys.exit("[EXIT] Calibration aborted by user.")

            elif key == ord("r"):
                self.pixel_pts = []
                print("  [RESET] Click 4 points again.")

            elif key in (13, ord(" ")):          # ENTER or SPACE
                if len(self.pixel_pts) < 4:
                    print(f"  [INFO] Need 4 points; have {len(self.pixel_pts)}.")
                    continue

                ok = self._prompt_world()
                if not ok:
                    self.pixel_pts = []
                    self.world_pts = []
                    continue

                if self._fit_homography():
                    cv2.destroyWindow(self._WIN)
                    return                       # success

                # Homography failed → let user redo
                self.pixel_pts = []
                self.world_pts = []


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def render_overlay(
    frame:       np.ndarray,
    masks:       list,
    merged_mask: "np.ndarray | None",
    wl_y:        "int | None",
    depth:       "float | None",
    ocr_val:     "float | None",
    calib_pts:   list,
    roi_rect:    "tuple | None" = None,
    ocr_warn:    bool = False,
) -> np.ndarray:
    """Composite all visual elements onto the frame."""
    vis = frame.copy()

    # ── water mask (semi-transparent, full frame) ────────────────────────────
    if merged_mask is not None:
        layer          = np.zeros_like(frame)
        layer[merged_mask > 0] = COL_MASK
        vis = cv2.addWeighted(vis, 1.0, layer, MASK_ALPHA, 0)

    # ── waterline (horizontal line across full width) ─────────────────────────
    if wl_y is not None:
        W = frame.shape[1]
        cv2.line(vis, (0, wl_y), (W, wl_y), COL_WATERLINE, 2)
        for x in range(0, W, 60):
            cv2.line(vis, (x, wl_y - 6), (x, wl_y + 6), COL_WATERLINE, 1)

    # ── ROI box around gauge (yellow dashed rectangle) ────────────────────────
    if roi_rect is not None:
        rx1, ry1, rx2, ry2 = roi_rect
        # Draw dashed yellow border to distinguish from waterline
        col_roi = (0, 220, 255)
        for seg_y in range(ry1, ry2, 20):
            y2 = min(seg_y + 10, ry2)
            cv2.line(vis, (rx1, seg_y), (rx1, y2), col_roi, 1)
            cv2.line(vis, (rx2, seg_y), (rx2, y2), col_roi, 1)
        cv2.line(vis, (rx1, ry1), (rx2, ry1), col_roi, 1)
        cv2.line(vis, (rx1, ry2 - 1), (rx2, ry2 - 1), col_roi, 1)
        cv2.putText(vis, "Gauge ROI", (rx1, ry1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_roi, 1)

    # ── calibration point markers ────────────────────────────────────────────
    for i, (px, py) in enumerate(calib_pts):
        px, py = int(px), int(py)
        cv2.circle(vis, (px, py), 5, COL_CALIB_PT, -1)
        cv2.circle(vis, (px, py), 7, (0, 0, 0), 1)
        cv2.putText(vis, f"C{i+1}", (px + 8, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_CALIB_PT, 1)

    # ── HUD (top-left panel) ─────────────────────────────────────────────────
    lines = []
    depth_str = f"{depth:.3f} m" if depth is not None else "-- m"
    lines.append(("Water Depth", depth_str, COL_TEXT))
    wl_str = f"Waterline Y: {wl_y} px" if wl_y is not None else "No detection"
    lines.append(("", wl_str, (160, 160, 160)))
    if ocr_val is not None:
        tag = "  [MISMATCH]" if ocr_warn else "  [OK]"
        col = COL_WARN if ocr_warn else (80, 220, 80)
        lines.append(("OCR(val)", f"{ocr_val:.2f} m{tag}", col))

    panel_h = 30 + 28 * len(lines)
    ov = vis.copy()
    cv2.rectangle(ov, (6, 6), (340, panel_h + 6), (20, 20, 20), -1)
    vis = cv2.addWeighted(vis, 0.35, ov, 0.65, 0)

    for i, (label, value, col) in enumerate(lines):
        y = 34 + i * 28
        if label:
            cv2.putText(vis, label + ":", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)
        cv2.putText(vis, value,
                    (12 if not label else 155, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72 if label == "Water Depth" else 0.52,
                    col, 2 if label == "Water Depth" else 1)

    # ── key-hint strip (bottom-right) ────────────────────────────────────────
    hint = "q=quit  r=recalib  p=pause  s=save"
    (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(vis, hint,
                (frame.shape[1] - tw - 8, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130, 130, 130), 1)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO SOURCE
# ══════════════════════════════════════════════════════════════════════════════

def open_video(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video source: {source}")
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    print(f"[OK] Source opened: {source}  ({W}×{H} @ {FPS:.1f} fps)")
    return cap


def grab_first_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """Read up to 30 frames until we get a good one."""
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
    sys.exit("[ERROR] Could not read any frames from the video source.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── components ─────────────────────────────────────────────────────────
    model      = load_yolo_model(args.model)
    ocr_reader = init_ocr() if args.ocr else None
    cap        = open_video(args.source)

    # ── grab a frame for calibration UI ────────────────────────────────────
    first_frame = grab_first_frame(cap)

    # ── calibration ─────────────────────────────────────────────────────────
    calib = Calibrator()
    if calib.load(args.calib):
        print("[INFO] Calibration loaded — press 'r' during detection to redo.")
    else:
        print("[INFO] No calibration found — launching interactive calibration.")
        calib.run_interactive(first_frame)
        calib.save(args.calib)

    H_mat       = calib.H                          # homography matrix
    calib_pts   = calib.pixel_pts                  # pixel coords for overlay
    ref_world_y = min(pt[1] for pt in calib.world_pts)  # ground reference Y

    # ── Determine gauge X (pixel column of the flood gauge) ─────────────────
    # Priority: CLI arg > inferred from calibration points > frame centre
    gauge_x = args.gauge_x if args.gauge_x is not None \
              else infer_gauge_x(calib_pts, first_frame.shape[1])
    roi_half = args.roi_width
    print(f"[INFO] Gauge X = {gauge_x}px  |  ROI half-width = {roi_half}px  "
          f"|  ref_world_y = {ref_world_y:.3f}m")

    # ── state ───────────────────────────────────────────────────────────────
    smoother    = Smoother(SMOOTH_BUF_SIZE)
    last_depth: "float | None" = None
    last_ocr:   "float | None" = None
    ocr_warn                   = False
    paused                     = False
    frame_idx                  = 0
    win                        = "Water Depth Detection"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win,
                     min(first_frame.shape[1], 1280),
                     min(first_frame.shape[0],  720))

    # Re-open if file (some frames were consumed for calibration)
    if not args.source.isdigit():
        cap.release()
        cap = open_video(args.source)

    print()
    print("[RUNNING] Detection active.  Press q to quit.")
    print()

    vis = first_frame   # initialise so 's' key works before first frame

    while True:
        # ── keyboard ────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\n[EXIT] User quit.")
            break

        if key == ord("p"):
            paused = not paused
            print(f"  {'[PAUSED]' if paused else '[RESUMED]'}")

        if key == ord("s"):
            fname = f"frame_{int(time.time())}.png"
            cv2.imwrite(fname, vis)
            print(f"\n  [SAVED] {fname}")

        if key == ord("r"):
            print("\n[RECALIBRATE] Opening calibration window …")
            ret, refresh = cap.read()
            ref_frame = refresh if ret else first_frame
            calib = Calibrator()
            calib.run_interactive(ref_frame)
            calib.save(args.calib)
            H_mat       = calib.H
            calib_pts   = calib.pixel_pts
            ref_world_y = min(pt[1] for pt in calib.world_pts)
            gauge_x     = args.gauge_x if args.gauge_x is not None \
                          else infer_gauge_x(calib_pts, frame.shape[1])
            print(f"[INFO] Gauge X updated → {gauge_x}px")
            smoother    = Smoother(SMOOTH_BUF_SIZE)
            last_depth  = None
            continue

        if paused:
            cv2.imshow(win, vis)
            continue

        # ── read frame ───────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("\n[INFO] Video ended / stream closed.")
            break

        frame_idx += 1
        t0 = time.perf_counter()

        # ── segmentation ─────────────────────────────────────────────────────
        masks = segment_frame(model, frame, args.conf)

        # ── ROI-based waterline detection (gauge column only) ─────────────────
        # This prevents bridge edges / reflections from being picked up as
        # the waterline — only the narrow gauge strip is examined.
        raw_y, roi_rect, merged_mask = detect_waterline_in_roi(
            masks, frame.shape, gauge_x, roi_half
        )
        sm_y = smoother.smooth_waterline(raw_y) if raw_y is not None else None

        # ── depth via homography ──────────────────────────────────────────────
        depth = None
        if sm_y is not None and H_mat is not None:
            raw_depth = waterline_depth(
                H_mat, sm_y, gauge_x, ref_world_y, debug=args.debug
            )
            depth = smoother.filter_depth(raw_depth)

        # Fall back to last stable depth when detection momentarily fails
        if depth is not None:
            last_depth = depth
        else:
            depth = last_depth

        # ── OCR validation only (every N frames to save CPU) ──────────────────
        # OCR result NEVER overrides homography — only prints a warning.
        if ocr_reader is not None and frame_idx % OCR_EVERY_N == 0:
            ocr_val = read_gauge_ocr(ocr_reader, frame, gauge_x)
            if ocr_val is not None:
                last_ocr = ocr_val
                if depth is not None:
                    diff = abs(ocr_val - depth)
                    ocr_warn = diff > OCR_DIFF_THRESH
                    if ocr_warn:
                        print(
                            f"\n  [OCR WARN] OCR={ocr_val:.2f}m  "
                            f"Hom={depth:.2f}m  diff={diff:.3f}m  "
                            f"→ keeping homography value"
                        )
                    else:
                        ocr_warn = False

        # ── terminal log ──────────────────────────────────────────────────────
        ms    = (time.perf_counter() - t0) * 1000
        d_str = f"{depth:.3f}m" if depth is not None else "N/A"
        print(
            f"  frame {frame_idx:5d} | wl_y={sm_y}px | depth={d_str} | {ms:.1f}ms",
            end="\r",
        )

        # ── draw ──────────────────────────────────────────────────────────────
        vis = render_overlay(
            frame       = frame,
            masks       = masks,
            merged_mask = merged_mask,
            wl_y        = sm_y,
            depth       = depth,
            ocr_val     = last_ocr,
            calib_pts   = calib_pts,
            roi_rect    = roi_rect,
            ocr_warn    = ocr_warn,
        )
        cv2.imshow(win, vis)

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Processed {frame_idx} frames total.")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()