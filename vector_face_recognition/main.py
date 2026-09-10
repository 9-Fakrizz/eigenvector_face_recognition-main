"""
main.py — Face ID System: Register + Recognize in one app
============================================================
A single Tkinter application that replaces the old split
register.py / recognize.py / main.py trio with one full-screen,
multi-page UI backed by one face-encoding database (face_store.py).

Pages
  - Recognize : live ESP32-triggered (or manual test button) scan,
                shows live camera + result panel, sends the matched
                ID back to the ESP32.
  - Register  : guided multi-pose capture wizard for enrolling a
                new person (or adding more samples to an existing one).
  - Manage    : list / search / delete enrolled people.

Performance notes (this used to slow down as more people were added)
  - The two big causes of the "gets laggy with more data" complaint
    were NOT the database lookup (that's a single vectorized numpy
    call in face_store.FaceStore.match, see that file) but:
      1. The UI thread and the recognition worker both called
         `cap.read()` on the *same* cv2.VideoCapture concurrently.
         OpenCV capture objects are not safe to share across threads
         like that; it silently serializes/stalls both readers under
         load. Fixed with a single CameraStream grabber thread that
         all consumers read a copy from.
      2. Face *detection* (HOG) ran on the full-resolution frame.
         Detection cost scales with pixel count, not roster size, but
         it dominates scan time. Fixed by detecting on a down-scaled
         copy of the frame and scaling the boxes back up before
         encoding on the full-res crop (same trick used during
         registration capture).
  - Recognition also now exits a scan early once a clear majority of
    votes has been reached, instead of always waiting for all
    SCAN_FRAMES frames.

Architecture
  SerialReaderThread ──► trigger_queue ──► FaceWorker(executor) ──► result_queue
  SerialWriterThread ◄──────────────────────────────────────────────┘
  CameraStreamThread ─► shared latest-frame buffer ─► UI + FaceWorker
  UIThread (main)    ◄── polls result_queue / drives pages
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import face_recognition
except ImportError:
    print("ERROR: face_recognition is not installed. Run: pip install face_recognition")
    sys.exit(1)

try:
    import serial
except ImportError:
    serial = None  # serial support becomes optional / test-mode only

from face_store import FaceStore

# ============================================================
# CONFIG
# ============================================================
SERIAL_PORT     = "/dev/ttyUSB0"   # e.g. "COM3" on Windows
SERIAL_BAUD     = 115200
CAMERA_INDEX    = 0
FACE_TOLERANCE  = 0.50             # lower = stricter match
SCAN_FRAMES     = 12               # max frames captured per scan attempt
MIN_VOTE_FRAMES = 4                # don't decide before at least this many votes
EARLY_EXIT_MARGIN = 3              # stop early once leader beats runner-up by this much
UNKNOWN_ID      = "000000000"      # sent back when no face matched
MAX_WORKERS     = 1                # recognition jobs are serialized: one camera, GIL-bound work
DETECTION_DOWNSCALE = 0.35         # detect on a smaller frame, encode on full-res crop
RESULT_HOLD_MS  = 3000             # how long a result stays on screen before returning to idle

GUIDED_STEPS = [
    ("Straight",  "Look STRAIGHT at the camera", 6),
    ("Left 45°",  "Turn your head ~45° to the LEFT", 5),
    ("Right 45°", "Turn your head ~45° to the RIGHT", 5),
    ("Tilt Up",   "Tilt your head slightly UP", 4),
    ("Tilt Down", "Tilt your head slightly DOWN", 4),
]

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)-8s] [%(name)-10s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MAIN")
log_serial = logging.getLogger("SERIAL")
log_face = logging.getLogger("FACE_REC")
log_worker = logging.getLogger("WORKER")
log_ui = logging.getLogger("UI")


# ============================================================
# Camera — single grabber thread, everyone else reads a copy
# ============================================================
class CameraStream:
    """
    Continuously reads frames from the camera in a dedicated thread and
    keeps only the latest one. All other threads call get_frame() to get
    a *copy*, instead of calling cap.read() themselves — sharing one
    cv2.VideoCapture across threads is not safe and was the main source
    of the "more load = more delay" symptom.
    """

    def __init__(self, index: int, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        log.info("CameraStream thread started")
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
            else:
                time.sleep(0.01)

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# ============================================================
# Face recognition helpers (shared by Recognize + Register pages)
# ============================================================
def detect_faces(frame_bgr: np.ndarray, downscale: float = DETECTION_DOWNSCALE):
    """
    Detect faces on a down-scaled copy of the frame (fast) and return
    face-location boxes scaled back up to full-frame coordinates, plus
    the full-resolution RGB frame to encode from.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (0, 0), fx=downscale, fy=downscale)
    small_locations = face_recognition.face_locations(small, model="hog")
    scale = 1.0 / downscale
    locations = [
        (int(top * scale), int(right * scale), int(bottom * scale), int(left * scale))
        for (top, right, bottom, left) in small_locations
    ]
    return rgb, locations


def encode_faces(frame_bgr: np.ndarray):
    """Detect + encode all faces in a frame. Returns list[(box, encoding)]."""
    rgb, locations = detect_faces(frame_bgr)
    if not locations:
        return []
    encodings = face_recognition.face_encodings(rgb, locations)
    return list(zip(locations, encodings))


# ============================================================
# Serial link — reader + writer threads, or a no-op stub in test mode
# ============================================================
class SerialLink:
    """
    Wraps ESP32 serial I/O. If the port can't be opened, runs in
    "test mode": read/write become no-ops but the rest of the app
    (including the manual TEST SCAN button) still works normally.
    """

    def __init__(self, port: str, baud: int, trigger_queue: "queue.Queue[str]"):
        self.trigger_queue = trigger_queue
        self.write_queue: "queue.Queue[str]" = queue.Queue()
        self.ser = None
        self.connected = False
        self._tx_count = 0

        if serial is None:
            log_serial.warning("pyserial not installed — running in TEST MODE (no hardware I/O)")
            return

        try:
            self.ser = serial.Serial(port, baud, timeout=0)
            self.connected = True
            log_serial.info(f"Serial open: {port} @ {baud} baud")
        except serial.SerialException as e:
            log_serial.warning(f"Serial error: {e} — running in TEST MODE (no hardware I/O)")

    def start(self):
        if not self.connected:
            return
        threading.Thread(target=self._reader_loop, daemon=True).start()
        threading.Thread(target=self._writer_loop, daemon=True).start()

    def send(self, msg: str):
        self._tx_count += 1
        if self.connected:
            self.write_queue.put(msg)
        else:
            log_serial.debug(f"[TEST MODE] would send: '{msg}' (msg #{self._tx_count})")

    def _reader_loop(self):
        log_serial.info("SerialReader thread started")
        buf = ""
        while True:
            try:
                if self.ser.in_waiting:
                    chunk = self.ser.read(self.ser.in_waiting).decode("utf-8", errors="ignore")
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line == "1":
                            log_serial.info(f"Trigger received at {datetime.now().strftime('%H:%M:%S')}")
                            self.trigger_queue.put("TRIGGER")
                else:
                    time.sleep(0.01)
            except Exception as e:
                log_serial.error(f"Reader error: {e}")
                time.sleep(0.5)

    def _writer_loop(self):
        log_serial.info("SerialWriter thread started")
        while True:
            try:
                msg = self.write_queue.get()
                self.ser.write((msg + "\n").encode())
                log_serial.info(f"Sent: '{msg}'")
            except Exception as e:
                log_serial.error(f"Writer error: {e}")


# ============================================================
# Recognition worker — consumes trigger_queue, produces result_queue
# ============================================================
class FaceWorker:
    def __init__(self, camera: CameraStream, store: FaceStore,
                 trigger_queue: "queue.Queue[str]", result_queue: "queue.Queue[dict]",
                 serial_link: SerialLink):
        self.camera = camera
        self.store = store
        self.trigger_queue = trigger_queue
        self.result_queue = result_queue
        self.serial_link = serial_link
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._job_count = 0

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        log_worker.info("FaceWorker thread started")
        while True:
            self.trigger_queue.get()
            self._job_count += 1
            job_id = self._job_count
            triggered_at = time.time()
            log_worker.info(f"Trigger received (#{job_id})")
            future = self.executor.submit(self._scan)
            future.add_done_callback(lambda f, jid=job_id, t=triggered_at: self._on_done(f, jid, t))

    def _scan(self) -> str:
        """Grabs frames, votes on identity, exits early on a clear majority."""
        votes: dict[str, int] = {}
        log_face.info(f"Starting scan (up to {SCAN_FRAMES} frames)")

        for i in range(SCAN_FRAMES):
            frame = self.camera.get_frame()
            if frame is None:
                continue

            for _box, enc in encode_faces(frame):
                pid, distance = self.store.match(enc, FACE_TOLERANCE)
                if pid is not None:
                    votes[pid] = votes.get(pid, 0) + 1
                    log_face.debug(f"  Frame {i+1}: MATCH '{pid}' (distance: {distance:.3f})")
                else:
                    votes[UNKNOWN_ID] = votes.get(UNKNOWN_ID, 0) + 1
                    d_str = f"{distance:.3f}" if distance is not None else "n/a"
                    log_face.debug(f"  Frame {i+1}: UNKNOWN (best distance: {d_str})")

            if votes and sum(votes.values()) >= MIN_VOTE_FRAMES:
                ranked = sorted(votes.values(), reverse=True)
                if len(ranked) == 1 or ranked[0] - ranked[1] >= EARLY_EXIT_MARGIN:
                    break

        if not votes:
            log_face.info("Final result: no face detected -> UNKNOWN")
            return UNKNOWN_ID

        winner = max(votes, key=votes.get)
        log_face.info(f"Final result: '{winner}' (votes: {votes})")
        return winner

    def _on_done(self, future, job_id: int, triggered_at: float):
        pid = future.result()
        elapsed = time.time() - triggered_at
        person = self.store.get(pid)
        display_name = person.name if person else None
        log_worker.info(f"Job #{job_id} complete: id='{pid}' name='{display_name}' time={elapsed:.2f}s")
        self.result_queue.put({"id": pid, "name": display_name, "elapsed": elapsed})
        self.serial_link.send(pid)


# ============================================================
# UI — shared look & feel helpers
# ============================================================
BG = "#0d0d0d"
PANEL_BG = "#1a1a1a"
FG_DIM = "#888888"
FG_LABEL = "#555555"
FG_TEXT = "#e0e0e0"
ACCENT = "#3a86ff"
GOOD = "#4dff91"
BAD = "#ff4444"
WARN = "#ff9f1c"
FONT_MONO = "Courier New"


class NavBar(tk.Frame):
    def __init__(self, parent, on_navigate, on_quit):
        super().__init__(parent, bg=BG, height=64)
        self.buttons: dict[str, tk.Button] = {}

        tk.Label(self, text=" FACE ID SYSTEM", font=(FONT_MONO, 18, "bold"),
                 fg=FG_TEXT, bg=BG).pack(side="left", padx=(4, 24), pady=14)

        for key, label in (("recognize", "RECOGNIZE"), ("register", "REGISTER"), ("manage", "MANAGE")):
            b = tk.Button(self, text=label, font=(FONT_MONO, 12, "bold"),
                          fg=FG_TEXT, bg="#222222", activebackground=ACCENT,
                          bd=0, padx=18, pady=10, cursor="hand2",
                          command=lambda k=key: on_navigate(k))
            b.pack(side="left", padx=6, pady=12)
            self.buttons[key] = b

        self.clock_label = tk.Label(self, text="", font=(FONT_MONO, 13), fg=FG_DIM, bg=BG)
        self.clock_label.pack(side="right", padx=(0, 10))

        tk.Button(self, text="QUIT", font=(FONT_MONO, 11, "bold"), fg=BAD, bg="#221111",
                  bd=0, padx=14, pady=10, cursor="hand2", command=on_quit).pack(side="right", padx=10)

        self._tick_clock()

    def _tick_clock(self):
        self.clock_label.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._tick_clock)

    def set_active(self, key: str):
        for k, b in self.buttons.items():
            b.config(bg=ACCENT if k == key else "#222222")


# ============================================================
# Page: Recognize
# ============================================================
class RecognizePage(tk.Frame):
    STATE_COLORS = {"idle": BG, "scanning": "#0f3460", "success": "#16213e", "unknown": "#3b0a0a"}
    BAR_COLORS = {"idle": ACCENT, "scanning": WARN, "success": GOOD, "unknown": BAD}

    def __init__(self, parent, app: "App"):
        super().__init__(parent, bg=BG)
        self.app = app
        self._state = "idle"
        self._build()
        self._update_camera()
        self._poll_results()

    def _build(self):
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=20)

        cam_frame = tk.Frame(main, bg=PANEL_BG, highlightbackground="#333", highlightthickness=2)
        cam_frame.pack(side="left", fill="both", expand=True)
        self.camera_label = tk.Label(cam_frame, bg=PANEL_BG)
        self.camera_label.pack(fill="both", expand=True)

        side = tk.Frame(main, bg=BG, width=380)
        side.pack(side="right", fill="y", padx=(20, 0))
        side.pack_propagate(False)

        self.status_bar = tk.Frame(side, height=8, bg=ACCENT)
        self.status_bar.pack(fill="x", pady=(0, 20))

        tk.Label(side, text="STATUS", font=(FONT_MONO, 11), fg=FG_LABEL, bg=BG).pack(anchor="w")
        self.status_label = tk.Label(side, text="WAITING", font=(FONT_MONO, 34, "bold"),
                                      fg=ACCENT, bg=BG, wraplength=360, justify="left")
        self.status_label.pack(anchor="w", pady=(4, 20))

        tk.Label(side, text="NAME", font=(FONT_MONO, 11), fg=FG_LABEL, bg=BG).pack(anchor="w")
        self.name_label = tk.Label(side, text="—", font=(FONT_MONO, 22, "bold"), fg=FG_TEXT, bg=BG)
        self.name_label.pack(anchor="w", pady=(4, 16))

        tk.Label(side, text="EMPLOYEE ID", font=(FONT_MONO, 11), fg=FG_LABEL, bg=BG).pack(anchor="w")
        self.id_label = tk.Label(side, text="—", font=(FONT_MONO, 22, "bold"), fg=FG_TEXT, bg=BG)
        self.id_label.pack(anchor="w", pady=(4, 16))

        tk.Label(side, text="LAST SCAN", font=(FONT_MONO, 11), fg=FG_LABEL, bg=BG).pack(anchor="w")
        self.elapsed_label = tk.Label(side, text="—", font=(FONT_MONO, 13), fg=FG_DIM, bg=BG)
        self.elapsed_label.pack(anchor="w", pady=(4, 16))

        tk.Label(side, text="QUEUE DEPTH", font=(FONT_MONO, 11), fg=FG_LABEL, bg=BG).pack(anchor="w")
        self.queue_label = tk.Label(side, text="0", font=(FONT_MONO, 20), fg=WARN, bg=BG)
        self.queue_label.pack(anchor="w", pady=(4, 16))

        tk.Label(side, text="DATABASE", font=(FONT_MONO, 11), fg=FG_LABEL, bg=BG).pack(anchor="w")
        self.db_label = tk.Label(side, text="—", font=(FONT_MONO, 13), fg=FG_DIM, bg=BG)
        self.db_label.pack(anchor="w", pady=(4, 16))

        serial_state = "LIVE" if self.app.serial_link.connected else "TEST MODE"
        serial_color = GOOD if self.app.serial_link.connected else WARN
        tk.Label(side, text="SERIAL", font=(FONT_MONO, 11), fg=FG_LABEL, bg=BG).pack(anchor="w")
        tk.Label(side, text=serial_state, font=(FONT_MONO, 13, "bold"), fg=serial_color, bg=BG).pack(anchor="w", pady=(4, 20))

        tk.Button(side, text="TEST SCAN", font=(FONT_MONO, 13, "bold"),
                  fg="#0d0d0d", bg=ACCENT, bd=0, pady=12, cursor="hand2",
                  command=self._manual_trigger).pack(fill="x", pady=(10, 0))

    def _manual_trigger(self):
        self.app.trigger_queue.put("TRIGGER")
        log_ui.info("Manual TEST SCAN triggered")

    def on_show(self):
        self.db_label.config(text=f"{len(self.app.store)} people / {self.app.store.sample_count()} samples")

    def _update_camera(self):
        frame = self.app.camera.get_frame()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            w = self.camera_label.winfo_width() or 640
            h = self.camera_label.winfo_height() or 480
            rgb = cv2.resize(rgb, (w, h))
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.camera_label.configure(image=img)
            self.camera_label.image = img

        self.queue_label.config(text=str(self.app.trigger_queue.qsize()))

        if self._state == "scanning":
            t = int(time.time() * 4) % 2
            self.status_bar.config(bg=WARN if t == 0 else BG)

        self.after(33, self._update_camera)

    def _poll_results(self):
        try:
            while True:
                result = self.app.result_queue.get_nowait()
                self._show_result(result)
        except queue.Empty:
            pass

        if not self.app.trigger_queue.empty() or self._state == "scanning":
            self._set_state("scanning")

        self.after(100, self._poll_results)

    def _show_result(self, result: dict):
        pid, name, elapsed = result["id"], result["name"], result["elapsed"]
        if pid == UNKNOWN_ID:
            self._set_state("unknown")
            self.status_label.config(text="UNKNOWN\nFACE", fg=BAD)
            self.id_label.config(text=pid, fg=BAD)
            self.name_label.config(text="—", fg=BAD)
        else:
            self._set_state("success")
            self.status_label.config(text="VERIFIED", fg=GOOD)
            self.id_label.config(text=pid, fg=GOOD)
            self.name_label.config(text=name or "—", fg=GOOD)

        self.elapsed_label.config(text=f"{elapsed:.2f}s")

        self.after(RESULT_HOLD_MS, lambda: (
            self._set_state("idle"),
            self.status_label.config(text="WAITING", fg=ACCENT),
            self.id_label.config(text="—", fg=FG_TEXT),
            self.name_label.config(text="—", fg=FG_TEXT),
        ))

    def _set_state(self, state: str):
        self._state = state
        self.status_bar.config(bg=self.BAR_COLORS.get(state, "#555"))


# ============================================================
# Page: Register
# ============================================================
class RegisterPage(tk.Frame):
    def __init__(self, parent, app: "App"):
        super().__init__(parent, bg=BG)
        self.app = app
        self._wizard_active = False
        self._build()

    def _build(self):
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=20)

        cam_frame = tk.Frame(main, bg=PANEL_BG, highlightbackground="#333", highlightthickness=2)
        cam_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(cam_frame, bg=PANEL_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        side = tk.Frame(main, bg=BG, width=380)
        side.pack(side="right", fill="y", padx=(20, 0))
        side.pack_propagate(False)

        tk.Label(side, text="ENROLL NEW PERSON", font=(FONT_MONO, 16, "bold"),
                 fg=FG_TEXT, bg=BG).pack(anchor="w", pady=(0, 16))

        tk.Label(side, text="9-DIGIT ID", font=(FONT_MONO, 11), fg=FG_LABEL, bg=BG).pack(anchor="w")
        self.id_var = tk.StringVar()
        tk.Entry(side, textvariable=self.id_var, font=(FONT_MONO, 16), bg="#222", fg=FG_TEXT,
                  insertbackground=FG_TEXT, bd=0).pack(fill="x", pady=(4, 12), ipady=6)

        tk.Label(side, text="DISPLAY NAME", font=(FONT_MONO, 11), fg=FG_LABEL, bg=BG).pack(anchor="w")
        self.name_var = tk.StringVar()
        tk.Entry(side, textvariable=self.name_var, font=(FONT_MONO, 16), bg="#222", fg=FG_TEXT,
                  insertbackground=FG_TEXT, bd=0).pack(fill="x", pady=(4, 20), ipady=6)

        self.start_btn = tk.Button(side, text="START CAPTURE", font=(FONT_MONO, 13, "bold"),
                                    fg="#0d0d0d", bg=GOOD, bd=0, pady=12, cursor="hand2",
                                    command=self._start_capture)
        self.start_btn.pack(fill="x")

        self.cancel_btn = tk.Button(side, text="CANCEL", font=(FONT_MONO, 12, "bold"),
                                     fg=FG_TEXT, bg="#333", bd=0, pady=10, cursor="hand2",
                                     command=self._cancel_capture, state="disabled")
        self.cancel_btn.pack(fill="x", pady=(8, 20))

        tk.Label(side, text="STEP", font=(FONT_MONO, 11), fg=FG_LABEL, bg=BG).pack(anchor="w")
        self.step_label = tk.Label(side, text="—", font=(FONT_MONO, 15, "bold"), fg=ACCENT, bg=BG,
                                    wraplength=360, justify="left")
        self.step_label.pack(anchor="w", pady=(4, 12))

        self.progress = ttk.Progressbar(side, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(0, 20))

        self.hint_label = tk.Label(side, text="Enter an ID and name, then press Start.",
                                    font=(FONT_MONO, 11), fg=FG_DIM, bg=BG, wraplength=360, justify="left")
        self.hint_label.pack(anchor="w")

        self._render_idle_camera()

    def on_show(self):
        pass

    def _render_idle_camera(self):
        if not self._wizard_active:
            frame = self.app.camera.get_frame()
            if frame is not None:
                self._draw_frame(frame)
            self.after(33, self._render_idle_camera)

    def _draw_frame(self, frame_bgr, boxes=None, overlay_text=None, countdown=None):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        w = self.canvas.winfo_width() or 640
        h = self.canvas.winfo_height() or 480
        rgb = cv2.resize(rgb, (w, h))
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self.canvas.image = img

        if overlay_text:
            self.canvas.create_rectangle(0, 0, w, 44, fill="#141414", outline="")
            self.canvas.create_text(12, 22, anchor="w", text=overlay_text, fill=ACCENT,
                                     font=(FONT_MONO, 13, "bold"))
        if countdown is not None:
            self.canvas.create_rectangle(0, 0, w, h, fill="black", stipple="gray50")
            self.canvas.create_text(w // 2, h // 2, text=str(countdown), fill="white",
                                     font=(FONT_MONO, 64, "bold"))

    def _start_capture(self):
        uid = self.id_var.get().strip()
        name = self.name_var.get().strip()

        if not (uid.isdigit() and len(uid) == 9):
            messagebox.showerror("Invalid ID", "ID must be exactly 9 digits.")
            return
        if not name:
            messagebox.showerror("Missing name", "Please enter a display name.")
            return

        existing = self.app.store.get(uid)
        if existing and not messagebox.askyesno(
                "Already registered",
                f"ID {uid} is already registered as '{existing.name}'.\n"
                f"Add more samples for this person?"):
            return

        self._wizard_active = True
        self.start_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self._captured: list = []
        self._abort = False
        threading.Thread(target=self._run_wizard, args=(uid, name), daemon=True).start()

    def _cancel_capture(self):
        self._abort = True

    def _run_wizard(self, uid: str, name: str):
        try:
            for step_idx, (step_name, instruction, target) in enumerate(GUIDED_STEPS):
                if not self._countdown(f"Get ready: {step_name}", 3):
                    return
                collected = 0
                while collected < target:
                    if self._abort:
                        self._finish(aborted=True)
                        return
                    frame = self.app.camera.get_frame()
                    if frame is None:
                        time.sleep(0.02)
                        continue

                    faces = encode_faces(frame)
                    boxes = [b for b, _ in faces]
                    header = f"STEP {step_idx+1}/{len(GUIDED_STEPS)} — {step_name}: {instruction}"
                    self.after(0, self._draw_progress_frame, frame, boxes, header, collected, target)

                    if faces:
                        # Take the largest detected face if several are in frame.
                        (top, right, bottom, left), enc = max(
                            faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][1] - f[0][3]))
                        self._captured.append(enc)
                        collected += 1
                    time.sleep(0.05)

            self.after(0, self._finish, False)
        except Exception as e:
            log.error(f"Registration wizard error: {e}")
            self.after(0, self._finish, True)

    def _draw_progress_frame(self, frame, boxes, header, collected, target):
        for (top, right, bottom, left) in boxes:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 230, 80), 2)
        self._draw_frame(frame, boxes, overlay_text=header)
        self.step_label.config(text=header)
        total_done = len(self._captured)
        total_needed = sum(s[2] for s in GUIDED_STEPS)
        pct = int(100 * total_done / total_needed)
        self.progress["value"] = pct
        self.hint_label.config(text=f"{collected}/{target} samples this step  •  {total_done}/{total_needed} total")

    def _countdown(self, message: str, seconds: int) -> bool:
        done = threading.Event()
        result = {"ok": True}

        def tick(remaining):
            if self._abort:
                result["ok"] = False
                done.set()
                return
            frame = self.app.camera.get_frame()
            if frame is not None:
                self._draw_frame(frame, overlay_text=message, countdown=remaining if remaining > 0 else None)
            if remaining <= 0:
                done.set()
                return
            self.after(1000, tick, remaining - 1)

        self.after(0, tick, seconds)
        done.wait()
        return result["ok"]

    def _finish(self, aborted: bool):
        self._wizard_active = False
        self.start_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")

        if aborted or not self._captured:
            self.hint_label.config(text="Registration cancelled — no data was saved.")
            self.step_label.config(text="—")
            self.progress["value"] = 0
        else:
            uid = self.id_var.get().strip()
            name = self.name_var.get().strip()
            self.app.store.add_person(uid, name, self._captured)
            log.info(f"Registered '{name}' ({uid}) with {len(self._captured)} samples")
            self.step_label.config(text="DONE")
            self.hint_label.config(
                text=f"Saved {len(self._captured)} samples for '{name}' ({uid}).\n"
                     f"Database now has {len(self.app.store)} people.")
            self.id_var.set("")
            self.name_var.set("")

        self._render_idle_camera()


# ============================================================
# Page: Manage
# ============================================================
class ManagePage(tk.Frame):
    def __init__(self, parent, app: "App"):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build()

    def _build(self):
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=20)

        top = tk.Frame(main, bg=BG)
        top.pack(fill="x", pady=(0, 12))
        tk.Label(top, text="ENROLLED PEOPLE", font=(FONT_MONO, 16, "bold"), fg=FG_TEXT, bg=BG).pack(side="left")
        tk.Button(top, text="REFRESH", font=(FONT_MONO, 11, "bold"), fg=FG_TEXT, bg="#333",
                  bd=0, padx=14, pady=8, cursor="hand2", command=self.refresh).pack(side="right")
        tk.Button(top, text="DELETE SELECTED", font=(FONT_MONO, 11, "bold"), fg="#0d0d0d", bg=BAD,
                  bd=0, padx=14, pady=8, cursor="hand2", command=self._delete_selected).pack(side="right", padx=(0, 10))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=PANEL_BG, fieldbackground=PANEL_BG,
                        foreground=FG_TEXT, rowheight=32, font=(FONT_MONO, 11))
        style.configure("Treeview.Heading", background="#222", foreground=FG_TEXT, font=(FONT_MONO, 11, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)])

        cols = ("id", "name", "samples", "created")
        self.tree = ttk.Treeview(main, columns=cols, show="headings", selectmode="extended")
        for c, label, w in (("id", "ID", 140), ("name", "Name", 220), ("samples", "Samples", 100), ("created", "Registered", 200)):
            self.tree.heading(c, text=label)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True)

    def on_show(self):
        self.refresh()

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        for p in self.app.store.list_people():
            created = datetime.fromtimestamp(p.created_at).strftime("%Y-%m-%d %H:%M")
            self.tree.insert("", "end", iid=p.id, values=(p.id, p.name, p.sample_count, created))

    def _delete_selected(self):
        selected = self.tree.selection()
        if not selected:
            return
        if not messagebox.askyesno("Confirm delete", f"Delete {len(selected)} selected people? This cannot be undone."):
            return
        for pid in selected:
            self.app.store.remove_person(pid)
        self.refresh()


# ============================================================
# App shell
# ============================================================
class App:
    def __init__(self):
        self.store = FaceStore()
        log.info(f"Loaded {len(self.store)} people / {self.store.sample_count()} samples")

        self.camera = CameraStream(CAMERA_INDEX)
        log.info("Camera opened successfully")

        self.trigger_queue: "queue.Queue[str]" = queue.Queue()
        self.result_queue: "queue.Queue[dict]" = queue.Queue()

        self.serial_link = SerialLink(SERIAL_PORT, SERIAL_BAUD, self.trigger_queue)
        self.serial_link.start()

        self.worker = FaceWorker(self.camera, self.store, self.trigger_queue,
                                  self.result_queue, self.serial_link)
        self.worker.start()

        self.root = tk.Tk()
        self.root.title("Face ID System")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg=BG)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.bind("<F11>", lambda e: self.root.attributes("-fullscreen", True))
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.nav = NavBar(self.root, self.show_page, self.quit)
        self.nav.pack(fill="x")

        self.container = tk.Frame(self.root, bg=BG)
        self.container.pack(fill="both", expand=True)

        self.pages = {
            "recognize": RecognizePage(self.container, self),
            "register": RegisterPage(self.container, self),
            "manage": ManagePage(self.container, self),
        }
        for page in self.pages.values():
            page.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_page("recognize")

    def show_page(self, key: str):
        page = self.pages[key]
        page.tkraise()
        self.nav.set_active(key)
        if hasattr(page, "on_show"):
            page.on_show()

    def quit(self):
        log.info("Shutting down")
        self.camera.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    log.info("=" * 60)
    log.info("Face ID System starting")
    log.info(f"Config: PORT={SERIAL_PORT}, BAUD={SERIAL_BAUD}, CAMERA={CAMERA_INDEX}")
    app = App()
    app.run()


if __name__ == "__main__":
    main()
