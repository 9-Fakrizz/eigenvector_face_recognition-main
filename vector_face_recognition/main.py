"""
main.py — Jetson Nano Face Recognition Node
============================================
Improvements:
  1. ESP32 serial comms via dedicated reader/writer threads
  2. Face recognition runs in a worker thread (ThreadPoolExecutor)
     → multiple triggers queue up; recognition never blocks serial I/O
  3. Fullscreen Tkinter UI with live camera feed + status overlay

Architecture:
  SerialReaderThread  ──► trigger_queue ──► FaceWorkerThread ──► result_queue
                                                                       │
  SerialWriterThread  ◄──────────────────────────────────────────────┘
  UIThread (main)     ◄──────────────────────────────────────────────┘
"""

import os
import cv2
import time
import queue
import serial
import pickle
import threading
import numpy as np
import face_recognition
import tkinter as tk
from PIL import Image, ImageTk
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ============================================================
# ⚙️  CONFIG
# ============================================================
SERIAL_PORT   = "/dev/ttyUSB0"   # Change to /dev/ttyUSB0 if using USB-Serial
SERIAL_BAUD   = 115200
CAMERA_INDEX  = 0
ENCODINGS_FILE = "encodings.pkl"   # pre-built with encode_faces.py
FACE_TOLERANCE = 0.50              # lower = stricter match
SCAN_FRAMES    = 10                # frames captured per scan attempt
UNKNOWN_ID     = "000000000"       # sent back when no face matched
MAX_WORKERS    = 2                 # parallel face-recognition jobs

# ============================================================
# 🗄️  Load known-face encodings once at startup
# ============================================================
def load_encodings(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Loaded {len(data['names'])} known faces from {path}")
        return data["encodings"], data["names"]
    print(f"⚠️  {path} not found — all scans will return UNKNOWN")
    return [], []

known_encodings, known_names = load_encodings(ENCODINGS_FILE)

# ============================================================
# 📬  Inter-thread queues
# ============================================================
trigger_queue: "queue.Queue[str]"  = queue.Queue()   # "TRIGGER" signals
result_queue:  "queue.Queue[dict]" = queue.Queue()   # scan results → UI + serial

# ============================================================
# 📡  Thread 1 — Serial Reader  (ESP32 → Jetson)
# ============================================================
def serial_reader(ser: serial.Serial):
    """
    Listens for '1\\n' from ESP32.
    Each '1' pushes one job into trigger_queue.
    Non-blocking: never stalls other threads.
    """
    print("🔌 SerialReader thread started")
    buf = ""
    while True:
        try:
            if ser.in_waiting:
                chunk = ser.read(ser.in_waiting).decode("utf-8", errors="ignore")
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line == "1":
                        print(f"[Serial RX] Trigger received at {datetime.now().strftime('%H:%M:%S')}")
                        trigger_queue.put("TRIGGER")
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"[SerialReader] Error: {e}")
            time.sleep(0.5)

# ============================================================
# 📡  Thread 2 — Serial Writer  (Jetson → ESP32)
# ============================================================
write_queue: "queue.Queue[str]" = queue.Queue()

def serial_writer(ser: serial.Serial):
    """
    Sends employee IDs back to ESP32.
    Dedicated thread so writes never block the face worker.
    """
    print("🔌 SerialWriter thread started")
    while True:
        try:
            msg = write_queue.get()          # blocks until there's something to send
            ser.write((msg + "\n").encode())
            print(f"[Serial TX] Sent: {msg}")
        except Exception as e:
            print(f"[SerialWriter] Error: {e}")

# ============================================================
# 📸  Face Recognition helper  (runs inside thread-pool)
# ============================================================
def do_face_recognition(cap: cv2.VideoCapture) -> str:
    """
    Grabs SCAN_FRAMES frames, runs face_recognition on each,
    returns the most-frequent matching name (or UNKNOWN_ID).
    Designed to be called from a thread-pool worker.
    """
    votes: dict[str, int] = {}

    for _ in range(SCAN_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use "hog" model for speed on CPU; change to "cnn" for GPU accuracy
        locations  = face_recognition.face_locations(rgb, model="hog")
        encodings  = face_recognition.face_encodings(rgb, locations)

        for enc in encodings:
            if not known_encodings:
                votes[UNKNOWN_ID] = votes.get(UNKNOWN_ID, 0) + 1
                continue
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx  = int(np.argmin(distances))
            if distances[best_idx] <= FACE_TOLERANCE:
                name = known_names[best_idx]
                votes[name] = votes.get(name, 0) + 1
            else:
                votes[UNKNOWN_ID] = votes.get(UNKNOWN_ID, 0) + 1

    if not votes:
        return UNKNOWN_ID
    return max(votes, key=votes.get)

# ============================================================
# 🧠  Thread 3 — Face Worker  (consumes trigger_queue)
# ============================================================
def face_worker(cap: cv2.VideoCapture):
    """
    Uses a ThreadPoolExecutor so up to MAX_WORKERS scans
    can run concurrently — fixes the 'more people = slower' problem.
    """
    print("🧠 FaceWorker thread started")
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def handle_future(future, triggered_at):
        emp_id = future.result()
        elapsed = time.time() - triggered_at
        print(f"[FaceWorker] Result={emp_id}  ({elapsed:.2f}s)")
        result_queue.put({"employee_id": emp_id, "elapsed": elapsed})
        write_queue.put(emp_id)         # send to ESP32

    while True:
        trigger_queue.get()             # wait for next trigger
        triggered_at = time.time()
        print("[FaceWorker] Submitting scan job to executor")
        future = executor.submit(do_face_recognition, cap)
        future.add_done_callback(lambda f, t=triggered_at: handle_future(f, t))

# ============================================================
# 🖥️  Thread 4 (Main) — Fullscreen Tkinter UI
# ============================================================
class FullscreenUI:
    """
    Industrial / utilitarian aesthetic — dark background,
    large mono status text, live camera feed.
    """

    STATUS_COLORS = {
        "idle":      "#1a1a2e",   # dark navy
        "scanning":  "#0f3460",   # deep blue
        "success":   "#16213e",   # dark
        "unknown":   "#3b0a0a",   # dark red
    }

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.root = tk.Tk()
        self.root.title("Employee Check-in System")
        self.root.attributes("-fullscreen", True)        # ← fullscreen
        self.root.configure(bg="#0d0d0d")
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.bind("<F11>",    lambda e: self.root.attributes("-fullscreen", True))

        self._build_layout()
        self._state = "idle"
        self._last_result = None
        self._update_camera()
        self._poll_results()

    # ----------------------------------------------------------
    def _build_layout(self):
        # ── Top bar ────────────────────────────────────────────
        top = tk.Frame(self.root, bg="#0d0d0d", height=60)
        top.pack(fill="x", padx=20, pady=(20, 0))

        tk.Label(top, text=" BUS EMPLOYEE CHECK-IN",
                 font=("Courier New", 18, "bold"),
                 fg="#e0e0e0", bg="#0d0d0d").pack(side="left")

        self.clock_label = tk.Label(top, text="",
                 font=("Courier New", 16),
                 fg="#888888", bg="#0d0d0d")
        self.clock_label.pack(side="right")
        self._tick_clock()

        # ── Main area: camera left, status right ───────────────
        main = tk.Frame(self.root, bg="#0d0d0d")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # Camera feed
        cam_frame = tk.Frame(main, bg="#1a1a1a", bd=2, relief="solid",
                             highlightbackground="#333", highlightthickness=2)
        cam_frame.pack(side="left", fill="both", expand=True)
        self.camera_label = tk.Label(cam_frame, bg="#1a1a1a")
        self.camera_label.pack(fill="both", expand=True)

        # Status panel
        side = tk.Frame(main, bg="#0d0d0d", width=400)
        side.pack(side="right", fill="y", padx=(20, 0))
        side.pack_propagate(False)

        # Status indicator bar
        self.status_bar = tk.Frame(side, height=8, bg="#3a86ff")
        self.status_bar.pack(fill="x", pady=(0, 20))

        tk.Label(side, text="STATUS", font=("Courier New", 11),
                 fg="#555", bg="#0d0d0d").pack(anchor="w")

        self.status_label = tk.Label(side, text="WAITING",
                 font=("Courier New", 38, "bold"),
                 fg="#3a86ff", bg="#0d0d0d", wraplength=380, justify="left")
        self.status_label.pack(anchor="w", pady=(4, 20))

        tk.Label(side, text="EMPLOYEE ID", font=("Courier New", 11),
                 fg="#555", bg="#0d0d0d").pack(anchor="w")

        self.id_label = tk.Label(side, text="—",
                 font=("Courier New", 28, "bold"),
                 fg="#e0e0e0", bg="#0d0d0d")
        self.id_label.pack(anchor="w", pady=(4, 20))

        tk.Label(side, text="LAST SCAN", font=("Courier New", 11),
                 fg="#555", bg="#0d0d0d").pack(anchor="w")

        self.elapsed_label = tk.Label(side, text="—",
                 font=("Courier New", 14),
                 fg="#888888", bg="#0d0d0d")
        self.elapsed_label.pack(anchor="w", pady=(4, 20))

        # Queue depth indicator
        tk.Label(side, text="QUEUE DEPTH", font=("Courier New", 11),
                 fg="#555", bg="#0d0d0d").pack(anchor="w")

        self.queue_label = tk.Label(side, text="0",
                 font=("Courier New", 22),
                 fg="#ff9f1c", bg="#0d0d0d")
        self.queue_label.pack(anchor="w", pady=(4, 0))

    # ----------------------------------------------------------
    def _tick_clock(self):
        self.clock_label.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    # ----------------------------------------------------------
    def _update_camera(self):
        """Reads one frame and updates the Label. Scheduled at ~30 fps."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to fill panel
            h = self.camera_label.winfo_height() or 480
            w = self.camera_label.winfo_width()  or 640
            frame = cv2.resize(frame, (w, h))
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.camera_label.configure(image=img)
            self.camera_label.image = img  # keep reference

        # Update queue depth indicator
        self.queue_label.config(text=str(trigger_queue.qsize()))

        if self._state == "scanning":
            # Pulse the status bar color while scanning
            t = int(time.time() * 4) % 2
            self.status_bar.config(bg="#3a86ff" if t == 0 else "#1a1a2e")

        self.root.after(33, self._update_camera)   # ~30 fps

    # ----------------------------------------------------------
    def _poll_results(self):
        """Polls result_queue every 100 ms (non-blocking)."""
        try:
            while True:
                result = result_queue.get_nowait()
                self._show_result(result)
        except queue.Empty:
            pass

        # Also reflect scanning state when trigger is pending
        if not trigger_queue.empty() or self._state == "scanning":
            self._set_state("scanning")

        self.root.after(100, self._poll_results)

    # ----------------------------------------------------------
    def _show_result(self, result: dict):
        emp_id  = result["employee_id"]
        elapsed = result["elapsed"]

        if emp_id == UNKNOWN_ID:
            self._set_state("unknown")
            self.status_label.config(text="UNKNOWN\nFACE", fg="#ff4444")
            self.id_label.config(text=emp_id, fg="#ff4444")
        else:
            self._set_state("success")
            self.status_label.config(text="VERIFIED ✓", fg="#4dff91")
            self.id_label.config(text=emp_id, fg="#4dff91")

        self.elapsed_label.config(text=f"Scan time: {elapsed:.2f}s")

        # Auto-return to idle after 3 s
        self.root.after(3000, lambda: (
            self._set_state("idle"),
            self.status_label.config(text="WAITING", fg="#3a86ff"),
            self.id_label.config(text="—", fg="#e0e0e0"),
        ))

    # ----------------------------------------------------------
    def _set_state(self, state: str):
        self._state = state
        bg = self.STATUS_COLORS.get(state, "#0d0d0d")
        bar_colors = {
            "idle":      "#3a86ff",
            "scanning":  "#ff9f1c",
            "success":   "#4dff91",
            "unknown":   "#ff4444",
        }
        self.status_bar.config(bg=bar_colors.get(state, "#555"))
        self.root.configure(bg=bg)

    # ----------------------------------------------------------
    def run(self):
        self.root.mainloop()


# ============================================================
# 🚀  Entry point
# ============================================================
def main():
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open camera")

    # Open serial port
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0)
        print(f"✅ Serial open: {SERIAL_PORT} @ {SERIAL_BAUD}")
    except serial.SerialException as e:
        print(f"⚠️  Serial error: {e}")
        ser = None

    # Start background threads (daemon=True → die when main thread exits)
    if ser:
        threading.Thread(target=serial_reader, args=(ser,), daemon=True).start()
        threading.Thread(target=serial_writer, args=(ser,), daemon=True).start()
    threading.Thread(target=face_worker,   args=(cap,), daemon=True).start()

    # Run fullscreen UI on main thread (Tkinter requirement)
    ui = FullscreenUI(cap)
    ui.run()

    cap.release()


if __name__ == "__main__":
    main()
