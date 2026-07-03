"""
main4.py — Jetson Nano Face Recognition Node with Test Button & Serial Debug
==========================================================================
Features:
  1. ESP32 serial comms via dedicated reader/writer threads
  2. Face recognition runs in a worker thread (ThreadPoolExecutor)
     → multiple triggers queue up; recognition never blocks serial I/O
  3. Windowed Tkinter UI with fixed camera size and Test Button
  4. COMPREHENSIVE DEBUG LOGGING for serial & face recognition testing
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
import sys

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
DEBUG_MODE     = True              # Enable extensive logging

# ============================================================
# �  Debug Logger
# ============================================================
def debug_log(component: str, message: str, level="INFO"):
    """Print debug messages with timestamp and component"""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level_str = f"[{level:7s}]"
        print(f"{timestamp} {level_str} [{component:12s}] {message}", file=sys.stdout, flush=True)

# ============================================================
# �🗄️  Load known-face encodings once at startup
# ============================================================
def load_encodings(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        num_faces = len(data['names'])
        debug_log("ENCODER", f"✅ Loaded {num_faces} known faces from {path}")
        debug_log("ENCODER", f"   Names: {data['names']}")
        return data["encodings"], data["names"]
    debug_log("ENCODER", f"⚠️  {path} not found — all scans will return UNKNOWN", "WARN")
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
    debug_log("SERIAL_RX", "🔌 SerialReader thread started")
    debug_log("SERIAL_RX", f"   Port: {ser.port} | Baud: {ser.baudrate} | Timeout: {ser.timeout}s")
    buf = ""
    rx_count = 0
    
    while True:
        try:
            if ser.in_waiting:
                chunk = ser.read(ser.in_waiting).decode("utf-8", errors="ignore")
                buf += chunk
                debug_log("SERIAL_RX", f"Received {len(chunk)} bytes: {repr(chunk)}", "DEBUG")
                
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        debug_log("SERIAL_RX", f"Line: '{line}'", "DEBUG")
                        rx_count += 1
                    
                    if line == "1":
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        debug_log("SERIAL_RX", f"✅ Trigger received (msg: '1') at {timestamp}", "INFO")
                        trigger_queue.put("TRIGGER")
            else:
                time.sleep(0.01)
        except Exception as e:
            debug_log("SERIAL_RX", f"❌ Error: {e}", "ERROR")
            time.sleep(0.5)

# ============================================================
# 📡  Thread 2 — Serial Writer  (Jetson → ESP32)
# ============================================================
write_queue: "queue.Queue[str]" = queue.Queue()

def serial_writer(ser: serial.Serial):
    debug_log("SERIAL_TX", "🔌 SerialWriter thread started")
    tx_count = 0
    
    while True:
        try:
            msg = write_queue.get()          # blocks until there's something to send
            tx_count += 1
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            
            # Debug output before sending
            debug_log("SERIAL_TX", f"Preparing to send: '{msg}' (msg #{tx_count})", "DEBUG")
            debug_log("SERIAL_TX", f"  - Bytes to send: {msg.encode() + b'\\n'}", "DEBUG")
            
            # Send data
            bytes_written = ser.write((msg + "\n").encode())
            ser.flush()  # Ensure data is physically sent
            
            # Debug output after sending
            debug_log("SERIAL_TX", f"✅ Sent: '{msg}' ({bytes_written} bytes) at {timestamp}")
            debug_log("SERIAL_TX", f"  - Queue remaining: {write_queue.qsize()}", "DEBUG")
            
        except Exception as e:
            debug_log("SERIAL_TX", f"❌ Error: {e}", "ERROR")
            debug_log("SERIAL_TX", f"   Exception type: {type(e).__name__}", "ERROR")

# ============================================================
# 📸  Face Recognition helper  (runs inside thread-pool)
# ============================================================
def do_face_recognition(cap: cv2.VideoCapture) -> str:
    debug_log("FACE_REC", f"Starting face recognition scan ({SCAN_FRAMES} frames)")
    votes: dict[str, int] = {}
    frame_num = 0

    for frame_idx in range(SCAN_FRAMES):
        ret, frame = cap.read()
        if not ret:
            debug_log("FACE_REC", f"  Frame {frame_idx}: Failed to read", "WARN")
            continue
        
        frame_num += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations  = face_recognition.face_locations(rgb, model="hog")
        encodings  = face_recognition.face_encodings(rgb, locations)
        
        debug_log("FACE_REC", f"  Frame {frame_num}: {len(locations)} face(s) detected", "DEBUG")

        for face_idx, enc in enumerate(encodings):
            if not known_encodings:
                votes[UNKNOWN_ID] = votes.get(UNKNOWN_ID, 0) + 1
                debug_log("FACE_REC", f"    Face {face_idx}: No encodings DB → UNKNOWN", "DEBUG")
                continue
            
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx  = int(np.argmin(distances))
            best_dist = distances[best_idx]
            
            if best_dist <= FACE_TOLERANCE:
                name = known_names[best_idx]
                votes[name] = votes.get(name, 0) + 1
                debug_log("FACE_REC", f"    Face {face_idx}: MATCH '{name}' (distance: {best_dist:.3f})", "DEBUG")
            else:
                votes[UNKNOWN_ID] = votes.get(UNKNOWN_ID, 0) + 1
                debug_log("FACE_REC", f"    Face {face_idx}: UNKNOWN (best distance: {best_dist:.3f})", "DEBUG")

    if not votes:
        debug_log("FACE_REC", f"No faces detected in any frame → UNKNOWN")
        return UNKNOWN_ID
    
    final_result = max(votes, key=votes.get)
    final_score = votes[final_result]
    debug_log("FACE_REC", f"Final result: '{final_result}' (votes: {final_score}, votes_dict: {votes})")
    return final_result

# ============================================================
# 🧠  Thread 3 — Face Worker  (consumes trigger_queue)
# ============================================================
def face_worker(cap: cv2.VideoCapture):
    debug_log("WORKER", "🧠 FaceWorker thread started")
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    job_count = 0

    def handle_future(future, triggered_at, job_id):
        try:
            emp_id = future.result()
            elapsed = time.time() - triggered_at
            debug_log("WORKER", f"Job #{job_id} completed: emp_id='{emp_id}' time={elapsed:.2f}s")
            result_queue.put({"employee_id": emp_id, "elapsed": elapsed})
            
            debug_log("WORKER", f"Queuing TX: '{emp_id}'", "DEBUG")
            write_queue.put(emp_id)         # send to ESP32
            
        except Exception as e:
            debug_log("WORKER", f"Job #{job_id} failed with exception: {e}", "ERROR")

    while True:
        try:
            trigger_queue.get()             # wait for next trigger
            job_count += 1
            triggered_at = time.time()
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            
            debug_log("WORKER", f"Trigger received (#{job_count}) at {timestamp}")
            debug_log("WORKER", f"  Submitting scan job to executor", "DEBUG")
            
            future = executor.submit(do_face_recognition, cap)
            future.add_done_callback(lambda f, t=triggered_at, jid=job_count: handle_future(f, t, jid))
            
        except Exception as e:
            debug_log("WORKER", f"Error processing trigger: {e}", "ERROR")

# ============================================================
# 🖥️  Thread 4 (Main) — Tkinter UI
# ============================================================
class FullscreenUI:
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
        
        # ปิด Fullscreen และกำหนดขนาดหน้าต่างตายตัวเพื่อป้องกันหน้าจอขยายเอง
        self.root.geometry("1024x600")
        self.root.configure(bg="#0d0d0d")
        
        debug_log("UI", "Initializing UI components...")

        self._build_layout()
        self._state = "idle"
        self._last_result = None
        self._update_camera()
        self._poll_results()
        
        debug_log("UI", "✅ UI initialization complete")

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

        # ปุ่มกดเพื่อจำลองการส่งสัญญาณจาก ESP32
        tk.Button(side, text="TEST SCAN (จำลอง ESP32)", 
                  font=("Courier New", 14, "bold"), 
                  bg="#ff9f1c", fg="#0d0d0d",
                  command=lambda: trigger_queue.put("TRIGGER")).pack(pady=(30, 0), fill="x")

    def _tick_clock(self):
        self.clock_label.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def _update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # ล็อกขนาดภาพให้คงที่ที่ 640x480 เพื่อแก้บั๊กหน้าจอขยายตัวเอง
            frame = cv2.resize(frame, (640, 480))
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.camera_label.configure(image=img)
            self.camera_label.image = img  # keep reference

        self.queue_label.config(text=str(trigger_queue.qsize()))

        if self._state == "scanning":
            t = int(time.time() * 4) % 2
            self.status_bar.config(bg="#3a86ff" if t == 0 else "#1a1a2e")

        self.root.after(33, self._update_camera)

    def _poll_results(self):
        try:
            while True:
                result = result_queue.get_nowait()
                debug_log("UI", f"Result received: {result}")
                self._show_result(result)
        except queue.Empty:
            pass

        if not trigger_queue.empty() or self._state == "scanning":
            self._set_state("scanning")

        self.root.after(100, self._poll_results)

    def _show_result(self, result: dict):
        emp_id  = result["employee_id"]
        elapsed = result["elapsed"]
        
        debug_log("UI", f"Displaying result: emp_id='{emp_id}', elapsed={elapsed:.2f}s", "DEBUG")

        if emp_id == UNKNOWN_ID:
            self._set_state("unknown")
            self.status_label.config(text="UNKNOWN\nFACE", fg="#ff4444")
            self.id_label.config(text=emp_id, fg="#ff4444")
            debug_log("UI", f"Status: UNKNOWN FACE")
        else:
            self._set_state("success")
            self.status_label.config(text="VERIFIED ✓", fg="#4dff91")
            self.id_label.config(text=emp_id, fg="#4dff91")
            debug_log("UI", f"Status: VERIFIED ✓")

        self.elapsed_label.config(text=f"Scan time: {elapsed:.2f}s")

        self.root.after(3000, lambda: (
            self._set_state("idle"),
            self.status_label.config(text="WAITING", fg="#3a86ff"),
            self.id_label.config(text="—", fg="#e0e0e0"),
        ))

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

    def run(self):
        self.root.mainloop()

# ============================================================
# 🚀  Entry point
# ============================================================
def main():
    debug_log("MAIN", "=" * 70)
    debug_log("MAIN", "Face Recognition System Starting")
    debug_log("MAIN", f"Debug Mode: {DEBUG_MODE}")
    debug_log("MAIN", f"Config: PORT={SERIAL_PORT}, BAUD={SERIAL_BAUD}, CAMERA={CAMERA_INDEX}")
    debug_log("MAIN", "=" * 70)
    
    # Initialize camera
    debug_log("MAIN", "Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        debug_log("MAIN", "❌ Cannot open camera", "ERROR")
        raise RuntimeError("❌ Cannot open camera")
    debug_log("MAIN", "✅ Camera opened successfully")

    # Initialize serial
    debug_log("MAIN", f"Opening serial port {SERIAL_PORT}...")
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0)
        debug_log("MAIN", f"✅ Serial open: {SERIAL_PORT} @ {SERIAL_BAUD} baud")
    except serial.SerialException as e:
        debug_log("MAIN", f"⚠️  Serial error: {e}", "WARN")
        debug_log("MAIN", "   Running WITHOUT serial communication (test mode)", "WARN")
        ser = None

    # Start worker threads
    debug_log("MAIN", "Starting worker threads...")
    
    if ser:
        threading.Thread(target=serial_reader, args=(ser,), daemon=True).start()
        debug_log("MAIN", "  ✓ Serial reader thread started")
        
        threading.Thread(target=serial_writer, args=(ser,), daemon=True).start()
        debug_log("MAIN", "  ✓ Serial writer thread started")
    else:
        debug_log("MAIN", "  ⊘ Serial threads skipped (no serial port)", "WARN")
    
    threading.Thread(target=face_worker, args=(cap,), daemon=True).start()
    debug_log("MAIN", "  ✓ Face worker thread started")
    
    debug_log("MAIN", "Starting UI...")
    ui = FullscreenUI(cap)
    debug_log("MAIN", "UI ready - entering main loop")
    
    try:
        ui.run()
    except KeyboardInterrupt:
        debug_log("MAIN", "Shutdown signal received (Ctrl+C)")
    finally:
        debug_log("MAIN", "Cleaning up...")
        cap.release()
        if ser:
            ser.close()
            debug_log("MAIN", "Serial port closed")
        debug_log("MAIN", "Shutdown complete")

if __name__ == "__main__":
    main()
