"""
main.py — Jetson Nano Auto-Scan Face Recognition
============================================
Improvements:
  1. Auto-Scan: Continuously detects faces in the background without waiting for triggers.
  2. Thread-Safe Camera: UI reads camera smoothly, FaceWorker safely gets a copy.
  3. Cooldown System: Prevents spamming the same ID to ESP32 if a person stands still.
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
from datetime import datetime

# ============================================================
# ⚙️  CONFIG (ตั้งค่าระบบ)
# ============================================================
SERIAL_PORT    = "/dev/ttyUSB0"    # พอร์ตสาย USB ที่ต่อกับ ESP32
SERIAL_BAUD    = 115200
CAMERA_INDEX   = 0
ENCODINGS_FILE = "encodings.pkl"   # ไฟล์ฐานข้อมูลใบหน้า
FACE_TOLERANCE = 0.50              # ความเข้มงวด (ค่าน้อย = ต้องเหมือนมาก)
UNKNOWN_ID     = "000000000"       # รหัสเมื่อไม่รู้จักใบหน้า
SCAN_COOLDOWN  = 4.0               # หน่วงเวลา 4 วินาที ไม่ให้ส่งรหัสคนเดิมซ้ำรัวๆ

# ============================================================
# 🗄️  ตัวแปรสำหรับแชร์ภาพระหว่าง Thread อย่างปลอดภัย
# ============================================================
latest_frame = None
frame_lock = threading.Lock()
result_queue = queue.Queue()
write_queue = queue.Queue()

# ============================================================
# 🗄️  Load known-face encodings
# ============================================================
def load_encodings(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Loaded {len(data['names'])} known faces from {path}")
        return data["encodings"], data["names"]
    print(f"⚠️  {path} not found — all scans will return UNKNOWN")
    return [], []

known_encodings, known_names = load_encodings(ENCODINGS_FILE)

# ============================================================
# 📡  Thread 1 — Serial Writer (ส่งข้อมูลไป ESP32)
# ============================================================
def serial_writer(ser):
    print("🔌 SerialWriter thread started")
    while True:
        try:
            msg = write_queue.get()
            ser.write((msg + "\n").encode())
            print(f"[Serial TX] Sent to ESP32: {msg}")
        except Exception as e:
            print(f"[SerialWriter] Error: {e}")

# ============================================================
# 🧠  Thread 2 — Auto-Scan Face Worker (ระบบสแกนอัตโนมัติ)
# ============================================================
def auto_face_worker():
    print("🧠 Auto-Scan FaceWorker thread started")
    last_seen_id = None
    last_scan_time = 0

    while True:
        time.sleep(0.05) # พักเบรกเล็กน้อยไม่ให้ CPU ร้อนเกินไป

        # 1. ดึงภาพล่าสุดมาอย่างปลอดภัย
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # 2. ย่อภาพลงครึ่งนึงเพื่อให้หาตำแหน่งใบหน้าไวขึ้น
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 3. ค้นหาว่ามีคนอยู่ในกล้องไหม? (ใช้โมเดล hog ซึ่งเร็ว)
        locations = face_recognition.face_locations(rgb_small, model="hog")
        
        if not locations:
            continue # ถ้าไม่มีคน ก็วนลูปรอต่อไปแบบเงียบๆ

        # 4. ถ้าเจอคน ค่อยเริ่มประมวลผลเปรียบเทียบใบหน้า
        start_time = time.time()
        
        # ขยายพิกัดกลับมาเป็นสัดส่วนเดิม
        locations = [(int(top*2), int(right*2), int(bottom*2), int(left*2)) for (top, right, bottom, left) in locations]
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        encodings = face_recognition.face_encodings(rgb_full, locations)

        found_id = UNKNOWN_ID
        for enc in encodings:
            if not known_encodings:
                break
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= FACE_TOLERANCE:
                found_id = known_names[best_idx]
                break # เอาหน้าแรกที่แมตช์เจอ

        # 5. ระบบกันการสแกนซ้ำ (Cooldown)
        if found_id == last_seen_id and (time.time() - last_scan_time) < SCAN_COOLDOWN:
            continue # ถ้าเพิ่งสแกนคนนี้ไป ก็ข้ามไปก่อน

        # 6. ส่งผลลัพธ์ไปที่หน้าจอ และส่งรหัสให้ ESP32
        elapsed = time.time() - start_time
        print(f"[Auto-Scan] Result = {found_id} ({elapsed:.2f}s)")
        
        result_queue.put({"employee_id": found_id, "elapsed": elapsed})
        
        # ส่งไปหา ESP32 เฉพาะรหัสที่รู้จัก
        if found_id != UNKNOWN_ID:
            write_queue.put(found_id)

        # จำไว้ว่าเพิ่งสแกนคนนี้ไป
        last_seen_id = found_id
        last_scan_time = time.time()

# ============================================================
# 🖥️  Thread 3 (Main) — Tkinter UI
# ============================================================
class AutoScanUI:
    STATUS_COLORS = {
        "idle":      "#1a1a2e",
        "scanning":  "#ff9f1c",
        "success":   "#16213e",
        "unknown":   "#3b0a0a",
    }

    def __init__(self, cap):
        self.cap = cap
        self.root = tk.Tk()
        self.root.title("Employee Auto-Scan System")
        self.root.geometry("1024x600")
        self.root.configure(bg="#0d0d0d")

        self._build_layout()
        self._state = "idle"
        self._update_camera()
        self._poll_results()

    def _build_layout(self):
        # ── Top bar ────────────────────────────────────────────
        top = tk.Frame(self.root, bg="#0d0d0d", height=60)
        top.pack(fill="x", padx=20, pady=(20, 0))

        tk.Label(top, text=" BUS EMPLOYEE CHECK-IN (AUTO-SCAN)",
                 font=("Courier New", 18, "bold"), fg="#e0e0e0", bg="#0d0d0d").pack(side="left")

        self.clock_label = tk.Label(top, font=("Courier New", 16), fg="#888888", bg="#0d0d0d")
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

        self.status_bar = tk.Frame(side, height=8, bg="#3a86ff")
        self.status_bar.pack(fill="x", pady=(0, 20))

        tk.Label(side, text="STATUS", font=("Courier New", 11), fg="#555", bg="#0d0d0d").pack(anchor="w")
        self.status_label = tk.Label(side, text="WAITING FOR FACE", font=("Courier New", 32, "bold"),
                                     fg="#3a86ff", bg="#0d0d0d", wraplength=380, justify="left")
        self.status_label.pack(anchor="w", pady=(4, 20))

        tk.Label(side, text="EMPLOYEE ID", font=("Courier New", 11), fg="#555", bg="#0d0d0d").pack(anchor="w")
        self.id_label = tk.Label(side, text="—", font=("Courier New", 28, "bold"), fg="#e0e0e0", bg="#0d0d0d")
        self.id_label.pack(anchor="w", pady=(4, 20))

        tk.Label(side, text="SCAN TIME", font=("Courier New", 11), fg="#555", bg="#0d0d0d").pack(anchor="w")
        self.elapsed_label = tk.Label(side, text="—", font=("Courier New", 14), fg="#888888", bg="#0d0d0d")
        self.elapsed_label.pack(anchor="w", pady=(4, 20))

    def _tick_clock(self):
        self.clock_label.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def _update_camera(self):
        global latest_frame
        ret, frame = self.cap.read()
        if ret:
            # ล็อกขนาดภาพป้องกันลูปการขยายหน้าต่าง
            frame = cv2.resize(frame, (640, 480))
            
            # บันทึกภาพล่าสุดให้ Worker Thread เอาไปประมวลผล
            with frame_lock:
                latest_frame = frame.copy()

            # อัปเดตขึ้นจอ UI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            self.camera_label.configure(image=img)
            self.camera_label.image = img

        self.root.after(33, self._update_camera)

    def _poll_results(self):
        try:
            while True:
                result = result_queue.get_nowait()
                self._show_result(result)
        except queue.Empty:
            pass

        self.root.after(100, self._poll_results)

    def _show_result(self, result):
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

        self.elapsed_label.config(text=f"{elapsed:.2f} seconds")

        # คืนค่าหน้าจอกลับเป็นสถานะรอคนต่อไปหลังจาก 3 วินาที
        self.root.after(3000, lambda: (
            self._set_state("idle"),
            self.status_label.config(text="WAITING FOR FACE", fg="#3a86ff"),
            self.id_label.config(text="—", fg="#e0e0e0"),
            self.elapsed_label.config(text="—")
        ))

    def _set_state(self, state):
        self._state = state
        bg = self.STATUS_COLORS.get(state, "#0d0d0d")
        bar_colors = {
            "idle":      "#3a86ff",
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
    # เปิดกล้อง
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open camera")

    # เปิดการเชื่อมต่อ ESP32
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0)
        print(f"✅ Serial open: {SERIAL_PORT} @ {SERIAL_BAUD}")
    except serial.SerialException as e:
        print(f"⚠️  Serial error: {e} (ระบบกล้องจะยังทำงานต่อไปได้)")
        ser = None

    # สตาร์ท Thread หลังบ้าน
    if ser:
        threading.Thread(target=serial_writer, args=(ser,), daemon=True).start()
    
    threading.Thread(target=auto_face_worker, daemon=True).start()

    # สตาร์ทหน้าจอ UI
    ui = AutoScanUI(cap)
    ui.run()

    cap.release()

if __name__ == "__main__":
    main()
