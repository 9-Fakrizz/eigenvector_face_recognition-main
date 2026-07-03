# QUICK START - main4.py Testing Guide

## 🚀 Quick Start (30 seconds)

```bash
cd vector_face_recognition
python main4.py
```

**What you'll see:**
- Console: Startup messages with timestamps
- Window: Live camera feed + "TEST SCAN" button
- Status: "WAITING"

---

## 🧪 Test Scenarios

### Scenario 1: Click Test Button

```
1. UI shows "WAITING"
2. Click "TEST SCAN (จำลอง ESP32)" button
3. Watch console:
   - [WORKER] Trigger received (#1) 
   - [FACE_REC] Frame X: Y face(s) detected
   - [FACE_REC] MATCH 'Name' (distance: 0.XX)
   - [WORKER] Job #1 completed: emp_id='12345' time=0.75s
   - [SERIAL_TX] ✅ Sent: '12345' (9 bytes)
4. UI shows "VERIFIED ✓" or "UNKNOWN FACE"
```

### Scenario 2: Watch Console Debug Output

The console will show detailed traces:

```
HH:MM:SS.mmm [INFO    ] [MAIN        ] Face Recognition System Starting
HH:MM:SS.mmm [INFO    ] [ENCODER     ] ✅ Loaded 5 known faces
HH:MM:SS.mmm [INFO    ] [MAIN        ] ✅ Camera opened successfully
HH:MM:SS.mmm [WARN    ] [MAIN        ] Serial error: No such file or directory
HH:MM:SS.mmm [WARN    ] [MAIN        ] Running WITHOUT serial communication (test mode)
HH:MM:SS.mmm [INFO    ] [SERIAL_RX   ] 🔌 SerialReader thread started
HH:MM:SS.mmm [INFO    ] [WORKER      ] 🧠 FaceWorker thread started
HH:MM:SS.mmm [INFO    ] [UI          ] ✅ UI initialization complete

[User clicks TEST SCAN]

HH:MM:SS.mmm [INFO    ] [WORKER      ] Trigger received (#1) at HH:MM:SS.mmm
HH:MM:SS.mmm [INFO    ] [FACE_REC    ] Starting face recognition scan (10 frames)
HH:MM:SS.mmm [DEBUG   ] [FACE_REC    ]   Frame 1: 1 face(s) detected
HH:MM:SS.mmm [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.385)
HH:MM:SS.mmm [INFO    ] [FACE_REC    ] Final result: 'John_Doe' (votes: 10, votes_dict: {'John_Doe': 10})
HH:MM:SS.mmm [INFO    ] [WORKER      ] Job #1 completed: emp_id='12345' time=0.75s
HH:MM:SS.mmm [DEBUG   ] [WORKER      ]   Queuing TX: '12345'
HH:MM:SS.mmm [INFO    ] [SERIAL_TX   ] ✅ Sent: '12345' (9 bytes) at HH:MM:SS.mmm
HH:MM:SS.mmm [DEBUG   ] [UI          ] Result received: {'employee_id': '12345', 'elapsed': 0.75}
HH:MM:SS.mmm [INFO    ] [UI          ] Status: VERIFIED ✓
```

---

## 📊 Debug Output Meanings

### ✅ Success Indicators
- `[INFO] [ENCODER] ✅ Loaded X known faces` → Database loaded
- `[INFO] [MAIN] ✅ Camera opened` → Camera working
- `[INFO] [SERIAL_RX] ✅ Trigger received` → ESP32 sent message
- `[INFO] [SERIAL_TX] ✅ Sent: 'ID'` → Employee ID sent to ESP32
- `[INFO] [FACE_REC] Final result: 'Name'` → Face matched

### ⚠️ Warnings (Still Works)
- `[WARN] Serial error: No such file or directory` → No serial port (test mode)
- `[WARN] encodings.pkl not found` → No face database (unknown faces only)

### ❌ Error Indicators
- `[ERROR] Cannot open camera` → Camera problem, won't work
- `[ERROR] SerialReader Error` → Serial reading problem
- `[ERROR] Job #X failed with exception` → Recognition failed

### 📍 Debug Details
- `[DEBUG] Received 1 bytes: '\x31\n'` → Raw bytes from ESP32 (hex)
- `[DEBUG] Frame 1: 2 face(s) detected` → Number of faces found
- `[DEBUG] Face 0: MATCH 'Name' (distance: 0.385)` → Face match score (< 0.50 = match)
- `[DEBUG] Queue remaining: 0` → Triggers still waiting to process

---

## 🎯 Key Debug Info You Can Check

### 1. **Is Face Database Loaded?**
Look for: `[ENCODER] ✅ Loaded X known faces`
- Shows number in `encodings.pkl`
- Lists all known names

### 2. **Is Camera Working?**
Look for: `[INFO] [MAIN] ✅ Camera opened successfully`
- Also check UI shows live video

### 3. **Is Face Detected?**
Look for: `[FACE_REC] Frame X: Y face(s) detected`
- 0 = no faces → "UNKNOWN"
- 1+ = faces found → try to match

### 4. **Is Face Matching Working?**
Look for: `[FACE_REC] Face 0: MATCH 'Name' (distance: X.XXX)`
- distance < 0.50 = MATCH ✓
- distance > 0.50 = UNKNOWN ✗
- Check if you're in frame when clicking TEST SCAN

### 5. **Is Serial Working?**
Look for: `[SERIAL_TX] ✅ Sent: 'ID'`
- Shows employee ID was sent
- Shows byte count (usually 9 bytes = 8 digit ID + newline)
- If not found: check serial port in config

---

## 📝 Typical Test Session Example

```bash
$ python main4.py
14:32:10.123 [INFO    ] [MAIN        ] ============================================
14:32:10.124 [INFO    ] [MAIN        ] Face Recognition System Starting
14:32:10.125 [INFO    ] [MAIN        ] Debug Mode: True
14:32:10.126 [INFO    ] [MAIN        ] Config: PORT=/dev/ttyUSB0, BAUD=115200, CAMERA=0
14:32:10.127 [INFO    ] [MAIN        ] ============================================
14:32:10.128 [INFO    ] [MAIN        ] Opening camera...
14:32:10.500 [INFO    ] [MAIN        ] ✅ Camera opened successfully
14:32:10.501 [INFO    ] [MAIN        ] Opening serial port /dev/ttyUSB0...
14:32:10.502 [WARN    ] [MAIN        ] ⚠️  Serial error: [Errno 2] No such file or directory: '/dev/ttyUSB0'
14:32:10.503 [WARN    ] [MAIN        ] Running WITHOUT serial communication (test mode)
14:32:10.504 [INFO    ] [ENCODER     ] ✅ Loaded 3 known faces from encodings.pkl
14:32:10.505 [INFO    ] [ENCODER     ] Names: ['John_Doe', 'Jane_Smith', 'Bob_Wilson']
14:32:10.506 [INFO    ] [SERIAL_RX   ] 🔌 SerialReader thread started
14:32:10.507 [INFO    ] [WORKER      ] 🧠 FaceWorker thread started
14:32:10.510 [INFO    ] [UI          ] Initializing UI components...
14:32:10.650 [INFO    ] [UI          ] ✅ UI initialization complete
14:32:10.651 [INFO    ] [MAIN        ] UI ready - entering main loop

# [User clicks TEST SCAN button with face in frame]

14:32:15.342 [INFO    ] [WORKER      ] Trigger received (#1) at 14:32:15.342
14:32:15.343 [DEBUG   ] [WORKER      ]   Submitting scan job to executor
14:32:15.344 [INFO    ] [FACE_REC    ] Starting face recognition scan (10 frames)
14:32:15.350 [DEBUG   ] [FACE_REC    ]   Frame 1: 1 face(s) detected
14:32:15.355 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.372)
14:32:15.360 [DEBUG   ] [FACE_REC    ]   Frame 2: 1 face(s) detected
14:32:15.365 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.385)
14:32:15.370 [DEBUG   ] [FACE_REC    ]   Frame 3: 1 face(s) detected
14:32:15.375 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.391)
14:32:15.380 [DEBUG   ] [FACE_REC    ]   Frame 4: 1 face(s) detected
14:32:15.385 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.380)
14:32:15.390 [DEBUG   ] [FACE_REC    ]   Frame 5: 1 face(s) detected
14:32:15.395 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.388)
14:32:15.400 [DEBUG   ] [FACE_REC    ]   Frame 6: 1 face(s) detected
14:32:15.405 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.379)
14:32:15.410 [DEBUG   ] [FACE_REC    ]   Frame 7: 1 face(s) detected
14:32:15.415 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.392)
14:32:15.420 [DEBUG   ] [FACE_REC    ]   Frame 8: 1 face(s) detected
14:32:15.425 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.381)
14:32:15.430 [DEBUG   ] [FACE_REC    ]   Frame 9: 1 face(s) detected
14:32:15.435 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.383)
14:32:15.440 [DEBUG   ] [FACE_REC    ]   Frame 10: 1 face(s) detected
14:32:15.445 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.389)
14:32:15.450 [INFO    ] [FACE_REC    ] Final result: 'John_Doe' (votes: 10, votes_dict: {'John_Doe': 10})
14:32:15.451 [INFO    ] [WORKER      ] Job #1 completed: emp_id='12345' time=1.107s
14:32:15.452 [DEBUG   ] [WORKER      ]   Queuing TX: '12345'
14:32:15.453 [DEBUG   ] [SERIAL_TX   ] Preparing to send: '12345' (msg #1)
14:32:15.454 [DEBUG   ] [SERIAL_TX   ]   Bytes to send: b'12345\n'
14:32:15.455 [INFO    ] [SERIAL_TX   ] ✅ Sent: '12345' (9 bytes) at 14:32:15.455
14:32:15.456 [DEBUG   ] [SERIAL_TX   ]   Queue remaining: 0
14:32:15.457 [DEBUG   ] [UI          ] Result received: {'employee_id': '12345', 'elapsed': 1.107}
14:32:15.458 [DEBUG   ] [UI          ] Displaying result: emp_id='12345', elapsed=1.107s
14:32:15.459 [INFO    ] [UI          ] Status: VERIFIED ✓

# [UI shows GREEN checkmark, then resets after 3 seconds]
```

---

## 💡 Pro Tips

1. **Save output to file for analysis:**
   ```bash
   python main4.py 2>&1 | tee test_$(date +%Y%m%d_%H%M%S).log
   ```

2. **Filter just errors:**
   ```bash
   grep ERROR debug_run.log
   ```

3. **Filter face matches:**
   ```bash
   grep "MATCH\|UNKNOWN" debug_run.log
   ```

4. **See serial communication:**
   ```bash
   grep "SERIAL_" debug_run.log
   ```

5. **Turn off debug for production:**
   Edit line in main4.py:
   ```python
   DEBUG_MODE = False
   ```

---

## 🔧 Config Changes You Might Need

Edit these in `main4.py`:

```python
SERIAL_PORT   = "/dev/ttyUSB0"    # Change if your port is different
SERIAL_BAUD   = 115200            # Must match ESP32 baud rate
CAMERA_INDEX  = 0                 # 0=default, try 1,2 if not working
FACE_TOLERANCE = 0.50             # Lower=stricter (fewer false matches)
SCAN_FRAMES    = 10                # More=more accurate (slower)
DEBUG_MODE     = True              # Set to False to hide debug output
```

---

**Ready to test? Run: `python main4.py` and click the TEST SCAN button!**

