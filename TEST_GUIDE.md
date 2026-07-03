# Face Recognition Test Guide

## Overview
This guide helps you test the face recognition system with button control and serial communication.

---

## File Descriptions

### main4.py - **TEST WITH BUTTON** (Current Focus)
- **Features**: Test button in UI + comprehensive debug logging
- **Status**: ✅ Enhanced with debug output
- **Use Case**: Debug serial communication, test face recognition logic
- **Button**: "TEST SCAN (จำลอง ESP32)" simulates trigger from ESP32
- **Debug Output**: Full logging of:
  - Serial RX/TX with timestamps
  - Face detection frames
  - Recognition votes
  - UI state changes

### main3.py - **AUTO SCAN**
- **Features**: Continuously scans without waiting for triggers
- **Status**: Original version (unchanged)
- **Use Case**: Production mode with automatic scanning
- **Note**: Implements cooldown to prevent spam

### main2.py - **WITH REGISTRATION**
- **Features**: Combined registration + recognition GUI
- **Status**: Original version (unchanged)
- **Use Case**: Enroll new faces, test recognition
- **Note**: Single-window app with mode switching

### main.py - **ORIGINAL VERSION**
- **Features**: Original trigger-based system
- **Status**: Unchanged
- **Use Case**: Reference/baseline

---

## Testing Workflow

### 1. **Test Button (main4.py)**

#### Step 1: Prepare
```bash
cd vector_face_recognition
python main4.py
```

#### Step 2: Watch Console Output
The debug output shows:
```
HH:MM:SS.mmm [INFO    ] [SERIAL_RX   ] Line: '1'
HH:MM:SS.mmm [INFO    ] [SERIAL_RX   ] ✅ Trigger received (msg: '1') at HH:MM:SS.mmm
HH:MM:SS.mmm [INFO    ] [WORKER      ] Trigger received (#1) at HH:MM:SS.mmm
HH:MM:SS.mmm [DEBUG   ] [FACE_REC    ]   Frame 1: 2 face(s) detected
HH:MM:SS.mmm [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.425)
HH:MM:SS.mmm [INFO    ] [FACE_REC    ] Final result: 'John_Doe' (votes: 4, votes_dict: {'John_Doe': 4})
HH:MM:SS.mmm [INFO    ] [WORKER      ] Job #1 completed: emp_id='12345' time=1.23s
HH:MM:SS.mmm [DEBUG   ] [WORKER      ]   Queuing TX: '12345'
HH:MM:SS.mmm [DEBUG   ] [SERIAL_TX   ] Preparing to send: '12345' (msg #1)
HH:MM:SS.mmm [INFO    ] [SERIAL_TX   ] ✅ Sent: '12345' (9 bytes) at HH:MM:SS.mmm
```

#### Step 3: Click "TEST SCAN" Button
- Watch the status change: "WAITING" → "SCANNING" → "VERIFIED ✓" or "UNKNOWN FACE"
- Check console for:
  - Face detection frames
  - Recognition matching
  - Serial TX messages

---

### 2. **Serial Communication Testing**

#### Check if Serial is Connected:
```bash
# Linux/Mac:
ls -la /dev/ttyUSB*

# Windows (in PowerShell):
Get-PnpDevice -FriendlyName "*USB*"
```

#### If No Serial Port:
- main4.py runs in **test mode** (no real serial communication)
- Button still works and shows "WAITING/SCANNING/RESULT"
- Debug output shows `[SERIAL_TX]` messages even without physical port
- You'll see: `Serial error: (FileNotFoundError) [Errno 2] No such file or directory: '/dev/ttyUSB0'`

#### Mock Serial Testing:
You can create a virtual serial port pair for testing:
```bash
# Linux (socat):
socat -d -d pty,raw,echo=0 pty,raw,echo=0

# This creates two virtual ports, e.g.:
# /dev/pts/10 (one side)
# /dev/pts/11 (other side)

# Then modify SERIAL_PORT in main4.py and test
```

---

### 3. **Debug Print Locations**

The enhanced main4.py logs:

| Component | What It Shows | Look For |
|-----------|--------------|----------|
| **ENCODER** | Loaded faces from encodings.pkl | Number of faces, names list |
| **SERIAL_RX** | Incoming trigger from ESP32 | "Trigger received" = "1" message |
| **SERIAL_TX** | Outgoing employee IDs | Employee ID being sent (9 bytes) |
| **FACE_REC** | Face detection & matching | Frame count, face locations, distances |
| **WORKER** | Job processing | Start/complete times, job numbers |
| **UI** | UI state changes | WAITING → SCANNING → VERIFIED |

---

### 4. **Common Test Cases**

#### Test Case 1: Known Face
1. Click "TEST SCAN"
2. Show your face to camera
3. **Expected Output**:
   - Console: `MATCH 'YourName' (distance: 0.4xx)`
   - UI: Green "VERIFIED ✓"
   - Console: `Sent: 'YOUR_ID'`

#### Test Case 2: Unknown Face
1. Click "TEST SCAN"
2. Show an unknown person's face or object
3. **Expected Output**:
   - Console: `distance: 0.6xx` (higher than threshold)
   - UI: Red "UNKNOWN FACE"
   - Console: `Sent: '000000000'`

#### Test Case 3: No Face
1. Click "TEST SCAN"
2. Point camera away / blank surface
3. **Expected Output**:
   - Console: `0 face(s) detected`
   - Console: `UNKNOWN` (UNKNOWN_ID)
   - UI: Red "UNKNOWN FACE"

#### Test Case 4: Multiple Triggers
1. Click "TEST SCAN" multiple times quickly
2. **Expected Output**:
   - Console: `Queue remaining: X` (should process queue)
   - Multiple job numbers: `Job #1`, `Job #2`, etc.
   - All results processed

---

### 5. **Configuration for Testing**

Edit the config at top of main4.py:

```python
SERIAL_PORT   = "/dev/ttyUSB0"    # Change port if needed
SERIAL_BAUD   = 115200             # Must match ESP32
CAMERA_INDEX  = 0                  # 0 = default camera
ENCODINGS_FILE = "encodings.pkl"  # Path to face database
FACE_TOLERANCE = 0.50              # Lower = stricter matching
SCAN_FRAMES    = 10                # More frames = more accurate
MAX_WORKERS    = 2                 # Parallel recognition jobs
DEBUG_MODE     = True              # ← Enable debug logging
```

---

## Next Steps: From Testing to Production

### Switch Versions as Needed:
1. **Testing** → Use `main4.py` (button + debug)
2. **Manual + Registration** → Use `main2.py` (combined mode)
3. **Auto Production** → Use `main3.py` (continuous scanning)
4. **Reference** → See `main.py` (original)

### Serial Debugging Tips:
- Watch `[SERIAL_TX]` lines to confirm data being sent
- Watch `[SERIAL_RX]` lines to confirm trigger received
- If no RX messages: Check ESP32 is sending "1\n"
- If TX not working: Check baudrate and port match

### Performance Notes:
- Face recognition: ~0.5-1.5 seconds per scan
- Queue depth shows how many triggers waiting
- Multiple triggers queue automatically

---

## Debug Output Levels

```
[INFO    ] - Important events (triggers, results, connections)
[DEBUG   ] - Detailed information (frame counts, distances, bytes)
[WARN    ] - Warnings (missing files, serial errors)
[ERROR   ] - Errors (exceptions, failures)
```

To disable debug output:
```python
DEBUG_MODE = False  # in main4.py
```

---

## Troubleshooting

### No Camera Image
- Check `CAMERA_INDEX` (try 0, 1, 2)
- Check camera permissions
- Console should show: `✅ Camera opened successfully`

### No Face Recognition
- Ensure `encodings.pkl` exists in same folder
- Check console: `Loaded X known faces`
- Try clicking TEST SCAN with known face

### Serial Not Working
- Check port: `ls /dev/ttyUSB*` or `com3`, `com4` on Windows
- Console shows: `Serial open: /dev/ttyUSB0 @ 115200`
- If error shown, still works in test mode (no real TX)

### Faces Not Matching
- Check `FACE_TOLERANCE` (0.50 = medium strict)
- Lower value = stricter (fewer false matches)
- Higher value = looser (more false matches)
- Watch console distances: should be < 0.50 for known faces

---

## Recording Test Results

When testing, save console output:
```bash
# Run with output to file:
python main4.py 2>&1 | tee test_run_$(date +%Y%m%d_%H%M%S).log

# Then analyze:
grep "MATCH\|UNKNOWN\|ERROR" test_run_*.log
```

---

**Happy Testing! 🚀**
