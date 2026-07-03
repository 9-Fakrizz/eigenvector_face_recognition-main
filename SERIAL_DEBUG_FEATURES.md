# SERIAL DEBUG FEATURES ADDED TO main4.py

## 📡 Serial Communication Debug Output

### Serial RX (ESP32 → Jetson)
```python
[HH:MM:SS.mmm] [INFO    ] [SERIAL_RX   ] ✅ Trigger received (msg: '1') at HH:MM:SS.mmm
[HH:MM:SS.mmm] [DEBUG   ] [SERIAL_RX   ] Received 1 bytes: '\x31\n'
[HH:MM:SS.mmm] [DEBUG   ] [SERIAL_RX   ] Line: '1'
```
- Logs every byte received
- Shows trigger messages clearly
- Displays timing of messages

### Serial TX (Jetson → ESP32)
```python
[HH:MM:SS.mmm] [DEBUG   ] [SERIAL_TX   ] Preparing to send: '12345' (msg #1)
[HH:MM:SS.mmm] [DEBUG   ] [SERIAL_TX   ] Bytes to send: b'12345\n'
[HH:MM:SS.mmm] [INFO    ] [SERIAL_TX   ] ✅ Sent: '12345' (9 bytes) at HH:MM:SS.mmm
[HH:MM:SS.mmm] [DEBUG   ] [SERIAL_TX   ] Queue remaining: 0
```
- Shows what will be sent before sending
- Confirms bytes written
- Shows queue depth

## 🎯 Face Recognition Debug Output

### Frame Processing
```python
[HH:MM:SS.mmm] [DEBUG   ] [FACE_REC    ]   Frame 1: 2 face(s) detected
[HH:MM:SS.mmm] [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.425)
[HH:MM:SS.mmm] [DEBUG   ] [FACE_REC    ]     Face 1: UNKNOWN (best distance: 0.620)
```
- Shows number of faces per frame
- Shows matching names and distances
- Shows why faces are rejected (distance > tolerance)

### Final Result
```python
[HH:MM:SS.mmm] [INFO    ] [FACE_REC    ] Final result: 'John_Doe' (votes: 4, votes_dict: {'John_Doe': 4})
[HH:MM:SS.mmm] [INFO    ] [WORKER      ] Job #1 completed: emp_id='12345' time=1.23s
```
- Shows voting results
- Shows final employee ID
- Shows timing

## 🔧 Component Logging

| Component | Function | Output |
|-----------|----------|--------|
| ENCODER | Load face database | ✅ 10 known faces loaded |
| SERIAL_RX | Receive trigger | Incoming bytes + "1" detection |
| SERIAL_TX | Send employee ID | Before/after send + bytes count |
| FACE_REC | Recognition logic | Faces per frame + matches + final result |
| WORKER | Job management | Job start/complete + timing |
| UI | UI updates | State changes + result display |

## 🧪 Test Features

### Test Button
- Located in UI: "TEST SCAN (จำลอง ESP32)"
- Simulates trigger from ESP32 (queues "TRIGGER")
- Full processing pipeline works same as real serial trigger

### Mock Serial Mode
- If serial port not found: shows warning but continues
- Test button still works
- All debug logging still shows TX messages
- Perfect for development without hardware

## 📊 Output Example: Full Test Cycle

```
13:45:22.123 [INFO    ] [MAIN        ] ============================================
13:45:22.124 [INFO    ] [MAIN        ] Face Recognition System Starting
13:45:22.125 [INFO    ] [MAIN        ] Config: PORT=/dev/ttyUSB0, BAUD=115200, CAMERA=0
13:45:22.126 [INFO    ] [ENCODER     ] ✅ Loaded 3 known faces from encodings.pkl
13:45:22.126 [INFO    ] [ENCODER     ] Names: ['John_Doe', 'Jane_Smith', 'Bob_Wilson']
13:45:22.200 [INFO    ] [MAIN        ] ✅ Camera opened successfully
13:45:22.201 [INFO    ] [MAIN        ] ✅ Serial open: /dev/ttyUSB0 @ 115200 baud
13:45:22.210 [INFO    ] [SERIAL_RX   ] 🔌 SerialReader thread started
13:45:22.211 [INFO    ] [SERIAL_TX   ] 🔌 SerialWriter thread started
13:45:22.212 [INFO    ] [WORKER      ] 🧠 FaceWorker thread started
13:45:22.300 [INFO    ] [UI          ] Initializing UI components...
13:45:22.450 [INFO    ] [UI          ] ✅ UI initialization complete

# User clicks TEST BUTTON
13:45:25.100 [INFO    ] [WORKER      ] Trigger received (#1) at 13:45:25.100
13:45:25.101 [DEBUG   ] [WORKER      ]   Submitting scan job to executor
13:45:25.102 [INFO    ] [FACE_REC    ] Starting face recognition scan (10 frames)
13:45:25.105 [DEBUG   ] [FACE_REC    ]   Frame 1: 1 face(s) detected
13:45:25.108 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.385)
13:45:25.110 [DEBUG   ] [FACE_REC    ]   Frame 2: 1 face(s) detected
13:45:25.113 [DEBUG   ] [FACE_REC    ]     Face 0: MATCH 'John_Doe' (distance: 0.391)
...more frames...
13:45:25.850 [INFO    ] [FACE_REC    ] Final result: 'John_Doe' (votes: 8, votes_dict: {'John_Doe': 8})
13:45:25.851 [INFO    ] [WORKER      ] Job #1 completed: emp_id='12345' time=0.75s
13:45:25.852 [DEBUG   ] [WORKER      ]   Queuing TX: '12345'
13:45:25.853 [DEBUG   ] [SERIAL_TX   ] Preparing to send: '12345' (msg #1)
13:45:25.854 [DEBUG   ] [SERIAL_TX   ]   Bytes to send: b'12345\n'
13:45:25.855 [INFO    ] [SERIAL_TX   ] ✅ Sent: '12345' (9 bytes) at 13:45:25.855
13:45:25.856 [DEBUG   ] [SERIAL_TX   ]   Queue remaining: 0
13:45:25.857 [DEBUG   ] [UI          ] Result received: {'employee_id': '12345', 'elapsed': 0.75}
13:45:25.858 [DEBUG   ] [UI          ] Displaying result: emp_id='12345', elapsed=0.75s
13:45:25.859 [INFO    ] [UI          ] Status: VERIFIED ✓
```

## 🎮 How to Use

### 1. Run with Debug
```bash
python main4.py
```
All output automatically logged to console.

### 2. Save Debug Output
```bash
python main4.py > debug_run.log 2>&1
```

### 3. Disable Debug (Production)
Edit main4.py:
```python
DEBUG_MODE = False  # Disables all debug_log() calls
```

### 4. Filter Output
```bash
# Watch only serial traffic
grep "SERIAL_" debug_run.log

# Watch only face recognition
grep "FACE_REC" debug_run.log

# Watch only errors
grep "ERROR" debug_run.log
```

## ✅ What to Look For When Testing

1. **Serial RX**: Do you see "Trigger received" messages?
2. **Face Detection**: Are frames being captured? How many faces?
3. **Face Matching**: Do distances look reasonable? (< 0.50 = match)
4. **Serial TX**: Does employee ID appear in TX messages?
5. **Timing**: How long does recognition take? (should be < 2 sec)
6. **Queue**: Multiple triggers queue properly?

