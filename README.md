# Face ID System — Register + Recognize + ESP32 Trigger

A full-screen Tkinter app that combines face **registration** and **recognition**
into one program, backed by dlib face encodings (via `face_recognition`) and an
ESP32 for the physical trigger/relay side.

## What's in this repo

```
vector_face_recognition/
  main.py         # the app — Recognize / Register / Manage pages
  face_store.py   # face-encoding database (encodings.pkl)
commu_function/
  esp32.cpp       # ESP32 firmware: trigger button -> serial -> Jetson -> ID reply -> LED/relay
  python.py       # small standalone script for manually sending a test serial message
```

`main.py` has three pages, switchable from the top nav bar:

- **Recognize** — live camera feed, waits for an ESP32 trigger (`"1\n"` over
  serial) or the on-screen **TEST SCAN** button, then shows the matched name/ID.
- **Register** — guided 5-pose capture wizard to enroll a new person (or add
  more samples to an existing one).
- **Manage** — list, search, and delete enrolled people.

## Requirements

- Python 3.9–3.11 (see the note below — 3.12+ is more likely to lack a
  prebuilt `dlib` wheel, and 3.13+ is very unlikely to have one at all).
- A webcam.
- Optional: an ESP32 connected over serial, running `commu_function/esp32.cpp`.
  Without one, the app runs fine in **test mode** (see below).

### Installing dependencies

**Linux (including Jetson / aarch64):**

```bash
sudo apt install build-essential cmake
pip install -r requirements.txt
```

`dlib` (a `face_recognition` dependency) builds from source here; the two
packages above are what it needs to compile.

**Windows:**

`pip install -r requirements.txt` will try to build `dlib` from source too,
which needs CMake + a matching Visual C++ toolchain and can fail (especially
on newer/preview Visual Studio versions that a given CMake release doesn't
recognize yet as a generator). The reliable path on Windows is conda-forge,
which ships a prebuilt `dlib` binary:

```bash
# Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html
conda create -n faceid --override-channels -c conda-forge python=3.10
conda install -n faceid --override-channels -c conda-forge opencv numpy pillow pyserial dlib face_recognition
```

Then run the app using that environment's Python (see below).

## Running it

```bash
# Linux / Jetson, after `pip install -r requirements.txt`
cd vector_face_recognition
python main.py
```

```bash
# Windows, using the conda env from above
cd vector_face_recognition
C:\Users\<you>\miniconda3\envs\faceid\python.exe main.py
```

The window opens full-screen. Press `Esc` to leave full-screen, `F11` to
return to it.

## Testing without an ESP32 connected

You don't need to change any config for this. `main.py` tries to open
`SERIAL_PORT` (set in `main.py`, default `/dev/ttyUSB0`); if that port
doesn't exist — which it won't unless you actually have the ESP32 plugged in
at that path — it logs a warning and the app runs in **test mode**:

- The Recognize page's sidebar shows `SERIAL: TEST MODE` instead of `LIVE`.
- A **TEST SCAN** button sits below that. Clicking it pushes the same
  "trigger" signal a real ESP32 would send over serial (`"1\n"`), running the
  exact same detect → match → result pipeline. The only difference is the
  final "send this ID back to the ESP32" step gets logged instead of written
  to a serial port.
- Register and Manage don't use serial at all, so they work identically
  either way.

Typical test flow:

1. Run `main.py`.
2. Go to **Register**, enter a 9-digit ID and a name, click **START CAPTURE**,
   and follow the 5 guided poses (straight / left / right / up / down).
3. Go to **Recognize**, click **TEST SCAN**, and look at the camera — you
   should see `VERIFIED` with your name and ID. Watch the console for
   `[FACE_REC]` / `[WORKER]` log lines showing the detection/match details.
4. Go to **Manage** to confirm the person is listed, or to delete them.

If you have pyserial installed but want to point at a real port later
(e.g. `COM3` on Windows, `/dev/ttyUSB0` on Linux), edit `SERIAL_PORT` at the
top of `main.py`.

## ESP32 firmware

`commu_function/esp32.cpp` implements the other end of the protocol:

- Debounced trigger button on `TRIGGER_PIN` → sends `"1\n"` to the Jetson.
- Waits for a 9-digit ID reply (`"000000000"` means unknown/no match).
- Drives a green/red LED and a relay (e.g. a door strike) based on the result.
- Has a retrigger lockout and a response timeout so a stuck button or a slow
  Jetson can't pile up multiple in-flight scans.

Flash it with the Arduino IDE / PlatformIO, wire up a push button (to GND,
using the internal pull-up) plus LEDs/relay per the pin constants at the top
of the file, and set `SERIAL_PORT`/`SERIAL_BAUD` in `main.py` to match.

## Notes on the face database

Enrolling someone through the Register page writes to
`vector_face_recognition/encodings.pkl`. That file contains other people's
biometric data, so it's git-ignored — don't commit it. Each machine running
this app builds up its own local database.
