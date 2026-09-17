# Face ID System — Register + Recognize + ESP32 Trigger

A full-screen Tkinter app that combines face **registration** and
**recognition** into one program, backed by dlib face encodings (via
`face_recognition`) and an ESP32 for the physical trigger/relay side.

```
vector_face_recognition/
  main.py         # the app — Recognize / Register / Manage / Help pages
  face_store.py   # face-encoding database (encodings.pkl)
commu_function/
  esp32.cpp       # ESP32 firmware: trigger button -> serial -> Jetson -> ID reply -> LED/relay
  python.py       # small standalone script for manually sending a test serial message
Dockerfile              # Jetson Nano deployment image
docker-requirements.txt # Python deps for the Docker image
docker-run.sh           # convenience script wrapping `docker run` with the right flags
```

Pick your setup path below, then jump to **How to use the app**.

- [Setup: Windows](#setup-windows)
- [Setup: Linux / Jetson (running directly, no Docker)](#setup-linux--jetson-running-directly-no-docker)
- [Setup: Docker on Jetson Nano](#setup-docker-on-jetson-nano)
- [How to use the app](#how-to-use-the-app)
- [ESP32 firmware](#esp32-firmware)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Setup: Windows

Windows can't reliably `pip install` this app's `dlib` dependency (it needs
to compile from source, which needs a matching Visual C++ toolchain and
often fails). Use conda-forge instead, which ships a prebuilt `dlib`.

1. **Install Miniconda**: https://docs.conda.io/en/latest/miniconda.html

2. **Create the environment:**
   ```bash
   conda create -n faceid --override-channels -c conda-forge python=3.10
   ```

3. **Install dependencies, pinning the CPU build of dlib.** Without the
   explicit `cpu_py310*` pin, conda's solver can pick a CUDA build instead
   and try to pull down ~1GB of NVIDIA cuDNN packages you don't need:
   ```bash
   conda install -n faceid --override-channels -c conda-forge "dlib=20.0.1=cpu_py310*" opencv numpy pillow pyserial face_recognition
   ```

4. **Fix a `pkg_resources` issue.** `face_recognition`'s model-data package
   still imports the old `pkg_resources` API, which newer setuptools no
   longer ships:
   ```bash
   conda run -n faceid pip install "setuptools<81"
   ```

5. **Run it:**
   ```bash
   cd vector_face_recognition
   C:\Users\<you>\miniconda3\envs\faceid\python.exe main.py
   ```

---

## Setup: Linux / Jetson (running directly, no Docker)

1. **Install build tools** (needed to compile `dlib` from source):
   ```bash
   sudo apt install build-essential cmake
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run it:**
   ```bash
   cd vector_face_recognition
   python main.py
   ```

If you'd rather containerize this for a Jetson Nano instead of installing
directly on the host, skip to [Setup: Docker on Jetson Nano](#setup-docker-on-jetson-nano).

---

## Setup: Docker on Jetson Nano

Builds a self-contained image for the original Jetson Nano running
**JetPack 4.6.x (L4T R32.7.x)**.

1. **Check your JetPack/L4T version** (skip if you already know it's 4.6.x):
   ```bash
   cat /etc/nv_tegra_release
   ```
   If it's not R32.7.x, open `Dockerfile` and change the
   `FROM nvcr.io/nvidia/l4t-base:r32.7.1` line to match — an l4t-base image
   needs to match the host's L4T release for camera/display passthrough to
   work.

2. **Add swap space.** Building `dlib` from source is memory-hungry and
   will likely OOM-kill on a stock 4GB Nano without it:
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

3. **Build the image** (do this on the Jetson itself — native ARM
   compilation is faster than cross-building via QEMU emulation for a
   CPU-heavy compile like dlib's):
   ```bash
   docker build -t faceid-system .
   ```
   Expect the `dlib` layer to take roughly 45-90 minutes the first time.
   It's cached in its own Docker layer, so rebuilding after editing
   `main.py` later takes seconds, not another hour.

4. **Run it:**
   ```bash
   chmod +x docker-run.sh
   ./docker-run.sh
   ```
   This script runs `xhost +local:docker` (so the container's GUI can
   reach your display), passes the camera device through, mounts `./data`
   on the host so enrolled people survive container restarts/rebuilds, and
   runs `--privileged` so the in-app serial port switcher (see
   [How to use the app](#how-to-use-the-app)) can reach whichever
   `/dev/ttyUSB*`/`/dev/ttyACM*` device the ESP32 actually shows up on.

5. **Override defaults** via environment variables if needed, e.g.:
   ```bash
   SERIAL_PORT=/dev/ttyACM0 CAMERA_DEVICE=/dev/video1 ./docker-run.sh
   ```

Once you've confirmed the real serial port in step 5, you can tighten
`docker-run.sh`'s `--privileged` to a specific `--device=/dev/ttyACM0` for
a locked-down production deployment.

---

## How to use the app

The window opens full-screen on Docker/kiosk deployments, or as a normal
window otherwise (`Esc` to leave full-screen, `F11` to enter it). Four
pages are reachable from the top nav bar: **Recognize**, **Register**,
**Manage**, **Help**.

### 1. Register a person

1. Go to the **Register** page.
2. Enter a 9-digit ID and a display name.
3. Click **START CAPTURE**.
4. Follow the 5 on-screen poses (straight, left, right, tilt up, tilt
   down) — hold each pose steady until that step's progress bar fills.
5. When all 5 steps finish, the person is saved. Nothing is saved if you
   click **CANCEL** partway through.

### 2. Recognize someone

You don't need an ESP32 connected to try this — the app runs fine in
**test mode** if no serial device is found.

1. Go to the **Recognize** page.
2. Click **TEST SCAN** (this simulates the ESP32 sending its trigger
   signal) and look at the camera.
3. You should see `VERIFIED` with the matched name/ID, or `UNKNOWN FACE`
   if nobody matches. Watch the console for `[FACE_REC]` / `[WORKER]` log
   lines showing the detection/match details.
4. With a real ESP32 wired up (see [ESP32 firmware](#esp32-firmware)),
   pressing its physical button does the same thing automatically —
   `SERIAL` shows `LIVE` (green) instead of `TEST MODE` (orange) once
   it's connected.

### 3. Point the app at the right serial port

If the ESP32 shows up on a different port than expected (common on
Linux/Jetson — e.g. `/dev/ttyACM0` instead of `/dev/ttyUSB0`, depending on
the board/USB-serial chip), you don't need to edit code or restart:

1. On the **Recognize** page, find the **SERIAL PORT** box.
2. Click **↻** to rescan currently-plugged-in ports, or just type the port
   name directly.
3. Click **RECONNECT**.

To set a fixed default instead (so you don't have to do this every
launch), set the `SERIAL_PORT` environment variable before starting the
app — see [Configuration reference](#configuration-reference).

### 4. Test the ESP32 independently of the camera

Use **MANUAL SEND TO ESP32** on the Recognize page to write any 9-digit ID
straight to the serial port, skipping face detection entirely — useful for
checking the ESP32's LED/relay wiring and firmware logic on its own.

### 5. Manage enrolled people

Go to the **Manage** page to see everyone currently enrolled (ID, name,
sample count, registration date). Select one or more rows and click
**DELETE SELECTED** to remove them. **REFRESH** re-reads the list.

### 6. In-app help

The **Help** page has this same usage guide plus troubleshooting notes,
available without leaving the app.

---

## ESP32 firmware

`commu_function/esp32.cpp` implements the other end of the protocol:

- Debounced trigger button on `TRIGGER_PIN` → sends `"1\n"` to the Jetson.
- Waits for a 9-digit ID reply (`"000000000"` means unknown/no match).
- Drives a green/red LED and a relay (e.g. a door strike) based on the result.
- Has a retrigger lockout and a response timeout so a stuck button or a
  slow Jetson can't pile up multiple in-flight scans.

To set it up:

1. Flash `esp32.cpp` with the Arduino IDE or PlatformIO.
2. Wire up a push button to GND (using the internal pull-up), plus
   LEDs/relay, per the pin constants at the top of the file.
3. Connect the ESP32 to the machine running `main.py` over USB.
4. Point the app at the right port — see
   [step 3 above](#3-point-the-app-at-the-right-serial-port).

---

## Configuration reference

Every setting below can be set as an environment variable before starting
the app (e.g. `SERIAL_PORT=/dev/ttyACM0 python main.py`), or edited
directly at the top of `vector_face_recognition/main.py`.

| Variable | Default | Meaning |
|---|---|---|
| `SERIAL_PORT` | `/dev/ttyUSB0` | ESP32 serial port. Also changeable live from the Recognize page. |
| `SERIAL_BAUD` | `115200` | Must match the ESP32 firmware's `Serial.begin(...)`. |
| `CAMERA_INDEX` | `0` | Which webcam to open. |
| `FACE_TOLERANCE` | `0.50` | Match distance threshold — lower = stricter. |
| `START_FULLSCREEN` | `0` (`1` in Docker) | `1` for a kiosk/door deployment; `0` keeps a normal window (so a dev console stays visible). |
| `FACE_DB_PATH` | `vector_face_recognition/encodings.pkl` | Where enrolled face data is stored — point this at a mounted volume in Docker. |

---

## Troubleshooting

- **"Recognize feels slow"**: on a CPU without a GPU-accelerated dlib
  build, each scan does real detection + encoding work per frame. Reduce
  `SCAN_FRAMES` or raise `EARLY_EXIT_MARGIN` at the top of `main.py` to
  trade accuracy for speed.
- **"Registration feels slow"**: the guided-capture loop only runs the
  expensive face-encoding step once per accepted sample (throttled by
  `CAPTURE_INTERVAL`), not on every video frame — if it's still slow,
  that's most likely this machine's CPU being the bottleneck, not a bug.
- **"Nobody matches" / distances always high**: lower `FACE_TOLERANCE` for
  stricter matching, or raise it if real matches are being rejected. Also
  make sure registration and recognition lighting are reasonably similar.
- **Queue depth keeps growing**: the ESP32 (or repeated TEST SCAN clicks)
  is sending triggers faster than scans complete. On the ESP32 side,
  `RETRIGGER_LOCKOUT_MS` enforces a minimum gap between triggers — don't
  lower it below how long a scan actually takes.
- **`dlib`/`face_recognition` won't install**: see the platform-specific
  [Setup](#setup-windows) sections above — this is almost always a missing
  prebuilt wheel for your Python version/OS, not a real build error.

## Notes on the face database

Enrolling someone through the Register page writes to
`vector_face_recognition/encodings.pkl` (or wherever `FACE_DB_PATH`
points). That file contains other people's biometric data, so it's
git-ignored — don't commit it. Each machine/deployment running this app
builds up its own local database.
