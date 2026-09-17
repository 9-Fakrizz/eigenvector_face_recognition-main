#!/usr/bin/env bash
# Run the Face ID System container on the Jetson.
#
# Usage:
#   ./docker-run.sh                          # defaults: /dev/ttyUSB0, camera 0
#   SERIAL_PORT=/dev/ttyACM0 ./docker-run.sh # ESP32 enumerated on a different port
#
# You can also skip picking a port here entirely and just leave the ESP32
# unplugged / on the wrong port — the app starts in TEST MODE either way,
# and the Recognize page's SERIAL PORT box lets you point it at the right
# device and hit RECONNECT once the container is already running, with
# --privileged (see below) giving it access to try any of them.
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-faceid-system}"
SERIAL_PORT="${SERIAL_PORT:-/dev/ttyUSB0}"
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"
DATA_DIR="${DATA_DIR:-$(pwd)/data}"

mkdir -p "$DATA_DIR"

# Let the container's X client connect to this user's X server. Only needs
# doing once per login session, but it's harmless to repeat.
xhost +local:docker >/dev/null 2>&1 || true

docker run -it --rm \
    --name faceid-system \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e SERIAL_PORT="$SERIAL_PORT" \
    -e SERIAL_BAUD="${SERIAL_BAUD:-115200}" \
    -e CAMERA_INDEX="${CAMERA_INDEX:-0}" \
    -e START_FULLSCREEN="${START_FULLSCREEN:-1}" \
    -e FACE_DB_PATH=/data/encodings.pkl \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v "$DATA_DIR:/data" \
    --device="$CAMERA_DEVICE" \
    --privileged \
    "$IMAGE_NAME"

# --privileged: gives the container access to every device on the Jetson
# (all /dev/ttyUSB*, /dev/ttyACM*, etc.), so the in-app RECONNECT feature
# can actually switch to whichever port the ESP32 shows up on without you
# needing to know it in advance or restart the container. Once you know
# the real port for good, you can tighten this to a specific
# `--device=/dev/ttyACM0` instead for a production deployment.
