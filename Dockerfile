# Jetson Nano deployment image for the Face ID System.
#
# Base image: matches JetPack 4.6.x (L4T R32.7.x), the standard/latest
# JetPack release for the original Jetson Nano. If your Jetson is on a
# different version, check it with:
#     cat /etc/nv_tegra_release
# and change the tag below to match (e.g. r32.6.1, r32.5.0, ...) — an
# l4t-base image needs to match the host's L4T release for device/driver
# passthrough (camera, display) to work correctly.
FROM nvcr.io/nvidia/l4t-base:r32.7.1

ENV DEBIAN_FRONTEND=noninteractive

# System packages:
#   python3-tk        - Tkinter, required by main.py's GUI
#   build-essential,
#   cmake, git         - needed to build dlib from source (no aarch64
#                        wheel on PyPI for it)
#   libopenblas-dev,
#   liblapack-dev      - linear algebra backend dlib/numpy use, matters a
#                        lot for face-encoding speed on a Nano's CPU
#   libjpeg-dev,
#   libpng-dev         - image codec support for opencv-python-headless
#   libsm6, libxext6,
#   libxrender1        - runtime libs opencv-python-headless still links
#                        against even without any GUI functions
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-tk \
        build-essential cmake git pkg-config \
        libopenblas-dev liblapack-dev \
        libjpeg-dev libpng-dev \
        libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# dlib's build is the slow, memory-hungry part (expect 45-90 minutes on a
# Nano; make sure you have swap configured — see README.md). It's kept in
# its own layer so editing application code later doesn't force a rebuild.
# CMAKE_BUILD_PARALLEL_LEVEL caps parallel compile jobs to reduce the peak
# memory the build needs; raise it if you have swap/RAM to spare.
ENV CMAKE_BUILD_PARALLEL_LEVEL=2
RUN python3 -m pip install --no-cache-dir dlib==20.0.1

# opencv-python-headless (not opencv-contrib-python): this app never calls
# cv2.imshow/waitKey, only capture/resize/color-convert/draw, and the
# headless build both has aarch64 wheels available and avoids pulling in
# a GUI stack we don't use.
COPY docker-requirements.txt .
RUN python3 -m pip install --no-cache-dir -r docker-requirements.txt

COPY vector_face_recognition/ ./vector_face_recognition/

# Defaults — override at `docker run` time with -e, or from the app's
# Recognize page (serial port can be changed live without restarting).
ENV SERIAL_PORT=/dev/ttyUSB0 \
    SERIAL_BAUD=115200 \
    CAMERA_INDEX=0 \
    START_FULLSCREEN=1 \
    FACE_DB_PATH=/data/encodings.pkl

WORKDIR /app/vector_face_recognition
CMD ["python3", "main.py"]
