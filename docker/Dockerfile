# A container that runs the whole pipeline with nothing to download.
#
# Every image this repository uses is rendered from a seed, so there is no
# dataset step and no volume to mount: `docker compose run --rm multicam`
# renders a miniature capture, calibrates, reconstructs and estimates pose.
#
# CPU only.  Torch comes from the CPU wheel index deliberately -- the default
# PyPI wheel carries the CUDA runtime, which is well over a gigabyte in the
# image and useless without `--gpus`.  Nothing here needs a GPU: the only
# trained model is small enough to train on CPU, which is what CI does.

FROM python:3.12-slim-bookworm

# OpenCV and Open3D need a handful of system libraries even in their headless
# builds.  libgomp is Open3D's OpenMP runtime; the rest are image and GL stubs.
# libegl1 is the one that is easy to miss: Open3D links against EGL even in a
# headless build, and without it `import open3d` fails with
# "libEGL.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libusb-1.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, in their own layer, so editing the source does not
# reinstall half a gigabyte of wheels.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch \
         --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=4

# Fail the build if the package does not even import, rather than at run time.
RUN python -c "import multicam, cv2, open3d, torch; print('multicam', multicam.__version__)"

CMD ["python", "scripts/demo.py", "--out", "/app/results/demo.json"]
