# The GenICam check, locally, in the same environment CI builds.
#
#   docker build -f docker/genicam.Dockerfile -t multicam-genicam .
#   docker run --rm --network none --cap-add NET_ADMIN multicam-genicam
#
# SIMULATED.  There are no cameras in here: three of Aravis's fake GigE Vision
# cameras run on the container's loopback interface.  `--network none` is
# enough, because GVCP and GVSP never leave loopback; NET_ADMIN is needed only
# to give `lo` one address per camera and applies to the container's own
# network namespace.
#
# Ubuntu 24.04 because that is what the CI runner is, so the Aravis build, the
# PyGObject build and Python 3.12 all match it.

FROM ubuntu:24.04

COPY scripts/genicam/setup_aravis.sh /tmp/genicam/scripts/genicam/setup_aravis.sh
COPY capture/fakecam/arv-scene-gv-camera.c /tmp/genicam/capture/fakecam/arv-scene-gv-camera.c
RUN sh /tmp/genicam/scripts/genicam/setup_aravis.sh apt \
    && apt-get install -y -qq --no-install-recommends python3-dev python3-venv \
         libegl1 libgl1 libgomp1 libusb-1.0-0 \
    && sh /tmp/genicam/scripts/genicam/setup_aravis.sh build \
    && rm -rf /var/lib/apt/lists/* /tmp/genicam

ENV PATH=/opt/venv/bin:/opt/aravis/bin:$PATH \
    ARAVIS_PREFIX=/opt/aravis \
    LD_LIBRARY_PATH=/opt/aravis/lib \
    GI_TYPELIB_PATH=/opt/aravis/lib/girepository-1.0 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MULTICAM_REQUIRE_GENICAM=1

WORKDIR /app
COPY requirements.txt requirements-genicam.txt /app/
RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt -r requirements-genicam.txt

COPY . /app

CMD ["sh", "-c", "sh scripts/genicam/loopback_addresses.sh 3 && python -m pytest tests/ -q -m genicam && python scripts/genicam_demo.py --out /tmp/genicam.json"]
