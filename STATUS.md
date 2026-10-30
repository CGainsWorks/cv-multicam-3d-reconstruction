# Status

As of 2026-10-16. Everything below that touches a camera is **simulated**: no
camera, lens, trigger line or turntable has ever been connected to this code.

## Complete

- **Calibration, reconstruction and pose pipeline** on images from the in-repo
  rasteriser: Zhang intrinsics, rig extrinsic bundle adjustment, multi-motion
  sparse reconstruction, plane-sweep MVS with Open3D fusion, generalised-camera
  PnP. CI runs it end to end on a miniature capture (`scripts/demo.py`) and
  asserts its regression guards.
- **GenICam / GigE Vision capture path, simulated.** `multicam.capture.genicam`
  drives cameras through Aravis 0.8.36 (built from its checksummed release
  tarball by `scripts/genicam/setup_aravis.sh`) and PyGObject 3.48.2: GenICam
  feature access over GVCP (`Width`, `Height`, `PixelFormat`, `TriggerSelector`,
  `TriggerMode`, `TriggerSource`, exposure, gain, the `*Auto` features, PTP where
  offered), `AcquisitionStart`, GVSP streaming, device timestamps, a software
  trigger plus explicit control-channel take and release. CI runs it against
  three of Aravis's own `arv-fake-gv-camera` simulators on the runner's loopback
  interface.
- **Timestamp synchronisation check, simulated.** Every frame set's exposure
  timestamps are compared and a set spread beyond the tolerance, or with a
  frame lost in transit, is dropped and re-triggered at the same station. CI
  provokes a late trigger and checks the set is rejected.
- **Captured frames fed into the pipeline, simulated.** `scripts/genicam_demo.py`
  renders a three-camera miniature capture, serves it from three
  `arv-scene-gv-camera` processes (Aravis's fake GigE Vision camera with its
  test pattern replaced by the rendered frames, `capture/fakecam/`), captures
  every frame set back through the GenICam path, requires every frame to be
  bit-identical to what was rendered, then runs calibration, reconstruction and
  pose on the captured frames against the demo's guards. One guard differs: the
  sparse cloud's median distance to CAD is held to 2.0 mm rather than 1.5 mm,
  because three cameras triangulate fewer points than four (1.60 mm on the
  first CI run). CI runs it.
- **Learned refiner and its ablation**, with the result that the network does
  not ship. The published numbers are in `results/published/`.
- **COLMAP reference run** (local, recorded in `results/published/colmap.json`).
- **Tests:** 152, of which 6 need Aravis and the loopback cameras. CI installs
  both and sets `MULTICAM_REQUIRE_GENICAM=1`, so those 6 fail rather than skip
  there; elsewhere they skip.

## In progress

Nothing.

## Not started

- **Any real camera.** The first things to check on a real GigE Vision rig:
  that `TriggerSource = Line1` with a wired common trigger gives exposures
  within the 0.2 ms default tolerance; that PTP (`GevIEEE1588` or `PtpEnable`)
  is available and locked, since without it device timestamps from different
  cameras are not comparable; packet size, receive buffers and inter-packet
  delay on a switch shared by four cameras; and whether every feature the
  backend sets exists under the same SFNC name on that vendor's cameras.
- **Turntable motion control.** `multicam.capture.Turntable.move_to` is an
  interface only; the simulated stage just tells the simulated cameras which
  frame to serve.
- **A fairer COLMAP comparison**: masking the static background, or giving
  COLMAP the calibrated rig. Neither was tried.

## Limitations

- **The ablation is not run in CI.** The trained network is published as
  release `refiner-v1` (SHA-256 `c9b849fd…a8e350b`, with the same history and
  configuration as `results/published/refiner_training.json`), so the ablation
  can be rerun locally without retraining. To retrain it instead, from a clean
  clone:

  ```bash
  make data                       # render the capture, about 20 minutes
  make calibrate reconstruct pose
  python scripts/train_refiner.py --data data/turntable --out results --models models \
      --epochs 40 --batch-size 12 --samples-per-frame 14 --seed 0
  python scripts/ablate.py --data data/turntable --out results
  ```

  These are the settings recorded in `results/published/refiner_training.json`.
  That training run took 1820 s (about 30 minutes) on a CUDA GPU; it has not
  been timed on a CPU. A retrained network will not match the published
  numbers to the last digit.
- **All images are synthetic**, from the in-repo rasteriser; the surface
  texture is the most flattering assumption in them. See
  [docs/limitations.md](docs/limitations.md).
- **Simulated synchronisation says nothing about a real rig.** Aravis's fake
  cameras share one host clock and expose on a 25 Hz grid of it, so their
  timestamp spreads are a property of the simulator. The simulated runs use a
  5 ms tolerance for that reason; 0.2 ms is the default for a hardware trigger.
- **The GenICam path needs Linux.** Aravis is built from source and the
  simulated cameras need loopback addresses `127.0.0.1` to `127.0.0.3`; on
  other systems `docker/genicam.Dockerfile` runs it with `--network none`.
- **"145 ms per station on cuda"** in the ablation table is from the local
  workstation's GPU, not from CI, which has no GPU.
- **The COLMAP comparison is one run with default settings.** It predates the
  fixed seed. An earlier run with a different registration count was not
  recorded, so the run-to-run spread is not measured.
- **One seed, one rig, one part**; no error bars on any number.
