# cv-multicam-3d-reconstruction

> ### Clean-room reimplementation
> This repository is an **independent reimplementation of a problem class**, built from
> scratch on personal time using public and synthetic data and open-source libraries. It
> is **not** derived from and contains no material belonging to any employer or client:
> no employer source code, schematics, calibration data, client data or internal
> documentation. It exists to demonstrate the architecture and engineering approach only.

A calibrated four-camera rig that reconstructs a machined part and finds its
6-DoF pose. Zhang calibration and a rig extrinsic bundle adjustment written from
the equations, multi-view triangulation, a sparse bundle adjuster with a Schur
complement over the landmarks, plane-sweep multi-view stereo, Open3D fusion and
meshing, generalised-camera PnP across the whole rig, and a learned
render-and-compare pose refiner that is ablated against the classical one.

**There is no hardware behind this.** Every image is rendered by a software
rasteriser written for the purpose, from a seed, in about twenty minutes. That
is a real limitation and it is the first entry in
[docs/limitations.md](docs/limitations.md). It is also what makes the ground
truth exact, which is the only reason a calibration can be scored against the
rig it was trying to recover rather than merely against its own residual.

<!-- table:headline -->
_Not generated yet: run `make experiments`._
<!-- /table:headline -->

### What it looks like

| | |
| --- | --- |
| ![summary](docs/plots/summary.png) | **Error through the pipeline, and what the ablation decided.** Left: the error at each stage, from the rig's extrinsics through the sparse and dense reconstructions to the final pose. Middle: how much of the part's surface each stage actually covers, which is the number that accuracy alone hides. Right: the ablation the whole repository is built around. |
| ![reconstruction](docs/plots/reconstruction.png) | **The part is whatever moves.** Right-hand panel: every rigid motion the cross-station alignment found. The static group is the background, and it is the *larger* one, so selecting by size reconstructs the room. Selecting the group that moves reconstructs the part, and nothing linked a background point across two stations. |

---

## What is here

| | |
| --- | --- |
| **Rig** | Four cameras on a 76-degree arc, 0.6 m from a turntable, 32-degree lens, 800 x 600. Each has its own focal length, principal point and radial-tangential distortion; the mounts are perturbed from nominal, which is what the extrinsic calibration has to find |
| **Calibration** | ChArUco detection, normalised-DLT homographies, Zhang's closed form for the absolute conic, then a joint Levenberg-Marquardt over nine intrinsics and every board pose. Rig extrinsics initialised over a maximum-weight spanning tree of the camera co-observation graph, then bundle adjusted. Cross-checked against `cv2.calibrateCamera` on identical detections |
| **Front end** | SIFT with the RootSIFT normalisation, ratio test, mutual consistency, and an epipolar gate that uses the calibrated rig to throw out matches no geometry can explain |
| **Sparse reconstruction** | Per-station multi-view triangulation, cross-station alignment by **multi-motion RANSAC** that keeps the group which moves and discards the static background however large it is, global track building, then bundle adjustment with a Schur complement over the points and a Huber kernel |
| **Dense** | Plane-sweep MVS bounded by the sparse cloud's own volume, ZNCC by box filters, sub-pixel parabola in inverse depth, N-view cross check, Open3D voxel fusion, normal orientation from the viewpoints, screened Poisson meshing with density trimming |
| **Pose** | Appearance model built from the reconstruction and filtered to the CAD surface; generalised-camera PnP under RANSAC over all four cameras at once; Levenberg-Marquardt refinement with analytic Jacobians and a first-order covariance |
| **Learned refiner** | A 0.6 M-parameter render-and-compare CNN over per-camera crops, predicting a twist applied with the same retraction the classical refiner uses. It is given the CAD model's *geometry* and never its appearance |
| **Reference** | COLMAP, through `pycolmap`, on identical images. It shares no code with any of the above |
| **Evaluation** | Reprojection residuals, point-to-surface distance to CAD, completeness, chamfer, ADD and ADD-S with recall at several thresholds, and per-class labelling so the fixture is not scored as the part |
| **Simulator** | A z-buffered rasteriser with near-plane clipping, perspective-correct interpolation and physically correct lens distortion, plus the rig, the scene and the capture planner |
| **Tests** | 131, including every Jacobian against finite differences, the renderer against the projection equations, the bundle adjuster against a known answer, and a firewall test that fails if any pipeline module so much as names a ground-truth accessor |

---

## Run it

```bash
./scripts/run_demo.sh
```

One command on a clean machine. It builds a virtual environment if there is not
one, renders a miniature capture, calibrates the rig from it, reconstructs the
part, estimates 6-DoF pose on held-out stations, and checks every number against
the ground truth it generated. About four minutes. Nothing is downloaded except
Python packages.

`docker compose -f docker/docker-compose.yml run --rm multicam` does the same in
a container.

```bash
make check                 # lint plus the full test suite, exactly what CI runs
make data                  # render the full capture, about 20 minutes
make experiments           # every number below, about 40 minutes after that
make plots tables docs-check
```

**The demo's numbers are worse than the ones below, and deliberately so.** Its
capture is a tenth the size, which calibrates coarsely — about 13 mm of rig
extrinsic error against 2 mm here — and that error passes straight through to
the pose. `make experiments` is what reproduces this page.

---

## Results

Full tables, every metric, every setting: **[docs/results.md](docs/results.md)**.

### Calibration

Per-camera intrinsics from that camera's own board sweep; rig extrinsics from 26
shared stations. The residual is what you would have on real hardware. The error
against truth is what you would not, and the gap between them is the point: a
calibration can have a beautiful residual and be wrong.

<!-- table:calibration -->
_Not generated yet: run `make experiments`._
<!-- /table:calibration -->

![calibration](docs/plots/calibration.png)

### Reconstruction

Against the CAD model, in the ground-truth frame, with **no fitting at all** —
the bundle adjustment holds the reference station fixed, so the reconstruction
frame is the rig frame at station 0 by construction and ground truth gives the
transform in closed form. ICP is run afterwards only as a diagnostic, and how
far it moves is reported in [docs/results.md](docs/results.md).

<!-- table:reconstruction -->
_Not generated yet: run `make experiments`._
<!-- /table:reconstruction -->

![dense](docs/plots/dense.png)

### 6-DoF pose

Held-out stations, arbitrary orientations drawn from the full rotation group,
against an appearance model built from the turntable scan.

<!-- table:pose -->
_Not generated yet: run `make experiments`._
<!-- /table:pose -->

![pose](docs/plots/pose.png)

---

## The ablation

The brief this was built to asks for one thing above the rest: does the learned
refiner earn its place, and if it does not, say so.

### Does it ship?

<!-- table:ablation -->
_Not generated yet: run `make experiments`._
<!-- /table:ablation -->

### The same question, controlled

Comparing a classical estimate against a refined classical estimate cannot
separate "the network is a better refiner" from "the classical estimate was
already at the noise floor and there was nothing left to correct". So both
refiners are started from the **same** deliberately perturbed pose, across four
magnitude bands.

<!-- table:head_to_head -->
_Not generated yet: run `make experiments`._
<!-- /table:head_to_head -->

![ablation](docs/plots/ablation.png)

The remaining two experiments — how the result moves with the number of
refinement iterations, and what happens as the classical stage is starved of
correspondences — are in [docs/results.md](docs/results.md).

---

## COLMAP, as an independent reference

COLMAP is the reference implementation for this problem and shares no code with
anything in `src/multicam/reconstruction`. Agreement is evidence that the
geometry is right; disagreement means one of the two has a bug.

The comparison is **not symmetric**, and the report says so. COLMAP solves for
every camera pose from scratch, which is a strictly harder problem than ours —
we are handed the calibrated rig — and it recovers geometry only up to a
similarity, so its result has to be scaled before it can be compared to a metric
one. That scale factor is reported, because it is the single number that says
what the rig calibration was worth.

<!-- table:colmap -->
_Not generated yet: run `make experiments`._
<!-- /table:colmap -->

---

## How it works

**[docs/design.md](docs/design.md)** is the architecture: the frame conventions
and the bug they prevent, why the optimiser works on a manifold and what broke
before it did, the Schur complement, why multi-motion alignment is necessary
rather than clever, and what the learned refiner is and is not allowed to see.

**[docs/calibration.md](docs/calibration.md)** is the calibration in detail.

A few things worth pulling out here, because they are the ones that cost real
time:

- **The optimiser parameterised poses by their own logarithm and stepped with
  `x + dx`.** That is wrong by the SE(3) right Jacobian, which is near the
  identity only for small poses. The symptom was not a crash: it was a solver
  that took one step, damped itself to a halt, reported "converged", and left
  intrinsics 36 px out in focal length.
- **The renderer applied lens distortion backwards.** A render warped the wrong
  way still looks like a plausible lens and still calibrates to a self-consistent
  model. Only a comparison against the projection equations shows it up; it put
  2 px of systematic error into every detected corner.
- **The dense stage's ambiguity test rejected 99.98 per cent of pixels**, because
  it compared the winning depth hypothesis against its own immediate neighbours
  in a cost volume that is smooth by construction.
- **The part had an accidental *appearance* symmetry** — not a geometric one —
  from the texture projection, which made the pose estimator land on the wrong
  limb on six of thirty-nine frames at exactly 90 degrees, each with a healthy
  inlier count and a 0.4 px residual. [Section 3a of the
  limitations](docs/limitations.md#3a-one-near-symmetry-was-an-artefact-and-it-cost-six-frames).

---

## Limitations

**[docs/limitations.md](docs/limitations.md)**, in full and in its own document.
The short version:

1. It is all synthetic, and the surface texture is the most flattering
   assumption in it. A real machined part is often shiny and nearly featureless.
2. The synchronised capture code has never been run, and everything here assumes
   the four exposures are the same instant.
3. The part is only weakly asymmetric, so ADD and ADD-S are both reported
   everywhere.
4. The turntable sees the part from one side, which caps completeness and means
   the Poisson mesh invents the underside.
5. The evaluation uses ground-truth segmentation to separate the part from the
   fixture it is bolted to, deliberately, labelled, and confined to
   `multicam.evaluation` by a test.
6. One seed, one rig, one part. The numbers have no error bars.

---

## Repository layout

```
src/multicam/
  geometry/        SO(3), SE(3), the pinhole-plus-radtan camera, triangulation
  sim/             meshes, procedural textures, the rasteriser, the rig, capture planning
  calibration/     ChArUco target, per-camera intrinsics, rig extrinsics
  reconstruction/  features, stations, multi-motion alignment, bundle adjustment
  fusion/          plane-sweep MVS, Open3D fusion and meshing
  pose/            generalised PnP, the classical refiner, the learned refiner
  evaluation/      pose and surface metrics, CAD registration, point labelling
  io/              the capture layout, the COLMAP bridge
  optim.py         manifold Levenberg-Marquardt with a robust kernel
capture/           synchronised capture for real hardware; never run, and says so
scripts/           one per stage, plus the demo, the plots and the table generator
tests/             131 tests
docs/              design, calibration, results, limitations
data/              rendered captures and DATASET_CARD.md; nothing committed
```

---

## Licence

MIT, [LICENSE](LICENSE). Everything in this repository — the code, the rendered
images, the object geometry, the textures — was generated by it. No third-party
dataset is downloaded, used or redistributed; see
[data/DATASET_CARD.md](data/DATASET_CARD.md).
