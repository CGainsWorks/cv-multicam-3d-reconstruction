# Design

How the pipeline is put together, and why each piece is the way it is. The
decisions worth defending are the ones where an obvious alternative exists and
was rejected.

---

## The shape of it

```
                 images (and nothing else)
                            |
   per-camera intrinsics ---+--- rig extrinsics        multicam.calibration
                            |
            per-station multi-view triangulation       multicam.reconstruction
                            |
          cross-station alignment (multi-motion)
                            |
       bundle adjustment (Schur complement)
                            |
            +---------------+----------------+
            |                                |
   plane-sweep MVS                   appearance model
   Open3D fusion                             |
   Poisson meshing              generalised PnP under RANSAC
   multicam.fusion                           |
                                   LM refinement (classical)
                                             |
                                   learned refinement (ablated)
                                        multicam.pose
                            |
                 evaluation against CAD      multicam.evaluation
```

Ground truth enters only at the bottom. That is enforced by a test, not by
discipline.

## Frames, once

Getting this wrong is the most expensive class of bug in the repository, so the
conventions are fixed in one place ([`src/multicam/types.py`](../src/multicam/types.py))
and never varied.

- `T_a_b` maps a point expressed in frame `b` into frame `a`. So
  `T_cam_world @ X_world == X_cam`, and `T_a_c == T_a_b @ T_b_c` reads left to
  right with the inner frames cancelling.
- The **rig frame is the reference camera's frame**, so `T_cam_rig[0]` is the
  identity by construction. Anchoring to a physical camera rather than an
  abstract frame removes a six-parameter gauge freedom from every solve.
- The **object frame is the rig frame at the reference station**. That choice
  costs nothing — the reconstruction is metric because the calibration is — and
  it means no ground truth is needed to define it.
- Perturbations are **right** perturbations: `T <- T exp(delta)`. Everywhere.

The one place a convention has to be translated is
[`multicam/io/colmap.py`](../src/multicam/io/colmap.py), because COLMAP orders a
twist rotation-first and this repository orders it translation-first. It happens
there and nowhere else.

### The bug this prevents, and did not

`T_a_b` from the cross-station alignment maps station `b` into station `a`. The
bundle adjuster wants the opposite direction. The first version chained them
without inverting, which produced a reconstruction that was internally
consistent — 0.29 px reprojection residual — and 35 mm from the CAD surface.
A low residual is not evidence of a correct frame; only a comparison against
something external is. The variable is now called `T_object_rig`, the inverse is
taken once and named, and the comment says which direction it maps.

## Optimisation on a manifold

[`multicam/optim.py`](../src/multicam/optim.py) takes an opaque state, a
Jacobian with respect to a *local* perturbation, and a `retract` callback.
Poses retract as `T <- T exp(delta)`.

**The rejected alternative** is to parameterise each pose by the six numbers of
its own logarithm and let the solver do `x + delta`. That is wrong, because
`exp(x + dx)` is not `exp(x) exp(dx)` — they differ by the right Jacobian
`Jr(x)`, which is close to the identity only for small `x`. Board poses half a
metre out with half a radian of rotation are not small.

The symptom was not a crash. It was a solver that took one step, damped itself
to a standstill, reported "converged: damping", and left a four-pixel residual
that looked almost plausible. The intrinsics it produced were 36 px out in focal
length with no distortion recovered at all.

A second, related bug lived in the same function: the Huber cost was being
compared across iterations using IRLS weights that had themselves changed, so
the accept-or-reject test compared two different objectives and rejected valid
steps. `huber_cost` is now evaluated explicitly and the weights only ever build
the normal equations.

With both fixed, the calibration agrees with `cv2.calibrateCamera` to 1e-7 px.

## Calibration

Textbook stages, written out rather than delegated, for one reason that is not
purity: the rig extrinsic solve refines extrinsics and board poses together and
needs the same projection Jacobian, the same parameterisation and the same
robust kernel as the per-camera solve. One implementation, checked against
finite differences, used in both places.

1. A homography per view, by normalised DLT.
2. Zhang's closed form for the image of the absolute conic, then `K` from it.
   This stage cannot see distortion at all, which is why its residual is around
   three pixels and the refined one is around a tenth.
3. Extrinsics per view from the homographies.
4. Joint Levenberg-Marquardt over the nine intrinsics and every board pose.

`cv2.calibrateCamera` is run alongside on identical detections as an
**independent referee**. It shares no code with any of the above, so agreement
is evidence and disagreement is a bug in one of the two. It is reported in
`results/calibration.json` and quoted in the README.

### Why ChArUco and not a checkerboard

Partial views are usable. Four cameras on a 76-degree arc cannot all see a whole
board at once, and a checkerboard detector returns all corners or none. ChArUco
identifies each corner through the markers around it, so a camera seeing half
the board contributes that half with its indices known.

### The spanning tree

Cameras 0 and 3 are 76 degrees apart and may never co-observe the board, so
there is no direct measurement of `T_c0_c3` at all. The extrinsics are
initialised by walking a **maximum-weight spanning tree** over the camera
co-observation graph, weighted by the number of shared views, so the chain runs
through the best-observed links rather than whichever is found first. Edges with
fewer than three shared views are dropped before the walk: a relative pose from
one or two views is dominated by the noise in a single board pose, and chaining
through it poisons everything downstream.

## Reconstruction

### A station at a time

A "station" is one turntable position. The rig is calibrated and rigid, so at a
station the relative poses of all four cameras are known exactly and the only
unknown is where each feature is. That makes a station a pure multi-view
triangulation with no pose estimation in it: fast, no gauge freedom, and it
produces a small metric point cloud that the next stage can treat as rigid-body
data.

The alternative — throw every image into one incremental SfM — works, and is
exactly what COLMAP does when it is run as the baseline. It also spends most of
its effort rediscovering the rig geometry that was already measured properly.

### The epipolar gate

Two cameras whose relative pose is known constrain a correspondence to a line.
A match off that line by more than the calibration residual is wrong however
good its descriptor distance was. On this data the gate removes about a sixth of
the matches that survive the ratio test, and nearly all of those are on
repetitive wall texture where the descriptor genuinely cannot tell one tile from
another and only the geometry can.

### Multi-motion alignment: the part is what moves

Between two stations there are **two** large, self-consistent groups of
correspondences. The floor and the walls insist the transform is the identity.
The part and the turntable insist it is a 20-degree rotation. A least-squares
fit lands between them and is wrong for everything.

A single RANSAC returns whichever group is larger — and which that is depends on
how much of the frame the part occupies, which is a property of the lens, not of
the algorithm. With a 58-degree lens the background won, and the pipeline
reconstructed the room: 1591 background points against 22 on the part.

So [`align.py`](../src/multicam/reconstruction/align.py) fits motions
*sequentially* — find the dominant one, remove its inliers, fit again — and
selects by the only criterion that does not require knowing the answer in
advance: **the part is the thing that moves**. The static group is discarded
however large it is. Every motion found, its size and its rotation, goes into
`results/sparse.json`, so the choice is visible rather than asserted.

Requiring a track to span two or more stations then removes the background
entirely, because nothing links its points across stations. The composition in
the results is the evidence: zero background points survive.

### Bundle adjustment

The normal equations have thousands of parameters and obvious structure: no
point sees another point, so the point-point block is block diagonal with 3x3
blocks. The Schur complement

```
S = H_ss - H_sp H_pp^-1 H_ps
```

reduces it to a dense matrix of six parameters per station — 102x102 for an
eighteen-station sweep — solved directly, with the point updates recovered by
back-substitution. Inverting `H_pp` is inverting a few thousand 3x3 matrices.

The reference station's pose is held fixed. Without it the reconstruction can
rotate and translate freely, `S` is singular to six decimal places, and the
damping quietly picks an arbitrary answer. Scale is *not* free: the rig
calibration fixed it, which is the whole reason for calibrating a rig.

## Dense reconstruction

Plane sweep, with three things that make it practical rather than merely
correct:

- **The sweep volume comes from the sparse cloud.** Each pixel's depth range is
  where its ray enters and leaves the sparse bounding box, not a global near and
  far plane. That is an order of magnitude fewer hypotheses and — more
  importantly — it is what removes the background, with no segmentation and no
  ground truth. A pixel whose ray never enters the box is not swept at all.
- **Uniform in inverse depth**, because stereo depth resolution is uniform in
  disparity.
- **ZNCC by box filters** over the whole image at once, so a hypothesis costs
  milliseconds rather than minutes.

### The ambiguity test that rejected everything

A winner that barely beats the runner-up along the same ray is a coin toss
between two textures rather than a measurement, so the margin is checked. The
first version compared the winner against the best *other* hypothesis, which
includes the hypotheses immediately either side of it — and a cost volume is
smooth, so those always score almost as well. It rejected every pixel in the
image: four valid depths out of twenty thousand, which reads exactly like a
matching failure.

The runner-up now has to be a different *peak*, outside an exclusion window
around the winner. The test that catches this compares a swept depth map against
the renderer's own depth buffer, which is the only kind of test that would.

## Pose

### Generalised PnP

The rig is calibrated and rigid, so the four cameras do not each estimate a pose
and then average. There is one unknown — the object's pose in the rig frame —
and all four cameras constrain it together. RANSAC draws its minimal sample from
a single camera, because P3P needs three correspondences in one central
projection, and then scores the hypothesis against every correspondence in every
camera. That is what makes it a generalised-camera RANSAC rather than four
independent ones.

It matters: the cameras span 76 degrees and between them see three sides of the
part, while any one camera sees one side and is correspondingly weak along its
viewing direction. There is a test that measures the joint solve against the
per-camera ones.

### The appearance model

A set of 3D points in the CAD frame, each carrying the descriptor observed
there, built from the sparse reconstruction — not from the CAD file, which has
geometry and no appearance. This is how a markerless model-based pose pipeline
is actually bootstrapped: scan the part once, register the scan to the drawing,
use the scan as the appearance model.

Points are kept only if they lie within a few millimetres of the CAD surface.
The turntable disc is rigidly attached during the scan and reconstructs
perfectly well, but it is not the part, and a pose estimator that locks onto the
fixture is not solving the problem.

### The learned refiner

Render-and-compare, in the DeepIM lineage, adapted to a calibrated multi-camera
rig: a shared CNN trunk over per-camera crops, a learned camera embedding, mean
pooling, and a six-vector twist applied as `T <- T exp(delta)` — the same
retraction the classical refiner uses, so the two are directly comparable.

**What it is given, and what it is not.** It sees the observed image, the
rendered silhouette of the CAD model at the current hypothesis, the rendered
depth, and the observed gradient magnitude. It is *not* given any rendering of
the part's appearance. The CAD model is a declared input to this problem, so its
geometry is fair; its appearance does not exist, and rendering the simulator's
own surface texture into the network's input would be handing it the answer key
and would not transfer to a real part for a moment.

It is small — about 0.6 M parameters — on purpose. The honest comparison is
against a classical refiner with six free parameters and perfect knowledge of
the geometry, and a network large enough to memorise this dataset would win the
benchmark and prove nothing.

The ablation that decides whether it ships is in
[results.md](results.md), and whatever it says is what the README says.

## Testing

131 tests, and the ones that earn their place are:

- **Every Jacobian against central finite differences.** A solver with a subtly
  wrong Jacobian converges on easy problems and fails on hard ones, which is
  very hard to debug later and very easy to catch here.
- **The renderer against the projection equations**, not against a saved image.
  That is what caught the distortion map being applied backwards — a render
  warped the wrong way still looks like a plausible lens and still calibrates to
  a self-consistent model.
- **The bundle adjuster against a problem with a known answer**, because a
  transposed block in a Schur complement still reduces the cost, just not as far.
- **The plane sweep against the renderer's depth buffer**, which is the only
  test that catches a warp that is off by a pixel.
- **The ground-truth firewall**, by parsing every pipeline module and failing if
  one names a ground-truth accessor. Every accuracy number rests on that
  separation, and it is a convention rather than a mechanism, so it is worth
  enforcing mechanically.
- **The documents against the results** (`make docs-check`), so a stale number
  in the README is a build failure rather than something a reader has to take on
  trust.
