# Calibration

The part of the pipeline where a quiet mistake is most expensive, because
everything downstream inherits it and almost nothing downstream reveals it.

---

## What gets calibrated, and in what order

Two stages, in the order you would do them in front of real hardware.

1. **Per-camera intrinsics**, one camera at a time, from that camera's own board
   sweep. Focal lengths, principal point, five radial-tangential distortion
   coefficients.
2. **Rig extrinsics**, from board stations that two or more cameras saw at the
   same instant. Five of the six degrees of freedom per camera are what the
   reconstruction needs; the sixth — the rig's own pose in the world — is never
   observable and is never estimated.

The reference camera's extrinsic is held at the identity throughout. That is not
a convenience: without it the problem has a six-parameter gauge freedom, the
normal equations are singular, and the covariance is meaningless.

## The target

A 7 x 5 ChArUco board, 25 mm squares, `DICT_4X4_50`, rendered by OpenCV's own
`cv2.aruco.CharucoBoard` and detected by OpenCV's own `CharucoDetector`.

**ChArUco rather than a plain checkerboard** for one reason that matters on a
rig: partial views are usable. Four cameras on a 76-degree arc cannot all see a
whole board at once, and a checkerboard detector returns all corners or none.
ChArUco identifies each corner through the markers around it, so a camera that
sees half the board contributes that half with its indices known, and the
extrinsic solve gets its co-observations without the board having to be waved
into a pose that suits every camera simultaneously.

**25 mm squares, not 40.** The board has to fit inside two or more frames at
once, and the frame is 380 mm across at the turntable. A 40 mm board overflows
it and *no station is usable* — which is how it failed the first time, with the
extrinsic solve reporting that cameras 1, 2 and 3 were not connected to the
reference by any edge.

### The board frame

The mesh is built directly in OpenCV's object-point frame: origin at the board's
first corner, `x` right along the printed sheet, `y` *down* it, `z` equal to
`x` cross `y`. That means `board.getChessboardCorners()` returns points in the
same frame as the mesh, so a pose recovered from detected corners is directly
comparable to the pose the renderer was given, with no compensating flip
anywhere. A test renders a board at a known pose and solves for it again.

The board's `+z` therefore points out of the *back* of the sheet. Aiming the
board at a camera with `look_at` renders the back of it: the geometry is right,
the image is mirrored, and ArUco decodes nothing. That cost an afternoon, and
the planner now reflects the target through the board centre with a comment
saying why.

## Planning the board poses

Both sessions plan every pose **geometrically before anything is rendered**.
Rendering is the expensive step and rejecting a station after paying for four
renders of it would triple the cost of a capture.

### Intrinsics

The target image position is drawn from a 3 x 3 grid over the frame rather than
uniformly. Uniform sampling plus a rejection test that demands the whole board
be inside concentrates every accepted pose in the middle of the image, which is
exactly where the distortion coefficients are least observable.

The depth range is **derived from the lens**, not hard-coded: the depths at
which the board covers between 28 and 80 per cent of the image width. A range
tuned for a 58-degree lens puts the board off the edge of the frame at every
pose on a 32-degree one, the planner rejects all of them, and the failure
surfaces as "only 0 intrinsic views planned" a long way from its cause.

Poses are also rejected if the board is seen nearly edge-on: the detector will
find it and the corner positions will be dominated by the resampling.

### Extrinsics

Board stations are placed in the shared working volume, oriented to face a point
on the camera arc, and accepted only if the whole board lands inside two or more
frames with a reasonable area. The extrinsic session is shot **in situ**, with
the cell's floor, walls and turntable present, because that is how you would
shoot it and because a board against a black background gives the ArUco adaptive
threshold a far easier problem than it will ever have on the shop floor.

## The solve

### Stage 1: a homography per view

Normalised DLT, with Hartley normalisation on both point sets. Without it the
algebraic cost mixes terms of order 1 with terms of order 10^5 and the smallest
singular vector is dominated by rounding.

### Stage 2: Zhang's closed form

Each view's homography gives two constraints on the image of the absolute conic
`B = K^-T K^-1`, from the orthonormality of the first two columns of the
rotation. Six unknowns, so three views suffice and more are better. `K` comes out
of `B` in closed form.

This stage **cannot see distortion at all**, which is why its residual is around
three pixels and the refined one is around a tenth. The skew it recovers is not
part of the camera model and is kept purely as a conditioning diagnostic: a well
posed set of views gives a skew of a few hundredths of a pixel, and a large one
means the board was never tilted enough.

### Stage 3: extrinsics from the homographies

The standard `lam K^-1 h_i` construction, with the two scale estimates combined
geometrically, the sign fixed by cheirality, and the result projected onto SO(3)
by SVD.

### Stage 4: joint refinement

Levenberg-Marquardt over the nine intrinsics and all six degrees of freedom of
every board pose at once, with analytic Jacobians and a Huber kernel at two
pixels.

The Jacobians are derived by hand in
[`geometry/camera.py`](../src/multicam/geometry/camera.py) and checked against
central finite differences in `tests/test_geometry.py`. That test is the reason
to trust the bundle adjuster: a solver with a subtly wrong Jacobian still
converges on easy problems and quietly fails on hard ones.

**Poses retract on the manifold.** See
[design.md](design.md#optimisation-on-a-manifold) for what happened when they
did not.

## Rig extrinsics

1. **Pairwise.** For every frame in which cameras `i` and `j` both see the board,
   `T_ci_cj = T_ci_board @ inv(T_cj_board)`. Each such frame is an independent
   estimate; they are averaged on the manifold, by summing the rotation matrices
   and projecting the sum back onto SO(3) with an SVD, which is the closed-form
   minimiser of the sum of squared Frobenius distances.
2. **A maximum-weight spanning tree** over the camera graph, weighted by the
   number of co-observations. Cameras 0 and 3 are 76 degrees apart and may never
   see the board together, so there is no direct `T_c0_c3` measurement at all and
   chaining through the middle cameras is the only way to initialise it.
   Weighting by co-observation count makes the chain run through the
   best-observed links rather than whichever is found first. Edges with fewer
   than three shared views are dropped entirely.
3. **A joint bundle adjustment** over every extrinsic and every board pose, with
   the reference camera fixed. This is where the reported residuals come from.

## How it is checked

Three independent checks, because a calibration that only checks itself is not
checked.

**Against OpenCV.** `cv2.calibrateCamera` runs on identical detections. It
shares no code with any of the above. The disagreement in focal length is
recorded in `results/calibration.json`; on this data it is around 1e-7 px, which
is agreement to the last bit either implementation can represent.

**Against the ground-truth rig.** Because the capture is synthetic, the rig that
generated it is on disk. The residual and the error against truth are both
reported, and the gap between them is the whole point: a calibration can have a
beautiful residual and be wrong. That gap is the thing you can never see on real
hardware.

**Against finite differences.** Every Jacobian, every time the tests run.

## What a bad calibration looks like downstream

Worth knowing, because the symptoms do not look like calibration.

A rig whose baselines are wrong by a fraction `e` produces a reconstruction
whose **scale** is wrong by `e`, and a pose that lands `e` times the stand-off
away from the truth. In the small capture that `scripts/demo.py` renders, the
extrinsics come out about 13 mm wrong on a 750 mm baseline — 1.7 per cent — and
the pose error is about 14 mm at a 0.66 m stand-off, which is 1.7 per cent of
0.66 m to within the noise.

Nothing in the pose estimator's own output hints at this. Its reprojection
residual is 0.4 px, its inlier ratio is healthy, and it is confidently wrong by
a fixed fraction. The only way to catch it is to calibrate well, or to check the
scale against something you measured with a ruler.
