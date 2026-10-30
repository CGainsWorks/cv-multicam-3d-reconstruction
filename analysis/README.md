# Analysis

Working notes on the things that were not obvious, kept because the reasoning
is worth more than the conclusion in most of these cases.

The finished numbers live in [docs/results.md](../docs/results.md); the record
files they are drawn from are in `results/published/`.

---

## Getting the pose refiner to learn anything at all

The learned refiner was the single most time-consuming part of this repository,
and almost none of that time went on the network. It went on four
configurations that all produced the same symptom: a training loss that did not
move from the identity baseline, which reads exactly like "this architecture
cannot learn this task" and was not that.

The diagnostic that made it tractable was an **overfit test**: take sixteen
samples, train on those sixteen for three hundred steps, and see whether the
loss goes to zero. It does — 3.72 mm to 0.65 mm — which rules out the inputs
being uninformative, the loss being wrong, the targets being misaligned with the
inputs, and the gradients not flowing. Everything after that was about
optimisation and parameterisation rather than architecture.

### 1. The learning rate was an order of magnitude too low

3e-4 is the reflex for AdamW on a vision model and it was wrong here. The head
is zero-initialised and the output is multiplied by `TRANSLATION_SCALE` and
`ROTATION_SCALE` before it is exponentiated, so the gradients arriving at the
trunk start around 2e-3 and the gradient clip at 1.0 never engages at all.

Worth noticing in general: **if your gradient clip never fires, your learning
rate is probably not what you think it is.**

### 2. A third of the training set carried no signal

The perturbation magnitude was drawn log-uniformly from 0.2 mm to 15 mm. At this
stand-off a 0.2 mm perturbation moves the part's silhouette by about a fifth of
a pixel: the rendered silhouette and the observed one are the same picture, and
the target is a twist of essentially zero.

Filling a large fraction of the training set with samples whose correct answer
is "do nothing" teaches the refiner that doing nothing is usually right, which
is precisely the behaviour it was stuck in. The lower bound is now 1 mm, which
is a quarter of the classical estimator's median error, so the range still
brackets what the refiner is actually handed.

### 3. The predicted twist was in a frame that made the task unlearnable

This is the interesting one.

The refiner predicts a correction to `T_rig_object`. The obvious composition is
`T_new = T exp(delta)`, which puts the twist in the **object's own frame**. The
network, however, sees *camera crops*. The map from "the rendered silhouette is
three pixels left of the observed one in camera 0" to an object-frame correction
depends on the object's current orientation — and the network is never given
that.

On a turntable sequence, where the object only ever rotates about one axis, this
would half work. The pose session draws orientations from the full rotation
group, so there is no consistent mapping to learn and the best available
prediction genuinely is zero.

`exp(delta) T` — the rig frame — fixes the orientation dependence and
introduces a different problem: it rotates the object about the *rig origin*,
0.6 m away, so a one-degree rotation drags the part about 10 mm sideways, and
rotation and translation become coupled through a lever arm that changes with
where the part is standing.

What works is a frame that is **centred on the object and oriented like the
rig**:

```
T_new = A exp(delta) A^-1 T,     A = translation(T @ centroid)
```

The map from image displacement to twist is then fixed per camera, which is
exactly what the learned camera embedding is able to encode. There are two tests
on this: one that the NumPy composition used at inference and the torch one used
in training agree to machine precision, and one that a pure rotation leaves the
object's centre where it was.

### 4. Optimiser steps, not epochs — and then it still barely learns

Generating a training sample means rendering the CAD model into four crops,
which costs about 120 ms on one core and is the binding constraint on the whole
exercise; the GPU is idle most of the time. Persistent `DataLoader` workers took
that to about 34 ms and made a few thousand steps affordable.

With a hundred stations, fourteen samples each and a batch of twelve, an epoch
is 116 steps. Three earlier configurations were abandoned at epoch four — 580
steps — on the reasonable-looking grounds that the loss had not moved. For a CNN
learning a dense alignment task from scratch that is not a verdict; it is barely
a warm-up.

So the fourth run was allowed to finish: 40 epochs, about 4600 steps, half an
hour. The loss is flat for **twenty-four epochs** and then begins to fall,
monotonically, for the rest of the run:

| epoch | validation | identity baseline | improvement |
| ---: | ---: | ---: | ---: |
| 0 | 5.570 mm | 5.564 mm | 0% |
| 12 | 5.570 mm | 5.564 mm | 0% |
| 24 | 5.533 mm | 5.564 mm | 1% |
| 32 | 5.504 mm | 5.564 mm | 1% |
| 39 | 5.498 mm | 5.564 mm | **1.2%** |

So it does learn. It learns far too slowly to be useful, and 1.2 per cent better
than predicting nothing is not a refiner. The trend was still downward when the
cosine schedule ran the learning rate to zero, so more steps would buy more —
but the gap to close is enormous and the slope is not promising.

**The honest reading** is that this is a data problem rather than a steps
problem. The refiner has a hundred distinct scenes to learn from. Every one of
the 1400 samples in an epoch is a perturbation of one of those hundred, so the
diversity the network actually sees is a hundred objects-in-context, not
fourteen hundred. Inferring the part's silhouette from a 96-pixel greyscale crop
in which the turntable has much the same texture as the part is a segmentation
problem in its own right, and a hundred examples is not enough to learn one.

**The lesson that generalises**: when a model is not learning, separate "cannot"
from "has not yet" — the overfit test answers that in ninety seconds — and then
separate "has not yet" from "has nothing to learn from". The second one took
considerably longer.

What the ablation did with this is in [results.md](../docs/results.md#5-the-ablation),
and it is a negative result.

---

## Why the rig has a 32-degree lens

The first version used 58 degrees, which seemed like a sensible machine-vision
lens and put the 187 mm part across about three per cent of the frame.

The consequence was not subtle. SIFT found roughly 1600 triangulated points per
station, of which **seven** were on the part and fifteen on the turntable. The
cross-station alignment then dutifully found that the dominant rigid motion
between stations was the identity — the room — and reconstructed the room.

An inspection cell frames the working volume, not the room. At 32 degrees the
turntable fills about eighty per cent of the frame width, the part carries
roughly 900 keypoints per view instead of 144, and the moving group is
comparable in size to the static one rather than one per cent of it.

The multi-motion selection in
[`align.py`](../src/multicam/reconstruction/align.py) is still necessary — it
is what makes the result independent of this choice — but the choice is what
makes the problem reasonable in the first place.

---

## What the COLMAP baseline turned out to measure

It was set up as a referee: run the reference implementation on the same images,
and treat agreement as evidence that the geometry is right. It did not referee
anything, and what it did instead is more useful.

COLMAP **cannot reconstruct the part from this sequence at all**. It registers a
minority of the 72 images, splits them across several models, and places none of
its points within 6 mm of the part's CAD surface. Its camera centres disagree
with ours by hundreds of millimetres after the best similarity alignment.

The tell is the instability. Two runs with identical settings registered 50 and
15 of the 72 images. An algorithm converging on a solution does not do that; an
algorithm with no solution to converge on does.

The reason is structural. Structure from motion assumes **one** rigid scene. A
turntable sequence contains two irreconcilable ones — a room that does not move
and a part that does — and there is nothing in the algorithm that can prefer the
part. It is precisely the failure that
[`align.py`](../src/multicam/reconstruction/align.py) exists to avoid, by fitting
several rigid motions and keeping the one that is not the identity.

So the honest summary is not "ours is more accurate than COLMAP". It is: given
identical images, COLMAP run with its default settings on unmasked images did
not recover the part while our pipeline, which is built for exactly this case,
did. That says something about the design decision and nothing about
implementation quality. It is also one recorded run: masking out the static
background or giving COLMAP the calibrated rig was not tried, and might change
the outcome.

The seed is now fixed. Two unseeded runs during development registered 50 and
15 of the 72 images; only the 15-image run's record was kept, so the spread
between runs is an observation and not a measurement.

---

## The three bugs that produced plausible-looking wrong answers

Collected because they share a shape: each one left every visible diagnostic
looking healthy.

| Bug | What the diagnostics said | What was actually true |
| --- | --- | --- |
| The distortion map applied backwards in the renderer | Calibration converged, residual 0.13 px, agreed with OpenCV to 1e-7 px | Every detected corner was about 2 px from where the camera model said it should be, and the calibration had recovered the distortion of the renderer rather than of the lens |
| Poses parameterised by their own logarithm, stepped with `x + dx` | "converged: damping" after one iteration | The focal length was 36 px out and no distortion had been recovered at all |
| Station poses composed without inverting | Bundle adjustment converged to 0.29 px reprojection residual | The reconstruction was 35 mm from the CAD surface and internally consistent |

The common thread is that **a residual only measures self-consistency**. All
three were caught by comparing against something external: the projection
equations, finite differences, and the CAD model respectively. None of them
would have been caught by looking harder at the residual.

---

## Reproducing anything here

Every script writes a JSON record carrying its settings, its full metrics, the
library versions and the commit it ran at.

```bash
make data            # render the capture from a seed, about 20 minutes
make experiments     # every number quoted anywhere in this repository
make plots tables docs-check
```
