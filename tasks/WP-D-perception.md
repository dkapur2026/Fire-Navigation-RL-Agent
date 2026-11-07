# WP-D: Neural perception front-end  (N2)

**Goal.** Make the "neuro" in neuro-symbolic literal. A CNN looks at a noisy, smoke-degraded picture of the house and
produces the symbolic state (which cells burn, where the dependents are); the planner then plans from the CNN's output.

## Read first
`src/emberpath/render.py` (`render_frame`), `world.py` (`Observation`, `observe(full=False)`), `env.py` (`encode`),
`planner.py` (what the planner consumes), `docs/ARCHITECTURE.md`.

## Deliverables
1. `src/emberpath/perception.py`*:
   * A rasterizer that turns a world into a small image tensor with realistic degradation: smoke attenuates colour,
     visibility radius masks distant cells, optional pixel noise. It must not leak ground truth (no direct state channels).
   * A dataset generator: roll out episodes (baseline and oracle agents, several blueprints and novelties) and save
     (image, per-cell labels) pairs.
   * A small CNN (PyTorch) predicting per-cell labels: burning / burnt / clear, plus dependent position and status.
   * A `PerceivedObservation` adapter that builds an `Observation` from the CNN's output so the existing planner runs unchanged.
2. Metrics: per-class precision / recall / F1; accuracy as a function of smoke density and of distance from the rescuer.
3. End-to-end study: baseline and oracle agents planning from CNN output vs ground truth; report the drop in score and in
   `gap_closed` once WP-F exists. Include a failure analysis (look at where perception errors cause bad decisions).
4. Debug and profile like an ML engineer would: loss curves, one genuine bug you found and how, throughput numbers.

## Done when
```bash
pip install -e ".[rl]"
python -m pytest -q          # adapter test + a tiny overfit-one-batch test that the CNN can fit
python -m emberpath.perception --help     # train / eval entry point, documented
```

## Constraints
Torch is optional; `import emberpath` must still work without it. Seeded and reproducible. Keep the model small enough to train
on CPU in minutes. No leakage: a unit test must show the rasterizer's output cannot be trivially decoded into ground truth
without the network (for example, no exact-colour state encoding).

## Docs
A "Perception" section in `docs/ARCHITECTURE.md` with the numbers. Flip N2 in `docs/SPEC.md`.
