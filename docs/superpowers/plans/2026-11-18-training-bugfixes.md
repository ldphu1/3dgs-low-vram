# Training correctness fixes

**Goal:** Fix the six reviewed bugs in priority order, with a failing regression
test before each implementation and a passing check before starting the next fix.

**Scope:** The existing Windows/PyTorch 2.5.1 setup, the two local trainers,
`convert.py`, and their shared logger. Preserve datasets, previous results, and
existing modifications in `gsplat_repo`. Execute sequentially in this chat.

**Verification:** Use standard-library `unittest` for project regression tests,
real gsplat topology operations for pruning, real checkpoint files in temporary
directories, and CUDA rendering when a compatible runtime is available. Run the
whole new regression suite and attempt the upstream test command; report any
environmental limitation explicitly.

## Tasks

1. **Pruning alignment** — `simple_trainer_low_vram.py`,
   `tests/test_training_regressions.py`.
   - [ ] Reproduce deletion and splitting that move Gaussian rows.
   - [ ] Store importance and visibility in scene signals so existing topology
     hooks transform them with the Gaussian rows.
   - [ ] Check controller pruning does not apply its mask twice and still keeps
     the intended most important Gaussians.
2. **Checkpoint evaluation** — `simple_trainer_low_vram.py`, regression tests.
   - [ ] Reproduce changed Gaussian counts and compressed SH reload.
   - [ ] Rebuild scene tensors and restore the actual SH degree, codebook, and
     indices before evaluation; handle uncompressed checkpoints and rank shards.
   - [ ] Verify loaded render colors agree with the saved model.
3. **Export parameterization** — `convert.py`, regression tests.
   - [ ] Exercise conversion from a temporary checkpoint and inspect exported
     scale/opacity values.
   - [ ] Pass log-scales/logit-opacities directly to the exporter; make the script
     import-safe and accept checkpoint/output paths for repeatable conversion.
4. **Evaluation logs** — `train_logger.py`, both trainers, regression tests.
   - [ ] Reproduce evaluation overwriting existing training/evaluation CSV files.
   - [ ] Give checkpoint evaluation separate log files and leave training logs
     and summaries untouched.
5. **Appearance rendering** — `simple_trainer_low_vram.py`, regression tests.
   - [ ] Reproduce RGB tensors being interpreted as SH coefficients.
   - [ ] Pass `sh_degree=None` for computed RGB and retain the existing SH path.
   - [ ] Check appearance rendering and backward on CUDA.
6. **Small K-Means inputs** — both local K-Means implementations and callers,
   regression tests.
   - [ ] Reproduce `N < K` and exercise the compression caller, not only K-Means.
   - [ ] Clamp effective cluster count and use it consistently in reshaping,
     padding and lookup; reject empty/invalid input clearly.
   - [ ] Verify small, ordinary, and invalid input plus CUDA phase transitions.

## Final checks

- [ ] Run all local regression tests and syntax checks.
- [ ] Run CUDA smoke checks through lazy SH upgrades, compression and reload.
- [ ] Attempt the upstream test command and record unavailable dependencies.
- [ ] Review the changes and document actual checks and remaining limits.
