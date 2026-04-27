# TODO

## Performance

- [ ] **Zero-copy DMA input/output** (`axrArgument.fd` instead of `.ptr`)
  - `axr_run_model_instance` currently takes 13.94 ms/call; the AIPU kernel alone is ~8 ms (`--explore latency`)
  - ~6 ms is DMA copy overhead (host→device input, device→host 3× feature outputs)
  - Fix: allocate input/output buffers as dmabuf and pass fd instead of ptr
  - Expected gain: brings search_enc from ~15 ms to ~9–10 ms → tracker from ~27 fps to ~35+ fps

- [ ] **xcorr_head on Metis hardware**
  - Currently the biggest bottleneck: 21.7 ms/frame, 58% of total time
  - Blocked by dynamic-weight grouped conv (weights come from template features, not static)
  - Possible approaches: rewrite xcorr as a static op, or fuse into a single compiled model
  - If xcorr moves to AIPU (~8 ms): total ~10 ms/frame → ~100 fps theoretical

## Correctness

- [ ] **Deterministic 3-run comparison (original vs split ONNX)**
  - Add `np.random.seed(42)` at top of `main()` in `track_pf_local.py` and `track_split.py`
  - Run each 3× on RunPod GPU; confirm bit-identical scores across runs
  - Frame-by-frame score diff between original and split: mean/max diff, first divergence > 0.01

## Evaluation

- [ ] **Quantitative tracking metrics on ir_crop.mp4**
  - No ground-truth annotations yet — need to label or use MuggleSAM output as proxy
  - Compute AUC, precision, success rate vs SAM2 reference
