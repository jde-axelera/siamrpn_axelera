# SiamRPN++ Object Tracker — Axelera Metis

SiamRPN++ tracker with ResNet-50 backbone, fine-tuned on IR/thermal drone datasets. Runs the backbone on Axelera Metis AIPU and the correlation head on CPU. Includes a particle filter for robust tracking under partial occlusion and motion blur.

> **Large files** (model checkpoints, ONNX weights, compiled Metis models, and test videos) are not included. Email **jaydeep.de@axelera.ai** to obtain them.

---

## Performance

Measured on a 640×480 @ 25 fps thermal drone sequence (5 966 frames).

| Script | Hardware | Particle Filter | fps |
|---|---|---|---|
| `track_split_axelera_nopf.py` | Metis AIPU + ARM CPU | No  | 23.9 |
| `track_split_axelera.py`      | Metis AIPU + ARM CPU | Yes | 19.3 |
| `track_split_axelera_cpp`     | Metis AIPU + ARM CPU | Yes | **26.8** |
| `track_split.py`              | RTX GPU + CPU        | Yes | 1.6  |
| `track_split_cpp`             | RTX GPU + CPU        | Yes | 44.0 |

---

## Quick Start — C++

```bash
source <SDK_ROOT>/venv/bin/activate
make -f Makefile_axelera

./track_split_axelera_cpp \
    --template_encoder compiled_template/compiled_model/model.json \
    --search_encoder   compiled_search/compiled_model/model.json \
    --xcorr_head       onnx/xcorr_head_ir8.onnx \
    --video            /path/to/input.mp4 \
    --output           /path/to/output.mp4 \
    --init_bbox        x,y,w,h
```

`--init_bbox` is optional — omit to select the target interactively on the first frame.  
`--display` opens a live tracking window.  
`--ratios r1,r2,...` sets anchor aspect ratios (default `0.37,0.56,0.79,1.11,2.26` for the IR model).

---

## Quick Start — Python

```bash
source <SDK_ROOT>/venv/bin/activate
export PYTHONPATH=$(pwd)/pysot:$PYTHONPATH

python track_split_axelera.py --config configs/track_config_axelera.json
```

Edit `configs/track_config_axelera.json` to set model paths, video path, and `init_bbox`.

---

## Files

```
├── track_split_axelera.py          Python inference — Metis + CPU xcorr + PF
├── track_split_axelera_nopf.py     Python inference — Metis + CPU xcorr, no PF
├── track_split_axelera.cpp         C++ inference   — Metis + CPU xcorr + PF
├── Makefile_axelera                C++ build recipe
├── configs/
│   ├── config_ir_siamese.yaml      training config
│   ├── config_ir_siamese_infer.yaml inference config
│   └── track_config_axelera.json   example Metis inference config
├── scripts/
│   ├── export_onnx_split.py        ONNX export from checkpoint
│   ├── train_siamrpn_aws.py        training script
│   └── track_split.py              Python inference (GPU reference)
└── env.yml                         conda environment spec
```

**Not in repo — request via email:**

| File | Description |
|---|---|
| `checkpoints/best_model.pth` | IR fine-tuned checkpoint |
| `onnx/xcorr_head_ir8.onnx` | xcorr head (IR version ≤ v9 for ORT 1.17.1) |
| `compiled_template/`, `compiled_search/` | Compiled Metis models |
| Test videos | `ir_crop.mp4` and other sequences |

---

## License

MIT License — Copyright (c) 2025 Axelera AI
