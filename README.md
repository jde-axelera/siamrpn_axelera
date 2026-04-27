# SiamRPN++ Object Tracker — Axelera Metis

> **Large files** (model checkpoints, ONNX weights, compiled Metis models, and test videos) are not included in this repository due to size. To obtain them, email **jaydeep.de@axelera.ai**.

SiamRPN++ tracker with ResNet-50 backbone, fine-tuned on IR/thermal drone datasets. Runs the backbone on Axelera Metis AIPU and the correlation head on CPU. Includes a particle filter for robust tracking under partial occlusion and motion blur.

---

## Performance

Measured on a 640×480 @ 25 fps thermal drone sequence (5 966 frames).

| Implementation | Hardware | fps |
|---|---|---|
| Python + ONNX Runtime | GPU | 1.6 |
| C++ + ONNX Runtime | GPU | 44.0 |
| **C++ + axruntime** | **Metis AIPU** | **26.8** |

### Per-stage breakdown — C++ Axelera

| Stage | ms/frame | % |
|---|---|---|
| Preprocessing | 0.3 | 0.8% |
| search\_encoder (Metis) | 15.3 | 41.0% |
| xcorr\_head (CPU) | 21.7 | **58.0%** ← bottleneck |
| Particle filter | 0.1 | 0.2% |

---

## Requirements

### Metis machine

- Axelera Voyager SDK 1.6 (installed at `/opt/voyager-sdk` or similar)
- OpenCV 4 (`pkg-config --exists opencv4`)
- ONNX Runtime 1.17.1 (bundled with SDK under `operators/`)

### Workstation (export + compile only)

- Python ≥ 3.10, PyTorch ≥ 2.0, `onnx`, `onnxruntime`
- [pysot](https://github.com/STVIR/pysot) on `PYTHONPATH`

---

## Quick Start — C++ Inference on Metis

### 1. Activate the SDK environment

```bash
source <SDK_ROOT>/venv/bin/activate
```

This sets the firmware environment variables (`AIPU_FIRMWARE_OMEGA`, `AIPU_RUNTIME_STAGE0_OMEGA`) required by axruntime.

### 2. Build

```bash
cd final_trained
# Edit SDK path in Makefile_axelera if your SDK is not at /home/ubuntu/1.6/voyager-sdk
make -f Makefile_axelera
```

### 3. Prepare xcorr\_head (one-time)

The ONNX export produces IR v10 but ORT 1.17.1 requires ≤ IR v9:

```bash
python3 -c "
import onnx
m = onnx.load('onnx/xcorr_head.onnx')
m.ir_version = 8
onnx.save(m, 'onnx/xcorr_head_ir8.onnx')
"
```

### 4. Run

```bash
./track_split_axelera_cpp \
    --template_encoder <SDK_ROOT>/customers/arquimea/compiled_template_v2/compiled_model/model.json \
    --search_encoder   <SDK_ROOT>/customers/arquimea/compiled_search_v2/compiled_model/model.json \
    --xcorr_head       onnx/xcorr_head_ir8.onnx \
    --video            /path/to/input.mp4 \
    --output           /path/to/output.mp4\
    --display
```

**`--init_bbox x,y,w,h`** — provide initial bounding box (top-left x/y, width, height). If omitted, the first frame opens in a window for mouse selection.

**`--display`** — show a live tracking window. Press **Q** or **ESC** to stop early.

**`--ratios r1,r2,...`** — anchor aspect ratios. Default `0.37,0.56,0.79,1.11,2.26` (IR-trained model). Use `0.33,0.5,1,2,3` for the standard `siamrpn_r50_l234_dwxcorr` pretrained weights.

**`--aipu_cores N`** — number of AIPU cores (default 4).

---

## Compiling Models for Metis

Use `axcompile` (part of Voyager SDK) to quantise and compile the ONNX encoders.

### Prepare calibration images

Extract frames from your target domain videos:

```bash
python3 scripts/extract_cal_images.py \
    --videos /path/to/video1.mp4 /path/to/video2.mp4 \
    --out    cal_images/ \
    --n      400
```

### Compile search encoder

```bash
source <SDK_ROOT>/venv/bin/activate
export PYTHONPATH=/path/to/transform/scripts:$PYTHONPATH

axcompile \
    --input        onnx/search_encoder.onnx \
    --imageset     cal_images/ \
    --transform    transform_search.py \
    --input-shape  1,3,255,255 \
    --input-data-layout NCHW \
    --color-format RGB \
    --imreader-backend PIL \
    --dataset-len  400 \
    --output       compiled_search/
```

### Compile template encoder

```bash
axcompile \
    --input        onnx/template_encoder.onnx \
    --imageset     cal_images/ \
    --transform    transform_template.py \
    --input-shape  1,3,127,127 \
    --input-data-layout NCHW \
    --color-format RGB \
    --imreader-backend PIL \
    --dataset-len  400 \
    --output       compiled_template/
```

The compiled model manifest (`compiled_model/model.json`) is what you pass to `--template_encoder` / `--search_encoder`.

### Transform scripts

`transform_search.py` and `transform_template.py` define `get_preprocess_transform(image)` — a function that converts a PIL image to the same format used at inference (BGR float32 CHW, values in [0, 255], no ImageNet normalisation). See the provided examples in the repo.

---

## ONNX Export (fine-tuned or custom checkpoint)

Splits a trained checkpoint into three ONNX models:

```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH

python scripts/export_onnx_split.py \
    --cfg   configs/config_ir_siamese_infer.yaml \
    --ckpt  checkpoints/best_model.pth \
    --out   onnx/
```

| Model | Input | Output | Runs on |
|---|---|---|---|
| `template_encoder.onnx` | `(1,3,127,127)` | `zf_0, zf_1, zf_2` | Metis (compiled) |
| `search_encoder.onnx` | `(1,3,255,255)` | `xf_0, xf_1, xf_2` | Metis (compiled) |
| `xcorr_head.onnx` | `zf_0..2, xf_0..2` | `cls, loc` | CPU (dynamic grouped conv) |

---

## Fine-tuning

To train or fine-tune on a new IR dataset:

```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH

torchrun --nproc_per_node=<NUM_GPUS> scripts/train_siamrpn_aws.py \
    --cfg       configs/config_ir_siamese.yaml \
    --pretrained pretrained/sot_resnet50.pth
```

Edit `configs/config_ir_siamese.yaml` to set `ROOT` (dataset root) and `ANCHOR.RATIOS`. To compute data-driven anchor ratios from your annotations:

```bash
python scripts/analyze_anchors.py \
    --data_root /path/to/datasets \
    --n_anchors 5
```

Checkpoints are saved to `pysot/snapshot/<experiment_name>/`. Resume training with `--resume <checkpoint.pth>`.

---

## Architecture

```
[Metis]  template_encoder  (1,3,127,127) → zf_0..2  (1,256,7,7)  — run once at init
[Metis]  search_encoder    (1,3,255,255) → xf_0..2  (1,256,31,31) — run per frame
[CPU]    xcorr_head         zf_0..2, xf_0..2 → cls (1,10,25,25), loc (1,20,25,25)
[CPU]    Particle filter + decode → tracked bounding box
```

Template features are center-cropped from the compiled model output `(1,256,15,15)` → `(1,256,7,7)` automatically inside the C++ binary.

---

## Files

```
├── track_split_axelera.cpp         C++ inference (Metis + CPU xcorr + PF)
├── Makefile_axelera                build recipe
├── configs/
│   ├── config_ir_siamese.yaml      training config
│   ├── config_ir_siamese_infer.yaml inference config
│   └── track_config_axelera.json   example Metis inference config
├── scripts/
│   ├── export_onnx_split.py        split ONNX export
│   ├── analyze_anchors.py          compute anchor ratios from annotations
│   ├── train_siamrpn_aws.py        training script
│   ├── track_split.py              Python inference (GPU)
│   ├── track_split_axelera.py      Python inference (Metis + CPU xcorr)
│   └── compare_sam2_vs_metis.py    side-by-side comparison video vs SAM2
└── env.yml                         conda environment spec
```

**Not in repo — request via email:**

| File | Description |
|---|---|
| `checkpoints/best_model.pth` | IR fine-tuned checkpoint (epoch 444) |
| `pretrained/sot_resnet50.pth` | SiamRPN++ SOT pretrained backbone |
| `onnx/template_encoder.onnx` | Exported template encoder |
| `onnx/search_encoder.onnx` | Exported search encoder |
| `onnx/xcorr_head.onnx` | Exported xcorr head |
| `onnx/xcorr_head_ir8.onnx` | IR v8 copy (required for C++ build) |
| Compiled Metis models | `compiled_template_v2/`, `compiled_search_v2/` |
| Test videos | `ir_crop.mp4`, `coyote.mp4`, and other Arquimea sequences |

Contact: **jaydeep.de@axelera.ai**
