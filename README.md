# SiamRPN++ Object Tracker — Axelera Metis

> **Large files** (model checkpoints, ONNX weights, compiled Metis models, and test videos) are not included in this repository due to size. Download them from the **[model release Google Drive](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing)**.

SiamRPN++ tracker with ResNet-50 backbone, fine-tuned on IR/thermal drone datasets. Runs the backbone on Axelera Metis AIPU and the correlation head on CPU. Includes a particle filter for robust tracking under partial occlusion and motion blur.

---

## Performance

Measured on a 640×480 @ 25 fps thermal drone sequence (5 966 frames).  
All Metis runs: backbone on Axelera AIPU, xcorr head on ARM CPU.  
All GPU runs: backbone on CUDA, xcorr head on CPU (dynamic grouped conv cannot run on GPU).

### Axelera Metis AIPU

| Script | Particle Filter | fps |
|---|---|---|
| `track_split_axelera_nopf.py` (Python) | No  | 23.9 |
| `track_split_axelera.py` (Python)      | Yes | 19.3 |
| `track_split_axelera_cpp` (C++)        | Yes | **26.8** |

### Reference

| Script | Hardware | Particle Filter | fps |
|---|---|---|---|
| `track_split.py` (Python) | x86 CPU only | Yes | ~0.8 |
| `track_split.py` (Python) | RTX GPU + CPU | Yes | 1.6 |
| `track_split_cpp` (C++)   | RTX GPU + CPU | Yes | **44.0** |

> The C++ Metis build (26.8 fps) outperforms the Python GPU build (1.6 fps) despite running on an embedded AIPU — the gap is Python interpreter overhead, not hardware. The C++ GPU build (44 fps) is faster because the GPU backbone is ~3× quicker than the AIPU for this model; the remaining bottleneck in both cases is `xcorr_head` on CPU.

### Per-stage breakdown — C++ Metis

| Stage | ms/frame | % |
|---|---|---|
| Preprocessing | 0.3 | 0.8% |
| search\_encoder (Metis AIPU) | 15.3 | 41.0% |
| xcorr\_head (ARM CPU) | 21.7 | **58.0%** ← bottleneck |
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

## Quick Start — Python Inference on Metis

### 1. Clone the repo and set up pysot

```bash
git clone git@github.com:jde-axelera/siamrpn_axelera.git
cd siamrpn_axelera
git clone https://github.com/STVIR/pysot.git
```

### 2. Obtain the checkpoint

Download `best_model.pth` from the **`ir_checkpoint/`** folder on the [model release Google Drive](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing). Place it at `checkpoints/best_model.pth` inside the repo.

If you also download the pre-compiled Metis models (`ir_Metis/`) and pre-exported ONNX files (`ir_onnx/`), skip steps 3–5 and go straight to step 6.

### 3. Export ONNX from checkpoint

Run this on a machine with PyTorch ≥ 2.0 (workstation or server, not required on Metis):

```bash
export PYTHONPATH=$(pwd)/pysot:$PYTHONPATH

python scripts/export_onnx_split.py \
    --cfg   configs/config_ir_siamese_infer.yaml \
    --ckpt  checkpoints/best_model.pth \
    --out   onnx/
```

This produces three models:

| File | Input shape | Output | Runs on |
|---|---|---|---|
| `onnx/template_encoder.onnx` | `(1,3,127,127)` | `zf_0, zf_1, zf_2` | Metis (compiled) |
| `onnx/search_encoder.onnx` | `(1,3,255,255)` | `xf_0, xf_1, xf_2` | Metis (compiled) |
| `onnx/xcorr_head.onnx` | `zf_0..2, xf_0..2` | `cls, loc` | CPU (ONNX Runtime) |

### 4. Fix xcorr\_head IR version (one-time)

ORT 1.17.1 on Metis supports IR ≤ v9; the export produces IR v10:

```bash
python3 -c "
import onnx
m = onnx.load('onnx/xcorr_head.onnx')
m.ir_version = 8
onnx.save(m, 'onnx/xcorr_head_ir8.onnx')
"
```

### 5. Compile encoders for Metis

Run on the Metis machine with Voyager SDK activated.

**Extract calibration images from your target-domain videos:**

```bash
python3 scripts/extract_cal_images.py \
    --videos /path/to/video1.mp4 /path/to/video2.mp4 \
    --out    cal_images/ \
    --n      400
```

**Compile both encoders:**

```bash
source <SDK_ROOT>/venv/bin/activate
export PYTHONPATH=$(pwd)/scripts:$PYTHONPATH

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

`transform_search.py` and `transform_template.py` in `scripts/` define `get_preprocess_transform(image)` — BGR float32 CHW, [0, 255], no ImageNet normalisation.

### 6. Activate the SDK environment and set PYTHONPATH

```bash
source <SDK_ROOT>/venv/bin/activate
export PYTHONPATH=$(pwd)/pysot:$PYTHONPATH
```

### 7. Edit the config

Copy and edit `configs/track_config_axelera.json` — update all paths to match your machine:

```json
{
  "paths": {
    "pysot_root":          "/path/to/siamrpn_axelera/pysot",
    "model_cfg":           "/path/to/siamrpn_axelera/configs/config_ir_siamese_infer.yaml",
    "checkpoint":          "/path/to/siamrpn_axelera/checkpoints/best_model.pth",
    "template_encoder_ax": "/path/to/siamrpn_axelera/compiled_template/compiled_model/model.json",
    "search_encoder_ax":   "/path/to/siamrpn_axelera/compiled_search/compiled_model/model.json",
    "xcorr_head":          "/path/to/siamrpn_axelera/onnx/xcorr_head_ir8.onnx",
    "video":               "/path/to/input.mp4",
    "output":              "/path/to/output.mp4"
  },
  "init_bbox": [x, y, w, h]
}
```

### 8. Run

With particle filter:

```bash
python track_split_axelera.py --config configs/track_config_axelera.json
```

Expected output:
```
Providers — template/search: Axelera Metis  |  xcorr_head: CPU
  template_encoder (Metis): 5.1 ms
  f=    0  score=1.000  cx=367.0 cy=189.0  ...
Processed 5966 frames in 308s  →  19.3 fps
```

Without particle filter (faster, single-panel output):

```bash
python track_split_axelera_nopf.py --config configs/track_config_axelera.json \
    --output output_nopf.mp4
```

Writes a `_stats.json` alongside the video with fps, frame count, and timing.

---

## Quick Start — C++ Inference on Metis

Complete **steps 1–5** from the Python section above, then:

### 1. Activate the SDK environment

```bash
source <SDK_ROOT>/venv/bin/activate
```

This sets the firmware environment variables (`AIPU_FIRMWARE_OMEGA`, `AIPU_RUNTIME_STAGE0_OMEGA`) required by axruntime.

### 2. Build

```bash
# Edit SDK path in Makefile_axelera to point to your Voyager SDK root
make -f Makefile_axelera
```

### 3. Run

```bash
./track_split_axelera_cpp \
    --template_encoder compiled_template/compiled_model/model.json \
    --search_encoder   compiled_search/compiled_model/model.json \
    --xcorr_head       onnx/xcorr_head_ir8.onnx \
    --video            /path/to/input.mp4 \
    --output           /path/to/output.mp4 \
    --display
```

**`--init_bbox x,y,w,h`** — provide initial bounding box (top-left x/y, width, height). If omitted, the first frame opens in a window for mouse selection.

**`--display`** — show a live tracking window. Press **Q** or **ESC** to stop early.

**`--ratios r1,r2,...`** — anchor aspect ratios. Default `0.37,0.56,0.79,1.11,2.26` (IR-trained model). Use `0.33,0.5,1,2,3` for the standard `siamrpn_r50_l234_dwxcorr` pretrained weights.

**`--aipu_cores N`** — number of AIPU cores (default 4).

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
├── track_split_axelera.py          Python inference — Metis + CPU xcorr + PF
├── track_split_axelera_nopf.py     Python inference — Metis + CPU xcorr, no PF
├── track_split_axelera.cpp         C++ inference   — Metis + CPU xcorr + PF
├── Makefile_axelera                C++ build recipe
├── configs/
│   ├── config_ir_siamese.yaml      training config
│   ├── config_ir_siamese_infer.yaml inference config
│   └── track_config_axelera.json   example Metis inference config
├── scripts/
│   ├── export_onnx_split.py        split ONNX export
│   ├── analyze_anchors.py          compute anchor ratios from annotations
│   ├── train_siamrpn_aws.py        training script
│   ├── track_split.py              Python inference (GPU, ONNX Runtime)
│   └── compare_sam2_vs_metis.py    side-by-side comparison video vs SAM2
└── env.yml                         conda environment spec
```

**Not in repo — download from Google Drive:**

| Drive | Folder | File | Description |
|---|---|---|---|
| [Models](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) | `ir_checkpoint/` | `best_model.pth` | IR fine-tuned checkpoint (epoch 444) |
| [Models](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) | `ir_onnx/` | `template_encoder.onnx` | Exported template encoder |
| [Models](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) | `ir_onnx/` | `search_encoder.onnx` | Exported search encoder |
| [Models](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) | `ir_onnx/` | `xcorr_head.onnx` | Exported xcorr head |
| [Models](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) | `ir_onnx/` | `xcorr_head_ir8.onnx` | IR v8 copy (required for C++ build) |
| [Models](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) | `ir_Metis/` | compiled models | `compiled_template/`, `compiled_search/` |
| [Models](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) | `LT_checkpoint/` | `lt.pth` | LT model pretrained weights |
| [Models](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) | `LT_onnx/` | `template_encoder_r50lt.onnx` | LT template encoder (CPU) |
| [Models](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) | `LT_onnx/` | `search_encoder_r50lt.onnx` | LT search encoder (for axcompile) |
| [Models](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) | `LT_onnx/` | `siamrpn_head_dyn.onnx` | LT head with dynamic spatial dims |
| [Models](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) | `LT_Metis/` | compiled search encoder | Compiled 255px search encoder for Metis |
| [Test videos](https://drive.google.com/drive/folders/1u9LNrtFh-FO3QFWKxQjHW6aSuus8aBZk?usp=sharing) | — | `ir_crop.mp4` | IR thermal drone sequence (110 MB) |
| [Test videos](https://drive.google.com/drive/folders/1u9LNrtFh-FO3QFWKxQjHW6aSuus8aBZk?usp=sharing) | — | `coyote_OrinMuggleSam.mp4` | RGB coyote sequence (76 MB) |

---

## License

MIT License

Copyright (c) 2025 Axelera AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
