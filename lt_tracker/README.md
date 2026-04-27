# SiamRPN++ on Axelera Metis — C++ Tracker

Real-time object tracking using SiamRPN++ (ResNet-50 LT backbone) with the search encoder
running on the Axelera Metis AIPU and xcorr/head running on the host CPU.

**Target machine:** `ubuntu@100.111.58.120` (password: `ubuntu`)  
**SDK:** `/home/ubuntu/1.6/voyager-sdk/` — always `source venv/bin/activate` first

---

## Performance

| Configuration | Metis encoder | fps (coyote.mp4, 2043 frames) |
|---|---|---|
| 255px encoder (default) | 18.5 ms/frame | **25.7 fps** |
| 351px encoder (legacy) | 37 ms/frame | 15.9 fps |

Time breakdown (255px, per frame):
- Preprocess: 0.3 ms
- Metis search encoder: 18.5 ms
- dw-xcorr (AVX2): 3.1 ms
- ORT head: 2.8 ms

---

## Files on Metis (`/home/ubuntu/siamrpn_poc/`)

| File | Description |
|---|---|
| `poc_siamrpn.cpp` | Main tracker source |
| `poc_siamrpn_cpp` | Compiled binary |
| `Makefile_poc` | Build file |
| `template_encoder_r50lt.onnx` | Template encoder (CPU, 6 outputs) |
| `siamrpn_head_dyn.onnx` | Head with dynamic spatial dims (25×25 or 37×37) |
| `siamrpn_head_37.onnx` | Head for 351px encoder only (static 37×37) |
| `build/siamrpn++onnx_255/` | Compiled 255px search encoder (18.5 ms) |
| `build/siamrpn++onnx/` | Compiled 351px search encoder (37 ms) |

---

## One-time Model Preparation

These steps run on a workstation with PyTorch + pysot installed, not on Metis.

### 1. Export template encoder

```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
python scripts/export_template_encoder_r50lt.py \
    --model_pth lt.pth \
    --config pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml \
    --out template_encoder_r50lt.onnx
```

This produces 6 outputs `k0`–`k5` with shapes `[1,128,5,5]×2`, `[1,256,5,5]×2`, `[1,512,5,5]×2`.

### 2. Fix head ONNX IR version (ORT 1.17.1 requires ≤ v9)

```bash
python3 -c "
import onnx
m = onnx.load('siamrpn_head_37.onnx')
m.ir_version = 8
onnx.save(m, 'siamrpn_head_37_ir8.onnx')
"
```

### 3. Make dynamic head (accepts both 25×25 and 37×37 score maps)

Run on Metis (or any machine with `onnx` installed):

```bash
python3 -c "
import onnx
m = onnx.load('siamrpn_head_37.onnx')
for t in list(m.graph.input) + list(m.graph.output):
    s = t.type.tensor_type.shape
    for d in [s.dim[2], s.dim[3]]:
        d.ClearField('dim_value')
        d.dim_param = 'S'
del m.graph.value_info[:]
onnx.save(m, 'siamrpn_head_dyn.onnx')
"
```

### 4. Compile search encoders (on Metis with SDK active)

```bash
source /home/ubuntu/1.6/voyager-sdk/venv/bin/activate

# 255×255 encoder (~17ms, recommended)
axcompile -i onnx_files/search_encoder.onnx \
    --input-shape 1,3,255,255 \
    -o compiled_search_255 --overwrite \
    --imageset cal_images --dataset-len 400 \
    --transform transform_search.py

# 351×351 encoder (~37ms, legacy)
axcompile -i onnx_files/search_encoder.onnx \
    --input-shape 1,3,351,351 \
    -o compiled_search_351 --overwrite \
    --imageset cal_images --dataset-len 400 \
    --transform transform_search.py
```

See `customers/arquimea/` for `transform_search.py` and calibration images.

---

## Build

On Metis:

```bash
ssh ubuntu@100.111.58.120  # password: ubuntu
source /home/ubuntu/1.6/voyager-sdk/venv/bin/activate
cd /home/ubuntu/siamrpn_poc
make -f Makefile_poc
```

### Makefile variables

| Variable | Default | Description |
|---|---|---|
| `SDK` | `/home/ubuntu/1.6/voyager-sdk` | Voyager SDK root |
| `AXR_INC` | `$(SDK)/venv/.../axelera/include` | Axelera runtime headers |
| `AXR_LIB` | `$(SDK)/venv/.../axelera/lib` | Axelera runtime library |
| `ORT_INC` | `$(SDK)/operators/onnxruntime-.../include` | ONNXRuntime headers |
| `ORT_LIB` | `$(SDK)/operators/lib` | ONNXRuntime library |

Build flags: `-O3 -mavx2 -mfma -std=c++17` — AVX2+FMA is required for vectorized dw-xcorr.

To rebuild from scratch:

```bash
make -f Makefile_poc clean && make -f Makefile_poc
```

---

## Run

Always activate the SDK environment first:

```bash
source /home/ubuntu/1.6/voyager-sdk/venv/bin/activate
cd /home/ubuntu/siamrpn_poc
```

### Default (255px encoder, 25.7 fps)

```bash
./poc_siamrpn_cpp \
    --search_encoder   build/siamrpn++onnx_255/siamrpn++onnx_255/1/model.json \
    --template_encoder template_encoder_r50lt.onnx \
    --head             siamrpn_head_dyn.onnx \
    --video            /home/ubuntu/Arquimea/wetransfer_test-videos-for-tracking_2025-11-11_0750/coyote.mp4 \
    --output           out_255.mp4 \
    --init_bbox        695,345,20,30
```

### Legacy 351px encoder (15.9 fps)

```bash
./poc_siamrpn_cpp \
    --search_encoder   build/siamrpn++onnx/siamrpn++onnx/1/model.json \
    --template_encoder template_encoder_r50lt.onnx \
    --head             siamrpn_head_37.onnx \
    --instance_size    351 \
    --video            /path/to/video.mp4 \
    --output           out_351.mp4 \
    --init_bbox        x,y,w,h
```

### Interactive ROI selection (omit `--init_bbox`)

```bash
./poc_siamrpn_cpp \
    --search_encoder   build/siamrpn++onnx_255/siamrpn++onnx_255/1/model.json \
    --template_encoder template_encoder_r50lt.onnx \
    --head             siamrpn_head_dyn.onnx \
    --video            /path/to/video.mp4 \
    --output           out.mp4 \
    --display
```

### Process a limited number of frames

```bash
./poc_siamrpn_cpp ... --max_frames 300
```

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--search_encoder PATH` | *(required)* | Axelera compiled model JSON |
| `--template_encoder PATH` | *(required)* | Template encoder ONNX |
| `--head PATH` | *(required)* | Head ONNX (`siamrpn_head_dyn.onnx` or `siamrpn_head_37.onnx`) |
| `--video PATH` | *(required)* | Input video |
| `--output PATH` | *(none)* | Output video path (skip to disable writing) |
| `--init_bbox x,y,w,h` | *(none)* | Initial bounding box; omit for interactive selection |
| `--instance_size N` | `255` | Search crop size. Use `255` with `siamrpn++onnx_255`, `351` with `siamrpn++onnx` |
| `--lost_instance_size N` | `2×instance_size` | LT mode search window size. Default: 510 for 255px, 702 for 351px |
| `--aipu_cores N` | `4` | AIPU cores for the normal-mode encoder |
| `--max_frames N` | `-1` (all) | Stop after N frames |
| `--display` | off | Show live tracking window |

---

## Tracking Parameters (Config struct defaults)

| Parameter | Default | Description |
|---|---|---|
| `exemplar_size` | 127 | Template crop size (fixed) |
| `instance_size` | 255 | Search crop size (`--instance_size`) |
| `lost_instance_size` | auto (2×) | LT window size (`--lost_instance_size`) |
| `num_tiles` | 2 | Tiles per dimension in LT mode (2×2 grid) |
| `penalty_k` | 0.05 | Scale/ratio change penalty |
| `window_influence` | 0.35 | Cosine window weight (normal mode) |
| `lr` | 0.28 | Bbox update learning rate |
| `conf_low` | 0.8 | Enter LT mode below this score |
| `conf_high` | 0.985 | Exit LT mode above this score |
| `enable_lt_mode` | true | Enable long-term re-detection |
| `aipu_cores` | 4 | AIPU cores for normal encoder |

---

## Architecture Notes

### score_size formula

```
score_size = (instance_size - exemplar_size) / anchor_stride + 1 + base_size
           = (255 - 127) / 8 + 1 + 8 = 25    (255px encoder)
           = (351 - 127) / 8 + 1 + 8 = 37    (351px encoder)
```

### Axelera sub-device budget

The Metis chip has 4 sub-devices. Allocation:
- 1 connection × 4 cores → normal-mode `srch_enc`
- 3 connections × 1 core → LT tile workers (worker 0 handles tiles {0,3} sequentially; workers 1 and 2 handle tiles {1} and {2} in parallel)

Total: 4 connections = 4 sub-devices. No headroom beyond this.

### Neck center-crop (template encoder only)

The Axelera compiled template encoder outputs `(1,256,15,15)` NHWC (backbone+neck without AdjustLayer).
The original ONNX output is `(1,256,7,7)` which includes a center crop (`l = (15-7)//2 = 4`, crop `[:,:,4:11,4:11]`).
This crop is applied in the C++ dequantize path — the `(1,256,7,7)` output feeds directly into xcorr.

### dw-xcorr implementation

Uses direct per-channel accumulation with `__restrict__` pointers and `#pragma GCC ivdep` to
trigger 8-wide AVX2 FMA vectorization. Avoids the ~240 MB im2col write that a BLAS approach
would require, keeping all working data (6.7 KB search + 5.4 KB output per channel) in L1 cache.

---

## Python tracker reference

`poc_siamrpn.py` is the Python equivalent using the same Axelera runtime.
It runs at ~19 fps with the 351px encoder (37ms Metis + 16ms xcorr_head on CPU).

```bash
source /home/ubuntu/1.6/voyager-sdk/venv/bin/activate
python poc_siamrpn.py
```

Edit `VIDEO`, `SRCH_JSON`, `INIT_BBOX` at the top of the file.
