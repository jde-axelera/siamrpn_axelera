# SiamRPN++ on Axelera Metis — C++ Tracker

Real-time object tracking using SiamRPN++ (ResNet-50 LT backbone) with the search encoder
running on the Axelera Metis AIPU and xcorr/head running on the host CPU.

**SDK:** `<SDK_ROOT>` (e.g. `/opt/voyager-sdk`) — always `source <SDK_ROOT>/venv/bin/activate` first

> **Raspberry Pi 5:** fully supported via Metis M.2 HAT. The Makefile auto-detects `aarch64` and uses `-mcpu=cortex-a76` (NEON) instead of `-mavx2`. Set `AXELERA_CONFIGURE_BOARD=,20` if inference crashes (RPi5 power budget). See the main README for full RPi5 setup instructions.

> **Model files** (checkpoint, ONNX, compiled Metis) are not in this repo. Download from **[Models Drive](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing)** — use the `LT_checkpoint/`, `LT_onnx/`, and `LT_Metis/` folders. Test videos: **[Videos Drive](https://drive.google.com/drive/folders/1u9LNrtFh-FO3QFWKxQjHW6aSuus8aBZk?usp=sharing)** (`ir_crop.mp4`, `coyote_OrinMuggleSam.mp4`).

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

## Files on Metis (`<WORK_DIR>/`)

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

> **Fastest path:** Download from [Google Drive](https://drive.google.com/drive/folders/1yt2IpE78SLc4MJjnIyn7J-VNmP0J6sQ2?usp=sharing) using gdown (Python API — `gdown -m` CLI is unreliable for large files):
>
> ```python
> import gdown, os
> os.makedirs('lt_tracker', exist_ok=True)
> # LT checkpoint
> gdown.download(id='10E55GLW9W2pltl9eWEOZuFarFiJWgsYf', output='lt_tracker/lt.pth')
> # LT head ONNX (siamrpn_head_37.onnx — then run step 3 to make dynamic version)
> gdown.download(id='1zEb7T6YZZEMTKD9roBlADdT9mvtNJLck', output='lt_tracker/siamrpn_head_37.onnx')
> # LT compiled search encoder (255px)
> gdown.download_folder(id='1j4eBYi6zqWi_dI57iVUY0XjiuWkX2f-H', output='lt_tracker/build/siamrpn++onnx_255')
> ```
>
> Then run step 3 (dynamic head) and steps 1–2 (export template/search encoders — requires PyTorch workstation). Skip step 4 if you downloaded the compiled encoder above.
>
> **Note:** The compiled encoder from Drive downloads into `build/siamrpn++onnx_255/1/model.json` (one directory level shallower than the default run command). Use `--search_encoder build/siamrpn++onnx_255/1/model.json` when running.

Steps 1–2 run on a workstation with PyTorch + pysot installed. Steps 3–4 run on Metis.

### 1. Export template encoder

```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
python scripts/export_template_encoder_r50lt.py \
    --model_pth lt.pth \
    --config pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml \
    --out template_encoder_r50lt.onnx
```

Produces 6 kernel outputs `k0`–`k5`: `[1,128,5,5]×2`, `[1,256,5,5]×2`, `[1,512,5,5]×2`. IR version auto-downgraded to 8.

### 2. Export search encoder

```bash
python scripts/export_search_encoder_r50lt.py \
    --model_pth lt.pth \
    --config pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml \
    --out search_encoder_r50lt.onnx \
    --size 255
```

Produces 3 feature outputs `xf0`–`xf2` (backbone + neck). Use `--size 351` for the legacy 351px encoder.

### 3. Make dynamic head (accepts both 25×25 and 37×37 score maps)

Starting from `siamrpn_head_37.onnx`:

```bash
python3 -c "
import onnx
m = onnx.load('siamrpn_head_37.onnx')
m.ir_version = 8
for t in list(m.graph.input) + list(m.graph.output):
    s = t.type.tensor_type.shape
    for d in [s.dim[2], s.dim[3]]:
        d.ClearField('dim_value')
        d.dim_param = 'S'
del m.graph.value_info[:]
onnx.save(m, 'siamrpn_head_dyn.onnx')
"
```

### 4. Compile search encoder on Metis (with SDK active)

Copy `search_encoder_r50lt.onnx` to Metis, then:

```bash
source <SDK_ROOT>/venv/bin/activate

# 255×255 encoder (~18.5ms, recommended)
axcompile -i onnx_files/search_encoder_r50lt.onnx \
    --input-shape 1,3,255,255 \
    -o compiled_search_255 --overwrite \
    --imageset cal_images --dataset-len 400 \
    --transform transform_search.py

# 351×351 encoder (~37ms, legacy)
axcompile -i onnx_files/search_encoder_r50lt.onnx \
    --input-shape 1,3,351,351 \
    -o compiled_search_351 --overwrite \
    --imageset cal_images --dataset-len 400 \
    --transform transform_search.py
```

`transform_search.py` and calibration images should be placed in the Metis SDK `customers/` directory.

---

## Build

On Metis:

```bash
source <SDK_ROOT>/venv/bin/activate
cd lt_tracker/          # inside the cloned repo
# Pass your SDK path (default in Makefile is /opt/voyager-sdk)
make -f Makefile_poc SDK=<SDK_ROOT>
```

### Makefile variables

| Variable | Default | Description |
|---|---|---|
| `SDK` | `<SDK_ROOT>` | Voyager SDK root |
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
source <SDK_ROOT>/venv/bin/activate
cd lt_tracker/      # inside cloned repo
```

### Default (255px encoder, ~30 fps)

If you downloaded the compiled encoder from Google Drive, the model.json is one level shallower:

```bash
./poc_siamrpn_cpp \
    --search_encoder   build/siamrpn++onnx_255/1/model.json \
    --template_encoder template_encoder_r50lt.onnx \
    --head             siamrpn_head_dyn.onnx \
    --video            /path/to/video.mp4 \
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
| `--conf_low F` | `0.8` | Score threshold to enter LT mode |
| `--conf_high F` | `0.985` | Score threshold to exit LT mode |
| `--no_pf` | off | Disable particle filter (normal mode) |
| `--pf_n N` | `200` | Number of particle filter particles |
| `--pf_std F` | `12.0` | PF motion noise std (pixels/frame) |
| `--lt_stuck_timeout N` | `30` | Frames stuck in LT before global scan (0 = disable) |
| `--lt_stuck_thresh F` | `0.3` | Score threshold for "stuck in LT" detection |
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

### dw-xcorr implementation

Uses direct per-channel accumulation with `__restrict__` pointers and `#pragma GCC ivdep` to
trigger 8-wide AVX2 FMA vectorization. Avoids the ~240 MB im2col write that a BLAS approach
would require, keeping all working data (6.7 KB search + 5.4 KB output per channel) in L1 cache.

---

## Python tracker reference

`poc_siamrpn.py` is the Python equivalent using the same Axelera runtime.
It runs at ~19 fps with the 351px encoder (37ms Metis + 16ms xcorr_head on CPU).

```bash
source <SDK_ROOT>/venv/bin/activate
python poc_siamrpn.py
```

Edit `VIDEO`, `SRCH_JSON`, `INIT_BBOX` at the top of the file.
