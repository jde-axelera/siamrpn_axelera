"""
Minimal frame-1 comparison script — mirrors exactly what poc_siamrpn_cpp does.
Prints the same debug stats for xf, zf, xcorr parts, and center-anchor scores.
"""
import numpy as np
import cv2
import onnxruntime as ort
import torch
import torch.nn.functional as F
import sys, os

VIDEO      = "/path/to/video.mp4"
TMPL_ONNX  = "template_encoder_r50lt.onnx"
SRCH_JSON  = "build/siamrpn++onnx_255/siamrpn++onnx_255/1/model.json"
HEAD_ONNX  = "siamrpn_head_dyn.onnx"

INIT_BBOX  = (695, 345, 20, 30)   # x,y,w,h
EXEMPLAR   = 127
INSTANCE   = 351
STRIDE     = 8
BASE_SIZE  = 8
CONTEXT    = 0.5
RATIOS     = [0.33, 0.5, 1.0, 2.0, 3.0]
SCALE      = 8.0

def get_subwindow(img, cx, cy, model_sz, orig_sz, avg):
    c = (orig_sz + 1) / 2
    xmin = int(cx - c);  xmax = xmin + orig_sz
    ymin = int(cy - c);  ymax = ymin + orig_sz
    H, W = img.shape[:2]
    lp = max(0, -xmin); tp = max(0, -ymin)
    rp = max(0, xmax - W); bp = max(0, ymax - H)
    xmin += lp; xmax += lp; ymin += tp; ymax += tp
    if lp or tp or rp or bp:
        padded = np.full((H+tp+bp, W+lp+rp, 3), avg, dtype=np.uint8)
        padded[tp:tp+H, lp:lp+W] = img
        patch = padded[ymin:ymax, xmin:xmax]
    else:
        patch = img[ymin:ymax, xmin:xmax]
    if model_sz != orig_sz:
        patch = cv2.resize(patch, (model_sz, model_sz))
    return patch

def to_nchw(img):
    H, W = img.shape[:2]
    t = np.zeros((1, 3, H, W), dtype=np.float32)
    for c in range(3):
        t[0, c] = img[:, :, c].astype(np.float32)
    return t

# ---------- Axelera search encoder ----------
from axelera.runtime import Context
ctx = Context()
mdl = ctx.load_model(SRCH_JSON)
conn = ctx.device_connect(None, 1)
inst = conn.load_model_instance(mdl, num_sub_devices=1, aipu_cores=4)

in_infos  = mdl.inputs()
out_infos = mdl.outputs()
in_buf    = np.zeros(in_infos[0].shape, dtype=np.int8)
out_bufs  = [np.zeros(o.shape, dtype=np.int8) for o in out_infos]

def quantize_img(img_hwc, info):
    q = np.clip(np.round(img_hwc.astype(np.float32) / info.scale + info.zero_point),
                -128, 127).astype(np.int8)
    if info.padding and any(p != (0,0) for p in info.padding):
        q = np.pad(q, info.padding[1:], constant_values=int(info.zero_point))
    return q

def dequantize_out(buf, info):
    out = buf
    if info.padding and any(p != (0,0) for p in info.padding):
        out = out[tuple(slice(b, -e or None) for b, e in info.padding)]
    result = (out.astype(np.float32) - info.zero_point) * info.scale
    return result.transpose(0,3,1,2) if result.ndim == 4 else result   # NHWC→NCHW

def run_search_enc(img_hwc):
    in_buf[:] = quantize_img(img_hwc, in_infos[0])
    inst.run([in_buf], out_bufs)
    return [dequantize_out(out_bufs[i], out_infos[i]) for i in range(len(out_infos))]

# ---------- Template encoder (ONNX Runtime) ----------
tmpl_sess = ort.InferenceSession(TMPL_ONNX, providers=['CPUExecutionProvider'])
head_sess  = ort.InferenceSession(HEAD_ONNX,  providers=['CPUExecutionProvider'])

# ---------- Load frames ----------
cap = cv2.VideoCapture(VIDEO)
ok, frame0 = cap.read()   # frame 0 — template
ok, frame1 = cap.read()   # frame 1 — first tracking frame
cap.release()
assert ok, "Could not read frames"

x, y, w, h = INIT_BBOX
cx0, cy0 = x + w/2.0, y + h/2.0

avg = np.mean(frame0, axis=(0,1)).astype(np.float32)

# ---------- Template crop & encoding ----------
wz = w + CONTEXT*(w+h)
hz = h + CONTEXT*(w+h)
s_z = round(np.sqrt(wz*hz))
z_crop = get_subwindow(frame0, cx0, cy0, EXEMPLAR, s_z, avg)
z_nchw = to_nchw(z_crop)

zf_list = tmpl_sess.run(None, {tmpl_sess.get_inputs()[0].name: z_nchw})
print(f"Template encoder: {len(zf_list)} outputs")
for i, z in enumerate(zf_list):
    print(f"  zf[{i}] shape={z.shape}")

# ---------- Search crop & encoding (frame 1) ----------
scale_z = EXEMPLAR / np.sqrt(wz*hz)
s_x = np.sqrt(wz*hz) * (INSTANCE / EXEMPLAR)
orig_sz = round(s_x)
x_crop = get_subwindow(frame1, cx0, cy0, INSTANCE, orig_sz, avg)

print(f"\nSearch crop: cx={cx0} cy={cy0} s_x={s_x:.3f} orig_sz={orig_sz}")
print(f"Search crop shape: {x_crop.shape}, mean_per_ch={np.mean(x_crop, axis=(0,1))}")

xf_list = run_search_enc(x_crop)
print(f"Search encoder: {len(xf_list)} outputs")
for i, xf in enumerate(xf_list):
    print(f"  xf[{i}] shape={xf.shape}")

# ---------- XCorr (same logic as C++ dw_xcorr) ----------
parts = []
for i in range(6):
    s = torch.from_numpy(xf_list[i])    # [1, C, 41, 41]
    k = torch.from_numpy(zf_list[i]).transpose(0,1)  # [C, 1, 5, 5]
    p = F.conv2d(s, k, groups=s.shape[1])   # [1, C, 37, 37]
    parts.append(p.numpy())

print("\nXCorr parts (frame 1):")
ss = 37
AN = len(RATIOS)
for i, p in enumerate(parts):
    C, H, W = p.shape[1], p.shape[2], p.shape[3]
    flat = p.flatten()
    amax = np.argmax(flat)
    c_idx = amax // (H*W); rem = amax % (H*W)
    r_idx = rem // W; col_idx = rem % W
    center_sum = float(np.sum(p[0, :, ss//2, ss//2]))
    print(f"  part[{i}] C={C} H={H} W={W} max={flat[amax]:.4f} @[c={c_idx},r={r_idx},c={col_idx}]  center_sum={center_sum:.4f}")
    if i == 0 or i == 5:
        cH, cW = ss//2, ss//2
        print(f"    part[{i}] center(r={cH},c={cW}) first 5 chans: {p[0, :5, cH, cW]}")
        print(f"    xf[{i}] center(r=20,c=20) first 5 chans: {xf_list[i][0, :5, 20, 20]}")
        print(f"    zf[{i}] center(r=2,c=2) first 5 chans: {zf_list[i][0, :5, 2, 2]}")
        # Print zf[i] channel 2 all 25 values and xf[i] ch2 window [18:23,18:23]
        print(f"    zf[{i}] ch2 (5x5): {zf_list[i][0, 2, :, :].flatten()}")
        print(f"    xf[{i}] ch2 window[18:23,18:23]: {xf_list[i][0, 2, 18:23, 18:23].flatten()}")
        # Manual xcorr check for ch2 at center
        xcorr_manual = float(np.sum(zf_list[i][0, 2] * xf_list[i][0, 2, 18:23, 18:23]))
        print(f"    MANUAL xcorr[{i}][2,18,18] = {xcorr_manual:.6f} (part shows {p[0, 2, cH, cW]:.6f}")

# ---------- Head + scores ----------
xcorr_cat = np.concatenate(parts, axis=1).astype(np.float32)
print(f"\nxcorr_cat shape: {xcorr_cat.shape}")
cls_out, loc_out = head_sess.run(None, {'xcorr_features': xcorr_cat})
print(f"cls_out shape: {cls_out.shape}")

# Decode scores
N = AN * ss * ss
cls_flat = cls_out[0].reshape(2*AN, ss*ss)
e = np.exp(cls_flat - cls_flat.max(axis=0, keepdims=True))
scores = e[AN:] / e.sum(axis=0)    # softmax for class 1
all_scores = np.concatenate([scores[k] for k in range(AN)])  # [N]

# Center anchor scores
ctr = ss//2
print(f"\nCenter anchor scores (row={ctr},col={ctr}):")
for k in range(AN):
    cidx = k * ss*ss + ctr*ss + ctr
    print(f"  k{k}={all_scores[cidx]:.4f}")

# Top-5
top5 = np.argsort(all_scores)[::-1][:5]
print("Top-5 raw scores:")
ori = -(ss//2) * STRIDE
for rank, idx in enumerate(top5):
    k = idx // (ss*ss); rem = idx % (ss*ss)
    r = rem // ss; c = rem % ss
    acx = ori + c * STRIDE
    acy = ori + r * STRIDE
    print(f"  [{rank+1}] idx={idx} k={k} r={r} c={c} score={all_scores[idx]:.4f} acx={acx} acy={acy}")
