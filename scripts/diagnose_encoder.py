#!/usr/bin/env python3
"""
diagnose_encoder.py
Compare raw feature map outputs from Axelera (INT8) vs CPU ONNX (FP32)
search encoder for the same input, using BOTH preprocessing styles:
  a) pysot-normalized:  /255, ImageNet mean/std  (what our tracker uses)
  b) raw uint8 pixels:  /255 only, no mean/std   (what poc_siamrpn.py uses)

Usage:
    python scripts/diagnose_encoder.py --config configs/track_config_axelera.json
"""
import argparse, json, os, sys
import numpy as np
import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)


class AxeleraQuantMixin:
    def _init_quant(self, model):
        self.input_infos    = model.inputs()
        self.output_infos   = model.outputs()
        self.input_info     = self.input_infos[0]
        self.input_buffers  = [np.zeros(t.shape, np.int8) for t in self.input_infos]
        self.output_buffers = [np.zeros(t.shape, np.int8) for t in self.output_infos]

    def _quantize(self, img):
        info = self.input_info
        if img.ndim == 4:
            img = img.transpose(0, 2, 3, 1).squeeze(0)
        elif img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        q = np.clip(np.round(img / info.scale + info.zero_point), -128, 127).astype(np.int8)
        if info.padding and any(p != (0, 0) for p in info.padding):
            q = np.pad(q, info.padding[1:], constant_values=info.zero_point)
        return q

    def _dequantize(self, idx):
        info = self.output_infos[idx]
        out  = self.output_buffers[idx]
        if info.padding and any(p != (0, 0) for p in info.padding):
            out = out[tuple(slice(b, -e or None) for b, e in info.padding)]
        result = (out.astype(np.float32) - info.zero_point) * info.scale
        return result.transpose(0, 3, 1, 2) if result.ndim == 4 else result

    @property
    def input_scale(self):
        return self.input_info.scale

    @property
    def input_zp(self):
        return self.input_info.zero_point

    @property
    def output_scales(self):
        return [(i.scale, i.zero_point) for i in self.output_infos]


class AxeleraEncoder(AxeleraQuantMixin):
    def __init__(self, ctx, model, aipu_cores=4):
        self._init_quant(model)
        conn = ctx.device_connect(None, 1)
        self.instance = conn.load_model_instance(model, num_sub_devices=1, aipu_cores=aipu_cores)

    def run(self, img_nchw):
        self.input_buffers[0][:] = self._quantize(img_nchw)
        self.instance.run(self.input_buffers, self.output_buffers)
        return [self._dequantize(i) for i in range(len(self.output_infos))]

    def run_raw_int8(self, img_nchw):
        """Return raw INT8 output buffers (before dequant) for debugging."""
        self.input_buffers[0][:] = self._quantize(img_nchw)
        self.instance.run(self.input_buffers, self.output_buffers)
        return [self.output_buffers[i].copy() for i in range(len(self.output_infos))]


def stats(arr, name):
    a = arr.flatten()
    print(f'    {name:30s}  shape={arr.shape}  '
          f'min={a.min():.4f}  max={a.max():.4f}  '
          f'mean={a.mean():.4f}  std={a.std():.4f}  '
          f'|nz|={np.count_nonzero(a):6d}/{a.size}')


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',        default='configs/track_config_axelera.json')
    ap.add_argument('--template-onnx', default='onnx/template_encoder.onnx')
    ap.add_argument('--search-onnx',   default='onnx/search_encoder.onnx')
    ap.add_argument('--frame-idx',     type=int, default=5)
    args = ap.parse_args()

    import onnxruntime as ort
    from pysot.core.config import cfg as pysot_cfg
    from pysot.models.model_builder import ModelBuilder
    from pysot.tracker.tracker_builder import build_tracker

    c = json.load(open(args.config))

    # ── Load frame ──
    cap = cv2.VideoCapture(c['paths']['video'])
    for _ in range(args.frame_idx + 1):
        ok, frame = cap.read()
    cap.release()
    print(f'Frame {args.frame_idx}: {frame.shape}  dtype={frame.dtype}')

    # ── Extract search patch using pysot ──
    pysot_cfg.defrost()
    pysot_cfg.merge_from_file(c['paths']['model_cfg'])
    pysot_cfg.CUDA = False
    pysot_cfg.freeze()

    model = ModelBuilder().eval()
    ckpt  = torch.load(c['paths']['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt.get('model', ckpt)))
    tracker = build_tracker(model)

    BBOX = c['init_bbox']
    tracker.init(frame, BBOX)
    cx0 = BBOX[0] + BBOX[2] / 2.0
    cy0 = BBOX[1] + BBOX[3] / 2.0

    tw, th  = float(tracker.size[0]), float(tracker.size[1])
    w_z = tw + pysot_cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
    h_z = th + pysot_cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
    s_x = float(np.sqrt(w_z * h_z)) * (pysot_cfg.TRACK.INSTANCE_SIZE / pysot_cfg.TRACK.EXEMPLAR_SIZE)

    # pysot-normalized patch (what the tracker actually uses)
    x_tensor = tracker.get_subwindow(frame, np.array([cx0, cy0]),
                                     pysot_cfg.TRACK.INSTANCE_SIZE,
                                     round(s_x), tracker.channel_average)
    x_pysot = x_tensor.cpu().numpy().astype(np.float32)   # (1,3,255,255) normalized

    # raw-pixel patch (what poc_siamrpn uses: just /255, no mean/std)
    raw_patch = tracker.get_subwindow(frame, np.array([cx0, cy0]),
                                      pysot_cfg.TRACK.INSTANCE_SIZE,
                                      round(s_x), tracker.channel_average)
    # Undo ImageNet normalization to get [0,1] image, then multiply back
    raw_np = raw_patch.cpu().numpy()[0].transpose(1, 2, 0)    # HWC
    raw_np = raw_np * STD + MEAN                               # un-normalize → [0,1]
    raw_np = raw_np.transpose(2, 0, 1)[np.newaxis]             # NCHW [0,1]
    x_raw = raw_np.astype(np.float32)                          # same range as poc

    print(f'\nInput tensor stats:')
    stats(x_pysot, 'x_pysot (normalized)')
    stats(x_raw,   'x_raw (/255 only, like POC)')

    # ── Build encoders ──
    opts = ort.SessionOptions(); opts.log_severity_level = 3
    srch_onnx = ort.InferenceSession(args.search_onnx, opts, providers=['CPUExecutionProvider'])
    tmpl_onnx = ort.InferenceSession(args.template_onnx, opts, providers=['CPUExecutionProvider'])

    from axelera.runtime import Context
    ctx        = Context()
    srch_model = ctx.load_model(c['paths']['search_encoder_ax'])
    tmpl_model = ctx.load_model(c['paths']['template_encoder_ax'])
    srch_ax = AxeleraEncoder(ctx, srch_model)
    tmpl_ax = AxeleraEncoder(ctx, tmpl_model)

    print(f'\nAxelera search encoder quantization:')
    print(f'  input  scale={srch_ax.input_scale:.6f}  zp={srch_ax.input_zp}')
    for i, (s, z) in enumerate(srch_ax.output_scales):
        print(f'  output[{i}]  scale={s:.6f}  zp={z}')

    # ── Run encoders with BOTH input styles ──
    print(f'\n{"="*70}')
    print('SEARCH ENCODER — pysot-normalized input')
    print('="*70')
    out_onnx_norm  = srch_onnx.run(None, {'search': x_pysot})
    out_ax_norm    = srch_ax.run(x_pysot)
    for i, (o, a) in enumerate(zip(out_onnx_norm, out_ax_norm)):
        print(f'\n  Feature[{i}]:')
        stats(o, f'  ONNX (FP32)')
        stats(a, f'  Axelera (INT8→deq)')
        print(f'    cosine_sim(ONNX, Axelera) = {cosine_sim(o, a):.6f}')
        print(f'    MAE = {np.mean(np.abs(o - a)):.6f}')
        print(f'    ONNX[:5] = {o.flatten()[:5]}')
        print(f'    AX  [:5] = {a.flatten()[:5]}')

    print(f'\n{"="*70}')
    print('SEARCH ENCODER — raw /255 input (like poc_siamrpn.py)')
    print('="*70')
    out_onnx_raw = srch_onnx.run(None, {'search': x_raw})
    out_ax_raw   = srch_ax.run(x_raw)
    for i, (o, a) in enumerate(zip(out_onnx_raw, out_ax_raw)):
        print(f'\n  Feature[{i}]:')
        stats(o, f'  ONNX (FP32)')
        stats(a, f'  Axelera (INT8→deq)')
        print(f'    cosine_sim(ONNX, Axelera) = {cosine_sim(o, a):.6f}')
        print(f'    MAE = {np.mean(np.abs(o - a)):.6f}')

    # ── Cross-input comparison: does normalization matter for Axelera? ──
    print(f'\n{"="*70}')
    print('CROSS-INPUT: does preprocessing matter?')
    print('="*70')
    for i in range(len(out_onnx_norm)):
        print(f'\n  Feature[{i}]:')
        print(f'    cosine_sim(ONNX_norm, ONNX_raw)  = {cosine_sim(out_onnx_norm[i], out_onnx_raw[i]):.6f}')
        print(f'    cosine_sim(AX_norm,   AX_raw)    = {cosine_sim(out_ax_norm[i],   out_ax_raw[i]):.6f}')
        print(f'    cosine_sim(ONNX_norm, AX_norm)   = {cosine_sim(out_onnx_norm[i], out_ax_norm[i]):.6f}')
        print(f'    cosine_sim(ONNX_raw,  AX_raw)    = {cosine_sim(out_onnx_raw[i],  out_ax_raw[i]):.6f}')

    # ── TEMPLATE ENCODER ──
    z_tensor = tracker.get_subwindow(frame, np.array([cx0, cy0]),
                                     pysot_cfg.TRACK.EXEMPLAR_SIZE,
                                     round(float(np.sqrt(w_z * h_z))),
                                     tracker.channel_average)
    z_pysot = z_tensor.cpu().numpy().astype(np.float32)
    z_raw   = z_pysot[0].transpose(1, 2, 0) * STD + MEAN
    z_raw   = z_raw.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    print(f'\n{"="*70}')
    print('TEMPLATE ENCODER — pysot-normalized input')
    print('="*70')
    out_tmpl_onnx = tmpl_onnx.run(None, {'template': z_pysot})
    out_tmpl_ax   = tmpl_ax.run(z_pysot)
    # Apply center crop to Axelera output (15→7)
    out_tmpl_ax_crop = []
    for f in out_tmpl_ax:
        if f.shape[-1] > 7:
            c7 = (f.shape[-1] - 7) // 2
            out_tmpl_ax_crop.append(f[:, :, c7:c7+7, c7:c7+7])
        else:
            out_tmpl_ax_crop.append(f)

    for i, (o, a) in enumerate(zip(out_tmpl_onnx, out_tmpl_ax_crop)):
        print(f'\n  Feature[{i}]:')
        stats(o, f'  ONNX (FP32)')
        stats(a, f'  Axelera (INT8→deq, cropped)')
        print(f'    cosine_sim = {cosine_sim(o, a):.6f}   MAE = {np.mean(np.abs(o - a)):.6f}')

    print(f'\nAxelera template encoder quantization:')
    print(f'  input  scale={tmpl_ax.input_scale:.6f}  zp={tmpl_ax.input_zp}')
    for i, (s, z) in enumerate(tmpl_ax.output_scales):
        print(f'  output[{i}]  scale={s:.6f}  zp={z}')


if __name__ == '__main__':
    main()
