#!/usr/bin/env python3
"""
export_xcorr_head_ax.py — Export xcorr_head_ax.onnx for Axelera pipeline
=========================================================================

The Axelera compiled template_encoder outputs raw neck features (1,256,15,15)
instead of post-conv_kernel weights (1,256,7,7). This script exports an
xcorr_head_ax.onnx that accepts the raw 15×15 template features and internally
applies conv_kernel before cross-correlation.

Inputs  (at runtime):
    zf_0, zf_1, zf_2  — (1, 256, 15, 15)  template neck features  [from Metis]
    xf_0, xf_1, xf_2  — (1, 256, 31, 31)  search  neck features   [from Metis]

Outputs:
    cls — (1, 10, 25, 25)
    loc — (1, 20, 25, 25)

Usage:
    python scripts/export_xcorr_head_ax.py \
        --cfg   configs/config_ir_siamese_infer.yaml \
        --ckpt  checkpoints/best_model.pth \
        --out   onnx/xcorr_head_ax.onnx
"""
import argparse, os, sys
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'pysot'))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder


class XCorrHeadAx(nn.Module):
    """xcorr_head that accepts raw 15×15 template neck features.

    Wraps MultiRPN.forward directly — each rpn level applies conv_kernel
    internally before cross-correlation, so we just pass the raw neck features.
    """
    def __init__(self, model):
        super().__init__()
        self.rpn_head = model.rpn_head

    def forward(self, zf_0, zf_1, zf_2, xf_0, xf_1, xf_2):
        z_fs = [zf_0, zf_1, zf_2]
        x_fs = [xf_0, xf_1, xf_2]
        cls, loc = self.rpn_head(z_fs, x_fs)
        return cls, loc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',   required=True)
    ap.add_argument('--ckpt',  required=True)
    ap.add_argument('--out',   default='onnx/xcorr_head_ax.onnx')
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.CUDA = False
    cfg.freeze()

    model = ModelBuilder()
    ckpt  = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt.get('model', ckpt)))
    model.eval()

    head = XCorrHeadAx(model).eval()

    # Template: 15×15 neck features (Metis output)
    # Search:   31×31 neck features (Metis output)
    zf_dummy = [torch.zeros(1, 256, 15, 15) for _ in range(3)]
    xf_dummy = [torch.zeros(1, 256, 31, 31) for _ in range(3)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f'Exporting xcorr_head_ax.onnx → {args.out}')
    with torch.no_grad():
        torch.onnx.export(
            head,
            tuple(zf_dummy + xf_dummy),
            args.out,
            opset_version=args.opset,
            input_names=['zf_0', 'zf_1', 'zf_2', 'xf_0', 'xf_1', 'xf_2'],
            output_names=['cls', 'loc'],
            dynamic_axes={n: {0: 'batch'} for n in
                          ['zf_0', 'zf_1', 'zf_2', 'xf_0', 'xf_1', 'xf_2', 'cls', 'loc']},
            dynamo=False,
        )

    # Verify
    onnx.checker.check_model(onnx.load(args.out))
    sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
    print('Inputs:')
    for i in sess.get_inputs():  print(f'  {i.name}: {i.shape}')
    print('Outputs:')
    for o in sess.get_outputs(): print(f'  {o.name}: {o.shape}')

    zf_np = [np.zeros((1, 256, 15, 15), np.float32) for _ in range(3)]
    xf_np = [np.zeros((1, 256, 31, 31), np.float32) for _ in range(3)]
    cls_out, loc_out = sess.run(None, {
        'zf_0': zf_np[0], 'zf_1': zf_np[1], 'zf_2': zf_np[2],
        'xf_0': xf_np[0], 'xf_1': xf_np[1], 'xf_2': xf_np[2],
    })
    print(f'cls: {cls_out.shape}  loc: {loc_out.shape}')
    print('Export OK')


if __name__ == '__main__':
    main()
