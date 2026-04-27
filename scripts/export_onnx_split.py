"""
export_onnx_split.py — Export SiamRPN++ as three separate ONNX models
======================================================================

Splits the tracker into:
  1. template_encoder.onnx  (Metis-compilable)
       input:  template (1, 3, 127, 127)
       output: zf_0, zf_1, zf_2

  2. search_encoder.onnx    (Metis-compilable)
       input:  search  (1, 3, 255, 255)
       output: xf_0, xf_1, xf_2

  3. xcorr_head.onnx        (CPU — dynamic-weight grouped conv)
       input:  zf_0, zf_1, zf_2, xf_0, xf_1, xf_2
       output: cls (1, 10, 25, 25),  loc (1, 20, 25, 25)

Usage:
    python scripts/export_onnx_split.py \\
        --cfg   configs/config_ir_siamese_infer.yaml \\
        --ckpt  checkpoints/best_model.pth \\
        --out   onnx/
"""
import argparse, os, sys
import torch
import torch.nn as nn
import onnx, onnxruntime as ort
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYSOT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))  # Arquimea/
sys.path.insert(0, os.path.join(PYSOT_ROOT, 'pysot'))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder


# ── Wrapper modules ───────────────────────────────────────────────────────────

class TemplateEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck     = model.neck

    def forward(self, z):
        return tuple(self.neck(self.backbone(z)))


class SearchEncoder(nn.Module):
    """Search branch: backbone + neck only. No xcorr."""
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck     = model.neck

    def forward(self, x):
        return tuple(self.neck(self.backbone(x)))


class XCorrHead(nn.Module):
    """RPN head: conv_kernel + conv_search + xcorr_depthwise + output heads."""
    def __init__(self, model):
        super().__init__()
        self.rpn_head = model.rpn_head

    def forward(self, zf0, zf1, zf2, xf0, xf1, xf2):
        cls, loc = self.rpn_head([zf0, zf1, zf2], [xf0, xf1, xf2])
        return cls, loc


# ── Export helper ─────────────────────────────────────────────────────────────

def export_and_verify(wrapper, dummy_inputs, input_names, output_names,
                      dynamic_axes, out_path, opset=17):
    wrapper = wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            out_path,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(out_path))

    sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    feed = {inp.name: t.numpy()
            for inp, t in zip(sess.get_inputs(),
                              dummy_inputs if isinstance(dummy_inputs, (list, tuple))
                              else [dummy_inputs])}
    outs = sess.run(None, feed)
    m = onnx.load(out_path)
    from collections import Counter
    ops = Counter(n.op_type for n in m.graph.node)
    grouped = [(n.name, next(a.i for a in n.attribute if a.name == 'group'))
               for n in m.graph.node if n.op_type == 'Conv'
               and any(a.name == 'group' and a.i > 1 for a in n.attribute)]
    print(f'  Nodes: {len(m.graph.node)}  |  ops: {dict(ops)}')
    print(f'  Grouped-conv nodes: {len(grouped)} {["group="+str(g) for _,g in grouped]}')
    print(f'  Output shapes: {[o.shape for o in outs]}')
    print(f'  OK → {out_path}')
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',   required=True)
    ap.add_argument('--ckpt',  required=True)
    ap.add_argument('--out',   default='onnx/')
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.CUDA = False
    cfg.freeze()

    model = ModelBuilder().eval()
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt.get('model', ckpt)), strict=True)
    print(f'Loaded: {args.ckpt}  epoch={ckpt.get("epoch","?")}  val_loss={ckpt.get("val_loss","?")}')

    os.makedirs(args.out, exist_ok=True)

    # ── Dummy inputs ──────────────────────────────────────────────────────────
    z = torch.zeros(1, 3, 127, 127)
    x = torch.zeros(1, 3, 255, 255)
    with torch.no_grad():
        zf = list(model.neck(model.backbone(z)))
        xf = list(model.neck(model.backbone(x)))
    print(f'Template features: {[f.shape for f in zf]}')
    print(f'Search  features:  {[f.shape for f in xf]}')

    # ── 1. template_encoder.onnx ──────────────────────────────────────────────
    print('\n[1/3] Exporting template_encoder.onnx ...')
    export_and_verify(
        TemplateEncoder(model),
        dummy_inputs=(z,),
        input_names=['template'],
        output_names=['zf_0', 'zf_1', 'zf_2'],
        dynamic_axes={'template': {0: 'batch'}},
        out_path=os.path.join(args.out, 'template_encoder.onnx'),
        opset=args.opset,
    )

    # ── 2. search_encoder.onnx ────────────────────────────────────────────────
    print('\n[2/3] Exporting search_encoder.onnx ...')
    export_and_verify(
        SearchEncoder(model),
        dummy_inputs=(x,),
        input_names=['search'],
        output_names=['xf_0', 'xf_1', 'xf_2'],
        dynamic_axes={'search': {0: 'batch'}},
        out_path=os.path.join(args.out, 'search_encoder.onnx'),
        opset=args.opset,
    )

    # ── 3. xcorr_head.onnx ────────────────────────────────────────────────────
    print('\n[3/3] Exporting xcorr_head.onnx ...')
    export_and_verify(
        XCorrHead(model),
        dummy_inputs=(zf[0], zf[1], zf[2], xf[0], xf[1], xf[2]),
        input_names=['zf_0', 'zf_1', 'zf_2', 'xf_0', 'xf_1', 'xf_2'],
        output_names=['cls', 'loc'],
        dynamic_axes={},
        out_path=os.path.join(args.out, 'xcorr_head.onnx'),
        opset=args.opset,
    )

    print('\nDone.')
    print('  template_encoder.onnx  →  Metis (run once at init / template update)')
    print('  search_encoder.onnx    →  Metis (run per frame)')
    print('  xcorr_head.onnx        →  CPU   (dynamic-weight grouped conv, 6 nodes)')


if __name__ == '__main__':
    main()
