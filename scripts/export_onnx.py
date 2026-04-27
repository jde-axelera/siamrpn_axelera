"""
Export best SiamRPN++ checkpoint to ONNX (opset 17).

Exports TWO models:
  1. template_encoder.onnx  — encodes the target template patch
       input:  template  (1, 3, 127, 127)
       output: zf_0, zf_1, zf_2   (multi-scale features from neck)

  2. tracker.onnx  — correlates template features with each search frame
       inputs: zf_0, zf_1, zf_2, search (1, 3, 255, 255)
       outputs: cls (1, 10, 25, 25),  loc (1, 20, 25, 25)

Usage:
    python export_onnx.py \
        --cfg pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
        --ckpt pysot/snapshot/all_datasets/best_model.pth \
        --out  exported/
"""
import argparse, os, sys
import torch
import torch.nn as nn
import onnx, onnxruntime as ort
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "pysot"))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder


# ── wrappers for clean ONNX graphs ───────────────────────────────────────────
class TemplateEncoder(nn.Module):
    """z -> (zf_0, zf_1, zf_2)"""
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck     = model.neck

    def forward(self, z):
        feats = self.backbone(z)          # list of 3 tensors
        zf    = self.neck(feats)          # adjusted features
        return tuple(zf)


class Tracker(nn.Module):
    """(zf_0, zf_1, zf_2, search) -> (cls, loc)"""
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck     = model.neck
        self.rpn_head = model.rpn_head

    def forward(self, zf0, zf1, zf2, search):
        zf   = [zf0, zf1, zf2]
        xf   = self.neck(self.backbone(search))
        cls, loc = self.rpn_head(zf, xf)
        return cls, loc


def export(model, wrapper_cls, dummy_inputs, input_names, output_names,
           dynamic_axes, out_path, opset=17):
    wrapper = wrapper_cls(model).eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy_inputs,
            out_path,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    # verify
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX model OK: {out_path}")

    # quick runtime check
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    feed = {inp.name: inp_arr.numpy()
            for inp, inp_arr in zip(sess.get_inputs(),
                                    dummy_inputs if isinstance(dummy_inputs, (list, tuple))
                                    else [dummy_inputs])}
    sess.run(None, feed)
    print(f"  Runtime check OK.")


def main():
    parser = argparse.ArgumentParser("SiamRPN++ ONNX export")
    parser.add_argument("--cfg",    required=True)
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--out",    default="exported")
    parser.add_argument("--opset",  type=int, default=17)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    os.makedirs(args.out, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    model = ModelBuilder().eval()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {args.ckpt} (epoch {ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss', '?')})")

    # ── dummy inputs ──────────────────────────────────────────────────────────
    z  = torch.zeros(1, 3, 127, 127)
    x  = torch.zeros(1, 3, 255, 255)
    with torch.no_grad():
        zf = list(model.neck(model.backbone(z)))   # get actual shapes
    zf0_shape, zf1_shape, zf2_shape = zf[0].shape, zf[1].shape, zf[2].shape
    print(f"Template feature shapes: {zf0_shape}, {zf1_shape}, {zf2_shape}")

    # ── export template encoder ───────────────────────────────────────────────
    enc_path = os.path.join(args.out, "template_encoder.onnx")
    print("\nExporting template_encoder.onnx ...")
    export(
        model,
        TemplateEncoder,
        dummy_inputs=(z,),
        input_names=["template"],
        output_names=["zf_0", "zf_1", "zf_2"],
        dynamic_axes={"template": {0: "batch"}},
        out_path=enc_path,
        opset=args.opset,
    )

    # ── export tracker ────────────────────────────────────────────────────────
    trk_path = os.path.join(args.out, "tracker.onnx")
    print("\nExporting tracker.onnx ...")
    zf_dummy = [torch.zeros(*s) for s in [zf0_shape, zf1_shape, zf2_shape]]
    export(
        model,
        Tracker,
        dummy_inputs=(*zf_dummy, x),
        input_names=["zf_0", "zf_1", "zf_2", "search"],
        output_names=["cls", "loc"],
        dynamic_axes={"search": {0: "batch"}, "cls": {0: "batch"}, "loc": {0: "batch"}},
        out_path=trk_path,
        opset=args.opset,
    )

    print(f"\nDone. Files in: {args.out}/")
    print(f"  template_encoder.onnx  — run once per target initialisation")
    print(f"  tracker.onnx           — run per frame")


if __name__ == "__main__":
    main()
