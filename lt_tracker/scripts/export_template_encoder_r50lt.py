#!/usr/bin/env python3
"""
Export TemplateEncoder from siamrpn_r50_l234_dwxcorr_lt to ONNX for use in poc_siamrpn_cpp.

Usage:
    export PYTHONPATH=/path/to/pysot:$PYTHONPATH
    python scripts/export_template_encoder_r50lt.py \
        --model_pth  model.pth \
        --config     pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml \
        --out        template_encoder_r50lt.onnx
"""
import sys, os, argparse
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pth', default='model.pth')
    parser.add_argument('--config',    default='pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
    parser.add_argument('--out',       default='template_encoder_r50lt.onnx')
    args = parser.parse_args()

    from pysot.core.config import cfg
    from pysot.models.model_builder import ModelBuilder

    cfg.merge_from_file(args.config)
    model = ModelBuilder()
    model.load_state_dict(torch.load(args.model_pth, map_location='cpu'))
    model.eval()

    class TemplateEncoder(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.backbone   = m.backbone
            self.neck       = getattr(m, 'neck', None)
            self.cls_kernels = nn.ModuleList([rpn.cls.conv_kernel for rpn in
                                              [m.rpn_head.rpn2, m.rpn_head.rpn3, m.rpn_head.rpn4]])
            self.loc_kernels = nn.ModuleList([rpn.loc.conv_kernel for rpn in
                                              [m.rpn_head.rpn2, m.rpn_head.rpn3, m.rpn_head.rpn4]])

        def forward(self, x):
            zf = self.backbone(x)
            if self.neck:
                zf = self.neck(zf)
            zf = list(zf.values()) if isinstance(zf, dict) else zf
            return [out for i, z in enumerate(zf)
                    for out in (self.cls_kernels[i](z), self.loc_kernels[i](z))]

    te = TemplateEncoder(model).eval()

    dummy = torch.zeros(1, 3, 127, 127)
    with torch.no_grad():
        outs = te(dummy)
    print(f"Template encoder: {len(outs)} outputs")
    for i, o in enumerate(outs):
        print(f"  k{i}: {tuple(o.shape)}")

    output_names = [f'k{i}' for i in range(len(outs))]
    torch.onnx.export(
        te, dummy, args.out,
        input_names=['template'],
        output_names=output_names,
        opset_version=11,
        do_constant_folding=True,
    )
    print(f"Saved: {args.out}")

    import onnx
    m = onnx.load(args.out)
    if m.ir_version > 9:
        m.ir_version = 8
        onnx.save(m, args.out)
        print("IR version downgraded to 8 for ORT 1.17.1 compatibility")


if __name__ == '__main__':
    main()
