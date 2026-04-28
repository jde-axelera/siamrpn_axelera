#!/usr/bin/env python3
"""
Export SearchEncoder from siamrpn_r50_l234_dwxcorr_lt to ONNX for Metis compilation.

Usage:
    export PYTHONPATH=/path/to/pysot:$PYTHONPATH
    python scripts/export_search_encoder_r50lt.py \
        --model_pth  lt.pth \
        --config     pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml \
        --out        search_encoder_r50lt.onnx \
        --size       255
"""
import argparse
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pth', default='lt.pth')
    parser.add_argument('--config',    default='pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
    parser.add_argument('--out',       default='search_encoder_r50lt.onnx')
    parser.add_argument('--size',      type=int, default=255, help='Search patch size (255 or 351)')
    args = parser.parse_args()

    from pysot.core.config import cfg
    from pysot.models.model_builder import ModelBuilder

    cfg.merge_from_file(args.config)
    model = ModelBuilder()
    model.load_state_dict(torch.load(args.model_pth, map_location='cpu'))
    model.eval()

    class SearchEncoder(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.backbone = m.backbone
            self.neck     = getattr(m, 'neck', None)

        def forward(self, x):
            feats = self.backbone(x)
            if self.neck:
                feats = self.neck(feats)
            return tuple(feats.values()) if isinstance(feats, dict) else tuple(feats)

    se = SearchEncoder(model).eval()

    dummy = torch.zeros(1, 3, args.size, args.size)
    with torch.no_grad():
        outs = se(dummy)
    print(f"Search encoder ({args.size}px): {len(outs)} outputs")
    for i, o in enumerate(outs):
        print(f"  xf{i}: {tuple(o.shape)}")

    output_names = [f'xf{i}' for i in range(len(outs))]
    torch.onnx.export(
        se, dummy, args.out,
        input_names=['search'],
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
        print("IR version downgraded to 8")


if __name__ == '__main__':
    main()
