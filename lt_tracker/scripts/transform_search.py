"""
Calibration transform for the siamrpn_r50_l234_dwxcorr_lt search encoder.

Used with axcompile --transform to preprocess calibration images for the
LT search encoder (255×255 input). Matches pysot get_subwindow: BGR float32
CHW [0, 255] — no ImageNet normalisation.

Place this file in the SDK customers/ directory before running axcompile:
    cp lt_tracker/scripts/transform_search.py <SDK_ROOT>/customers/<your_dir>/
"""
import numpy as np
import cv2
import torch


def get_preprocess_transform(image) -> torch.Tensor:
    """Preprocess a calibration frame for the LT SiamRPN++ search encoder.

    Accepts a PIL Image (RGB) or an HWC BGR uint8 numpy array.
    Center-crops to square, resizes to 255×255, returns a (3, 255, 255)
    BGR float32 tensor in [0, 255] — no ImageNet normalisation.
    """
    from PIL import Image as _PILImage

    if isinstance(image, _PILImage.Image):
        img = np.array(image.convert('RGB'))[:, :, ::-1].copy()  # RGB → BGR
    elif hasattr(image, 'numpy'):                                  # axelera wrapper
        img = np.array(image.numpy())
    else:
        img = np.asarray(image).copy()

    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    img = img[y0:y0 + side, x0:x0 + side]
    img = cv2.resize(img, (255, 255), interpolation=cv2.INTER_LINEAR)

    # HWC uint8 BGR → CHW float32 [0, 255]
    return torch.from_numpy(img.copy()).permute(2, 0, 1).float()


# ── Fix 1: register module so pickle can find this function ──────────────────
import sys as _sys, types as _types
if 'transform_search' not in _sys.modules:
    _m = _types.ModuleType('transform_search')
    _m.__dict__.update(globals())
    _sys.modules['transform_search'] = _m

# ── Fix 2: replace ForkingPickler with CloudPickler so axcompile's internal
# lambdas can be serialised across worker processes. ─────────────────────────
try:
    import pickle as _pickle
    import cloudpickle as _cp
    import multiprocessing as _mp
    import multiprocessing.reduction as _mpr
    import multiprocessing.queues as _mpq

    if not getattr(_pickle, '_cp_patched', False):
        _pickle.dumps = _cp.dumps
        _pickle.dump  = _cp.dump

        _OrigFP = _mpr.ForkingPickler

        class _CPForkingPickler(_cp.CloudPickler):
            _extra_reducers = _OrigFP._extra_reducers
            _copyreg_dispatch_table = _OrigFP._copyreg_dispatch_table

        _mpr.ForkingPickler                        = _CPForkingPickler
        _mpq.ForkingPickler                        = _CPForkingPickler
        _mpr.AbstractReducer.ForkingPickler        = _CPForkingPickler
        if hasattr(_mp, 'reduction'):
            _mp.reduction.ForkingPickler           = _CPForkingPickler

        _pickle._cp_patched = True
except Exception:
    pass
