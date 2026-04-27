"""
train_siamrpn_aws.py  —  SiamRPN++ fine-tuning on IR/thermal datasets
======================================================================
Trains on a weighted combination of 9 IR/thermal datasets (Anti-UAV410,
MSRS, VT-MOT, MassMIND, MVSS, DUT-VTUAV, Anti-UAV300, BIRDSAI, HIT-UAV).
Datasets missing their annotation JSON are skipped automatically.

Features:
  - Stochastic pair sampling: 10,000 (template, search) pairs per epoch
  - Warmup (5 ep) → SGDR cosine annealing (T0=50, T_mult=2)
  - ReduceLROnPlateau rescue (×0.3 after 15 stagnant epochs)
  - Backbone frozen for first 10 epochs, then fine-tuned at 0.1× LR
  - Early stopping: patience=50, relative delta=1e-4
  - Validation loss after every epoch; best checkpoint saved
  - Resume from any checkpoint
  - TensorBoard logging

Usage
-----
  conda activate pysot
  # Single GPU:
  python train_siamrpn_aws.py \
      --cfg  pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
      --pretrained pretrained/sot_resnet50.pth

  # Multi-GPU (4x T4) via DistributedDataParallel:
  torchrun --nproc_per_node=4 train_siamrpn_aws.py \
      --cfg  pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
      --pretrained pretrained/sot_resnet50.pth
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, json, logging, math, os, random, re, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import cv2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ── PySOT on path ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYSOT_DIR  = os.path.join(SCRIPT_DIR, "pysot")
sys.path.insert(0, PYSOT_DIR)

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.utils.model_load import load_pretrain
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.utils.bbox import center2corner, Center

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


# ── device helpers ────────────────────────────────────────────────────────────
def init_distributed():
    """Initialize DDP if launched via torchrun. set_device BEFORE init_process_group."""
    if "RANK" not in os.environ:
        return 0, 0, 1, True
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)        # MUST be before NCCL init
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, local_rank, world_size, (rank == 0)


def get_device(local_rank=0):
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        logger.info(f"CUDA: {n} GPU(s), using cuda:{local_rank}")
        return torch.device(f"cuda:{local_rank}")
    logger.warning("No CUDA found, falling back to CPU.")
    return torch.device("cpu")


def to_dev(t, device):
    return t.to(device=device, dtype=torch.float32) if t.is_floating_point() \
           else t.to(device)



def _get_center_crop(image, anno, output_size, exemplar_size):
    """
    Extract a square crop of `output_size` pixels centered on `anno`,
    padding with the image mean at boundaries.

    This pre-crops the full frame so that the annotation target is at the
    center of the returned image — a prerequisite for PySOT's Augmentation
    pipeline, which assumes the target is at the image center.

    Args:
        image       : BGR uint8 numpy array (H×W×3)
        anno        : [x1,y1,x2,y2] or [w,h] in image pixel coords
        output_size : desired square output size (e.g. 254 for template, 510 for search)
        exemplar_size: cfg.TRAIN.EXEMPLAR_SIZE (127)

    Returns:
        crop    : uint8 BGR array of shape (output_size, output_size, 3)
        new_anno: [x1,y1,x2,y2] in crop pixel coords (target at center)
    """
    if len(anno) == 4:
        ax1, ay1, ax2, ay2 = [float(v) for v in anno]
        w, h = ax2 - ax1, ay2 - ay1
        cx, cy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    else:
        w, h = float(anno[0]), float(anno[1])
        cx, cy = image.shape[1] / 2.0, image.shape[0] / 2.0

    w = max(w, 1.0); h = max(h, 1.0)  # guard against degenerate boxes

    context = 0.5 * (w + h)
    s_z = np.sqrt((w + context) * (h + context))  # PySOT exemplar window size

    # How many original-image pixels should map to output_size pixels
    # so that the target fills ~exemplar_size pixels in the output.
    orig_crop_size = s_z * (output_size / exemplar_size)

    avg_chans = image.mean(axis=(0, 1)).tolist()
    H, W = image.shape[:2]

    half = orig_crop_size / 2.0
    x0 = int(np.floor(cx - half + 0.5))
    y0 = int(np.floor(cy - half + 0.5))
    x1c = x0 + int(orig_crop_size)
    y1c = y0 + int(orig_crop_size)

    lp = max(0, -x0);   tp = max(0, -y0)
    rp = max(0, x1c - W); bp = max(0, y1c - H)

    if any([lp, tp, rp, bp]):
        img_pad = cv2.copyMakeBorder(
            image, tp, bp, lp, rp,
            cv2.BORDER_CONSTANT, value=avg_chans)
        crop = img_pad[y0+tp : y1c+tp, x0+lp : x1c+lp]
    else:
        crop = image[y0:y1c, x0:x1c]

    if crop.shape[0] == 0 or crop.shape[1] == 0:
        crop = np.full((output_size, output_size, 3), avg_chans, dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (output_size, output_size))

    # Annotation in crop coords: target should be at (output_size/2, output_size/2)
    scale = output_size / orig_crop_size
    new_cx, new_cy = output_size / 2.0, output_size / 2.0
    new_w, new_h = w * scale, h * scale
    new_anno = [new_cx - new_w/2, new_cy - new_h/2,
                new_cx + new_w/2, new_cy + new_h/2]

    return crop, new_anno


# ── Dataset ───────────────────────────────────────────────────────────────────
class AntiUAV410Dataset(Dataset):
    def __init__(self, root, anno_path, frame_range=50, epoch_len=None):
        super().__init__()
        self.root = root
        self.frame_range = frame_range

        with open(anno_path) as f:
            self.labels = json.load(f)

        self.sequences = []
        for seq, tracks in self.labels.items():
            frames = sorted(int(k) for k in tracks["0"].keys())
            if len(frames) > 1:
                self.sequences.append((seq, frames))

        self.anchor_target = AnchorTarget()
        self.template_aug = Augmentation(
            cfg.DATASET.TEMPLATE.SHIFT, cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,  cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR)
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT, cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,  cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR)

        n = len(self.sequences)
        self.length = epoch_len if epoch_len and epoch_len > 0 else n
        logger.info(f"Dataset ({root.split('/')[-1]}): {n} seqs, length={self.length}")

    def __len__(self):
        return self.length

    def _get_bbox(self, image, anno):
        if len(anno) == 4:
            x1, y1, x2, y2 = anno
            w, h = x2 - x1, y2 - y1
        else:
            w, h = anno
        ctx = 0.5
        ez  = cfg.TRAIN.EXEMPLAR_SIZE
        s   = np.sqrt((w + ctx*(w+h)) * (h + ctx*(w+h)))
        sc  = ez / s
        cx, cy = image.shape[1]//2, image.shape[0]//2
        return center2corner(Center(cx, cy, w*sc, h*sc))

    def __getitem__(self, index):
        seq, frames = self.sequences[index % len(self.sequences)]
        track = self.labels[seq]["0"]

        t_idx = random.randint(0, len(frames)-1)
        lo    = max(t_idx - self.frame_range, 0)
        hi    = min(t_idx + self.frame_range, len(frames)-1) + 1
        s_idx = random.randint(lo, hi-1)
        tf, sf = frames[t_idx], frames[s_idx]

        t_img = cv2.imread(os.path.join(self.root, seq, f"{tf:06d}.jpg"))
        s_img = cv2.imread(os.path.join(self.root, seq, f"{sf:06d}.jpg"))
        if t_img is None or s_img is None:
            t_img = cv2.imread(os.path.join(self.root, seq, f"{frames[0]:06d}.jpg"))
            s_img = t_img.copy()
            tf = sf = frames[0]

        t_anno = track[f"{tf:06d}"]
        s_anno = track[f"{sf:06d}"]
        gray   = cfg.DATASET.GRAY and cfg.DATASET.GRAY > random.random()

        # Pre-crop around annotation center so Augmentation receives a square
        # image with the target at center (PySOT's required input format).
        t_crop, t_anno_c = _get_center_crop(t_img, t_anno,
                                             cfg.TRAIN.EXEMPLAR_SIZE * 2,
                                             cfg.TRAIN.EXEMPLAR_SIZE)
        s_crop, s_anno_c = _get_center_crop(s_img, s_anno,
                                             cfg.TRAIN.SEARCH_SIZE * 2,
                                             cfg.TRAIN.EXEMPLAR_SIZE)
        template, _ = self.template_aug(t_crop, self._get_bbox(t_crop, t_anno_c),
                                        cfg.TRAIN.EXEMPLAR_SIZE, gray=gray)
        search, bbox = self.search_aug(s_crop, self._get_bbox(s_crop, s_anno_c),
                                       cfg.TRAIN.SEARCH_SIZE, gray=gray)
        cls, delta, delta_weight, _ = self.anchor_target(
            bbox, cfg.TRAIN.OUTPUT_SIZE, neg=False)

        return {
            "template":         template.transpose(2,0,1).astype(np.float32),
            "search":           search.transpose(2,0,1).astype(np.float32),
            "label_cls":        cls,
            "label_loc":        delta,
            "label_loc_weight": delta_weight,
            "bbox":             np.array(bbox),
        }


# ── Base class shared by all extra datasets ───────────────────────────────────
class IRTrackingDatasetBase(Dataset):
    """
    Generic dataset loader for any dataset converted to PySOT JSON format.
    Subclasses override _load_image() to handle modality-specific reading.
    """
    def __init__(self, root, anno_path, frame_range=50, epoch_len=None, name=""):
        super().__init__()
        self.root        = root
        self.frame_range = frame_range
        self.name        = name

        if not os.path.isfile(anno_path):
            logger.warning(f"[{name}] annotation file not found: {anno_path} — dataset skipped.")
            self.sequences = []
            self.length    = 0
            return

        with open(anno_path) as f:
            self.labels = json.load(f)

        self.sequences = []
        for seq, tracks in self.labels.items():
            frames = sorted(int(k) for k in tracks["0"].keys())
            if len(frames) >= 1:
                self.sequences.append((seq, frames))

        self.anchor_target = AnchorTarget()
        self.template_aug  = Augmentation(
            cfg.DATASET.TEMPLATE.SHIFT, cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,  cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR)
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT, cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,  cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR)

        n = len(self.sequences)
        # IMPORTANT: keep length=0 when empty so CombinedDataset filters it out
        self.length = epoch_len if (epoch_len and epoch_len > 0 and n > 0) else n
        logger.info(f"[{name}] {n} sequences, epoch_len={self.length}")

    def __len__(self):
        return self.length

    def _load_image(self, path):
        """Load image and ensure 3-channel uint8 BGR (OpenCV convention)."""
        img = cv2.imread(path)
        if img is None:
            return None
        if len(img.shape) == 2:           # grayscale LWIR → replicate to 3ch
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _get_bbox(self, image, anno):
        if len(anno) == 4:
            x1, y1, x2, y2 = anno
            w, h = x2 - x1, y2 - y1
        else:
            w, h = anno
        ctx = 0.5
        ez  = cfg.TRAIN.EXEMPLAR_SIZE
        s   = np.sqrt((w + ctx*(w+h)) * (h + ctx*(w+h)))
        sc  = ez / max(s, 1e-3)
        cx, cy = image.shape[1]//2, image.shape[0]//2
        return center2corner(Center(cx, cy, w*sc, h*sc))

    def _find_image(self, seq, frame_id):
        """Try common extensions for a given sequence/frame."""
        for ext in (".jpg", ".png", ".bmp", ".jpeg"):
            p = os.path.join(self.root, seq, f"{frame_id:06d}{ext}")
            if os.path.isfile(p):
                return p
        return None

    def __getitem__(self, index):
        if not self.sequences:
            raise IndexError("Dataset is empty (check annotation path).")
        seq, frames = self.sequences[index % len(self.sequences)]
        track = self.labels[seq]["0"]

        # Sample a template and search frame within frame_range
        t_idx = random.randint(0, len(frames)-1)
        lo    = max(t_idx - self.frame_range, 0)
        hi    = min(t_idx + self.frame_range, len(frames)-1) + 1
        s_idx = random.randint(lo, hi-1)
        tf, sf = frames[t_idx], frames[s_idx]

        t_path = self._find_image(seq, tf)
        s_path = self._find_image(seq, sf)

        # Fallback: use first valid frame for both
        if t_path is None or s_path is None:
            tf = sf = frames[0]
            t_path = s_path = self._find_image(seq, tf)

        t_img = self._load_image(t_path) if t_path else None
        s_img = self._load_image(s_path) if s_path else None

        if t_img is None:
            # Return zeros so training can continue.
            # Shapes must match AnchorTarget output exactly to avoid collate errors:
            #   cls   : (num_anchors, OUTPUT_SIZE, OUTPUT_SIZE)
            #   delta : (4*num_anchors, OUTPUT_SIZE, OUTPUT_SIZE)
            _na = cfg.ANCHOR.ANCHOR_NUM   # typically 5
            _os = cfg.TRAIN.OUTPUT_SIZE
            z   = np.zeros((3, cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE), dtype=np.float32)
            x   = np.zeros((3, cfg.TRAIN.SEARCH_SIZE,   cfg.TRAIN.SEARCH_SIZE),   dtype=np.float32)
            cls   = np.zeros((_na, _os, _os),    dtype=np.int64)
            delta = np.zeros((4, _na, _os, _os), dtype=np.float32)
            dw    = np.zeros((_na, _os, _os),    dtype=np.float32)  # same shape as cls
            return {"template": z, "search": x, "label_cls": cls,
                    "label_loc": delta, "label_loc_weight": dw,
                    "bbox": np.zeros(4, dtype=np.float32)}

        t_anno = track.get(f"{tf:06d}", track[next(iter(track))])
        s_anno = track.get(f"{sf:06d}", track[next(iter(track))])
        gray   = cfg.DATASET.GRAY and cfg.DATASET.GRAY > random.random()

        # Pre-crop around annotation center so Augmentation receives a square
        # image with the target at center (PySOT's required input format).
        s_img_eff = s_img if s_img is not None else t_img
        t_crop, t_anno_c = _get_center_crop(t_img, t_anno,
                                             cfg.TRAIN.EXEMPLAR_SIZE * 2,
                                             cfg.TRAIN.EXEMPLAR_SIZE)
        s_crop, s_anno_c = _get_center_crop(s_img_eff, s_anno,
                                             cfg.TRAIN.SEARCH_SIZE * 2,
                                             cfg.TRAIN.EXEMPLAR_SIZE)
        template, _ = self.template_aug(t_crop, self._get_bbox(t_crop, t_anno_c),
                                        cfg.TRAIN.EXEMPLAR_SIZE, gray=gray)
        search, bbox = self.search_aug(s_crop, self._get_bbox(s_crop, s_anno_c),
                                       cfg.TRAIN.SEARCH_SIZE, gray=gray)
        cls, delta, delta_weight, _ = self.anchor_target(
            bbox, cfg.TRAIN.OUTPUT_SIZE, neg=False)

        return {
            "template":         template.transpose(2,0,1).astype(np.float32),
            "search":           search.transpose(2,0,1).astype(np.float32),
            "label_cls":        cls,
            "label_loc":        delta,
            "label_loc_weight": delta_weight,
            "bbox":             np.array(bbox),
        }


# ── MSRS dataset (paired IR/visible, 480×640) ─────────────────────────────────
class MSRSDataset(IRTrackingDatasetBase):
    """
    MSRS: Multi-Spectral Road Scene.
    Uses the IR channel only. Images live in <root>/ir/<name>.png.
    Pseudo-sequences: pairs of consecutive IR images.
    """
    def __init__(self, root, anno_path, frame_range=1, epoch_len=None):
        # Override root to point to the ir/ subfolder
        ir_root = root  # anno image paths use full root; _find_image adds seq/frame
        super().__init__(ir_root, anno_path, frame_range, epoch_len, name="MSRS")
        self.ir_dir = os.path.join(root, "ir") if os.path.isdir(
            os.path.join(root, "ir")) else root

    def _find_image(self, seq, frame_id):
        # seq = msrs_<split>_NNNNN — encode the pair start index N
        # frame_id 1 -> files[N], frame_id 2 -> files[N+1]
        imgs = sorted(f for f in os.listdir(self.ir_dir)
                      if f.lower().endswith((".png", ".jpg", ".bmp")))
        try:
            seq_idx = int(seq.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            seq_idx = 0
        idx = seq_idx + (frame_id - 1)
        idx = min(idx, len(imgs) - 1)
        if idx < 0 or not imgs:
            return None
        return os.path.join(self.ir_dir, imgs[idx])


# ── VT-MOT / PFTrack dataset (RGB+IR multi-object tracking) ──────────────────
class VTMOTDataset(IRTrackingDatasetBase):
    """
    VT-MOT: Visible-Thermal Multiple Object Tracking.
    Each sequence has infrared/ and visible/ subfolders.
    Uses the infrared channel. Annotations converted from MOT gt.txt.
    seq format in JSON: '<seq_folder>_obj<id>' — we split on '_obj' to
    recover the folder name.
    """
    def __init__(self, root, anno_path, frame_range=30, epoch_len=None):
        super().__init__(root, anno_path, frame_range, epoch_len, name="VT-MOT")

    def _find_image(self, seq, frame_id):
        # seq is like 'seq001_obj003'; extract folder name
        folder = seq.rsplit("_obj", 1)[0]
        for subdir in ("infrared", "ir", "thermal"):
            for ext in (".jpg", ".png"):
                p = os.path.join(self.root, folder, subdir, f"{frame_id:06d}{ext}")
                if os.path.isfile(p):
                    return p
        return None


# ── MassMIND dataset (LWIR maritime, single-channel 640×512) ─────────────────
class MassMINDDataset(IRTrackingDatasetBase):
    """
    MassMIND: 2,916 Long Wave Infrared maritime images.
    Single-channel LWIR → replicated to 3ch in _load_image().
    Pseudo-sequences: same image used as template and search.
    """
    def __init__(self, root, anno_path, frame_range=1, epoch_len=None):
        super().__init__(root, anno_path, frame_range, epoch_len, name="MassMIND")

    def _find_image(self, seq, frame_id):
        # seq is 'massmind_<stem>_cls<id>'; the image is root/**/<stem>.png
        stem = seq.split("_cls")[0].replace("massmind_", "", 1)
        for dirpath, _, filenames in os.walk(self.root):
            for fn in filenames:
                if os.path.splitext(fn)[0] == stem:
                    return os.path.join(dirpath, fn)
        return None


# ── MVSS-Baseline dataset (RGB-thermal video with seg labels) ────────────────
class MVSSDataset(IRTrackingDatasetBase):
    """
    MVSS-Baseline: Multi-modal Video Semantic Segmentation.
    Uses the thermal channel. Annotations derived from semantic seg masks.
    seq format: '<sequence_folder>_cls<id>'
    """
    def __init__(self, root, anno_path, frame_range=20, epoch_len=None):
        super().__init__(root, anno_path, frame_range, epoch_len, name="MVSS")

    def _find_image(self, seq, frame_id):
        folder = seq.rsplit("_cls", 1)[0]
        for subdir in ("thermal", "ir", "infrared", "images"):
            for ext in (".png", ".jpg"):
                p = os.path.join(self.root, folder, subdir, f"{frame_id:06d}{ext}")
                if os.path.isfile(p):
                    return p
                # try zero-padded with different widths
                p = os.path.join(self.root, folder, subdir,
                                 f"{frame_id:04d}{ext}")
                if os.path.isfile(p):
                    return p
        return None


# ── DUT-VTUAV dataset (RGB+Thermal UAV, 500 seqs 1920×1080) ──────────────────
class DUTVTUAVDataset(IRTrackingDatasetBase):
    """
    DUT-VTUAV: each sequence has infrared/ and visible/ subdirs.
    We use the infrared channel exclusively for IR-domain training.
    groundtruth.txt: one 'x y w h' line per frame.
    """
    def __init__(self, root, anno_path, frame_range=50, epoch_len=None):
        super().__init__(root, anno_path, frame_range, epoch_len, name="DUT-VTUAV")

    def _find_image(self, seq, frame_id):
        for ext in (".jpg", ".png"):
            p = os.path.join(self.root, seq, "infrared", f"{frame_id:06d}{ext}")
            if os.path.isfile(p):
                return p
        return None



# ── Anti-UAV 300 dataset ──────────────────────────────────────────────────────
class AntiUAV300Dataset(IRTrackingDatasetBase):
    """
    Anti-UAV300: sequences stored as infrared.mp4 + infrared.json per video.
    The converter (convert_antiuav300) extracts frames to <seq>/<i:06d>.jpg
    and writes train_pysot.json / val_pysot.json in PySOT format.
    Images live at <root>/<seq>/<frame>.jpg  (same layout as AntiUAV410).
    """
    def __init__(self, root, anno_path, frame_range=50, epoch_len=None):
        super().__init__(root, anno_path, frame_range, epoch_len, name="AntiUAV300")

    def _find_image(self, seq, frame_id):
        for ext in (".jpg", ".png"):
            p = os.path.join(self.root, seq, f"{frame_id:06d}{ext}")
            if os.path.isfile(p):
                return p
        return None


# ── BIRDSAI dataset (TIR aerial wildlife, MOT-derived SOT) ───────────────────
class BIRDSAIDataset(IRTrackingDatasetBase):
    """
    BIRDSAI: frames/ subdir holds the TIFF/JPG thermal images.
    Annotation JSON keys look like '<seq>_obj<id>'; the folder name is
    extracted by splitting on '_obj'.
    """
    def __init__(self, root, anno_path, frame_range=30, epoch_len=None):
        super().__init__(root, anno_path, frame_range, epoch_len, name="BIRDSAI")

    def _find_image(self, seq, frame_id):
        folder = seq.rsplit("_obj", 1)[0]
        for subdir in ("frames", "ir", "thermal", ""):
            for ext in (".jpg", ".png", ".tiff", ".tif"):
                p = os.path.join(self.root, folder, subdir, f"{frame_id:06d}{ext}")
                if os.path.isfile(p):
                    return p
        return None


# ── HIT-UAV dataset (IR detection → pseudo-SOT) ───────────────────────────────
class HITUAVDataset(IRTrackingDatasetBase):
    """
    HIT-UAV: images are flat inside images/<seq_prefix>_<frame>.jpg.
    The JSON key is '<prefix>_cls<id>'; image lookup strips the _cls<id>
    suffix and rebuilds the original filename.
    """
    def __init__(self, root, anno_path, frame_range=5, epoch_len=None):
        super().__init__(root, anno_path, frame_range, epoch_len, name="HIT-UAV")

    def _find_image(self, seq, frame_id):
        # seq = '<prefix>_cls<id>' — strip class suffix to get image prefix
        prefix = re.sub(r"_cls\d+$", "", seq)
        for ext in (".jpg", ".png"):
            # flat layout: images/<prefix>_<frame_id>.jpg
            p = os.path.join(self.root, f"{prefix}_{int(frame_id):06d}{ext}")
            if os.path.isfile(p):
                return p
            # nested layout: images/<prefix>/<frame_id>.jpg
            p = os.path.join(self.root, prefix, f"{frame_id:06d}{ext}")
            if os.path.isfile(p):
                return p
        return None


# ── CombinedDataset: weighted mix of all active datasets ─────────────────────
class CombinedDataset(Dataset):
    """
    Combines multiple IR tracking datasets with per-dataset sampling weights.
    Datasets whose annotation files are missing are silently skipped.
    """
    def __init__(self, datasets, weights=None, total_len=10000):
        super().__init__()
        # Filter out empty datasets
        self.datasets = [d for d in datasets if len(d) > 0]
        if not self.datasets:
            raise RuntimeError("All datasets are empty — check annotation paths.")

        if weights is None:
            weights = [1.0] * len(self.datasets)
        else:
            weights = [w for d, w in zip(datasets, weights) if len(d) > 0]

        total_w = sum(weights)
        self.probs = [w / total_w for w in weights]
        self.total_len = total_len

        logger.info("CombinedDataset:")
        for ds, p in zip(self.datasets, self.probs):
            logger.info(f"  {ds.name if hasattr(ds,'name') else type(ds).__name__:20s}"
                        f"  size={len(ds):6d}  sample_prob={p:.3f}")
        logger.info(f"  Total epoch length: {self.total_len}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        ds = random.choices(self.datasets, weights=self.probs, k=1)[0]
        return ds[random.randint(0, len(ds) - 1)]


# ── LR strategy: Warmup → Cosine-Warm-Restarts (SGDR) + Plateau rescue ───────
#
#  Why this combination?
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │ Phase 1 — Linear Warmup (epochs 0..warmup_epochs)                      │
#  │   Gradually ramps LR from near-zero to base_lr so early noisy          │
#  │   gradients don't destroy the pretrained backbone weights.             │
#  │                                                                         │
#  │ Phase 2 — Cosine Annealing with Warm Restarts / SGDR                   │
#  │   (Loshchilov & Hutter, 2017 — https://arxiv.org/abs/1608.03983)       │
#  │   T_0=50, T_mult=2  →  restarts at ep 50, 150, 350                     │
#  │   Each restart period doubles, giving more exploitation time as the    │
#  │   model matures.  The periodic cosine resets help escape sharp local   │
#  │   minima and explore flat loss regions.                                 │
#  │                                                                         │
#  │ Phase 3 — ReduceLROnPlateau (plateau rescue, runs in parallel)         │
#  │   If val_loss does not improve by ≥ min_delta for `patience` epochs,   │
#  │   multiply the current LR by `factor`.  Acts as a safety net between   │
#  │   SGDR restarts — catches cases where even a warm restart cannot        │
#  │   drive the loss down further.                                          │
#  │   patience=15, factor=0.3, min_delta=1e-4, min_lr=1e-7, cooldown=5    │
#  │                                                                         │
#  │  Timeline (500 ep, warmup=5):                                          │
#  │   0─5   warmup ramp                                                    │
#  │   5─55  cosine cycle 1 (T=50)                                          │
#  │   55─155 cosine cycle 2 (T=100)                                        │
#  │   155─355 cosine cycle 3 (T=200)                                       │
#  │   355─500 cosine tail                                                   │
#  │   plateau rescue fires whenever val_loss stalls ≥15 ep anywhere        │
#  └─────────────────────────────────────────────────────────────────────────┘

class WarmupCosineWarmRestarts:
    """
    Linear warmup for `warmup_epochs`, then hands off to
    CosineAnnealingWarmRestarts (SGDR) with T_0 and T_mult.
    Respects per-param-group `lr_scale` so backbone / neck / head
    can have different base LRs.
    """
    def __init__(self, optimizer, warmup_epochs, base_lr,
                 T_0=50, T_mult=2, min_lr=1e-6):
        self.opt           = optimizer
        self.warmup        = warmup_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self._warmup_done  = False

        # Build the inner SGDR scheduler.
        # We initialise it once warmup ends; store params for deferred creation.
        self._T_0   = T_0
        self._T_mult = T_mult
        self._sgdr  = None   # created at end of warmup

    def _get_lrs(self):
        """Return current LR of first param-group (for logging)."""
        return self.opt.param_groups[0]["lr"]

    def step(self, epoch):
        if epoch < self.warmup:
            # Linear ramp  0 → base_lr
            frac = (epoch + 1) / max(self.warmup, 1)
            lr   = self.base_lr * frac
            for pg in self.opt.param_groups:
                pg["lr"] = lr * pg.get("lr_scale", 1.0)
            return lr

        # First step after warmup: create SGDR and reset param-group lrs
        if self._sgdr is None:
            for pg in self.opt.param_groups:
                pg["lr"]         = self.base_lr * pg.get("lr_scale", 1.0)
                pg["initial_lr"] = self.base_lr * pg.get("lr_scale", 1.0)
            self._sgdr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.opt,
                T_0=self._T_0,
                T_mult=self._T_mult,
                eta_min=self.min_lr,
            )
            logger.info(f"  [LR] Warmup done. SGDR started: "
                        f"T_0={self._T_0}, T_mult={self._T_mult}, "
                        f"min_lr={self.min_lr:.2e}")

        # SGDR counts from 0 after warmup
        sgdr_epoch = epoch - self.warmup
        self._sgdr.step(sgdr_epoch)
        return self._get_lrs()


# ── Early Stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training when val_loss has not improved by more than `min_delta`
    (relative) for `patience` consecutive epochs.

    Design notes
    ─────────────
    • Uses *relative* improvement so the threshold scales automatically as
      loss magnitudes shrink during training (avoids being too lenient early
      and too strict late).
    • Counter is reset to 0 on any improvement, no matter how small (as long
      as it exceeds min_delta).  This means a single good epoch is enough to
      earn another `patience` epochs of grace.
    • The counter and best value are serialised into every checkpoint so
      early-stopping state is fully restored on resume — no phantom patience
      spending after a restart.
    • `stopped_epoch` records the exact epoch for clean log messages.
    """

    def __init__(self, patience=50, min_delta=1e-4):
        """
        Args:
            patience  (int):   Number of epochs without improvement before
                               stopping.  Default 50.
            min_delta (float): Minimum *relative* improvement required to
                               count as progress.
                               improvement = (prev_best - val_loss) / prev_best
                               Must be > min_delta to reset the counter.
        """
        self.patience      = patience
        self.min_delta     = min_delta
        self.best          = float("inf")
        self.counter       = 0           # epochs since last improvement
        self.stopped_epoch = None        # set when triggered
        self.triggered     = False

    def step(self, val_loss):
        """
        Call once per epoch with the current validation loss.
        Returns True if training should stop, False otherwise.
        """
        if self.best == float("inf"):
            # first epoch — always counts as improvement
            self.best    = val_loss
            self.counter = 0
            return False

        relative_improvement = (self.best - val_loss) / max(abs(self.best), 1e-12)

        if relative_improvement > self.min_delta:
            # Genuine improvement — reset counter
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.triggered = True
        return self.triggered

    def state_dict(self):
        return {
            "patience":      self.patience,
            "min_delta":     self.min_delta,
            "best":          self.best,
            "counter":       self.counter,
            "stopped_epoch": self.stopped_epoch,
            "triggered":     self.triggered,
        }

    def load_state_dict(self, d):
        self.patience      = d["patience"]
        self.min_delta     = d["min_delta"]
        self.best          = d["best"]
        self.counter       = d["counter"]
        self.stopped_epoch = d["stopped_epoch"]
        self.triggered     = d["triggered"]

    def status(self):
        """Human-readable status line for logging."""
        return (f"EarlyStopping: counter={self.counter}/{self.patience}  "
                f"best_val={self.best:.4f}  "
                f"min_delta={self.min_delta:.1e}")


def build_schedulers(optimizer, cfg, start_epoch=0):
    """
    Returns (primary_scheduler, plateau_scheduler).

    primary_scheduler  — WarmupCosineWarmRestarts (SGDR)
    plateau_scheduler  — ReduceLROnPlateau acting as plateau rescue

    Caller must:
      lr = primary.step(epoch)          # every epoch, before train
      plateau.step(val_loss)            # every epoch, after val
    """
    primary = WarmupCosineWarmRestarts(
        optimizer,
        warmup_epochs=5,
        base_lr=cfg.TRAIN.BASE_LR,
        T_0=50,
        T_mult=2,
        min_lr=1e-6,
    )
    # If resuming, fast-forward warmup state
    if start_epoch > 0:
        for ep in range(start_epoch):
            primary.step(ep)

    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",           # monitor val loss
        factor=0.3,           # multiply LR by 0.3 on plateau  (aggressive but fast)
        patience=15,          # wait 15 epochs before firing
        threshold=1e-4,       # min improvement to count as "better"
        threshold_mode="rel", # relative improvement
        cooldown=5,           # wait 5 epochs after firing before watching again
        min_lr=1e-7,          # hard floor — never go below this
        # verbose removed in PyTorch 2.x — we do our own logging
    )
    return primary, plateau


# ── optimizer ─────────────────────────────────────────────────────────────────
def build_optimizer(model):
    params = []
    params += [{"params": filter(lambda p: p.requires_grad,
                                  model.backbone.parameters()),
                "lr": cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR,
                "lr_scale": cfg.BACKBONE.LAYERS_LR}]
    if cfg.ADJUST.ADJUST:
        params += [{"params": model.neck.parameters(),
                    "lr": cfg.TRAIN.BASE_LR, "lr_scale": 1.0}]
    params += [{"params": model.rpn_head.parameters(),
                "lr": cfg.TRAIN.BASE_LR, "lr_scale": 1.0}]
    return torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM,
                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)



# ── one epoch forward ─────────────────────────────────────────────────────────
def run_epoch(model, loader, device, optimizer=None, epoch=0):
    training = optimizer is not None
    model.train() if training else model.eval()

    raw_model = model.module if hasattr(model, "module") else model
    if training and epoch < cfg.BACKBONE.TRAIN_EPOCH:
        for m in raw_model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    accum_steps = getattr(cfg.TRAIN, "GRAD_ACCUM_STEPS", 1)
    total, count = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        if training:
            optimizer.zero_grad()
        for step, data in enumerate(loader):
            batch = {
                "template":         to_dev(data["template"],         device),
                "search":           to_dev(data["search"],           device),
                "label_cls":        data["label_cls"].to(device),
                "label_loc":        to_dev(data["label_loc"],        device),
                "label_loc_weight": to_dev(data["label_loc_weight"], device),
                "bbox":             to_dev(data["bbox"],             device),
            }
            outputs = model(batch)
            loss    = outputs["total_loss"]

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                logger.warning(f"Bad loss at step {step}, skipping.")
                continue

            if training:
                # Scale loss by 1/accum_steps so the accumulated gradient
                # matches what a single forward pass on the full batch would give
                (loss / accum_steps).backward()
                is_update_step = (step + 1) % accum_steps == 0 or (step + 1) == len(loader)
                if is_update_step:
                    clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                    optimizer.step()
                    optimizer.zero_grad()

            total += loss.item()
            count += 1

            if training and (step + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                cls_l = outputs.get("cls_loss", torch.tensor(0.)).item()
                loc_l = outputs.get("loc_loss", torch.tensor(0.)).item()
                logger.info(f"Epoch[{epoch+1}] step[{step+1}/{len(loader)}] "
                            f"loss={loss.item():.4f} cls={cls_l:.4f} loc={loc_l:.4f}")

    return total / max(count, 1)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("SiamRPN++ AWS training")
    parser.add_argument("--cfg",        required=True)
    parser.add_argument("--pretrained", default="",
                        help="Pretrained backbone .pth (sot_resnet50)")
    parser.add_argument("--resume",     default="",
                        help="Full checkpoint .pth to resume from")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Run all EPOCH epochs — no early stopping")
    parser.add_argument("--no_plateau",    action="store_true",
                        help="Use fixed log-decay LR — no ReduceLROnPlateau")
    parser.add_argument("--smoke-test", action="store_true",
                        help=(
                            "Run a single epoch with a tiny dataset slice "
                            "(64 samples) to verify the full pipeline "
                            "without committing to a real training run. "
                            "Early stopping and checkpoint rotation are "
                            "disabled in this mode."
                        ))
    args = parser.parse_args()

    rank, local_rank, world_size, is_main = init_distributed()
    # Per-rank seed so each DDP process samples different pairs
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # ── smoke-test overrides (applied after freeze via cfg.defrost) ────────────
    if args.smoke_test:
        cfg.defrost()
        cfg.TRAIN.EPOCH          = 1      # single epoch
        cfg.DATASET.VIDEOS_PER_EPOCH = 64 # tiny slice — fast loader warm-up
        cfg.TRAIN.BATCH_SIZE     = min(cfg.TRAIN.BATCH_SIZE, 4)   # fits any GPU
        cfg.TRAIN.NUM_WORKERS    = 0      # no multiprocessing — easier tracebacks
        cfg.freeze()
        logger.info("=" * 60)
        logger.info("  🔥  SMOKE-TEST MODE")
        logger.info("      epochs=1  |  samples=64  |  batch=4  |  workers=0")
        logger.info("      Early stopping + checkpoint rotation disabled.")
        logger.info("=" * 60)

    device = get_device(local_rank)

    if is_main:
        os.makedirs(cfg.TRAIN.LOG_DIR,      exist_ok=True)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    if world_size > 1:
        dist.barrier()   # wait for rank-0 to create dirs
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR) if is_main else None

    # ── model ─────────────────────────────────────────────────────────────────
    model = ModelBuilder()

    pretrained = args.pretrained or cfg.TRAIN.PRETRAINED
    if pretrained and os.path.isfile(pretrained):
        logger.info(f"Loading pretrained backbone: {pretrained}")
        load_pretrain(model.backbone, pretrained)

    # freeze backbone; layers unlock after BACKBONE_TRAIN_EPOCH
    for p in model.backbone.parameters():
        p.requires_grad = False

    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
        if is_main:
            logger.info(f"DDP enabled: {world_size} GPUs (local_rank={local_rank})")
    else:
        n_gpus = torch.cuda.device_count()
        logger.info(f"Found {n_gpus} GPU(s) â single-GPU training "
                    f"(accum_steps={cfg.TRAIN.GRAD_ACCUM_STEPS})")

    raw_model = model.module if isinstance(model, DDP) else model

    # ── datasets ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Building training datasets...")
    logger.info("=" * 60)

    def ds_cfg(name):
        """Safely return a dataset config node, or None if not present."""
        try:
            return getattr(cfg.DATASET, name)
        except AttributeError:
            return None

    # Primary IR SOT dataset
    antiuav_train = AntiUAV410Dataset(
        cfg.DATASET.ANTIUAV410.ROOT,
        cfg.DATASET.ANTIUAV410.ANNO,
        frame_range=cfg.DATASET.ANTIUAV410.FRAME_RANGE,
    )

    # Extra datasets — each skips gracefully if annotations are missing
    c = ds_cfg("MSRS")
    msrs_train = MSRSDataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else MSRSDataset("", "")

    c = ds_cfg("VTMOT")
    vtmot_train = VTMOTDataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else VTMOTDataset("", "")

    c = ds_cfg("MASSMIND")
    massmind_train = MassMINDDataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else MassMINDDataset("", "")

    c = ds_cfg("MVSS")
    mvss_train = MVSSDataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else MVSSDataset("", "")

    c = ds_cfg("DUTVTUAV")
    dutvtuav_train = DUTVTUAVDataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else DUTVTUAVDataset("", "")

    c = ds_cfg("ANTIUAV300")
    antiuav300_train = AntiUAV300Dataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else AntiUAV300Dataset("", "")

    c = ds_cfg("BIRDSAI")
    birdsai_train = BIRDSAIDataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else BIRDSAIDataset("", "")

    c = ds_cfg("HITUAV")
    hituav_train = HITUAVDataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else HITUAVDataset("", "")

    # Combine with config-driven weights (default 1.0 if WEIGHT key absent)
    def w(name, default=1.0):
        try:
            return getattr(cfg.DATASET, name).WEIGHT
        except AttributeError:
            return default

    train_ds = CombinedDataset(
        datasets=[
            antiuav_train, msrs_train,    vtmot_train,    massmind_train, mvss_train,
            dutvtuav_train, antiuav300_train, birdsai_train, hituav_train,
        ],
        weights=[
            w("ANTIUAV410", 3.0), w("MSRS",       1.0), w("VTMOT",    2.0),
            w("MASSMIND",   1.0), w("MVSS",        1.5), w("DUTVTUAV", 2.5),
            w("ANTIUAV300",  2.5), w("BIRDSAI",  1.5),
            w("HITUAV",     1.0),
        ],
        total_len=cfg.DATASET.VIDEOS_PER_EPOCH // max(world_size, 1),
    )

    # Validation — AntiUAV410 + MSRS + VTMOT + new val splits
    antiuav_val = AntiUAV410Dataset(
        cfg.DATASET.ANTIUAV410_VAL.ROOT,
        cfg.DATASET.ANTIUAV410_VAL.ANNO,
        frame_range=cfg.DATASET.ANTIUAV410_VAL.FRAME_RANGE,
    )
    c = ds_cfg("MSRS_VAL")
    msrs_val = MSRSDataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else MSRSDataset("", "")

    c = ds_cfg("VTMOT_VAL")
    vtmot_val = VTMOTDataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else VTMOTDataset("", "")

    c = ds_cfg("DUTVTUAV_VAL")
    dutvtuav_val = DUTVTUAVDataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else DUTVTUAVDataset("", "")

    c = ds_cfg("ANTIUAV300_VAL")
    antiuav300_val = AntiUAV300Dataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else AntiUAV300Dataset("", "")

    c = ds_cfg("BIRDSAI_VAL")
    birdsai_val = BIRDSAIDataset(
        c.ROOT, c.ANNO, frame_range=c.FRAME_RANGE) if c else BIRDSAIDataset("", "")

    val_ds = CombinedDataset(
        datasets=[antiuav_val, msrs_val, vtmot_val,
                  dutvtuav_val, antiuav300_val, birdsai_val],
        weights=[3.0, 1.0, 2.0, 2.5, 2.5, 1.5],
        total_len=1000,
    )

    logger.info("=" * 60)
    logger.info(f"Train combined size : {len(train_ds)}")
    logger.info(f"Val   combined size : {len(val_ds)}")
    logger.info("=" * 60)

    # Cap workers to physical CPU count to avoid DataLoader freeze warning
    safe_workers = min(cfg.TRAIN.NUM_WORKERS, os.cpu_count() or 1)
    if safe_workers != cfg.TRAIN.NUM_WORKERS:
        logger.info(
            f"NUM_WORKERS capped {cfg.TRAIN.NUM_WORKERS} → {safe_workers} "
            f"(machine has {os.cpu_count()} CPUs)"
        )

    train_loader = DataLoader(train_ds, batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=safe_workers,
                              pin_memory=True, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=safe_workers,
                              pin_memory=True, shuffle=False, drop_last=False)

    # ── optimizer ─────────────────────────────────────────────────────────────
    optimizer = build_optimizer(raw_model)

    # ── resume (must happen before schedulers so we can fast-forward) ─────────
    start_epoch   = 0
    best_val_loss = float("inf")
    best_ckpt     = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, "best_model.pth")
    plateau_state = None   # saved state_dict for plateau scheduler
    es_state      = None   # saved state_dict for early stopping

    resume = args.resume or cfg.TRAIN.RESUME
    if resume and os.path.isfile(resume):
        logger.info(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location="cpu")
        raw_model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt["epoch"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        plateau_state = ckpt.get("plateau_scheduler", None)
        es_state      = ckpt.get("early_stopping",   None)
        logger.info(f"Resumed epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── schedulers (built after resume so fast-forward is correct) ────────────
    primary_sched, plateau_sched = build_schedulers(optimizer, cfg, start_epoch)
    if args.no_plateau:
        import numpy as _np
        _w = _np.linspace(cfg.TRAIN.BASE_LR * 0.02, cfg.TRAIN.BASE_LR, 5)
        _d = _np.logspace(
            _np.log10(cfg.TRAIN.BASE_LR),
            _np.log10(cfg.TRAIN.LR.KWARGS.get("end_lr", 1e-5)),
            max(1, cfg.TRAIN.EPOCH - 5))
        _lrs = list(_w) + list(_d)
        class _LogSched:
            def step(self, epoch):
                idx = min(int(epoch), len(_lrs) - 1)
                lr  = _lrs[idx]
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * pg.get("lr_scale", 1.0)
                return lr
        primary_sched = _LogSched()
        logger.info(f"  [LR] Log-decay schedule: {cfg.TRAIN.BASE_LR:.2e} → "
                    f"{_lrs[-1]:.2e} over {cfg.TRAIN.EPOCH} epochs")
    if plateau_state is not None:
        plateau_sched.load_state_dict(plateau_state)
        logger.info("  Restored ReduceLROnPlateau state from checkpoint.")

    # ── early stopping ────────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=50, min_delta=1e-4)
    if es_state is not None:
        early_stopping.load_state_dict(es_state)
        logger.info(
            f"  Restored EarlyStopping state from checkpoint. "
            f"counter={early_stopping.counter}/{early_stopping.patience}, "
            f"best={early_stopping.best:.4f}"
        )

    logger.info(
        f"LR schedule  : warmup(5ep) → SGDR(T0=50, Tmult=2) + "
        f"ReduceLROnPlateau(patience=15, factor=0.3, min_lr=1e-7)"
    )
    logger.info(
        f"Early stopping: patience=50 epochs, min_delta=1e-4 (relative)"
    )
    logger.info(f"Training {cfg.TRAIN.EPOCH} epochs, "
                f"batch={cfg.TRAIN.BATCH_SIZE}, steps≈{len(train_loader)}")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.TRAIN.EPOCH):

        # unlock backbone after BACKBONE_TRAIN_EPOCH
        if epoch == cfg.BACKBONE.TRAIN_EPOCH:
            if is_main:
                logger.info("=" * 60)
                logger.info("Unfreezing backbone layers for fine-grained training.")
            for layer_name in cfg.BACKBONE.TRAIN_LAYERS:
                for p in getattr(raw_model.backbone, layer_name).parameters():
                    p.requires_grad = True
            # Rebuild optimizer with backbone now included, then rebuild schedulers
            optimizer = build_optimizer(raw_model)
            primary_sched, plateau_sched = build_schedulers(optimizer, cfg, start_epoch=0)
            logger.info("  Schedulers reset: SGDR T_0=50 restarts from backbone-unfreeze epoch.")
            logger.info("=" * 60)

        # ── step primary schedule (warmup → SGDR) ─────────────────────────────
        lr = primary_sched.step(epoch)
        lr_before = optimizer.param_groups[0]["lr"]  # after warmup/SGDR — plateau detection only

        train_loss = run_epoch(model, train_loader, device, optimizer, epoch)
        val_loss   = run_epoch(model, val_loader,   device, optimizer=None, epoch=epoch)

        # ── step plateau rescue AFTER observing val_loss ──────────────────────
        if not args.no_plateau:
            plateau_sched.step(val_loss)
        lr_after = optimizer.param_groups[0]["lr"]

        # Detect and log if plateau scheduler fired (LR changed from its action)
        if is_main and lr_after < lr_before * 0.99:
            logger.info(
                f"  ⚡ [ReduceLROnPlateau] LR reduced: {lr_before:.2e} → {lr_after:.2e}  "
                f"(val_loss stalled for {plateau_sched.patience} epochs)"
            )
            tb_writer.add_scalar("lr/plateau_event", lr_after, epoch + 1)

        # Use the post-plateau LR for logging (most accurate current value)
        lr_log = lr_after

        # ── early stopping check ──────────────────────────────────────────────
        should_stop = early_stopping.step(val_loss)
        if is_main:
            tb_writer.add_scalar("early_stopping/counter", early_stopping.counter, epoch + 1)

        # Log epoch summary — include early-stopping counter so it's always visible
        if is_main:
            logger.info(
                f"Epoch [{epoch+1:>3}/{cfg.TRAIN.EPOCH}]  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={lr_log:.2e}  "
                f"best_val={best_val_loss:.4f}  "
                f"ES={early_stopping.counter}/{early_stopping.patience}"
            )
            tb_writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch + 1)
            tb_writer.add_scalar("lr/current",  lr_log,  epoch + 1)
            tb_writer.add_scalar("lr/sgdr",     lr,      epoch + 1)

        # save periodic checkpoint every 10 epochs — keep only the last 2
        # (skipped entirely in smoke-test mode)
        if is_main and not args.smoke_test and (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR,
                                     f"checkpoint_e{epoch+1}.pth")
            torch.save({"epoch": epoch+1,
                        "state_dict":        raw_model.state_dict(),
                        "optimizer":         optimizer.state_dict(),
                        "plateau_scheduler": plateau_sched.state_dict(),
                        "early_stopping":    early_stopping.state_dict(),
                        "train_loss":        train_loss,
                        "val_loss":          val_loss,
                        "best_val_loss":     best_val_loss}, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

            # ── rolling window: delete checkpoints older than the last 2 ──
            import glob as _glob
            all_ckpts = sorted(
                _glob.glob(os.path.join(cfg.TRAIN.SNAPSHOT_DIR, "checkpoint_e*.pth")),
                key=lambda p: int(os.path.basename(p)
                                    .replace("checkpoint_e", "")
                                    .replace(".pth", ""))
            )
            # all_ckpts is sorted oldest→newest; keep only the last 2
            to_delete = all_ckpts[:-2]
            for old_ckpt in to_delete:
                os.remove(old_ckpt)
                logger.info(f"  ✗ Removed old checkpoint: {os.path.basename(old_ckpt)}")
            if to_delete:
                remaining = [os.path.basename(p) for p in all_ckpts[-2:]]
                logger.info(f"  ✔ Keeping last 2 checkpoints: {remaining}")

        # save best model
        if is_main and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch+1,
                        "state_dict":        raw_model.state_dict(),
                        "optimizer":         optimizer.state_dict(),
                        "plateau_scheduler": plateau_sched.state_dict(),
                        "early_stopping":    early_stopping.state_dict(),
                        "train_loss":        train_loss,
                        "val_loss":          val_loss,
                        "best_val_loss":     best_val_loss}, best_ckpt)
            logger.info(f"★ New best model (val={val_loss:.4f}) saved: {best_ckpt}")

        # ── early stopping — skipped in smoke-test mode
        if not args.smoke_test and not args.no_early_stop and should_stop:
            early_stopping.stopped_epoch = epoch + 1
            logger.info("")
            logger.info("=" * 60)
            logger.info("  ⛔  EARLY STOPPING TRIGGERED")
            logger.info(f"     Val loss did not improve by >{early_stopping.min_delta:.1e} "
                        f"(relative) for {early_stopping.patience} consecutive epochs.")
            logger.info(f"     Stopped at epoch {early_stopping.stopped_epoch} "
                        f"/ {cfg.TRAIN.EPOCH}")
            logger.info(f"     Best val loss : {best_val_loss:.4f}  "
                        f"(epoch {early_stopping.stopped_epoch - early_stopping.patience})")
            logger.info(f"     Best model    : {best_ckpt}")
            logger.info("=" * 60)
            tb_writer.add_scalar("early_stopping/stopped_epoch",
                                 early_stopping.stopped_epoch, early_stopping.stopped_epoch)
            break

    tb_writer.close()

    if early_stopping.triggered:
        logger.info(f"Training ended early at epoch {early_stopping.stopped_epoch}.")
    else:
        logger.info(f"Training complete — ran all {cfg.TRAIN.EPOCH} epochs.")
    logger.info(f"Best val loss : {best_val_loss:.4f}")
    logger.info(f"Best model    : {best_ckpt}")


if __name__ == "__main__":
    main()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
