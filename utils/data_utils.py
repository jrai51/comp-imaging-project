import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

#############################################################################################
# TODO: MOVE THIS ALL TO INDEPENDENT .PY FILE, JUPYTER NOTEBOOK ONLY FOR EASY VISUALIZATION #
#############################################################################################

class KITTIDepthInpaintingDataset(Dataset):
    """
    KITTI Depth Inpainting Dataset 

    For each depth map D:
      - D is from data_depth_annotated (semi-dense KITTI fused depth)
      - Convert 16-bit PNG to float32 depth in meters
      - Make a corrupted copy by blanking out random rectangular regions
      - Return (corrupted_depth, original_depth, valid_mask)

    Returned tensors:
      input_depth:  [1,H,W] float32, meters, 0 where either originally invalid OR masked out
      target_depth: [1,H,W] float32, meters, original semi-dense depth
      valid_mask:   [1,H,W] bool, True where KITTI has actual ground truth depth
    """

    def __init__(
        self,
        annotated_root,
        split='train',
        resize_to=(256, 1216),   # (H,W) or None to keep native
        n_rectangles=(1, 3),     # random number of holes per sample
        hole_frac_range=(0.05, 0.20),  # each hole covers 5%-20% of the image area hole_frac_range=(0.01, 0.05) (0.05, 0.20)

        normalize=False,
        max_depth_m=80.0,
        seed=None
    ):
        """
        annotated_root: path to 'data_depth_annotated'
        split: 'train' or 'val'
        resize_to: (H,W) final spatial size for network input/output. If None, keep original KITTI size.
        n_rectangles: tuple(int,int), inclusive range for number of synthetic occlusion rectangles
        hole_frac_range: tuple(float,float), area fraction of each rectangle w.r.t. full image
        normalize: if True, we divide depth maps by max_depth_m so values lie in [0,1] approximately
        max_depth_m: depth cap for normalization
        seed: optional random seed for reproducibility (mask locations etc.)
        """
        self.annotated_root = Path(annotated_root) / split
        self.resize_to = resize_to
        self.n_rectangles = n_rectangles
        self.hole_frac_range = hole_frac_range
        self.normalize = normalize
        self.max_depth_m = max_depth_m

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Collect all .png files in this split
        self.depth_paths = sorted(self.annotated_root.rglob('*.png'))
        if len(self.depth_paths) == 0:
            raise RuntimeError(
                f"No depth maps found under {self.annotated_root}. "
                "Check that annotated_root points to data_depth_annotated/"
            )

    def __len__(self):
        return len(self.depth_paths)

    def _load_depth_meters(self, p):
        """
        KITTI depth completion annotated images are 16-bit PNGs.
        Pixel value 0   => invalid / no measurement.
        Pixel value v>0 => depth (in meters) == v / 256.0

        We load as uint16, convert to float32 meters, optionally resize.
        """
        depth16 = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # HxW, uint16
        if depth16 is None:
            raise RuntimeError(f"Failed to read depth map {p}")

        # optional resize (NEAREST so we don't interpolate depth values weirdly)
        if self.resize_to is not None:
            H, W = self.resize_to
            depth16 = cv2.resize(
                depth16,
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )

        depth_m = depth16.astype(np.float32) / 256.0  # meters
        return depth_m  # shape (H,W), float32

    def _apply_random_holes(self, depth, valid_mask):
        """
        depth:      (H,W) float32, meters
        valid_mask: (H,W) bool, True where depth>0

        We create a corrupted copy by zeroing out rectangular regions, simulating occlusions.
        Importantly: we are zeroing the depth values. That looks like "missing data."
        """
        H, W = depth.shape
        corrupted = depth.copy()

        # we don't actually change valid_mask for training,
        # because we WANT supervision inside the holes.
        # valid_mask still represents "ground truth exists here in the original data".
        num_rects = random.randint(self.n_rectangles[0], self.n_rectangles[1])
        img_area = H * W

        for _ in range(num_rects):
            # pick area fraction for this hole
            frac = random.uniform(self.hole_frac_range[0], self.hole_frac_range[1])
            target_area = frac * img_area

            # random aspect ratio between about square and elongated
            aspect = random.uniform(0.5, 2.0)
            hole_h = int(np.sqrt(target_area / aspect))
            hole_w = int(aspect * hole_h)

            hole_h = max(1, min(H, hole_h))
            hole_w = max(1, min(W, hole_w))

            y0 = random.randint(0, H - hole_h)
            x0 = random.randint(0, W - hole_w)

            # wipe that region
            corrupted[y0:y0+hole_h, x0:x0+hole_w] = 0.0

        return corrupted

    def __getitem__(self, idx):
        depth_m = self._load_depth_meters(self.depth_paths[idx])  # (H,W) float32

        # valid pixels are where KITTI has any measurement at all
        valid_mask = depth_m > 0.0  # (H,W) bool

        # create corrupted version with synthetic occlusions
        corrupted_m = self._apply_random_holes(depth_m, valid_mask)

        # optional normalization to ~[0,1] (helps training stability sometimes)
        if self.normalize:
            target_depth = np.clip(depth_m / self.max_depth_m, 0.0, 1.0)
            input_depth  = np.clip(corrupted_m / self.max_depth_m, 0.0, 1.0)
        else:
            target_depth = depth_m
            input_depth  = corrupted_m

        # (C,H,W) tensors for PyTorch
        input_depth  = torch.from_numpy(input_depth).unsqueeze(0).float()   # [1,H,W]
        target_depth = torch.from_numpy(target_depth).unsqueeze(0).float()  # [1,H,W]
        valid_mask_t = torch.from_numpy(valid_mask).unsqueeze(0).bool()     # [1,H,W]

        return {
            "input_depth":  input_depth,      # corrupted, holes zeroed
            "target_depth": target_depth,     # original semi-dense depth
            "valid_mask":   valid_mask_t,     # True where ground truth exists
            "path":         str(self.depth_paths[idx])
        }


def get_inpainting_loader(
    annotated_root,
    split='train',
    batch_size=4,
    shuffle=True,
    num_workers=0,
    resize_to=(256,1216),
    normalize=False,
    seed=None
):
    """
    Convenience wrapper to create a DataLoader.
    """
    ds = KITTIDepthInpaintingDataset(
        annotated_root=annotated_root,
        split=split,
        resize_to=resize_to,
        normalize=normalize,
        seed=seed
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train' and shuffle),
        num_workers=num_workers,
        pin_memory=True
    )

    return dl
