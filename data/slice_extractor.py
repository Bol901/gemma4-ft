"""Extract and preprocess axial slices from an MRI NIfTI volume.

Module 1 of the GBM VLM finetune pipeline. Pure numpy / nibabel / PIL,
no deep-learning dependencies, so it can be unit tested standalone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _load_canonical(path: Path) -> np.ndarray:
    """Load a NIfTI file and return its array in closest-canonical (RAS+) order."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return np.asarray(img.dataobj, dtype=np.float32)


def _percentile_normalize(
    vol: np.ndarray, brain_mask: np.ndarray, p_low: float = 1.0, p_high: float = 99.0
) -> np.ndarray:
    """Clip to [p_low, p_high] percentiles of the brain region, scale to [0, 1]."""
    brain_vals = vol[brain_mask]
    if brain_vals.size == 0:
        # Degenerate volume (all zeros): nothing to normalize against.
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(brain_vals, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    out = (np.clip(vol, lo, hi) - lo) / (hi - lo)
    return out.astype(np.float32)


def _select_z_indices(
    vol: np.ndarray,
    mask: Optional[np.ndarray],
    n_slices: int,
    z_trim_top: float,
    z_trim_bottom: float,
    mask_context: int = 4,
) -> List[int]:
    """Pick ``n_slices`` z positions, ordered superior -> inferior (descending z).

    In RAS+ canonical orientation axis 2 increases toward superior, so a
    descending z order yields slices from the top of the head downward, which
    matches the wording used by the prompt builder.
    """
    z_dim = vol.shape[2]

    if mask is not None and mask.any():
        zs_with_tumor = np.where(mask.any(axis=(0, 1)))[0]
        z_min = max(0, int(zs_with_tumor.min()) - mask_context)
        z_max = min(z_dim - 1, int(zs_with_tumor.max()) + mask_context)
    else:
        z_min = int(round(z_trim_bottom * z_dim))
        z_max = int(round((1.0 - z_trim_top) * z_dim)) - 1

    z_min = max(0, min(z_min, z_dim - 1))
    z_max = max(0, min(z_max, z_dim - 1))
    if z_max < z_min:
        z_min, z_max = z_max, z_min

    span = z_max - z_min + 1
    if span <= n_slices:
        selected = list(range(z_min, z_max + 1))
    else:
        selected = sorted(
            set(int(round(v)) for v in np.linspace(z_min, z_max, n_slices))
        )

    # Superior -> inferior.
    selected.sort(reverse=True)
    return selected


def extract_slices(
    nifti_path: str,
    mask_path: Optional[str] = None,
    n_slices: int = 16,
    z_trim_top: float = 0.10,
    z_trim_bottom: float = 0.15,
    apply_lr_flip: bool = True,
) -> Tuple[List[Image.Image], List[int]]:
    """Extract ``n_slices`` preprocessed axial slices from an MRI volume.

    Returns a list of RGB uint8 PIL images and the corresponding z indices
    (ordered superior -> inferior). See IMPLEMENTATION.md Module 1 for the
    full step list.
    """
    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI not found: {nifti_path}")

    vol = _load_canonical(nifti_path)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape}")

    mask = None
    if mask_path is not None:
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = _load_canonical(mask_path) > 0
        if mask.shape != vol.shape:
            raise ValueError(
                f"Mask shape {mask.shape} != volume shape {vol.shape}"
            )

    brain_mask = vol > 0
    vol_norm = _percentile_normalize(vol, brain_mask)

    z_indices = _select_z_indices(
        vol_norm, mask, n_slices, z_trim_top, z_trim_bottom
    )

    slices: List[Image.Image] = []
    for z in z_indices:
        slice_2d = vol_norm[:, :, z]
        if apply_lr_flip:
            slice_2d = np.fliplr(slice_2d)
        slice_uint8 = np.clip(slice_2d * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([slice_uint8] * 3, axis=-1)
        slices.append(Image.fromarray(rgb, mode="RGB"))

    logger.info(
        "Extracted %d slices from %s (z=%s, mask=%s)",
        len(slices),
        nifti_path.name,
        z_indices,
        mask_path is not None and mask is not None and mask.any(),
    )
    return slices, z_indices
