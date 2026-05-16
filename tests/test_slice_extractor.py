"""Minimal tests for data/slice_extractor.py.

No real BraTS case is available in this environment, so we synthesize
NIfTI volumes with a known geometry and an asymmetric marker to verify
slice count, ordering, normalization range and the LR-flip behaviour.
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.slice_extractor import extract_slices  # noqa: E402

SHAPE = (192, 192, 144)


def _save_nifti(arr: np.ndarray, path: Path) -> None:
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4)), str(path))


@pytest.fixture
def synth_case(tmp_path: Path):
    rng = np.random.default_rng(0)
    # Brain: a centered ellipsoid of positive intensity, zeros outside.
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, SHAPE[0]),
        np.linspace(-1, 1, SHAPE[1]),
        np.linspace(-1, 1, SHAPE[2]),
        indexing="ij",
    )
    brain = (xx**2 + yy**2 + zz**2) < 0.8
    vol = np.zeros(SHAPE, dtype=np.float32)
    vol[brain] = 400.0 + rng.normal(0, 20, size=brain.sum())
    # Asymmetric bright marker on the patient's left half (high x index).
    vol[150:170, 90:110, 60:80] = 3000.0

    mask = np.zeros(SHAPE, dtype=np.float32)
    mask[140:160, 80:120, 50:70] = 1.0  # tumor confined to z in [50, 70)

    nifti = tmp_path / "t1c.nii.gz"
    maskf = tmp_path / "tumor_mask.nii.gz"
    _save_nifti(vol, nifti)
    _save_nifti(mask, maskf)
    return nifti, maskf, vol


def test_count_dtype_and_size(synth_case):
    nifti, _, _ = synth_case
    slices, z = extract_slices(str(nifti), n_slices=16)
    assert len(slices) == 16
    assert len(z) == 16
    for img in slices:
        assert img.mode == "RGB"
        assert img.size == (SHAPE[1], SHAPE[0])  # PIL size = (width=cols, height=rows)
        a = np.asarray(img)
        assert a.dtype == np.uint8
        # 3 channels identical (grayscale replicated).
        assert np.array_equal(a[..., 0], a[..., 1])
        assert np.array_equal(a[..., 1], a[..., 2])


def test_superior_to_inferior_order(synth_case):
    nifti, _, _ = synth_case
    _, z = extract_slices(str(nifti), n_slices=16)
    assert z == sorted(z, reverse=True), "z indices must be superior->inferior"
    assert all(0 <= zi < SHAPE[2] for zi in z)


def test_no_mask_uses_trim_range(synth_case):
    nifti, _, _ = synth_case
    _, z = extract_slices(
        str(nifti), n_slices=16, z_trim_top=0.10, z_trim_bottom=0.15
    )
    z_dim = SHAPE[2]
    assert min(z) >= round(0.15 * z_dim)
    assert max(z) <= round(0.90 * z_dim) - 1


def test_mask_restricts_to_tumor_plus_context(synth_case):
    nifti, maskf, _ = synth_case
    _, z = extract_slices(str(nifti), mask_path=str(maskf), n_slices=16)
    # Tumor z in [50, 70), context +-4 -> allowed [46, 73].
    assert min(z) >= 46
    assert max(z) <= 73


def test_lr_flip_actually_flips(synth_case):
    nifti, _, _ = synth_case
    z_target = 70  # inside the bright marker's z range [60, 80)
    no_flip, z1 = extract_slices(
        str(nifti), n_slices=144, apply_lr_flip=False
    )
    flip, z2 = extract_slices(str(nifti), n_slices=144, apply_lr_flip=True)
    i = z1.index(z_target)
    assert z1 == z2
    a = np.asarray(no_flip[i])[..., 0]
    b = np.asarray(flip[i])[..., 0]
    assert np.array_equal(np.fliplr(a), b)
    assert not np.array_equal(a, b), "marker is asymmetric; flip must change image"


def test_normalization_range(synth_case):
    nifti, _, _ = synth_case
    slices, _ = extract_slices(str(nifti), n_slices=8)
    allpx = np.concatenate([np.asarray(s).ravel() for s in slices])
    assert allpx.min() >= 0 and allpx.max() <= 255
    assert allpx.max() > 200, "bright marker should saturate near 255 after p99 clip"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_slices("/nonexistent/foo.nii.gz")
