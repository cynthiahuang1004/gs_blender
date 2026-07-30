# === Colab Cell: Post-process renders (depth → dmaps/norms PNGs) ===
# Paste this entire block into a single Colab cell and run it.
#
# Prerequisites: Google Drive mounted at /content/drive

from google.colab import drive
drive.mount('/content/drive')

import os, shutil
import numpy as np
import cv2
from pathlib import Path

root_dir = Path('/content/drive/MyDrive/HDR_Lab/gs_blender_renders')


def dmap2norm(dmap):
    zx = cv2.Sobel(dmap, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(dmap, cv2.CV_64F, 0, 1, ksize=5)
    normals = np.dstack((-zx, -zy, np.ones_like(dmap)))
    length = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= length
    normals += 1
    normals /= 2
    return normals[:, :, ::-1].astype(np.float32)


sensor_dirs = sorted(root_dir.glob('*/*/session_*/sensor_*'))
print(f'Found {len(sensor_dirs)} sensor directories')

total_processed = 0
for change_dir in sensor_dirs:
    raw_depth_dir = change_dir / 'raw_data'
    dmaps_dir = change_dir / 'dmaps'
    norms_dir = change_dir / 'norms'

    if not raw_depth_dir.exists():
        continue

    raw_depths = sorted(raw_depth_dir.glob('*.npy'))
    if not raw_depths:
        continue

    if norms_dir.exists() and dmaps_dir.exists():
        existing = len(list(norms_dir.glob('*.png')))
        if existing >= len(raw_depths):
            continue

    if norms_dir.exists():
        shutil.rmtree(norms_dir)
    if dmaps_dir.exists():
        shutil.rmtree(dmaps_dir)
    norms_dir.mkdir(exist_ok=True)
    dmaps_dir.mkdir(exist_ok=True)

    rel_path = change_dir.relative_to(root_dir)
    print(f'Processing {rel_path}: {len(raw_depths)} files')

    for raw_path in raw_depths:
        raw = raw_path.name
        dmap_path = dmaps_dir / raw.replace('.npy', '.png')
        norm_path = norms_dir / raw.replace('.npy', '.png')

        data = np.load(str(raw_path))
        dmap = data.copy()
        dmin = dmap[dmap > 0].min() if np.any(dmap > 0) else 0
        dmax = dmap.max()
        if dmax > dmin:
            dmap_norm = np.zeros_like(dmap)
            dmap_norm[dmap > 0] = (dmap[dmap > 0] - dmin) / (dmax - dmin)
        else:
            dmap_norm = np.zeros_like(dmap)

        cv2.imwrite(str(dmap_path), (dmap_norm * 255).astype(np.uint8))

        if '_gt' in raw:
            norm = dmap2norm(dmap_norm)
        else:
            norm = dmap2norm(1.0 - dmap_norm)

        norm = np.clip(norm, 0, 1)
        cv2.imwrite(str(norm_path), (norm * 255).astype(np.uint8))

    total_processed += len(raw_depths)

print(f'\nDone! Processed {total_processed} depth files across {len(sensor_dirs)} sensors.')
