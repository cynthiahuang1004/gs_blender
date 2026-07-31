"""Post-process renders: depth npy → dmaps/norms PNGs (parallel version)."""
import os, shutil, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2


def dmap2norm(dmap):
    zx = cv2.Sobel(dmap, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(dmap, cv2.CV_64F, 0, 1, ksize=5)
    normals = np.dstack((-zx, -zy, np.ones_like(dmap)))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    normals = (normals + 1) / 2
    return normals[:, :, ::-1].astype(np.float32)


def process_sensor(sensor_dir):
    sensor_dir = Path(sensor_dir)
    raw_dir = sensor_dir / 'raw_data'
    dmaps_dir = sensor_dir / 'dmaps'
    norms_dir = sensor_dir / 'norms'

    if not raw_dir.exists():
        return 0

    raw_depths = sorted(raw_dir.glob('*.npy'))
    if not raw_depths:
        return 0

    if norms_dir.exists() and dmaps_dir.exists():
        existing = len(list(norms_dir.glob('*.png')))
        if existing >= len(raw_depths):
            return 0

    if norms_dir.exists():
        shutil.rmtree(norms_dir)
    if dmaps_dir.exists():
        shutil.rmtree(dmaps_dir)
    norms_dir.mkdir(exist_ok=True)
    dmaps_dir.mkdir(exist_ok=True)

    count = 0
    for raw_path in raw_depths:
        dmap = np.load(raw_path)
        nz = dmap[dmap > 0]
        dmin = nz.min() if len(nz) else 0
        dmax = dmap.max()
        if dmax > dmin:
            dmap_norm = np.zeros_like(dmap)
            mask = dmap > 0
            dmap_norm[mask] = (dmap[mask] - dmin) / (dmax - dmin)
        else:
            dmap_norm = np.zeros_like(dmap)

        cv2.imwrite(str(dmaps_dir / raw_path.name.replace('.npy', '.png')),
                     (dmap_norm * 255).astype(np.uint8))

        if '_gt' in raw_path.name:
            norm = dmap2norm(dmap_norm)
        else:
            norm = dmap2norm(1.0 - dmap_norm)

        cv2.imwrite(str(norms_dir / raw_path.name.replace('.npy', '.png')),
                     (np.clip(norm, 0, 1) * 255).astype(np.uint8))
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'renders_v3'))
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    root = Path(args.root)
    sensor_dirs = sorted(root.glob('*/session_*/sensor_*'))
    if not sensor_dirs:
        sensor_dirs = sorted(root.glob('session_*/sensor_*'))
    print(f'Found {len(sensor_dirs)} sensor directories, {args.workers} workers')

    t0 = time.time()
    total = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_sensor, str(sd)): sd for sd in sensor_dirs}
        for f in as_completed(futures):
            n = f.result()
            total += n
            done += 1
            if n > 0:
                sd = futures[f]
                rel = sd.relative_to(root)
                print(f'[{done}/{len(sensor_dirs)}] {rel}: {n} files', flush=True)

    elapsed = time.time() - t0
    print(f'\nDone! {total} depth files in {elapsed:.1f}s ({total/max(elapsed,1):.0f} files/s)')


if __name__ == '__main__':
    main()
