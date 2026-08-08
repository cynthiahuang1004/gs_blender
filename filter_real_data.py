"""
filter_real_data.py
===================
Filter real tactile data by comparing contact area (extracted from tactile image
via RGB gradient) with GT depth mask. Removes samples where IoU < threshold.

Method: RGB Grad 0.7x Otsu
  - Compute per-channel Sobel gradient magnitude
  - Otsu threshold * 0.7 for binarization
  - Dilate + morphological close to fill contact region
  - Compare with GT depth > 0 mask via IoU

Usage:
    python filter_real_data.py                          # default: IoU >= 0.30
    python filter_real_data.py --threshold 0.25         # custom threshold
    python filter_real_data.py --dry-run                # preview only, don't delete
    python filter_real_data.py --root /path/to/data     # custom data root
"""

import os, argparse, shutil
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


def rgb_grad_mask(tac_arr, otsu_scale=0.7):
    img = tac_arr.astype(np.float32)
    total_mag = np.zeros(img.shape[:2], dtype=np.float64)
    for c in range(3):
        gx = cv2.Sobel(img[:, :, c], cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(img[:, :, c], cv2.CV_64F, 0, 1, ksize=5)
        total_mag += np.sqrt(gx**2 + gy**2)
    mag_u8 = np.clip(total_mag / total_mag.max() * 255, 0, 255).astype(np.uint8)
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mag_u8 = cv2.dilate(mag_u8, kernel_d, iterations=1)
    otsu_val, _ = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (mag_u8 > max(1, int(otsu_val * otsu_scale))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def compute_iou(mask, depth):
    m = mask > 0
    d = depth > 0
    inter = np.logical_and(m, d).sum()
    union = np.logical_or(m, d).sum()
    return inter / (union + 1e-6)


def process_sample(args):
    obj, sess, idx, sensor = args
    tac_path = os.path.join(sensor, 'samples', f'{idx}.png')
    tac_arr = np.array(Image.open(tac_path))
    depth_path = os.path.join(sensor, 'raw_data', f'{idx}_gt.npy')
    if not os.path.exists(depth_path):
        depth_path = os.path.join(sensor, 'raw_data', f'{idx}.npy')
    depth = np.load(depth_path)
    mask = rgb_grad_mask(tac_arr, 0.7)
    iou = compute_iou(mask, depth)
    return obj, sess, idx, sensor, iou


def delete_sample(sensor, idx):
    for subdir, suffix in [
        ('samples', '.png'),
        ('rgb', '.png'),
        ('raw_data', '.npy'),
        ('raw_data', '_gt.npy'),
        ('raw_data', '_pose.json'),
        ('dmaps', '.png'),
        ('dmaps', '_gt.png'),
        ('norms', '.png'),
        ('norms', '_gt.png'),
    ]:
        path = os.path.join(sensor, subdir, f'{idx}{suffix}')
        if os.path.exists(path):
            os.remove(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=os.path.join(os.path.dirname(__file__), 'real_filtered'))
    parser.add_argument('--threshold', type=float, default=0.30)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    root = args.root
    thresh = args.threshold

    all_samples = []
    for obj in sorted(os.listdir(root)):
        obj_dir = os.path.join(root, obj)
        if not os.path.isdir(obj_dir):
            continue
        for sess in sorted(os.listdir(obj_dir)):
            if not sess.startswith('session_'):
                continue
            sensor = os.path.join(obj_dir, sess, 'sensor_0000')
            samples_dir = os.path.join(sensor, 'samples')
            if not os.path.isdir(samples_dir):
                continue
            for f in sorted(os.listdir(samples_dir)):
                if f.endswith('.png'):
                    all_samples.append((obj, sess, f.replace('.png', ''), sensor))

    print(f'Total samples: {len(all_samples)}')
    print(f'Threshold: IoU >= {thresh}')
    print(f'Workers: {args.workers}')
    print(f'Dry run: {args.dry_run}\n')

    results_by_obj = {}
    to_delete = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_sample, s): s for s in all_samples}
        done = 0
        for f in as_completed(futures):
            obj, sess, idx, sensor, iou = f.result()
            if obj not in results_by_obj:
                results_by_obj[obj] = {'keep': 0, 'filter': 0, 'total': 0}
            results_by_obj[obj]['total'] += 1
            if iou >= thresh:
                results_by_obj[obj]['keep'] += 1
            else:
                results_by_obj[obj]['filter'] += 1
                to_delete.append((sensor, idx))
            done += 1
            if done % 1000 == 0:
                print(f'  {done}/{len(all_samples)} processed')

    print(f'\n{"object":>35} {"total":>6} {"keep":>6} {"filter":>7} {"keep%":>6}')
    print('-' * 65)
    total_all = keep_all = filter_all = 0
    for obj in sorted(results_by_obj.keys()):
        r = results_by_obj[obj]
        pct = r['keep'] / r['total'] * 100
        print(f'{obj:>35} {r["total"]:>6} {r["keep"]:>6} {r["filter"]:>7} {pct:>5.1f}%')
        total_all += r['total']
        keep_all += r['keep']
        filter_all += r['filter']
    print('-' * 65)
    print(f'{"TOTAL":>35} {total_all:>6} {keep_all:>6} {filter_all:>7} {keep_all/total_all*100:>5.1f}%')

    if args.dry_run:
        print(f'\nDry run — no files deleted.')
        return

    print(f'\nDeleting {len(to_delete)} samples...')
    for sensor, idx in to_delete:
        delete_sample(sensor, idx)
    print('Done!')


if __name__ == '__main__':
    main()
