"""
make_tune_grid.py
=================
Generate comparison grids for sim2real RGB tuning verification.
Output: calibration/tune_output/tune_grid.png

Two modes:

1. sampled (default) — random rgb_tuned samples vs a random real image
   per object. Verifies the batch output of post_process_rgb_tuned.py,
   including the per-image reference variability.

       python make_tune_grid.py
       python make_tune_grid.py --objs pattern_32 pattern_35 button

2. paired — pose-matched sim/real pairs (raw pose matching, no transform),
   three columns: original sim | tuned sim (on-the-fly) | real.
   Used during parameter tuning.

       python make_tune_grid.py --mode paired

All images are shown with the training-time fixed 1/sqrt(2) center crop so
the grid reflects what the model actually sees.
"""
import os, sys, math, json, random, argparse
import numpy as np
from PIL import Image, ImageDraw
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

SIM_ROOT = os.path.join(ROOT_DIR, 'renders_v3')
REAL_ROOT = os.path.join(ROOT_DIR, 'real_filtered')
OUT_DIR = os.path.join(SCRIPT_DIR, 'tune_output')

FIXED_CROP = 1.0 / math.sqrt(2.0)

DEFAULT_OBJS = ['pattern_01_2_lines_angle_2', 'pattern_04_3_lines_angle_1',
                'pattern_33', 'pattern_32', 'pattern_35', 'button']


def fixed_center_crop(img):
    H, W = img.shape[:2]
    side = int(math.floor(min(H, W) * FIXED_CROP))
    oy, ox = (H - side) // 2, (W - side) // 2
    return cv2.resize(img[oy:oy+side, ox:ox+side], (W, H),
                      interpolation=cv2.INTER_LINEAR)


def list_pngs(root, obj, subdir):
    paths = []
    obj_dir = os.path.join(root, obj)
    if not os.path.isdir(obj_dir):
        return paths
    for sess in sorted(os.listdir(obj_dir)):
        if not sess.startswith('session_'):
            continue
        d = os.path.join(obj_dir, sess, 'sensor_0000', subdir)
        if os.path.isdir(d):
            paths += [os.path.join(d, f) for f in sorted(os.listdir(d))
                      if f.endswith('.png')]
    return paths


def load_crop(path, size):
    img = fixed_center_crop(np.array(Image.open(path).convert('RGB')))
    return Image.fromarray(img).resize((size, size), Image.LANCZOS)


def make_sampled_grid(objs, n_tuned=3, seed=7):
    """Random rgb_tuned samples vs random real per object."""
    rng = random.Random(seed)
    T, P = 170, 3
    COLS = n_tuned + 1
    ROWS = len(objs)
    W = COLS * T + (COLS + 1) * P
    H = ROWS * (T + 15) + (ROWS + 1) * P + 22
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    headers = [f'Tuned sample {i+1}' for i in range(n_tuned)] + ['Real (random)']
    for c, hdr in enumerate(headers):
        draw.text((P + c * (T + P) + T // 2, 4), hdr, fill=(0, 0, 0), anchor='mt')

    for row, obj in enumerate(objs):
        y = 22 + row * (T + 15 + P)
        tuned = list_pngs(SIM_ROOT, obj, 'rgb_tuned')
        for c, p in enumerate(rng.sample(tuned, min(n_tuned, len(tuned)))):
            canvas.paste(load_crop(p, T), (P + c * (T + P), y))
        real = list_pngs(REAL_ROOT, obj, 'rgb')
        if real:
            canvas.paste(load_crop(rng.choice(real), T), (P + n_tuned * (T + P), y))
        else:
            draw.text((P + n_tuned * (T + P) + T // 2, y + T // 2), 'no real',
                      fill=(150, 150, 150), anchor='mm')
        draw.text((P, y + T + 1), obj, fill=(80, 80, 80))
    return canvas


def load_raw_poses(root, obj):
    """Raw pose [cos(ez), sin(ez), sample_x, sample_y] — matching convention:
    real GT pose has NO rz0/half transform."""
    poses = []
    obj_dir = os.path.join(root, obj)
    for sess in sorted(os.listdir(obj_dir)):
        if not sess.startswith('session_'):
            continue
        raw = os.path.join(obj_dir, sess, 'sensor_0000', 'raw_data')
        if not os.path.isdir(raw):
            continue
        for f in sorted(os.listdir(raw)):
            if f.endswith('_pose.json') and '_gt' not in f:
                with open(os.path.join(raw, f)) as fh:
                    p = json.load(fh)
                idx = f.replace('_pose.json', '')
                if not os.path.exists(os.path.join(obj_dir, sess, 'sensor_0000',
                                                   'rgb', f'{idx}.png')):
                    continue
                ez = p['rotation_euler'][2]
                poses.append((sess, idx, np.array([
                    math.cos(ez), math.sin(ez), p['sample_x'], p['sample_y']])))
    return poses


def make_paired_grid(objs):
    """Pose-matched pairs: original sim | tuned (on-the-fly) | real."""
    from scipy.spatial import cKDTree
    from post_process_rgb_tuned import build_references, tune_rgb, BLACK_OBJS

    T, P = 200, 4
    COLS, ROWS = 3, len(objs)
    W = COLS * T + (COLS + 1) * P
    H = ROWS * (T + 16) + (ROWS + 1) * P + 25
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for c, hdr in enumerate(['Sim (crop)', 'Tuned (crop)', 'Real (crop)']):
        draw.text((P + c * (T + P) + T // 2, 6), hdr, fill=(0, 0, 0), anchor='mt')

    for row, obj in enumerate(objs):
        sp = load_raw_poses(SIM_ROOT, obj)
        rp = load_raw_poses(REAL_ROOT, obj)
        if not sp or not rp:
            continue
        weights = np.array([1, 1, 200, 200])
        tree = cKDTree(np.array([p[2] for p in sp]) * weights)
        dists, idxs = tree.query(np.array([p[2] for p in rp]) * weights, k=1)
        bi = int(np.argmin(dists))
        s_sess, s_idx, _ = sp[idxs[bi]]
        r_sess, r_idx, _ = rp[bi]

        sim = np.array(Image.open(os.path.join(
            SIM_ROOT, obj, s_sess, 'sensor_0000', 'rgb', f'{s_idx}.png')).convert('RGB'))
        real = np.array(Image.open(os.path.join(
            REAL_ROOT, obj, r_sess, 'sensor_0000', 'rgb', f'{r_idx}.png')).convert('RGB'))
        refs = build_references(obj)
        tuned = tune_rgb(sim, refs[0], obj in BLACK_OBJS)

        y = 25 + row * (T + 16 + P)
        for c, img in enumerate([sim, tuned, real]):
            crop = fixed_center_crop(img)
            canvas.paste(Image.fromarray(crop).resize((T, T), Image.LANCZOS),
                         (P + c * (T + P), y))
        draw.text((P, y + T + 1), f'{obj}  (d={dists[bi]:.3f})', fill=(80, 80, 80))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sampled', 'paired'], default='sampled')
    parser.add_argument('--objs', nargs='*', default=DEFAULT_OBJS)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', default=os.path.join(OUT_DIR, 'tune_grid.png'))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.mode == 'sampled':
        grid = make_sampled_grid(args.objs, seed=args.seed)
    else:
        grid = make_paired_grid(args.objs)
    grid.save(args.out)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
