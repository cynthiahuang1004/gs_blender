"""
post_process_rgb_tuned.py
=========================
Generate rgb_tuned/ from rgb/ for sim2real domain adaptation. Pure post-process,
no re-rendering. Output is NOT fixed-center-cropped (VisTacFusion does that).

Pipeline (validated on pose-matched sim/real pairs, calibration/tune_output):
  1. Zoom 1.25 (FOV correction: sim FOV slightly wider than real)
  2. Reinhard LAB transfer to a real reference image of the same object
     (chroma boost 1.2, L darken 6)
  3. Blue-core correction (blue objects only): align most-saturated blue
     pixels to real blue core in LAB, extra L -12
  4. Center saturation boost (1.3x at center)
  5. Radial haze (center 0.03, edge 0.25)
  6. Radial blur (center sigma 0.5, edge sigma 2.4)

Reference selection: per sim image, a real image of the same object is chosen
deterministically (path-hash seed) from up to 50 refs -> adds real-data
variability like augmentation. Objects without real data use the pooled
blue-pattern references (same blue plate material).

Usage:
    python post_process_rgb_tuned.py                    # all objects
    python post_process_rgb_tuned.py --obj pattern_32   # specific object(s)
    python post_process_rgb_tuned.py --workers 16
"""
import os, math, argparse, hashlib, random
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT = os.path.join(SCRIPT_DIR, 'renders_v3')
REAL_ROOT = os.path.join(SCRIPT_DIR, 'real_filtered')

BLACK_OBJS = {'pattern_04_3_lines_angle_2', 'pattern_06_5_lines_angle_1',
              'pattern_31_rod', 'pattern_32', 'pattern_35', 'pattern_36', 'pattern_37'}
BLUE_REF_POOL_OBJS = ['pattern_01_2_lines_angle_1_2', 'pattern_01_2_lines_angle_2',
                      'pattern_01_2_lines_angle_3', 'pattern_04_3_lines_angle_1', 'pattern_33']

FIXED_CROP = 1.0 / math.sqrt(2.0)
N_REFS = 50


def fixed_center_crop(img):
    """Only used to build reference stats (matches training view)."""
    H, W = img.shape[:2]
    side = int(math.floor(min(H, W) * FIXED_CROP))
    oy, ox = (H - side) // 2, (W - side) // 2
    return cv2.resize(img[oy:oy+side, ox:ox+side], (W, H), interpolation=cv2.INTER_LINEAR)


def get_blue_core_lab(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 0] > 95) & (hsv[:, :, 0] < 135) & (hsv[:, :, 1] > 40)
    if mask.sum() < 100:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    sats = hsv[:, :, 1][mask]
    pix = lab[mask]
    top = np.argsort(sats)[-len(sats) // 4:]
    return pix[top].mean(axis=0)


def build_references(obj):
    """Return list of dicts: {lab_mean, lab_std, blue_core} from real images."""
    src_objs = [obj]
    real_dir = os.path.join(REAL_ROOT, obj)
    if not os.path.isdir(real_dir):
        src_objs = BLUE_REF_POOL_OBJS  # fallback pool for objects without real data
    paths = []
    for so in src_objs:
        od = os.path.join(REAL_ROOT, so)
        if not os.path.isdir(od):
            continue
        for sess in sorted(os.listdir(od)):
            if not sess.startswith('session_'):
                continue
            rd = os.path.join(od, sess, 'sensor_0000', 'rgb')
            if not os.path.isdir(rd):
                continue
            for f in sorted(os.listdir(rd)):
                if f.endswith('.png'):
                    paths.append(os.path.join(rd, f))
    rng = random.Random(42)
    paths = rng.sample(paths, min(N_REFS, len(paths)))
    refs = []
    for p in paths:
        img = fixed_center_crop(np.array(Image.open(p).convert('RGB')))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
        refs.append({
            'lab_mean': lab.mean(axis=0),
            'lab_std': lab.std(axis=0),
            'blue_core': get_blue_core_lab(img),
        })
    return refs


def tune_rgb(img, ref, is_black, zoom=1.25, chroma=1.2, darken=6.0,
             blue_extra_dark=12.0, center_sat=1.3):
    h, w = img.shape[:2]
    # 1. Zoom (FOV correction)
    side = int(w / zoom)
    off = (w - side) // 2
    img = cv2.resize(img[off:off+side, off:off+side], (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Reinhard LAB transfer
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean = lab.reshape(-1, 3).mean(axis=0)
    src_std = lab.reshape(-1, 3).std(axis=0)
    target_std = ref['lab_std'].copy()
    target_std[1:] *= chroma
    target_mean = ref['lab_mean'].copy()
    target_mean[0] -= darken
    lab = (lab - src_mean) / np.maximum(src_std, 1e-6) * target_std + target_mean
    img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # 3. Blue-core correction (blue objects only)
    if not is_black and ref['blue_core'] is not None:
        sim_blue = get_blue_core_lab(img)
        if sim_blue is not None:
            delta = ref['blue_core'] - sim_blue
            delta[0] -= blue_extra_dark
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            blue_w = np.clip(hsv[:, :, 1] / 80.0, 0, 1)
            for c in range(3):
                lab[:, :, c] += delta[c] * blue_w
            img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    img = img.astype(np.float32)
    cy, cx = h / 2, w / 2
    Y, X = np.mgrid[:h, :w]
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    center_w = np.clip(1.0 - dist, 0, 1)[:, :, None]

    # 4. Center saturation boost
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * (1.0 + (center_sat - 1.0) * center_w)

    # 5. Radial haze
    glow = cv2.GaussianBlur(img, (0, 0), 11.0)
    ha = 0.03 + 0.22 * (1 - center_w)
    img = img * (1 - ha) + glow * ha + ha * 18.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 6. Radial blur
    blurred = cv2.GaussianBlur(img, (0, 0), 2.4)
    light = cv2.GaussianBlur(img, (0, 0), 0.5)
    cw2 = np.clip(1.0 - dist, 0, 1)[:, :, None]
    return (light.astype(np.float32) * cw2 + blurred.astype(np.float32) * (1 - cw2)).astype(np.uint8)


def process_sensor(args):
    sensor_dir, obj, refs = args
    rgb_dir = os.path.join(sensor_dir, 'rgb')
    out_dir = os.path.join(sensor_dir, 'rgb_tuned')
    if not os.path.isdir(rgb_dir):
        return 0
    files = sorted(f for f in os.listdir(rgb_dir) if f.endswith('.png'))
    if not files:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    existing = set(os.listdir(out_dir))
    if len(existing) >= len(files):
        return 0
    is_black = obj in BLACK_OBJS
    n = 0
    for f in files:
        out_path = os.path.join(out_dir, f)
        if f in existing:
            continue
        img = np.array(Image.open(os.path.join(rgb_dir, f)).convert('RGB'))
        # Deterministic per-image reference pick (path-hash seed)
        seed = int(hashlib.md5(os.path.join(sensor_dir, f).encode()).hexdigest()[:8], 16)
        ref = refs[seed % len(refs)]
        out = tune_rgb(img, ref, is_black)
        Image.fromarray(out).save(out_path)
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=SIM_ROOT)
    parser.add_argument('--obj', nargs='*', default=None)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    objs = sorted(d for d in os.listdir(args.root)
                  if os.path.isdir(os.path.join(args.root, d)))
    if args.obj:
        objs = [o for o in objs if o in set(args.obj)]

    print('Building real references per object...')
    obj_refs = {o: build_references(o) for o in objs}
    for o in objs:
        src = 'own real' if os.path.isdir(os.path.join(REAL_ROOT, o)) else 'blue pool'
        print(f'  {o}: {len(obj_refs[o])} refs ({src})')

    tasks = []
    for obj in objs:
        obj_dir = os.path.join(args.root, obj)
        for sess in sorted(os.listdir(obj_dir)):
            if not sess.startswith('session_'):
                continue
            sensor = os.path.join(obj_dir, sess, 'sensor_0000')
            if os.path.isdir(os.path.join(sensor, 'rgb')):
                tasks.append((sensor, obj, obj_refs[obj]))

    print(f'\n{len(tasks)} sensor dirs, {args.workers} workers')
    import time
    t0 = time.time()
    total = done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_sensor, t): t for t in tasks}
        for f in as_completed(futures):
            n = f.result()
            total += n
            done += 1
            if n > 0:
                sensor = futures[f][0]
                print(f'[{done}/{len(tasks)}] {os.path.relpath(sensor, args.root)}: {n} images', flush=True)
    print(f'\nDone! {total} images in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
