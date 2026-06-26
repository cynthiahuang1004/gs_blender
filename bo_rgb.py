"""
bo_rgb.py
=========
Bayesian optimization to calibrate RGB rendering parameters against
a reference real image (real_data/rgb_images/47.jpg).

Pipeline
--------
1. Load 47.jpg as target (128x128)
2. For each BO trial:
   a. Write params JSON to temp file
   b. Call Blender with gs_rgb_render.py (fixed scene: session_000/sensor_0000/0000_pose.json)
   c. Apply post-FX in Python (barrel distortion + radial blur + vignette)
   d. loss = 0.3 * MSE + 0.7 * per-channel Bhattacharyya histogram distance
3. forest_minimize over 14-D parameter space
4. Save bo_results/best_rgb_params.json + best_rgb_render.png + rgb_convergence.png

Usage
-----
    pip install scikit-optimize opencv-python matplotlib
    python bo_rgb.py
"""

import os, json, subprocess, tempfile, time, shutil
import numpy as np
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from skopt import forest_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
except ImportError:
    raise SystemExit('scikit-optimize not found.\nInstall with:  pip install scikit-optimize')

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BLENDER_PATH  = r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'
TARGET_IMAGE  = os.path.join(SCRIPT_DIR, 'real_data', 'rgb_images', '47.jpg')
RESULTS_DIR   = os.path.join(SCRIPT_DIR, 'bo_results')
RENDER_SCRIPT = os.path.join(SCRIPT_DIR, 'gs_rgb_render.py')
BLEND_FILE    = os.path.join(SCRIPT_DIR, 'gelsight_sampler.blend')
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_SIZE = (128, 128)
LOSS_THRESHOLD = 0.05

# ── Load target ────────────────────────────────────────────────
def _load_target():
    img = cv2.imread(TARGET_IMAGE)
    if img is None:
        raise RuntimeError(f'Cannot load target image: {TARGET_IMAGE}')
    img = cv2.resize(img, TARGET_SIZE).astype(np.float32) / 255.0
    cv2.imwrite(os.path.join(RESULTS_DIR, 'rgb_target.png'),
                (img * 255).astype(np.uint8))
    print(f'Target: {TARGET_IMAGE}  →  {RESULTS_DIR}/rgb_target.png')
    return img

TARGET = _load_target()

# ── Post-FX (applied in Python, not in Blender) ───────────────

def _barrel(rgb, k1):
    H, W = rgb.shape[:2]
    cx, cy = W / 2., H / 2.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx - cx) / cx;  yn = (yy - cy) / cy
    fac = 1. + k1 * (xn**2 + yn**2)
    xs = np.clip(xn * fac * cx + cx, 0, W - 1)
    ys = np.clip(yn * fac * cy + cy, 0, H - 1)
    x0 = np.floor(xs).astype(int);  x1 = np.minimum(x0 + 1, W - 1)
    y0 = np.floor(ys).astype(int);  y1 = np.minimum(y0 + 1, H - 1)
    wx = (xs - x0)[:, :, None];     wy = (ys - y0)[:, :, None]
    return np.clip(
        rgb[y0, x0] * (1 - wx) * (1 - wy) + rgb[y0, x1] * wx * (1 - wy) +
        rgb[y1, x0] * (1 - wx) * wy        + rgb[y1, x1] * wx * wy, 0, 1)

def _gblur(rgb, sigma):
    if sigma < 0.05:
        return rgb.copy()
    sz = max(3, int(6 * sigma + 1) | 1)
    x = np.arange(sz) - sz // 2
    k = np.exp(-x**2 / (2 * sigma**2));  k /= k.sum()
    out = np.empty_like(rgb)
    for c in range(rgb.shape[2]):
        h = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1, rgb[:, :, c])
        out[:, :, c] = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 0, h)
    return np.clip(out, 0, 1)

def _apply_fx(img_uint8, p):
    """Apply barrel + radial blur + vignette. Returns float32 [0,1]."""
    rgb = img_uint8.astype(np.float32) / 255.
    H, W = rgb.shape[:2]
    rgb = _barrel(rgb, float(p.get('barrel_k1', 0.07)))
    blurred = _gblur(rgb, float(p.get('blur_sigma', 1.25)))
    cy, cx = H / 2., W / 2.
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    w = np.clip(dist ** 1.5, 0, 1)[:, :, None]   # blur_falloff fixed at 1.5
    rgb = rgb * (1 - w) + blurred * w
    mask = np.clip(1. - dist ** 2 * float(p.get('vignette', 0.25)), 0, 1)[:, :, None]
    return np.clip(rgb * mask, 0, 1)

# ── Loss ───────────────────────────────────────────────────────

def _image_loss(rendered, target):
    """Both inputs: float32 BGR [0,1] at TARGET_SIZE."""
    mse = float(np.mean((rendered - target) ** 2))
    hist_d = 0.0
    for c in range(3):
        h_r = cv2.calcHist([rendered], [c], None, [64], [0., 1.])
        h_t = cv2.calcHist([target],   [c], None, [64], [0., 1.])
        cv2.normalize(h_r, h_r);  cv2.normalize(h_t, h_t)
        hist_d += cv2.compareHist(h_r, h_t, cv2.HISTCMP_BHATTACHARYYA)
    return 0.3 * mse + 0.7 * (hist_d / 3.0)

# ── Parameter space (14-D) ─────────────────────────────────────
PARAM_SPACE = [
    # World / ambient lighting
    Real(0.5,   4.0,  name='world_strength'),
    # Camera depth of field
    Real(0.8,   3.0,  name='dof_fstop'),
    # Object material (blue plastic)
    Real(0.0,   0.2,  name='obj_r'),
    Real(0.05,  0.4,  name='obj_g'),
    Real(0.4,   1.0,  name='obj_b'),
    Real(0.1,   0.8,  name='obj_roughness'),
    # Platform material (metal)
    Real(0.0,   0.4,  name='plat_r'),
    Real(0.1,   0.6,  name='plat_g'),
    Real(0.1,   0.6,  name='plat_b'),
    Real(0.1,   0.6,  name='plat_roughness'),
    Real(0.4,   1.0,  name='plat_metallic'),
    # Post-FX
    Real(0.0,   0.15, name='barrel_k1'),
    Real(0.5,   2.5,  name='blur_sigma'),
    Real(0.0,   0.5,  name='vignette'),
]

# Warm-start: current hardcoded values from render_rgb_batch.py
X0_MANUAL = [
    2.0,                                        # world_strength
    1.2,                                        # dof_fstop
    0.03, 0.18, 0.75, 0.35,                    # object (blue plastic)
    26/255, 115/255, 106/255, 0.25, 0.85,      # platform (teal metal)
    0.07, 1.25, 0.25,                           # post-FX
]

# ── Temp paths ─────────────────────────────────────────────────
PARAMS_TMP  = os.path.join(tempfile.gettempdir(), 'gs_rgb_params.json')
RENDER_BASE = os.path.join(tempfile.gettempdir(), 'gs_rgb_render')
RENDER_PNG  = RENDER_BASE + '.png'
# Use PID in the blend copy name to avoid file-lock conflicts across retries
BLEND_COPY  = os.path.join(tempfile.gettempdir(), f'gs_rgb_bo_{os.getpid()}.blend')

_call_count = [0]
_best_loss  = [float('inf')]
_history    = []


@use_named_args(PARAM_SPACE)
def objective(**params):
    _call_count[0] += 1
    idx = _call_count[0]

    clean = {k: float(v) for k, v in params.items()}
    with open(PARAMS_TMP, 'w') as f:
        json.dump(clean, f)

    for cand in (RENDER_PNG, RENDER_BASE + '0001.png'):
        if os.path.exists(cand):
            os.remove(cand)

    shutil.copy(BLEND_FILE, BLEND_COPY)
    env = os.environ.copy()
    env['RGB_PARAMS']     = PARAMS_TMP
    env['RGB_OUT']        = RENDER_BASE
    env['RGB_SCRIPT_DIR'] = SCRIPT_DIR

    t0 = time.time()
    proc = subprocess.run(
        [BLENDER_PATH, '--background', BLEND_COPY,
         '--python', RENDER_SCRIPT],
        cwd=SCRIPT_DIR, env=env,
        capture_output=True, timeout=240,
    )
    elapsed = time.time() - t0

    rendered_path = None
    for cand in (RENDER_PNG, RENDER_BASE + '0001.png'):
        if os.path.exists(cand):
            rendered_path = cand;  break

    if proc.returncode != 0 or rendered_path is None:
        tail = (proc.stderr or b'')[-400:].decode('utf-8', errors='replace')
        print(f'  [{idx:3d}] FAILED rc={proc.returncode}  ({elapsed:.1f}s)\n{tail}')
        _history.append((idx, 1.0))
        return 1.0

    img_raw = cv2.imread(rendered_path)
    if img_raw is None:
        print(f'  [{idx:3d}] FAILED: cannot read {rendered_path}')
        _history.append((idx, 1.0))
        return 1.0

    img_fx  = _apply_fx(img_raw, clean)
    img_fx  = cv2.resize(img_fx.astype(np.float32), TARGET_SIZE)
    loss    = _image_loss(img_fx, TARGET)
    _history.append((idx, loss))

    is_best = loss < _best_loss[0]
    if is_best:
        _best_loss[0] = loss
        cv2.imwrite(os.path.join(RESULTS_DIR, 'best_rgb_render.png'),
                    (img_fx * 255).astype(np.uint8))
        with open(os.path.join(RESULTS_DIR, 'best_rgb_params.json'), 'w') as f:
            json.dump(clean, f, indent=2)

    marker = ' ★ NEW BEST' if is_best else ''
    print(f'  [{idx:3d}] loss={loss:.5f}  best={_best_loss[0]:.5f}  ({elapsed:.1f}s){marker}')
    if is_best:
        print('         ' + '  '.join(f'{k}={v:.3g}' for k, v in clean.items()))
    return loss


def _early_stop(result):
    if result.fun <= LOSS_THRESHOLD:
        print(f'\n  ★ Early stop: best loss {result.fun:.5f} ≤ {LOSS_THRESHOLD}')
        return True
    return False


def main():
    print('=' * 65)
    print('GelSight RGB Bayesian Optimization')
    print(f'  Target  : {os.path.basename(TARGET_IMAGE)}  ({TARGET_SIZE[0]}x{TARGET_SIZE[1]})')
    print(f'  Params  : {len(PARAM_SPACE)} dimensions')
    print(f'  Loss    : 0.3*MSE + 0.7*histogram  (threshold={LOSS_THRESHOLD})')
    print(f'  Blender : {BLENDER_PATH}')
    print(f'  Results : {RESULTS_DIR}')
    print('=' * 65)

    result = forest_minimize(
        objective,
        PARAM_SPACE,
        x0=[X0_MANUAL],
        n_calls=500,
        n_initial_points=20,
        acq_func='EI',
        random_state=42,
        verbose=False,
        callback=_early_stop,
    )

    print(f'\n{"=" * 65}')
    print(f'Optimization complete.  Best loss: {result.fun:.6f}')
    best = {p.name: float(v) for p, v in zip(PARAM_SPACE, result.x)}
    print('Best parameters:')
    for k, v in best.items():
        print(f'  {k:16s}: {v:.4f}')

    final_path = os.path.join(RESULTS_DIR, 'best_rgb_params.json')
    with open(final_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f'\nSaved → {final_path}')

    # Convergence plot
    iters  = [h[0] for h in _history]
    losses = [h[1] for h in _history]
    bests  = [min(losses[:i + 1]) for i in range(len(losses))]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(iters, losses, 'o', alpha=0.35, ms=3, label='trial loss')
    ax.plot(iters, bests, '-', lw=2, label='best so far')
    ax.axhline(LOSS_THRESHOLD, ls='--', color='r', alpha=0.6, label=f'threshold={LOSS_THRESHOLD}')
    ax.set_xlabel('Trial');  ax.set_ylabel('Loss');  ax.legend()
    plt.tight_layout()
    conv_path = os.path.join(RESULTS_DIR, 'rgb_convergence.png')
    plt.savefig(conv_path, dpi=120)
    print(f'Convergence plot → {conv_path}')


if __name__ == '__main__':
    main()
