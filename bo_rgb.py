"""
bo_rgb.py
=========
Bayesian optimization to calibrate RGB rendering parameters against
all real RGB images in real_data/rgb_images/ (mean loss across all targets).

Pipeline
--------
1. Load 47.jpg as target (128x128)
2. For each BO trial:
   a. Write params JSON to temp file
   b. Call Blender with gs_rgb_render.py (fixed scene)
   c. Apply post-FX in Python:
      barrel distortion + radial blur + vignette +
      radial warm tint + haze (gel optics simulation)
   d. loss = 0.2*MSE + 0.5*Bhattacharyya + 0.3*(1-SSIM)
3. forest_minimize over 22-D parameter space
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
BLENDER_PATH  = '/home/shared/blender-4.2.0-linux-x64/blender'
RESULTS_DIR   = os.path.join(SCRIPT_DIR, 'bo_results')
RENDER_SCRIPT = os.path.join(SCRIPT_DIR, 'gs_rgb_render.py')
BLEND_FILE    = os.path.join(SCRIPT_DIR, 'gelsight_sampler.blend')
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_SIZE = (128, 128)
LOSS_THRESHOLD = 0.05

# ── Load targets (all real RGB images) ─────────────────────────
def _load_targets():
    targets = []
    img_dir = os.path.join(SCRIPT_DIR, 'real_data', 'rgb_images')
    fnames = sorted(f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    for fname in fnames:
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        img = cv2.resize(img, TARGET_SIZE).astype(np.float32) / 255.0
        targets.append(img)
    if not targets:
        raise RuntimeError(f'No images found in {img_dir}')
    mean_img = np.mean(targets, axis=0)
    cv2.imwrite(os.path.join(RESULTS_DIR, 'rgb_target_mean.png'),
                (mean_img * 255).astype(np.uint8))
    print(f'Targets: {len(targets)} images from real_data/rgb_images/')
    print(f'  Mean target → {RESULTS_DIR}/rgb_target_mean.png')
    return targets

TARGETS = _load_targets()

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

def _boost_sat(rgb, factor, dist):
    """Boost saturation in center, fade toward edges. rgb: float32 BGR [0,1]."""
    if abs(factor - 1.0) < 0.01:
        return rgb
    gray = 0.114 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.299 * rgb[:,:,2]
    gray = gray[:,:,None]
    center_w = np.clip(1.0 - dist, 0, 1)[:,:,None] ** 2
    local_fac = 1.0 + (factor - 1.0) * center_w
    return np.clip(gray + (rgb - gray) * local_fac, 0, 1)

def _apply_fx(img_uint8, p):
    """Apply barrel + radial blur + vignette + asymmetric warm tint + haze + sat boost."""
    rgb = img_uint8.astype(np.float32) / 255.
    H, W = rgb.shape[:2]
    rgb = _barrel(rgb, float(p.get('barrel_k1', 0.07)))
    blurred = _gblur(rgb, float(p.get('blur_sigma', 1.25)))
    cy, cx = H / 2., W / 2.
    yy, xx = np.mgrid[0:H, 0:W]

    # Asymmetric tint center: tint_cx/cy shift the "clear" center
    tcx = cx + float(p.get('tint_cx', 0.0)) * cx
    tcy = cy + float(p.get('tint_cy', 0.0)) * cy
    dist = np.sqrt(((yy - tcy) / cy) ** 2 + ((xx - tcx) / cx) ** 2)

    w = np.clip(dist ** 1.5, 0, 1)[:, :, None]
    rgb = rgb * (1 - w) + blurred * w

    # Radial warm tint: blend toward a warm color at edges (simulates gel optics)
    tint_color = np.array([[[
        float(p.get('tint_b', 0.15)),   # BGR order for cv2
        float(p.get('tint_g', 0.55)),
        float(p.get('tint_r', 0.95)),
    ]]], dtype=np.float32)
    tint_str = float(p.get('tint_strength', 0.4))
    tint_w = np.clip(dist ** 2 * tint_str, 0, 1)[:, :, None]
    rgb = rgb * (1 - tint_w) + tint_color * tint_w

    # Haze layer: semi-transparent warm fog over edges
    haze_opacity = float(p.get('haze_opacity', 0.2))
    haze_w = np.clip(dist ** 1.5 * haze_opacity, 0, 1)[:, :, None]
    rgb = rgb * (1 - haze_w) + tint_color * 0.7 * haze_w

    # Center saturation boost
    sat_boost = float(p.get('sat_boost', 1.4))
    rgb = _boost_sat(rgb, sat_boost, dist)

    # Vignette (brightness falloff)
    mask = np.clip(1. - dist ** 2 * float(p.get('vignette', 0.25)), 0, 1)[:, :, None]
    return np.clip(rgb * mask, 0, 1)

# ── Loss ───────────────────────────────────────────────────────

def _ssim(img1, img2):
    """Mean SSIM over 3 channels. Inputs: float32 [0,1] same shape."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = 7
    s = 1.5
    sz = max(3, k | 1)
    x = np.arange(sz) - sz // 2
    g = np.exp(-x ** 2 / (2 * s ** 2));  g /= g.sum()
    def _conv2d(arr):
        out = np.empty_like(arr)
        for c in range(arr.shape[2]):
            h = np.apply_along_axis(lambda r: np.convolve(r, g, mode='same'), 1, arr[:, :, c])
            out[:, :, c] = np.apply_along_axis(lambda r: np.convolve(r, g, mode='same'), 0, h)
        return out
    mu1 = _conv2d(img1);  mu2 = _conv2d(img2)
    mu1_sq = mu1 ** 2;  mu2_sq = mu2 ** 2;  mu12 = mu1 * mu2
    s1_sq = _conv2d(img1 ** 2) - mu1_sq
    s2_sq = _conv2d(img2 ** 2) - mu2_sq
    s12   = _conv2d(img1 * img2) - mu12
    num   = (2 * mu12 + C1) * (2 * s12 + C2)
    den   = (mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2)
    return float(np.mean(num / den))


def _image_loss(rendered, target):
    """Both inputs: float32 BGR [0,1] at TARGET_SIZE."""
    mse = float(np.mean((rendered - target) ** 2))
    hist_d = 0.0
    for c in range(3):
        h_r = cv2.calcHist([rendered], [c], None, [64], [0., 1.])
        h_t = cv2.calcHist([target],   [c], None, [64], [0., 1.])
        cv2.normalize(h_r, h_r);  cv2.normalize(h_t, h_t)
        hist_d += cv2.compareHist(h_r, h_t, cv2.HISTCMP_BHATTACHARYYA)
    ssim_val = _ssim(rendered, target)
    return 0.2 * mse + 0.5 * (hist_d / 3.0) + 0.3 * (1.0 - ssim_val)

# ── Parameter space (22-D) ─────────────────────────────────────
PARAM_SPACE = [
    # World / ambient lighting
    Real(0.5,   4.0,  name='world_strength'),
    # Camera depth of field
    Real(0.8,   3.0,  name='dof_fstop'),
    # Object material (blue plastic)
    Real(0.05,  0.55, name='obj_r'),
    Real(0.15,  0.65, name='obj_g'),
    Real(0.3,   1.0,  name='obj_b'),
    Real(0.1,   0.8,  name='obj_roughness'),
    # Platform material (metal)
    Real(0.0,   0.4,  name='plat_r'),
    Real(0.1,   0.6,  name='plat_g'),
    Real(0.1,   0.6,  name='plat_b'),
    Real(0.1,   0.6,  name='plat_roughness'),
    Real(0.4,   1.0,  name='plat_metallic'),
    # Post-FX: barrel + blur
    Real(0.0,   0.15, name='barrel_k1'),
    Real(0.5,   2.5,  name='blur_sigma'),
    Real(0.0,   0.5,  name='vignette'),
    # Radial warm tint (gel edge color)
    Real(0.6,   1.0,  name='tint_r'),
    Real(0.3,   0.8,  name='tint_g'),
    Real(0.0,   0.4,  name='tint_b'),
    Real(0.1,   0.8,  name='tint_strength'),
    # Asymmetric tint center offset (-0.5..+0.5 of half-width)
    Real(-0.5,  0.5,  name='tint_cx'),
    Real(-0.5,  0.5,  name='tint_cy'),
    # Center saturation boost
    Real(1.0,   2.5,  name='sat_boost'),
    # Haze (gel fog)
    Real(0.0,   0.6,  name='haze_opacity'),
]

# Warm-start from v4 best
X0_MANUAL = [
    2.370,                                      # world_strength
    0.865,                                      # dof_fstop
    0.165, 0.249, 0.386,                        # obj RGB
    0.364,                                      # obj_roughness
    0.226, 0.296, 0.468, 0.524, 0.582,         # platform
    0.048, 1.607, 0.006,                        # barrel, blur, vignette
    0.837, 0.734, 0.317, 0.311,                # tint
    -0.394, 0.129,                              # tint center offset
    1.806,                                      # sat_boost
    0.047,                                      # haze_opacity
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
    loss    = float(np.mean([_image_loss(img_fx, t) for t in TARGETS]))
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
    print(f'  Targets : {len(TARGETS)} real images  ({TARGET_SIZE[0]}x{TARGET_SIZE[1]})')
    print(f'  Params  : {len(PARAM_SPACE)} dimensions')
    print(f'  Loss    : 0.2*MSE + 0.5*histogram + 0.3*(1-SSIM)  (threshold={LOSS_THRESHOLD})')
    print(f'  Blender : {BLENDER_PATH}')
    print(f'  Results : {RESULTS_DIR}')
    print('=' * 65)

    result = forest_minimize(
        objective,
        PARAM_SPACE,
        x0=[X0_MANUAL],
        n_calls=500,
        n_initial_points=50,
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
