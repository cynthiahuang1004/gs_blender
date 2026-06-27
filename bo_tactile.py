"""
bo_tactile.py
=============
Bayesian optimization to find GelSight sensor parameters that minimize the
visual difference between a simulated background render and real sensor
baseline images.

Pipeline
--------
1. Load & average all real_data/base_tactile_images/*.jpg  → TARGET
2. For each BO trial:
   a. Write a params JSON to a temp file
   b. Call Blender (background mode) via the BG_RENDER_PATH env hook in scripting.py
   c. Load the rendered PNG, resize to TARGET_SIZE
   d. Compute loss  =  0.6 * MSE  +  0.4 * per-channel histogram distance
3. forest_minimize (scikit-optimize) searches the 24-D parameter space
4. Save  bo_results/final_params.json  +  best_render.png  +  convergence.png

Usage
-----
    pip install scikit-optimize opencv-python matplotlib
    python bo_tactile.py

After convergence, use the saved final_params.json as the fixed sensor
configuration (set GELSIGHT_FIXED_PARAMS env var before running Blender).
"""

import os, json, subprocess, tempfile, time, shutil
import numpy as np
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── scikit-optimize ───────────────────────────────────────────────────────────
try:
    from skopt import forest_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence
except ImportError:
    raise SystemExit(
        'scikit-optimize not found.\n'
        'Install with:  pip install scikit-optimize'
    )

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
BLENDER_PATH    = r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'
BASE_IMAGES_DIR = os.path.join(SCRIPT_DIR, 'real_data', 'base_tactile_images')
RESULTS_DIR     = os.path.join(SCRIPT_DIR, 'bo_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_SIZE = (128, 128)   # resize everything to this for speed

# ── Early-stopping threshold ──────────────────────────────────────────────────
# loss = 0.6*MSE + 0.4*Bhattacharyya, both in [0,1].
# 0.12 = current (noticeable colour/brightness gap)
# 0.06 = target  (visually close, sim-to-real usable)
# 0.03 = physics lower bound (nearly unreachable due to sim/real domain gap)
LOSS_THRESHOLD = 0.06

# ── Load target image (mean of all base images) ───────────────────────────────

def _load_target():
    imgs = []
    for fname in sorted(os.listdir(BASE_IMAGES_DIR)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(BASE_IMAGES_DIR, fname))
            if img is None:
                continue
            img = cv2.resize(img, TARGET_SIZE).astype(np.float32) / 255.0
            imgs.append(img)
    if not imgs:
        raise RuntimeError(f'No images found in {BASE_IMAGES_DIR}')
    target = np.mean(imgs, axis=0)
    # Save target for visual reference
    cv2.imwrite(os.path.join(RESULTS_DIR, 'target_mean.png'),
                (target * 255).astype(np.uint8))
    print(f'Target: averaged {len(imgs)} base images → {RESULTS_DIR}/target_mean.png')
    return target


TARGET = _load_target()


# ── Parameter space (24 dimensions) ──────────────────────────────────────────
#
# 6 independent emitters, each with strength + RGB colour  (4 params × 6 = 24).
# Colour intents (warm-started from manual tuning in X0_MANUAL):
#     Top  / LeftGreen   → gray-green
#     Bottom             → weak blue-green
#     Left               → weak red
#     Right              → strong red
#     RightGreen         → dominant gray-green
#
# NOT searched here (fixed to best-known values inside scripting.py, since BO
# only ever sees a flat no-contact render):
#     scale=0.4918, light_rot_z=-3.14159, light_z=-0.004139, fov=60.0,
#     length=0.008751, gel_roughness=0.4455, gel_fac=0.2971,
#     green_str=39.69, smoothness=30
#
# Ranges derived from scripting.py manual tuning, with slack for BO to explore.
# Order here must match X0_MANUAL below.
PARAM_SPACE = [
    # TopEmittor (actual image: 左下) — gray-green
    Real(40.0,  150.0,  name='top_str'),
    Real(0.0,   0.6,    name='top_r'),
    Real(0.4,   1.0,    name='top_g'),
    Real(0.0,   0.6,    name='top_b'),

    # BottomEmittor (actual image: 右上) — weak blue-green
    Real(20.0,  100.0,  name='bot_str'),
    Real(0.0,   0.5,    name='bot_r'),
    Real(0.2,   0.8,    name='bot_g'),
    Real(0.5,   1.0,    name='bot_b'),

    # LeftEmittor (actual image: 左上) — weak red
    Real(15.0,  80.0,   name='left_str'),
    Real(0.5,   1.0,    name='left_r'),
    Real(0.0,   0.4,    name='left_g'),
    Real(0.0,   0.3,    name='left_b'),

    # RightEmittor (actual image: 右下) — strong red
    Real(60.0,  160.0,  name='right_str'),
    Real(0.7,   1.0,    name='right_r'),
    Real(0.0,   0.4,    name='right_g'),
    Real(0.0,   0.2,    name='right_b'),

    # LeftGreenEmittor (image: 右) — gray-green
    Real(20.0,  100.0,  name='lg_str'),
    Real(0.0,   0.5,    name='lg_r'),
    Real(0.4,   1.0,    name='lg_g'),
    Real(0.0,   0.6,    name='lg_b'),

    # RightGreenEmittor (image: 左) — dominant gray-green
    Real(60.0,  180.0,  name='rg_str'),
    Real(0.0,   0.5,    name='rg_r'),
    Real(0.5,   1.0,    name='rg_g'),
    Real(0.0,   0.5,    name='rg_b'),
]

# Warm-start: matches user-specified colour intent (all 24 values verified
# to lie within their corresponding PARAM_SPACE bounds).
X0_MANUAL = [
    80.0,  0.3,  0.65, 0.3,    # top: gray-green       (actual image: 左下)
    40.0,  0.1,  0.5,  0.9,    # bot: weak blue-green  (actual image: 右上)
    30.0,  0.9,  0.05, 0.05,   # left: weak red        (actual image: 左上)
    120.0, 1.0,  0.0,  0.0,    # right: strong red     (actual image: 右下)
    60.0,  0.3,  0.65, 0.3,    # lg: gray-green        (右)
    120.0, 0.3,  0.7,  0.3,    # rg: dominant gray-green (左)
]


# ── Loss function ─────────────────────────────────────────────────────────────

def _image_loss(rendered: np.ndarray, target: np.ndarray) -> float:
    """
    Weighted combination of:
      - MSE        (captures spatial gradient pattern)
      - Per-channel Bhattacharyya histogram distance (captures colour distribution)
    Both inputs are float32 BGR images in [0, 1] at TARGET_SIZE.
    """
    mse = float(np.mean((rendered - target) ** 2))

    hist_d = 0.0
    for c in range(3):
        h_r = cv2.calcHist([rendered], [c], None, [64], [0.0, 1.0])
        h_t = cv2.calcHist([target],   [c], None, [64], [0.0, 1.0])
        cv2.normalize(h_r, h_r)
        cv2.normalize(h_t, h_t)
        hist_d += cv2.compareHist(h_r, h_t, cv2.HISTCMP_BHATTACHARYYA)
    hist_d /= 3.0

    return 0.6 * mse + 0.4 * hist_d


# ── Objective function (called by forest_minimize) ────────────────────────────

_call_count = [0]
_best_loss  = [float('inf')]
_history    = []   # list of (call_idx, loss)

PARAMS_TMP  = os.path.join(tempfile.gettempdir(), 'gs_fixed_params.json')
RENDER_BASE = os.path.join(tempfile.gettempdir(), 'gs_bg_render')
RENDER_PNG  = RENDER_BASE + '.png'
# Temporary copy of the blend file — never touch the original
BLEND_COPY  = os.path.join(tempfile.gettempdir(), 'gs_bo_render.blend')


@use_named_args(PARAM_SPACE)
def objective(**params):
    _call_count[0] += 1
    idx = _call_count[0]

    # ── serialise params (skopt may pass numpy scalars) ──
    clean = {k: (int(v) if isinstance(v, (np.integer, int)) else float(v))
             for k, v in params.items()}
    with open(PARAMS_TMP, 'w') as f:
        json.dump(clean, f)

    # ── remove stale render ──
    for candidate in (RENDER_PNG, RENDER_BASE + '0001.png'):
        if os.path.exists(candidate):
            os.remove(candidate)

    # ── call Blender (use a temp copy so the original .blend is never modified) ──
    shutil.copy(os.path.join(SCRIPT_DIR, 'gelsight_sampler.blend'), BLEND_COPY)

    env = os.environ.copy()
    env['GELSIGHT_FIXED_PARAMS'] = PARAMS_TMP
    env['GELSIGHT_BG_RENDER']    = RENDER_BASE

    t0 = time.time()
    proc = subprocess.run(
        [BLENDER_PATH, '--background', BLEND_COPY,
         '--python', os.path.join(SCRIPT_DIR, 'scripting_bo.py')],
        cwd=SCRIPT_DIR,
        env=env,
        capture_output=True,
        timeout=240,
    )
    elapsed = time.time() - t0

    # ── find rendered file (handle frame-numbered variants) ──
    rendered_path = None
    for candidate in (RENDER_PNG, RENDER_BASE + '0001.png'):
        if os.path.exists(candidate):
            rendered_path = candidate
            break

    if proc.returncode != 0 or rendered_path is None:
        stderr_tail = proc.stderr[-400:].decode('utf-8', errors='replace') if proc.stderr else ''
        print(f'  [{idx:3d}] FAILED rc={proc.returncode}  ({elapsed:.1f}s)\n{stderr_tail}')
        _history.append((idx, 1.0))
        return 1.0

    rendered = cv2.imread(rendered_path).astype(np.float32) / 255.0
    rendered = cv2.resize(rendered, TARGET_SIZE)
    loss = _image_loss(rendered, TARGET)

    _history.append((idx, loss))

    is_best = loss < _best_loss[0]
    if is_best:
        _best_loss[0] = loss
        cv2.imwrite(os.path.join(RESULTS_DIR, 'best_render.png'),
                    (rendered * 255).astype(np.uint8))
        with open(os.path.join(RESULTS_DIR, 'best_params.json'), 'w') as f:
            json.dump(clean, f, indent=2)

    marker = ' ★ NEW BEST' if is_best else ''
    print(f'  [{idx:3d}] loss={loss:.5f}  best={_best_loss[0]:.5f}  ({elapsed:.1f}s){marker}')
    if is_best:
        print('         ' + '  '.join(f'{k}={v:.3g}' for k, v in clean.items()))

    return loss


# ── Main ──────────────────────────────────────────────────────────────────────

def _early_stop_callback(result):
    """Stop as soon as best loss drops below LOSS_THRESHOLD."""
    if result.fun <= LOSS_THRESHOLD:
        print(f'\n  ★ Early stop: best loss {result.fun:.5f} ≤ threshold {LOSS_THRESHOLD}')
        return True   # returning True tells skopt to stop
    return False


def main():
    print('=' * 65)
    print('GelSight Sensor Bayesian Optimization')
    print(f'  Target    : avg of {len(os.listdir(BASE_IMAGES_DIR))} base images '
          f'@ {TARGET_SIZE[0]}×{TARGET_SIZE[1]}')
    print(f'  Params    : {len(PARAM_SPACE)} dimensions')
    print(f'  Threshold : {LOSS_THRESHOLD}  (stops early if reached)')
    print(f'  Max calls : 500  (safety net)')
    print(f'  Blender   : {BLENDER_PATH}')
    print(f'  Results   : {RESULTS_DIR}')
    print('=' * 65)

    result = forest_minimize(
        objective,
        PARAM_SPACE,
        x0=[X0_MANUAL],        # warm start (single initial point)
        n_calls=500,
        n_initial_points=50,   # random exploration before surrogate kicks in (24D)
        acq_func='EI',
        random_state=42,
        verbose=False,
        callback=_early_stop_callback,
    )

    print(f'\n{"=" * 65}')
    print(f'Optimization complete.  Best loss: {result.fun:.6f}')
    best_params = {
        p.name: int(v) if isinstance(v, (np.integer, int)) else float(v)
        for p, v in zip(PARAM_SPACE, result.x)
    }
    print('Best parameters:')
    for k, v in best_params.items():
        print(f'  {k:16s}: {v}')

    # ── Save final params ──
    final_path = os.path.join(RESULTS_DIR, 'final_params.json')
    with open(final_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    # ── Save convergence plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    plot_convergence(result, ax=ax)
    ax.set_title('BO Convergence (min loss per call)')

    ax2 = axes[1]
    calls, losses = zip(*_history) if _history else ([], [])
    ax2.plot(calls, losses, 'o-', alpha=0.6, markersize=3)
    ax2.set_xlabel('Call index')
    ax2.set_ylabel('Loss')
    ax2.set_title('All evaluations')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    conv_path = os.path.join(RESULTS_DIR, 'convergence.png')
    fig.savefig(conv_path, dpi=120)
    plt.close()

    # ── Side-by-side comparison ──
    best_render_path = os.path.join(RESULTS_DIR, 'best_render.png')
    target_path      = os.path.join(RESULTS_DIR, 'target_mean.png')
    if os.path.exists(best_render_path) and os.path.exists(target_path):
        best_img   = cv2.imread(best_render_path)
        target_img = cv2.imread(target_path)
        comparison = np.hstack([target_img, best_img])
        cv2.imwrite(os.path.join(RESULTS_DIR, 'comparison.png'), comparison)

    print(f'\nResults saved to  {RESULTS_DIR}/')
    print('  final_params.json  — use with GELSIGHT_FIXED_PARAMS env var')
    print('  best_render.png    — best matching background render')
    print('  target_mean.png    — averaged real baseline target')
    print('  comparison.png     — target | best render side by side')
    print('  convergence.png    — loss curves')
    print(f'\nTo use the optimized params in a normal render run:')
    print(f'  set GELSIGHT_FIXED_PARAMS={final_path}')
    print(f'  python run_blender.py')


if __name__ == '__main__':
    main()