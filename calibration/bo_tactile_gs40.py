"""Single-frame Bayesian-optimisation calibration of the Blender tactile renderer for a NEW sensor
(written for GelSlim 4.0; works for any sensor whose background image you have).

Differences from bo_tactile_v2.py (which was tuned for GelSlim 5.0):
  * all six emitters are free (strength + RGB, 24 params) + 3 post-fx scalars: no sensor-A prior;
  * marker-robust objective: dark marker dots are detected in the target (black-hat + threshold),
    inpainted, and excluded from the score; the score is computed on Gaussian-blurred images
    (sigma 4 px at 128 px) so it fits the illumination FIELD, not texture or dots:
        score = 0.5 * (1 - LAB-MSE_blur*50) + 0.3 * SSIM_blur + 0.2 * (1 - Bhattacharyya(hist))
  * objective / bounds / n_iter configurable from the command line.

usage (from gs_blender/calibration, needs `pip install bayesian-optimization scikit-image opencv-python`):
    GELSIGHT_BO_TARGET=../real_data_gs40/background_0.jpg GELSIGHT_BO_RESULTS=../bo_results/tactile_gs40 \
    BLENDER_PATH=/path/to/blender python bo_tactile_gs40.py --init 30 --n_iter 250
Outputs best_params.json (drop-in for GELSIGHT_FIXED_PARAMS), best_render.png, comparison.png, dots_mask.png.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from bayes_opt import BayesianOptimization
from skimage.metrics import structural_similarity as ssim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
BLENDER_PATH = os.environ.get('BLENDER_PATH', '/home/shared/blender-4.2.0-linux-x64/blender')
TARGET_PATH = os.environ.get('GELSIGHT_BO_TARGET', os.path.join(ROOT_DIR, 'real_data_gs40', 'background_0.jpg'))
RESULTS_DIR = os.environ.get('GELSIGHT_BO_RESULTS', os.path.join(ROOT_DIR, 'bo_results', 'tactile_gs40'))
os.makedirs(RESULTS_DIR, exist_ok=True)
TARGET_SIZE = (128, 128)
BLUR_SIGMA = 4.0

# geometry / gel parameters are kept (sensor-independent enough for a first fit; tune by hand afterwards)
FIXED_PARAMS = {'scale_y': 0.4918, 'light_z': -0.004139, 'rot_z': -3.14159, 'fov': 60.0, 'length': 0.008751,
                'gel_roughness': 0.4455, 'gel_fac': 0.2971, 'smoothness': 30}
EMITTERS = ['top', 'bot', 'left', 'right', 'lg', 'rg']
PBOUNDS = {}
for e in EMITTERS:
    PBOUNDS[f'{e}_str'] = (0.0, 150.0)
    for c in 'rgb':
        PBOUNDS[f'{e}_{c}'] = (0.0, 1.0)
PBOUNDS.update({'saturation': (0.6, 2.0), 'brightness': (0.6, 1.5), 'contrast': (0.7, 1.8)})

PARAMS_TMP = os.path.join(tempfile.gettempdir(), 'gs_bo40_params.json')
RENDER_BASE = os.path.join(tempfile.gettempdir(), 'gs_bo40_render')
RENDER_PNG = RENDER_BASE + '.png'
BLEND_COPY = os.path.join(tempfile.gettempdir(), 'gs_bo40_sampler.blend')


def marker_mask(img_u8, min_area=6, max_area=400):
    """Dark blob (marker) mask via black-hat on the gray image; returns bool [H,W] and the inpainted image."""
    g = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    _, m = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area:
            keep[lab == i] = 255
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
    inp = cv2.inpaint(img_u8, keep, 5, cv2.INPAINT_TELEA)
    return keep > 0, inp


def load_target():
    img = cv2.imread(TARGET_PATH)
    if img is None:
        raise RuntimeError(f'Cannot load: {TARGET_PATH}')
    img = cv2.resize(img, TARGET_SIZE)
    mask, inp = marker_mask(img)
    cv2.imwrite(os.path.join(RESULTS_DIR, 'dots_mask.png'), np.hstack([img, cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), inp]))
    print(f'marker mask: {mask.mean() * 100:.1f}% of pixels masked')
    return inp.astype(np.float32) / 255.0, mask


def apply_post_fx(img_u8, saturation, brightness, contrast):
    img = img_u8
    if abs(saturation - 1.0) > 0.01:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if abs(brightness - 1.0) > 0.01:
        img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    if abs(contrast - 1.0) > 0.01:
        img = np.clip((img.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
    return img


def render_blender(params):
    with open(PARAMS_TMP, 'w') as f:
        json.dump({**FIXED_PARAMS, **params}, f)
    for c in (RENDER_PNG, RENDER_BASE + '0001.png'):
        if os.path.exists(c):
            os.remove(c)
    shutil.copy(os.path.join(ROOT_DIR, 'gelsight_sampler.blend'), BLEND_COPY)
    env = os.environ.copy()
    env['GELSIGHT_FIXED_PARAMS'] = PARAMS_TMP
    env['GELSIGHT_BG_RENDER'] = RENDER_BASE
    subprocess.run([BLENDER_PATH, '--background', BLEND_COPY, '--python', os.path.join(SCRIPT_DIR, 'scripting_bo.py')],
                   cwd=SCRIPT_DIR, env=env, capture_output=True, timeout=600)
    for c in (RENDER_PNG, RENDER_BASE + '0001.png'):
        if os.path.exists(c):
            return cv2.imread(c)
    return None


def blur(img):
    return cv2.GaussianBlur(img, (0, 0), BLUR_SIGMA)


def score(rendered_u8, target, mask, saturation, brightness, contrast):
    rendered = cv2.resize(apply_post_fx(rendered_u8, saturation, brightness, contrast), TARGET_SIZE).astype(np.float32) / 255.0
    rb, tb = blur(rendered), blur(target)
    r_lab = cv2.cvtColor((rb * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor((tb * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_mse = np.mean(((r_lab - t_lab) ** 2)[~mask]) / (255.0 ** 2)
    s = ssim((tb * 255).astype(np.uint8), (rb * 255).astype(np.uint8), channel_axis=2)
    bh = 0.0
    for c in range(3):
        h1 = cv2.calcHist([(target * 255).astype(np.uint8)], [c], (~mask).astype(np.uint8), [32], [0, 256]).ravel()
        h2 = cv2.calcHist([(rendered * 255).astype(np.uint8)], [c], None, [32], [0, 256]).ravel()
        bh += cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA) / 3
    return 0.5 * (1.0 - min(lab_mse * 50, 1.0)) + 0.3 * s + 0.2 * (1.0 - bh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_iter', type=int, default=250)
    ap.add_argument('--init', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    target, mask = load_target()
    print(f'Target: {TARGET_PATH}\nBO: {a.init} init + {a.n_iter} iter, {len(PBOUNDS)} params', flush=True)
    best = [-1.0]; it = [0]

    def objective(**kw):
        it[0] += 1
        r = render_blender({k: v for k, v in kw.items() if k not in ('saturation', 'brightness', 'contrast')})
        if r is None:
            print(f'  [{it[0]:4d}] RENDER FAILED', flush=True); return 0.0
        s = score(r, target, mask, kw['saturation'], kw['brightness'], kw['contrast'])
        if s > best[0]:
            best[0] = s
            cv2.imwrite(os.path.join(RESULTS_DIR, 'best_render.png'), apply_post_fx(r, kw['saturation'], kw['brightness'], kw['contrast']))
            json.dump({**FIXED_PARAMS, **{k: round(float(v), 4) for k, v in kw.items()}}, open(os.path.join(RESULTS_DIR, 'best_params.json'), 'w'), indent=2)
            print(f'  [{it[0]:4d}] NEW BEST score={s:.4f}', flush=True)
        elif it[0] % 10 == 0:
            print(f'  [{it[0]:4d}] score={s:.4f}  best={best[0]:.4f}', flush=True)
        return s

    opt = BayesianOptimization(f=objective, pbounds=PBOUNDS, random_state=a.seed, verbose=0)
    opt.maximize(init_points=a.init, n_iter=a.n_iter)
    bi = cv2.imread(os.path.join(RESULTS_DIR, 'best_render.png'))
    ti = cv2.resize(cv2.imread(TARGET_PATH), (bi.shape[1], bi.shape[0]))
    cv2.imwrite(os.path.join(RESULTS_DIR, 'comparison.png'), np.hstack([ti, bi]))
    print(f'\nBest score {opt.max["target"]:.4f}; results in {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
