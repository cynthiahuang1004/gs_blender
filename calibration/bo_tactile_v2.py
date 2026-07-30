"""
bo_tactile_v2.py
================
BO for base tactile image, searching only:
  - Left/Bottom/Right emitter color + strength (12 params)
  - Post-processing: saturation, brightness, contrast (3 params)
Green emitters (Top, LeftGreen, RightGreen) are fixed to manually tuned values.

Target: real_data_test/base_tactile_images/0.jpg

Usage:
    python bo_tactile_v2.py [--n_iter 150] [--init 20]
"""

import os, sys, json, subprocess, shutil, tempfile, argparse, time
import numpy as np
import cv2

try:
    from bayes_opt import BayesianOptimization
except ImportError:
    raise SystemExit('Install: pip install bayesian-optimization')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
BLENDER_PATH = '/home/shared/blender-4.2.0-linux-x64/blender'
TARGET_PATH = os.path.join(ROOT_DIR, 'real_data_test', 'base_tactile_images', '0.jpg')
RESULTS_DIR = os.path.join(ROOT_DIR, 'bo_results', 'tactile_v2')
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_SIZE = (128, 128)

# Fixed green emitters (manually tuned)
FIXED_PARAMS = {
    'top_str': 54.06, 'top_r': 0.05, 'top_g': 0.65, 'top_b': 0.02,
    'lg_str': 28.38, 'lg_r': 0.05, 'lg_g': 0.65, 'lg_b': 0.09,
    'rg_str': 176.33, 'rg_r': 0.03, 'rg_g': 0.71, 'rg_b': 0.04,
    'scale_y': 0.4918,
    'light_z': -0.004139,
    'rot_z': -3.14159,
    'fov': 60.0,
    'length': 0.008751,
    'gel_roughness': 0.4455,
    'gel_fac': 0.2971,
    'smoothness': 30,
}

# Baseline values for BO params (current manual tune)
BASELINE = {
    'bot_str': 50.0, 'bot_r': 0.27, 'bot_g': 0.21, 'bot_b': 0.85,
    'left_str': 18.0, 'left_r': 0.70, 'left_g': 0.30, 'left_b': 0.02,
    'right_str': 84.83, 'right_r': 0.92, 'right_g': 0.08, 'right_b': 0.05,
    'saturation': 1.3, 'brightness': 1.0, 'contrast': 1.3,
}

# Search bounds (reasonable range around baseline)
PBOUNDS = {
    'bot_str':    (20.0, 100.0),
    'bot_r':      (0.0, 0.5),
    'bot_g':      (0.0, 0.5),
    'bot_b':      (0.5, 1.0),
    'left_str':   (5.0, 50.0),
    'left_r':     (0.4, 1.0),
    'left_g':     (0.0, 0.5),
    'left_b':     (0.0, 0.3),
    'right_str':  (40.0, 150.0),
    'right_r':    (0.6, 1.0),
    'right_g':    (0.0, 0.3),
    'right_b':    (0.0, 0.3),
    'saturation': (0.8, 2.0),
    'brightness': (0.7, 1.4),
    'contrast':   (0.8, 1.8),
}

# Blender temp files
PARAMS_TMP = os.path.join(tempfile.gettempdir(), 'gs_bo2_params.json')
RENDER_BASE = os.path.join(tempfile.gettempdir(), 'gs_bo2_render')
RENDER_PNG = RENDER_BASE + '.png'
BLEND_COPY = os.path.join(tempfile.gettempdir(), 'gs_bo2.blend')


def load_target():
    img = cv2.imread(TARGET_PATH)
    if img is None:
        raise RuntimeError(f'Cannot load: {TARGET_PATH}')
    return cv2.resize(img, TARGET_SIZE).astype(np.float32) / 255.0


def apply_post_fx(img_u8, saturation, brightness, contrast):
    img = img_u8.copy()
    if abs(saturation - 1.0) > 0.01:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if abs(brightness - 1.0) > 0.01:
        img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    if abs(contrast - 1.0) > 0.01:
        img = np.clip((img.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
    return img


def render_blender(params_dict):
    all_params = {**FIXED_PARAMS, **params_dict}
    with open(PARAMS_TMP, 'w') as f:
        json.dump(all_params, f)

    for c in (RENDER_PNG, RENDER_BASE + '0001.png'):
        if os.path.exists(c):
            os.remove(c)

    shutil.copy(os.path.join(ROOT_DIR, 'gelsight_sampler.blend'), BLEND_COPY)

    env = os.environ.copy()
    env['GELSIGHT_FIXED_PARAMS'] = PARAMS_TMP
    env['GELSIGHT_BG_RENDER'] = RENDER_BASE

    proc = subprocess.run(
        [BLENDER_PATH, '--background', BLEND_COPY,
         '--python', os.path.join(SCRIPT_DIR, 'scripting_bo.py')],
        cwd=SCRIPT_DIR, env=env, capture_output=True, timeout=300,
    )

    for c in (RENDER_PNG, RENDER_BASE + '0001.png'):
        if os.path.exists(c):
            return cv2.imread(c)
    return None


def score(rendered_u8, target, saturation, brightness, contrast):
    processed = apply_post_fx(rendered_u8, saturation, brightness, contrast)
    rendered = cv2.resize(processed, TARGET_SIZE).astype(np.float32) / 255.0

    # LAB MSE
    r_lab = cv2.cvtColor((rendered * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_mse = np.mean((r_lab - t_lab) ** 2) / (255.0 ** 2)

    # SSIM
    from skimage.metrics import structural_similarity as ssim
    s = ssim((target * 255).astype(np.uint8), (rendered * 255).astype(np.uint8), channel_axis=2)

    return 0.5 * s + 0.5 * (1.0 - min(lab_mse * 50, 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=150)
    parser.add_argument('--init', type=int, default=20)
    args = parser.parse_args()

    target = load_target()
    print(f'Target: {TARGET_PATH}')
    print(f'BO: {args.init} init + {args.n_iter} iter, 15 params')

    best_score = -1
    best_params = None
    iteration = [0]

    def objective(**kwargs):
        nonlocal best_score, best_params
        iteration[0] += 1

        blender_params = {k: kwargs[k] for k in kwargs if k not in ('saturation', 'brightness', 'contrast')}
        rendered_u8 = render_blender(blender_params)

        if rendered_u8 is None:
            print(f'  [{iteration[0]:4d}] RENDER FAILED')
            return 0.0

        s = score(rendered_u8, target, kwargs['saturation'], kwargs['brightness'], kwargs['contrast'])

        if s > best_score:
            best_score = s
            best_params = dict(kwargs)
            processed = apply_post_fx(rendered_u8, kwargs['saturation'], kwargs['brightness'], kwargs['contrast'])
            cv2.imwrite(os.path.join(RESULTS_DIR, 'best_render.png'), processed)
            with open(os.path.join(RESULTS_DIR, 'best_params.json'), 'w') as f:
                json.dump({k: round(v, 4) for k, v in best_params.items()}, f, indent=2)
            print(f'  [{iteration[0]:4d}] NEW BEST score={s:.4f}')
        elif iteration[0] % 10 == 0:
            print(f'  [{iteration[0]:4d}] score={s:.4f}  best={best_score:.4f}')

        return s

    optimizer = BayesianOptimization(
        f=objective, pbounds=PBOUNDS, random_state=42, verbose=0,
    )

    # Probe baseline first
    optimizer.probe(params=BASELINE, lazy=True)

    print('Starting BO...')
    optimizer.maximize(init_points=args.init, n_iter=args.n_iter)

    best = optimizer.max
    print(f'\nBest score: {best["target"]:.4f}')
    print('Best params:')
    for k, v in sorted(best['params'].items()):
        print(f'  {k}: {v:.4f}')

    # Save comparison
    best_img = cv2.imread(os.path.join(RESULTS_DIR, 'best_render.png'))
    target_img = cv2.resize(cv2.imread(TARGET_PATH), (best_img.shape[1], best_img.shape[0]))
    comp = np.hstack([target_img, best_img])
    cv2.imwrite(os.path.join(RESULTS_DIR, 'comparison.png'), comp)
    print(f'\nResults saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
