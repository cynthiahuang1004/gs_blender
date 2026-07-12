"""
bo_gel_effect.py
================
Bayesian Optimization to find post-processing parameters that transform
a no-gel RGB image into a gel-like RGB image.

Usage:
    python bo_gel_effect.py [--n_iter 100] [--init 10]
"""

import numpy as np
import cv2
import json
import argparse
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from bayes_opt import BayesianOptimization

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent
NO_GEL_PATH = ROOT_DIR / 'real_data' / 'no_gel.png'
GEL_PATH = ROOT_DIR / 'real_data' / 'gel.png'
OUT_DIR = ROOT_DIR / 'bo_results' / 'gel_fx'
OUT_DIR.mkdir(exist_ok=True)


def load_images():
    no_gel = cv2.imread(str(NO_GEL_PATH))
    gel = cv2.imread(str(GEL_PATH))
    # Resize to match if needed
    if no_gel.shape != gel.shape:
        gel = cv2.resize(gel, (no_gel.shape[1], no_gel.shape[0]))
    no_gel = cv2.cvtColor(no_gel, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gel = cv2.cvtColor(gel, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return no_gel, gel


def gaussian_blur(img, sigma):
    if sigma < 0.3:
        return img.copy()
    return cv2.GaussianBlur(img, (0, 0), sigma)


def barrel_distort(img, k1):
    if abs(k1) < 1e-4:
        return img.copy()
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx - cx) / cx
    yn = (yy - cy) / cy
    r2 = xn**2 + yn**2
    fac = 1.0 + k1 * r2
    xs = np.clip(xn * fac * cx + cx, 0, W - 1)
    ys = np.clip(yn * fac * cy + cy, 0, H - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, W - 1)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, H - 1)
    wx = (xs - x0)[:, :, None]; wy = (ys - y0)[:, :, None]
    return np.clip(img[y0,x0]*(1-wx)*(1-wy) + img[y0,x1]*wx*(1-wy) +
                   img[y1,x0]*(1-wx)*wy + img[y1,x1]*wx*wy, 0, 1)


def apply_gel_fx(img, params):
    """Apply all post-processing effects to transform no-gel → gel."""
    H, W = img.shape[:2]
    rgb = img.copy()

    # 1. Barrel distortion
    rgb = barrel_distort(rgb, params['barrel_k1'])

    # 2. Refraction distortion (secondary)
    rgb = barrel_distort(rgb, params['refraction_k2'])

    # 3. Radial blur (center sharp, edge blurry)
    if params['blur_sigma'] > 0.3:
        blurred = gaussian_blur(rgb, params['blur_sigma'])
        cy, cx = H / 2.0, W / 2.0
        yy, xx = np.mgrid[0:H, 0:W]
        dist = np.sqrt(((yy - cy) / cy)**2 + ((xx - cx) / cx)**2)
        weight = np.clip(dist ** 1.5, 0, 1)[:,:,None]
        rgb = rgb * (1 - weight) + blurred * weight
    else:
        cy, cx = H / 2.0, W / 2.0
        yy, xx = np.mgrid[0:H, 0:W]
        dist = np.sqrt(((yy - cy) / cy)**2 + ((xx - cx) / cx)**2)

    # 4. Blue shift
    rgb[:,:,0] = np.clip(rgb[:,:,0] - params['blue_shift'] * 0.3, 0, 1)
    rgb[:,:,1] = np.clip(rgb[:,:,1] - params['blue_shift'] * 0.1, 0, 1)
    rgb[:,:,2] = np.clip(rgb[:,:,2] + params['blue_shift'] * 0.2, 0, 1)

    # 5. Color tint (radial)
    tint = np.array([[[params['tint_r'], params['tint_g'], params['tint_b']]]], dtype=np.float32)
    tint_w = np.clip(dist ** 2 * params['tint_strength'], 0, 1)[:,:,None]
    rgb = rgb * (1 - tint_w) + tint * tint_w

    # 6. Haze
    haze_w = np.clip(dist ** 1.5 * params['haze_opacity'], 0, 1)[:,:,None]
    rgb = rgb * (1 - haze_w) + tint * 0.7 * haze_w

    # 7. Contrast boost
    rgb = np.clip((rgb - 0.5) * params['contrast_boost'] + 0.5, 0, 1)

    # 8. Gamma
    rgb = np.clip(np.power(np.maximum(rgb, 0), 1.0 / params['gamma']), 0, 1)

    # 9. Clarity (local contrast)
    if abs(params['clarity']) > 0.1:
        soft = gaussian_blur(rgb, 3.0)
        detail = rgb - soft
        rgb = np.clip(rgb + detail * params['clarity'], 0, 1)

    # 10. Saturation boost
    gray = (0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2])[:,:,None]
    rgb = np.clip(gray + (rgb - gray) * params['sat_boost'], 0, 1)

    # 11. Brightness / exposure
    rgb = np.clip(rgb * params['brightness'], 0, 1)

    # 12. Specular highlight (center)
    if params['specular_strength'] > 0.01:
        spec_cx = cx + params['spec_cx'] * cx
        spec_cy = cy + params['spec_cy'] * cy
        spec_dist = np.sqrt(((yy - spec_cy) / cy)**2 + ((xx - spec_cx) / cx)**2)
        spec = np.exp(-spec_dist**2 / (2 * max(params['specular_size'], 0.001)**2))
        rgb = np.clip(rgb + spec[:,:,None] * params['specular_strength'], 0, 1)

    # 13. Edge darkening
    if params['edge_darkening'] > 0.01:
        edge_mask = np.clip(dist ** 3 * params['edge_darkening'], 0, 0.5)[:,:,None]
        rgb = rgb * (1 - edge_mask)

    # 14. Gel thickness gradient
    if abs(params['gel_gradient_strength']) > 0.01:
        angle_rad = params['gel_gradient_angle'] * np.pi / 180.0
        grad = ((xx - cx) / cx * np.cos(angle_rad) +
                (yy - cy) / cy * np.sin(angle_rad))
        grad_mask = (1.0 + grad * params['gel_gradient_strength'] * 0.3)[:,:,None]
        rgb = np.clip(rgb * grad_mask, 0, 1)

    # 15. Vignette
    mask = np.clip(1.0 - dist**2 * params['vignette'], 0, 1)[:,:,None]
    rgb = np.clip(rgb * mask, 0, 1)

    # 16. White balance (R/G/B channel multipliers)
    rgb[:,:,0] = np.clip(rgb[:,:,0] * params['wb_r'], 0, 1)
    rgb[:,:,1] = np.clip(rgb[:,:,1] * params['wb_g'], 0, 1)
    rgb[:,:,2] = np.clip(rgb[:,:,2] * params['wb_b'], 0, 1)

    return np.clip(rgb, 0, 1)


def score(no_gel, gel, params):
    """LAB-space MSE + SSIM score."""
    result = apply_gel_fx(no_gel, params)
    result_u8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    gel_u8 = (np.clip(gel, 0, 1) * 255).astype(np.uint8)

    # Convert to LAB for perceptually uniform comparison
    result_lab = cv2.cvtColor(result_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    gel_lab = cv2.cvtColor(gel_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    # LAB MSE (normalized by channel range: L=[0,100], A=[-128,127], B=[-128,127])
    lab_mse = (((result_lab[:,:,0] - gel_lab[:,:,0]) / 100)**2).mean() + \
              (((result_lab[:,:,1] - gel_lab[:,:,1]) / 255)**2).mean() + \
              (((result_lab[:,:,2] - gel_lab[:,:,2]) / 255)**2).mean()
    lab_mse /= 3

    # SSIM on RGB
    s = ssim(gel_u8, result_u8, channel_axis=2)

    return 0.5 * s + 0.5 * (1.0 - min(lab_mse * 50, 1.0))


# Parameter search space
PBOUNDS = {
    'barrel_k1':          (-0.1, 0.15),
    'refraction_k2':      (-0.05, 0.1),
    'blur_sigma':         (0.0, 3.0),
    'blue_shift':         (-0.2, 0.5),
    'tint_r':             (0.3, 1.0),
    'tint_g':             (0.3, 1.0),
    'tint_b':             (0.3, 1.0),
    'tint_strength':      (0.0, 0.5),
    'haze_opacity':       (0.0, 0.3),
    'contrast_boost':     (0.8, 2.5),
    'gamma':              (0.5, 1.5),
    'clarity':            (0.0, 5.0),
    'sat_boost':          (0.5, 2.5),
    'brightness':         (0.7, 1.5),
    'specular_strength':  (0.0, 0.5),
    'specular_size':      (0.02, 0.3),
    'spec_cx':            (-0.3, 0.3),
    'spec_cy':            (-0.3, 0.3),
    'edge_darkening':     (0.0, 1.5),
    'gel_gradient_strength': (0.0, 0.8),
    'gel_gradient_angle': (0.0, 360.0),
    'vignette':           (0.0, 0.8),
    'wb_r':               (0.7, 1.3),
    'wb_g':               (0.7, 1.3),
    'wb_b':               (0.7, 1.5),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--init', type=int, default=30)
    args = parser.parse_args()

    print('Loading images...')
    no_gel, gel = load_images()
    print(f'  no_gel: {no_gel.shape}, gel: {gel.shape}')

    best_score = -1
    best_params = None
    iteration = [0]

    def objective(**kwargs):
        s = score(no_gel, gel, kwargs)
        nonlocal best_score, best_params
        iteration[0] += 1
        if s > best_score:
            best_score = s
            best_params = dict(kwargs)
            print(f'  [{iteration[0]:4d}] NEW BEST score={s:.4f}')
        elif iteration[0] % 20 == 0:
            print(f'  [{iteration[0]:4d}] score={s:.4f}  best={best_score:.4f}')
        return s

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=PBOUNDS,
        random_state=42,
        verbose=0,
    )

    print(f'Running BO: {args.init} init + {args.n_iter} iterations...')
    optimizer.maximize(init_points=args.init, n_iter=args.n_iter)

    best = optimizer.max
    best_params = best['params']
    print(f'\nBest score: {best["target"]:.4f}')
    print(f'Best params:')
    for k, v in sorted(best_params.items()):
        print(f'  {k}: {v:.4f}')

    # Save best params
    out_path = OUT_DIR / 'best_gel_fx_params.json'
    with open(out_path, 'w') as f:
        json.dump({k: round(v, 4) for k, v in best_params.items()}, f, indent=2)
    print(f'\nSaved to {out_path}')

    # Save comparison image
    result = apply_gel_fx(no_gel, best_params)
    comparison = np.hstack([no_gel, result, gel])
    comparison = (np.clip(comparison, 0, 1) * 255).astype(np.uint8)
    comp_path = OUT_DIR / 'gel_fx_comparison.png'
    cv2.imwrite(str(comp_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f'Saved comparison to {comp_path}')

    # Save result alone
    result_path = OUT_DIR / 'gel_fx_result.png'
    cv2.imwrite(str(result_path), cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f'Saved result to {result_path}')


if __name__ == '__main__':
    main()
