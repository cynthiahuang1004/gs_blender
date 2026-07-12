"""
bo_gel_effect_v2.py — BO with v6 params as starting point
"""
import numpy as np, cv2, json, argparse
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from bayes_opt import BayesianOptimization
from bo_gel_effect import load_images, apply_gel_fx, score

OUT_DIR = Path(__file__).parent.parent / 'bo_results' / 'gel_fx'

# v6 baseline params
BASELINE = {
    'barrel_k1': -0.02, 'refraction_k2': 0.02, 'blur_sigma': 1.5,
    'blue_shift': 0.25, 'tint_r': 0.75, 'tint_g': 0.70, 'tint_b': 0.15,
    'tint_strength': 0.25, 'haze_opacity': 0.3, 'contrast_boost': 1.5,
    'gamma': 1.1, 'clarity': 1.5, 'sat_boost': 0.5, 'brightness': 1.0,
    'specular_strength': 0.5, 'specular_size': 0.02, 'spec_cx': 0.3,
    'spec_cy': 0.27, 'edge_darkening': 0.0, 'gel_gradient_strength': 0.8,
    'gel_gradient_angle': 99.5, 'vignette': 0.0,
    'wb_r': 1.05, 'wb_g': 1.05, 'wb_b': 1.2,
}

# Tighter search range around baseline
PBOUNDS = {}
for k, v in BASELINE.items():
    if k == 'gel_gradient_angle':
        PBOUNDS[k] = (0.0, 360.0)
    else:
        margin = max(abs(v) * 0.5, 0.1)
        lo = max(v - margin, 0.0) if k not in ['barrel_k1', 'refraction_k2', 'blue_shift'] else v - margin
        hi = v + margin
        PBOUNDS[k] = (round(lo, 4), round(hi, 4))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--init', type=int, default=20)
    args = parser.parse_args()

    no_gel, gel = load_images()
    print(f'Images: {no_gel.shape}')
    best_score = -1
    best_params = None
    it = [0]

    def objective(**kwargs):
        nonlocal best_score, best_params
        s = score(no_gel, gel, kwargs)
        it[0] += 1
        if s > best_score:
            best_score = s
            best_params = dict(kwargs)
            print(f'  [{it[0]:4d}] NEW BEST score={s:.4f}')
        elif it[0] % 20 == 0:
            print(f'  [{it[0]:4d}] score={s:.4f}  best={best_score:.4f}')
        return s

    optimizer = BayesianOptimization(f=objective, pbounds=PBOUNDS, random_state=42, verbose=0)
    # Probe baseline first
    optimizer.probe(params=BASELINE, lazy=True)
    print(f'BO: {args.init} init + {args.n_iter} iter (baseline probed)')
    optimizer.maximize(init_points=args.init, n_iter=args.n_iter)

    best = optimizer.max
    bp = best['params']
    print(f'\nBest score: {best["target"]:.4f}')
    for k, v in sorted(bp.items()):
        print(f'  {k}: {v:.4f}')

    with open(OUT_DIR / 'best_gel_fx_params.json', 'w') as f:
        json.dump({k: round(v, 4) for k, v in bp.items()}, f, indent=2)

    result = apply_gel_fx(no_gel, bp)
    comp = np.hstack([no_gel, result, gel])
    cv2.imwrite(str(OUT_DIR / 'gel_fx_comparison.png'),
                cv2.cvtColor((np.clip(comp, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT_DIR / 'gel_fx_result.png'),
                cv2.cvtColor((np.clip(result, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print('Saved results')

if __name__ == '__main__':
    main()
