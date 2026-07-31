"""
tune_rgb.py
===========
Manual tuning helper for the RGB renders (high camera + contact camera).
Blender renders the RAW image; all post FX (the _gel_fx chain from
scripting.py) are applied here with cv2 for fast iteration.

Usage:
    python tune_rgb.py                                # defaults, pattern_33
    python tune_rgb.py params.json                    # custom params
    python tune_rgb.py params.json pattern_35         # custom object
    python tune_rgb.py params.json pattern_35 --post-only   # reuse cached raw render

Scene params (need Blender re-render):
    world_strength, obj_roughness, plat_r/g/b, plat_roughness, plat_metallic,
    rgb_fov, obj_location, obj_rotation_override

Post FX params (--post-only is enough):
    barrel_k1, refraction_k2, blur_sigma, tint_r/g/b, tint_strength,
    tint_cx/cy, haze_opacity, blue_shift, contrast_boost, gamma, clarity,
    sat_boost, specular_strength/size, edge_darkening,
    gel_gradient_strength/angle, vignette

Output:
    calibration/tune_output/tuned_rgb.png          (high camera)
    calibration/tune_output/tuned_rgb_contact.png  (contact camera)
"""

import os, sys, json, subprocess, shutil, tempfile
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
BLENDER_PATH = '/home/shared/blender-4.2.0-linux-x64/blender'
MESHES_DIR = os.path.join(ROOT_DIR, 'meshes')
RENDERS_DIR = os.path.join(ROOT_DIR, 'renders')
OUT_DIR = os.path.join(SCRIPT_DIR, 'tune_output')
os.makedirs(OUT_DIR, exist_ok=True)

RENDER_BASE = os.path.join(tempfile.gettempdir(), 'gs_tune_rgb_render')

DEFAULTS = {
    # Scene
    'world_strength': 2.37,
    'obj_roughness': 0.25,
    'plat_r': 0.226, 'plat_g': 0.35, 'plat_b': 0.40,
    'plat_roughness': 0.35, 'plat_metallic': 0.582,
    'rgb_fov': 40.0,
    # Object placement (default: pattern_33 real sample 0050 pose)
    'obj_location': [0.02997, -0.04357, -0.001853],
    'obj_rotation_override': [0.0, 0.0, 1.367562362997865],
    'press_depth': 0.0002,
    # Post FX (defaults = bo_results/rgb/best_rgb_params.json values)
    'barrel_k1': 0.048,
    'refraction_k2': 0.0,
    'global_blur_sigma': 0.0,
    'blur_sigma': 1.607,
    'blur_falloff': 1.5,
    'tint_r': 0.85, 'tint_g': 0.70, 'tint_b': 0.25,
    'tint_strength': 0.25,
    'tint_cx': -0.15, 'tint_cy': 0.05,
    'haze_opacity': 0.10,
    'blue_shift': 0.0,
    'contrast_boost': 1.0,
    'gamma': 1.0,
    'clarity': 0.0,
    'sat_boost': 1.2,
    'specular_strength': 0.0,
    'specular_size': 0.05,
    'edge_darkening': 0.0,
    'gel_gradient_strength': 0.0,
    'gel_gradient_angle': 0.0,
    'vignette': 0.006,
    'brightness': 1.0,
    'spec_cx': 0.0, 'spec_cy': 0.0,
    'wb_r': 1.0, 'wb_g': 1.0, 'wb_b': 1.0,
}


def _barrel(img, k1):
    if abs(k1) < 1e-4:
        return img
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx - cx) / cx
    yn = (yy - cy) / cy
    fac = 1.0 + k1 * (xn**2 + yn**2)
    xs = np.clip(xn * fac * cx + cx, 0, W - 1).astype(np.float32)
    ys = np.clip(yn * fac * cy + cy, 0, H - 1).astype(np.float32)
    return cv2.remap(img, xs, ys, cv2.INTER_LINEAR)


def apply_gel_fx(img_u8, p):
    """Port of scripting.py _gel_fx (same order of effects). img_u8: BGR uint8."""
    rgb = img_u8[:, :, ::-1].astype(np.float32) / 255.0  # to RGB float
    H, W = rgb.shape[:2]

    # 1. Barrel + 2. refraction
    rgb = _barrel(rgb, p['barrel_k1'])
    rgb = _barrel(rgb, p['refraction_k2'])

    # 2.5 Global gaussian blur (uniform, camera softness)
    if p.get('global_blur_sigma', 0.0) > 0.1:
        rgb = cv2.GaussianBlur(rgb, (0, 0), p['global_blur_sigma'])

    # 3. Radial blur (center sharp, edge blurry; dist from tint center)
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    tcx = cx + p['tint_cx'] * cx
    tcy = cy + p['tint_cy'] * cy
    dist = np.sqrt(((yy - tcy) / cy)**2 + ((xx - tcx) / cx)**2)
    if p['blur_sigma'] > 0.3:
        blurred = cv2.GaussianBlur(rgb, (0, 0), p['blur_sigma'])
        weight = np.clip(dist ** p.get('blur_falloff', 1.5), 0, 1)[:, :, None]
        rgb = rgb * (1 - weight) + blurred * weight

    # 4. Tint
    tint = np.array([[[p['tint_r'], p['tint_g'], p['tint_b']]]], dtype=np.float32)
    tint_w = np.clip(dist ** 2 * p['tint_strength'], 0, 1)[:, :, None]
    rgb = rgb * (1 - tint_w) + tint * tint_w

    # 5. Haze
    haze_w = np.clip(dist ** 1.5 * p['haze_opacity'], 0, 1)[:, :, None]
    rgb = rgb * (1 - haze_w) + tint * 0.7 * haze_w

    # 6. Blue shift
    if abs(p['blue_shift']) > 1e-4:
        rgb[:, :, 0] = np.clip(rgb[:, :, 0] - p['blue_shift'] * 0.3, 0, 1)
        rgb[:, :, 1] = np.clip(rgb[:, :, 1] - p['blue_shift'] * 0.1, 0, 1)
        rgb[:, :, 2] = np.clip(rgb[:, :, 2] + p['blue_shift'] * 0.2, 0, 1)

    # 7. Contrast
    if abs(p['contrast_boost'] - 1.0) > 1e-4:
        rgb = np.clip((rgb - 0.5) * p['contrast_boost'] + 0.5, 0, 1)

    # 8. Gamma
    if abs(p['gamma'] - 1.0) > 1e-4:
        rgb = np.clip(np.power(np.maximum(rgb, 0), 1.0 / p['gamma']), 0, 1)

    # 9. Clarity
    if abs(p['clarity']) > 1e-4:
        soft = cv2.GaussianBlur(rgb, (0, 0), 3.0)
        rgb = np.clip(rgb + (rgb - soft) * p['clarity'], 0, 1)

    # 10. Saturation boost (center-weighted)
    if abs(p['sat_boost'] - 1.0) > 0.01:
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])[:, :, None]
        center_w = np.clip(1.0 - dist, 0, 1)[:, :, None] ** 2
        local_fac = 1.0 + (p['sat_boost'] - 1.0) * center_w
        rgb = np.clip(gray + (rgb - gray) * local_fac, 0, 1)

    # 10.5 Brightness / exposure
    if abs(p.get('brightness', 1.0) - 1.0) > 1e-4:
        rgb = np.clip(rgb * p['brightness'], 0, 1)

    # 11. Specular (with optional center offset)
    if p['specular_strength'] > 1e-4:
        scx = cx + p.get('spec_cx', 0.0) * cx
        scy = cy + p.get('spec_cy', 0.0) * cy
        spec_dist = np.sqrt(((yy - scy) / cy)**2 + ((xx - scx) / cx)**2)
        spec = np.exp(-spec_dist**2 / (2 * max(p['specular_size'], 1e-3)**2))
        rgb = np.clip(rgb + spec[:, :, None] * p['specular_strength'], 0, 1)

    # 12. Edge darkening
    if abs(p['edge_darkening']) > 1e-4:
        edge_mask = np.clip(dist ** 3 * p['edge_darkening'], 0, 0.5)[:, :, None]
        rgb = rgb * (1 - edge_mask)

    # 13. Gel thickness gradient
    if abs(p['gel_gradient_strength']) > 1e-4:
        ang = p['gel_gradient_angle'] * np.pi / 180.0
        grad = ((xx - cx) / cx * np.cos(ang) + (yy - cy) / cy * np.sin(ang))
        rgb = np.clip(rgb * (1.0 + grad * p['gel_gradient_strength'] * 0.3)[:, :, None], 0, 1)

    # 14. Vignette
    mask = np.clip(1.0 - dist**2 * p['vignette'], 0, 1)[:, :, None]
    rgb = np.clip(rgb * mask, 0, 1)

    # 15. White balance
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] * p.get('wb_r', 1.0), 0, 1)
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] * p.get('wb_g', 1.0), 0, 1)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2] * p.get('wb_b', 1.0), 0, 1)

    return (rgb[:, :, ::-1] * 255).astype(np.uint8)  # back to BGR


def render_blender(params, obj_name):
    mesh_path = os.path.join(MESHES_DIR, f'{obj_name}.obj')
    session_path = os.path.join(RENDERS_DIR, obj_name, 'session_000', 'session.json')
    with open(session_path) as f:
        sess = json.load(f)

    params_tmp = os.path.join(tempfile.gettempdir(), 'gs_tune_rgb_params.json')
    with open(params_tmp, 'w') as f:
        json.dump(params, f)
    blend_copy = os.path.join(tempfile.gettempdir(), 'gs_tune_rgb.blend')
    shutil.copy(os.path.join(ROOT_DIR, 'gelsight_sampler.blend'), blend_copy)

    env = os.environ.copy()
    env['GELSIGHT_FIXED_PARAMS'] = params_tmp
    env['GELSIGHT_BG_RENDER'] = RENDER_BASE
    env['TUNE_OBJ_FILE'] = mesh_path
    env['TUNE_OBJ_ROTATION'] = json.dumps(params['obj_rotation_override'])
    env['TUNE_OBJ_SCALE'] = str(sess['fixed_scale'])
    env['TUNE_OBJ_LOCATION'] = json.dumps(params['obj_location'])

    print('Rendering with Blender (2 cameras)...')
    proc = subprocess.run(
        [BLENDER_PATH, '--background', blend_copy,
         '--python', os.path.join(SCRIPT_DIR, 'scripting_tune_rgb.py')],
        cwd=SCRIPT_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    for line in proc.stdout.decode('utf-8', errors='replace').split('\n'):
        if '[tune]' in line:
            print(f'  {line.strip()}')
    return proc.returncode == 0


def main():
    params = dict(DEFAULTS)
    obj_name = 'pattern_33'
    post_only = '--post-only' in sys.argv
    argv = [a for a in sys.argv[1:] if a != '--post-only']

    if len(argv) > 0:
        with open(argv[0]) as f:
            params.update(json.load(f))
        print(f'Loaded overrides from {argv[0]}')
    if len(argv) > 1:
        obj_name = argv[1]

    print(f'Object: {obj_name}, post_only={post_only}')

    if not post_only:
        if not render_blender(params, obj_name):
            print('Blender render failed!')
            return

    for tag in ['high', 'contact']:
        raw_path = f'{RENDER_BASE}_{tag}.png'
        if not os.path.exists(raw_path):
            raw_path = f'{RENDER_BASE}_{tag}0001.png'
        img = cv2.imread(raw_path)
        if img is None:
            print(f'Missing raw render: {raw_path} (run without --post-only first)')
            continue
        out = apply_gel_fx(img, params)
        suffix = '' if tag == 'high' else '_contact'
        out_path = os.path.join(OUT_DIR, f'tuned_rgb{suffix}.png')
        cv2.imwrite(out_path, out)
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
