"""
tune_base_tactile.py
====================
Manual tuning helper for base tactile image.
Renders one flat-gel background with given emitter parameters.

Usage:
    python tune_base_tactile.py                     # use current defaults
    python tune_base_tactile.py params.json         # load from JSON file

Parameters (24 emitter params + 7 fixed params):

  Emitter params (6 emitters × 4 = 24):
    top_str, top_r, top_g, top_b           # TopEmittor (image: 左下)
    bot_str, bot_r, bot_g, bot_b           # BottomEmittor (image: 右上)
    left_str, left_r, left_g, left_b      # LeftEmittor (image: 左上)
    right_str, right_r, right_g, right_b  # RightEmittor (image: 右下)
    lg_str, lg_r, lg_g, lg_b             # LeftGreenEmittor (image: 右)
    rg_str, rg_r, rg_g, rg_b             # RightGreenEmittor (image: 左)

  Fixed params (can also be tuned):
    scale_y        (light array Y scale, default 0.4918)
    light_z        (light Z height, default -0.004139)
    rot_z          (light array rotation, default -3.14159)
    fov            (camera FOV, default 60.0)
    length         (camera half-width, default 0.008751)
    gel_roughness  (gel surface roughness, default 0.4455)
    gel_fac        (gel mix factor, default 0.2971)
    smoothness     (corrective smooth iterations, default 30)

Output:
    calibration/tune_output/tuned.png
"""

import os, sys, json, subprocess, shutil, tempfile
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
BLENDER_PATH = '/home/shared/blender-4.2.0-linux-x64/blender'
OUT_DIR = os.path.join(SCRIPT_DIR, 'tune_output')
os.makedirs(OUT_DIR, exist_ok=True)

DEFAULTS = {
    # Emitter params
    'top_str': 80.0,  'top_r': 0.3,  'top_g': 0.65, 'top_b': 0.3,
    'bot_str': 40.0,  'bot_r': 0.1,  'bot_g': 0.5,  'bot_b': 0.9,
    'left_str': 30.0, 'left_r': 0.9, 'left_g': 0.05,'left_b': 0.05,
    'right_str':120.0,'right_r':1.0, 'right_g':0.0, 'right_b':0.0,
    'lg_str': 60.0,   'lg_r': 0.3,   'lg_g': 0.65,  'lg_b': 0.3,
    'rg_str': 120.0,  'rg_r': 0.3,   'rg_g': 0.7,   'rg_b': 0.3,
    # Fixed params
    'scale_y': 0.4918,
    'light_z': -0.004139,
    'rot_z': -3.14159,
    'fov': 60.0,
    'length': 0.008751,
    'gel_roughness': 0.4455,
    'gel_fac': 0.2971,
    'smoothness': 30,
    # Post-processing
    'saturation': 1.0,   # 1.0 = no change, >1 = more saturated, <1 = less
    'brightness': 1.0,   # 1.0 = no change
    'contrast': 1.0,     # 1.0 = no change
}


def render(params):
    params_tmp = os.path.join(tempfile.gettempdir(), 'gs_tune_params.json')
    render_base = os.path.join(tempfile.gettempdir(), 'gs_tune_render')
    render_png = render_base + '.png'
    blend_copy = os.path.join(tempfile.gettempdir(), 'gs_tune.blend')

    with open(params_tmp, 'w') as f:
        json.dump(params, f)

    if os.path.exists(render_png):
        os.remove(render_png)

    shutil.copy(os.path.join(ROOT_DIR, 'gelsight_sampler.blend'), blend_copy)

    env = os.environ.copy()
    env['GELSIGHT_FIXED_PARAMS'] = params_tmp
    env['GELSIGHT_BG_RENDER'] = render_base

    print('Rendering with Blender...')
    proc = subprocess.run(
        [BLENDER_PATH, '--background', blend_copy,
         '--python', os.path.join(SCRIPT_DIR, 'scripting_bo.py')],
        cwd=SCRIPT_DIR,
        env=env,
        capture_output=True,
        timeout=300,
    )

    if proc.returncode != 0:
        print(f'Blender failed (rc={proc.returncode})')
        stderr = proc.stderr.decode('utf-8', errors='replace')[-500:]
        print(stderr)
        return None

    # Find output and apply post-processing
    for candidate in (render_png, render_base + '0001.png'):
        if os.path.exists(candidate):
            out_path = os.path.join(OUT_DIR, 'tuned.png')
            img = cv2.imread(candidate)
            # Saturation
            if abs(params.get('saturation', 1.0) - 1.0) > 0.01:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation'], 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            # Brightness
            if abs(params.get('brightness', 1.0) - 1.0) > 0.01:
                img = np.clip(img.astype(np.float32) * params['brightness'], 0, 255).astype(np.uint8)
            # Contrast
            if abs(params.get('contrast', 1.0) - 1.0) > 0.01:
                img = np.clip((img.astype(np.float32) - 127.5) * params['contrast'] + 127.5, 0, 255).astype(np.uint8)
            # Haze (milky gel scattering)
            if params.get('haze', 0.0) > 0.01:
                a = params['haze']
                f = img.astype(np.float32)
                glow = cv2.GaussianBlur(f, (0, 0), 8.0)
                f = f * (1 - a) + glow * a + a * 30.0
                img = np.clip(f, 0, 255).astype(np.uint8)
            cv2.imwrite(out_path, img)
            print(f'Saved: {out_path}')
            return out_path

    print('No output file found!')
    return None


def main():
    params = dict(DEFAULTS)

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        with open(json_path) as f:
            overrides = json.load(f)
        params.update(overrides)
        print(f'Loaded overrides from {json_path}')

    print('Current parameters:')
    print('  === Emitters ===')
    for name, label in [('top', 'Top (image:左下)'), ('bot', 'Bottom (image:右上)'),
                        ('left', 'Left (image:左上)'), ('right', 'Right (image:右下)'),
                        ('lg', 'LeftGreen (image:右)'), ('rg', 'RightGreen (image:左)')]:
        s, r, g, b = params[f'{name}_str'], params[f'{name}_r'], params[f'{name}_g'], params[f'{name}_b']
        print(f'    {label:30s}  str={s:6.1f}  R={r:.2f}  G={g:.2f}  B={b:.2f}')
    print('  === Fixed ===')
    for k in ['scale_y', 'light_z', 'rot_z', 'fov', 'length', 'gel_roughness', 'gel_fac', 'smoothness']:
        print(f'    {k:16s}: {params[k]}')
    print('  === Post-processing ===')
    for k in ['saturation', 'brightness', 'contrast']:
        print(f'    {k:16s}: {params[k]}')

    render(params)


if __name__ == '__main__':
    main()
