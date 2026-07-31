"""
tune_tactile.py
===============
Manual tuning helper for tactile contact appearance.
Renders a tactile image with an object pressed into the gel.

Usage:
    python tune_tactile.py                          # defaults + default object
    python tune_tactile.py params.json              # load from JSON
    python tune_tactile.py params.json button       # specific object

Parameters:

  Contact appearance:
    gel_roughness       (0=mirror, 1=fully diffuse, default 0.4455)
    gel_fac             (0=transparent, 1=fully reflective, default 0.2971)
    smoothness          (corrective smooth iterations, default 30)
    dark_base_gain      (background brightness, 1.0=white, 0.0=black, default 0.9)
    contact_r/g/b       (contact area color, default 60/255 each = dark gray)
    contact_z_thresh    (Z below this = fully dark contact, default -0.001)
    bg_z_thresh         (Z above this = fully bright background, default 0.0005)
    press_depth         (how deep to press in meters, default 0.001 = 1mm)

  Emitter params (inherited from base tactile tuning):
    top_str/r/g/b, bot_str/r/g/b, left_str/r/g/b,
    right_str/r/g/b, lg_str/r/g/b, rg_str/r/g/b

  Fixed sensor params:
    scale_y, light_z, rot_z, fov, length

  Post-processing:
    saturation, brightness, contrast

Output:
    calibration/tune_output/tuned_contact.png
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

DEFAULTS = {
    # Emitter params (from manual tuning)
    'top_str': 54.06, 'top_r': 0.05, 'top_g': 0.65, 'top_b': 0.02,
    'bot_str': 50.0,  'bot_r': 0.27, 'bot_g': 0.21, 'bot_b': 0.85,
    'left_str': 18.0, 'left_r': 0.70, 'left_g': 0.30, 'left_b': 0.02,
    'right_str': 84.83, 'right_r': 0.92, 'right_g': 0.08, 'right_b': 0.05,
    'lg_str': 28.38,  'lg_r': 0.05,  'lg_g': 0.65,  'lg_b': 0.09,
    'rg_str': 176.33, 'rg_r': 0.03,  'rg_g': 0.71,  'rg_b': 0.04,
    # Fixed sensor
    'scale_y': 0.4918,
    'light_z': -0.004139,
    'rot_z': -3.14159,
    'fov': 60.0,
    'length': 0.008751,
    # Contact appearance
    'gel_roughness': 0.4455,
    'gel_fac': 0.2971,
    'smoothness': 30,
    'dark_base_gain': 0.9,
    'contact_r': 60.0 / 255.0,
    'contact_g': 60.0 / 255.0,
    'contact_b': 60.0 / 255.0,
    'contact_z_thresh': -0.001,
    'bg_z_thresh': 0.0005,
    'press_depth': 0.001,
    # Post-processing
    'saturation': 1.3,
    'brightness': 1.0,
    'contrast': 1.3,
}

# Object presets: rotation and scale per object type
OBJ_PRESETS = {}


def _load_obj_preset(obj_name):
    """Load rotation/scale from session_000/session.json if available."""
    session_path = os.path.join(RENDERS_DIR, obj_name, 'session_000', 'session.json')
    if os.path.exists(session_path):
        with open(session_path) as f:
            s = json.load(f)
        return {
            'rotation': s['base_rotation'],
            'scale': s.get('fixed_scale', 1.0),
            'z_anchor': s.get('z_anchor', 0),
        }
    return {'rotation': [0, 0, 0], 'scale': 1.0, 'z_anchor': 0}


def render(params, obj_name='button'):
    mesh_path = os.path.join(MESHES_DIR, f'{obj_name}.obj')
    if not os.path.exists(mesh_path):
        print(f'ERROR: mesh not found: {mesh_path}')
        return None

    preset = _load_obj_preset(obj_name)

    params_tmp = os.path.join(tempfile.gettempdir(), 'gs_tune_contact_params.json')
    render_base = os.path.join(tempfile.gettempdir(), 'gs_tune_contact_render')
    render_png = render_base + '.png'
    blend_copy = os.path.join(tempfile.gettempdir(), 'gs_tune_contact.blend')

    with open(params_tmp, 'w') as f:
        json.dump(params, f)

    if os.path.exists(render_png):
        os.remove(render_png)

    shutil.copy(os.path.join(ROOT_DIR, 'gelsight_sampler.blend'), blend_copy)

    env = os.environ.copy()
    env['GELSIGHT_FIXED_PARAMS'] = params_tmp
    env['GELSIGHT_BG_RENDER'] = render_base
    env['TUNE_OBJ_FILE'] = mesh_path
    env['TUNE_PRESS_DEPTH'] = str(params.get('press_depth', 0.001))
    rotation = params.get('obj_rotation_override', preset['rotation'])
    env['TUNE_OBJ_ROTATION'] = json.dumps(rotation)
    env['TUNE_OBJ_SCALE'] = str(preset['scale'])
    env['TUNE_Z_ANCHOR'] = str(preset.get('z_anchor', 0))
    env['TUNE_OBJ_OFFSET'] = json.dumps(params.get('offset', [0, 0]))
    if params.get('obj_location') is not None:
        env['TUNE_OBJ_LOCATION'] = json.dumps(params['obj_location'])

    print(f'Rendering {obj_name} (press={params.get("press_depth", 0.001)*1000:.1f}mm)...')
    proc = subprocess.run(
        [BLENDER_PATH, '--background', blend_copy,
         '--python', os.path.join(SCRIPT_DIR, 'scripting_tune_tactile.py')],
        cwd=SCRIPT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    for line in proc.stdout.decode('utf-8', errors='replace').split('\n'):
        if '[tune]' in line:
            print(f'  {line.strip()}')

    if proc.returncode != 0:
        print(f'Blender failed (rc={proc.returncode})')
        stderr = proc.stderr.decode('utf-8', errors='replace')[-500:]
        print(stderr)
        return None

    # Find output and apply post-processing
    for candidate in (render_png, render_base + '0001.png'):
        if os.path.exists(candidate):
            out_path = os.path.join(OUT_DIR, 'tuned_contact.png')
            img = cv2.imread(candidate)

            if abs(params.get('saturation', 1.0) - 1.0) > 0.01:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation'], 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if abs(params.get('brightness', 1.0) - 1.0) > 0.01:
                img = np.clip(img.astype(np.float32) * params['brightness'], 0, 255).astype(np.uint8)
            if abs(params.get('contrast', 1.0) - 1.0) > 0.01:
                img = np.clip((img.astype(np.float32) - 127.5) * params['contrast'] + 127.5, 0, 255).astype(np.uint8)
            if params.get('sharpen', 0.0) > 0.01:
                soft = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
                img = np.clip(img.astype(np.float32) + (img.astype(np.float32) - soft) * params['sharpen'], 0, 255).astype(np.uint8)
            if params.get('haze', 0.0) > 0.01:
                # milky gel scattering: blend with blurred version + slight white lift
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
    obj_name = 'button'

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        with open(json_path) as f:
            overrides = json.load(f)
        params.update(overrides)
        print(f'Loaded overrides from {json_path}')

    if len(sys.argv) > 2:
        obj_name = sys.argv[2]

    print(f'\nObject: {obj_name}')
    print('Current parameters:')
    print('  === Contact Appearance ===')
    for k in ['gel_roughness', 'gel_fac', 'smoothness', 'dark_base_gain',
              'contact_r', 'contact_g', 'contact_b',
              'contact_z_thresh', 'bg_z_thresh', 'press_depth']:
        print(f'    {k:20s}: {params[k]}')
    print('  === Emitters ===')
    for name, label in [('top', 'Top (image:左下)'), ('bot', 'Bottom (image:右上)'),
                        ('left', 'Left (image:左上)'), ('right', 'Right (image:右下)'),
                        ('lg', 'LeftGreen (image:右)'), ('rg', 'RightGreen (image:左)')]:
        s, r, g, b = params[f'{name}_str'], params[f'{name}_r'], params[f'{name}_g'], params[f'{name}_b']
        print(f'    {label:30s}  str={s:6.1f}  R={r:.2f}  G={g:.2f}  B={b:.2f}')
    print('  === Post-processing ===')
    for k in ['saturation', 'brightness', 'contrast']:
        print(f'    {k:16s}: {params[k]}')

    render(params, obj_name)


if __name__ == '__main__':
    main()
