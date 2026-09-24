"""Re-render existing simulated sessions with a RANDOM sensor design per session (domain-randomisation
baseline: 'do not calibrate, randomise'). Uses the same poses as renders_v3 (raw_data/XXXX_pose.json), so
depth/normal/pose labels are reused unchanged; only samples/ (the tactile image) and calibration/0000.png
(that session's background) are written.

Run inside Blender from the gs_blender root (GELSIGHT_FIXED_PARAMS must NOT point to an existing file):
    blender -t 8 --background gelsight_sampler.blend --python render_random_sensor.py -- \
        --obj pattern_33 --sessions 0 1 2 --out /path/renders_v3_dr [--seed 0] [--every 1]
Output: <out>/<obj>/session_XXX/sensor_0000/{samples,calibration}/...; use as sim root with tactile_subdir samples.
"""
import argparse, json, os, random, sys
_dir = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _dir)
os.environ['GELSIGHT_FIXED_PARAMS'] = '/nonexistent'      # -> create_sensor.randomize() draws random params
import bpy  # noqa: E402
import scripting as S  # noqa: E402
SIM_ROOT = os.environ.get('SIM_ROOT', os.path.join(_dir, 'renders_v3'))


def enable_gpu():
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'; prefs.get_devices()
        for d in prefs.devices: d.use = d.type == 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'
    except Exception as e:  # noqa
        print('[dr] GPU enable failed:', e)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(); ap.add_argument('--obj', required=True); ap.add_argument('--sessions', nargs='+', type=int, required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--every', type=int, default=3)
    a = ap.parse_args(argv)
    if not os.environ.get('GELSIGHT_FORCE_CPU'): enable_gpu()
    mesh_dir = os.path.join(_dir, 'meshes'); obj_file = a.obj + '.obj'
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=os.path.join(mesh_dir, obj_file), directory=mesh_dir, files=[{"name": obj_file}])
    new = [o.name for o in bpy.data.objects if o.name not in before]; obj = bpy.data.objects[new[0]]; obj.hide_render = True
    for si in a.sessions:
        sess = f'session_{si:03d}'; unit_in = f'{SIM_ROOT}/{a.obj}/{sess}/sensor_0000'; unit_out = f'{a.out}/{a.obj}/{sess}/sensor_0000'
        os.makedirs(f'{unit_out}/samples', exist_ok=True); os.makedirs(f'{unit_out}/calibration', exist_ok=True)
        random.seed(a.seed * 1000 + si * 37 + hash(a.obj) % 1000)      # one random sensor per (obj, session)
        sensor = S.create_sensor(); sensor.randomize(); sensor.apply()
        # background of this random sensor
        S.move_object('IndenterSurface', (0, 0, -1), (0, 0, 0)) if 'IndenterSurface' in bpy.data.objects else None
        obj.location = (0, 0, 1.0); bpy.context.scene.frame_set(0)
        bpy.context.scene.render.filepath = f'{unit_out}/calibration/0000'; bpy.ops.render.render(write_still=True)
        S._tactile_post_fx(f'{unit_out}/calibration/0000.png')
        poses = sorted(f for f in os.listdir(f'{unit_in}/raw_data') if f.endswith('_pose.json'))[::a.every]
        done = 0
        for pf in poses:
            tag = pf[:4]; out_png = f'{unit_out}/samples/{tag}.png'
            if os.path.exists(out_png) and os.path.getsize(out_png) > 1000: continue
            p = json.load(open(f'{unit_in}/raw_data/{pf}'))
            obj.location = p['location']; obj.rotation_euler = p['rotation_euler']; obj.scale = p['scale']; obj.hide_render = True
            bpy.data.objects['GelSurface'].modifiers['Shrinkwrap'].target = obj; bpy.context.scene.frame_set(0)
            bpy.context.scene.render.filepath = out_png[:-4]; bpy.ops.render.render(write_still=True)
            gt = f'{unit_in}/raw_data/{tag}_gt.npy'; S._tactile_post_fx(out_png, dmap_path=gt if os.path.exists(gt) else None); done += 1
        print(f'[dr] {a.obj}/{sess}: {done} rendered', flush=True)
    print('[dr] done', flush=True)


main()
