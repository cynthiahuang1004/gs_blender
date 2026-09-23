"""Render the calibrated Blender *tactile* image at real GT poses (physics render for the
physics-guided GAN, method "phys-cond").

Run inside Blender (cwd = gs_blender):
    GELSIGHT_FIXED_PARAMS=bo_results/tactile_v2/best_params.json \
    blender --background gelsight_sampler.blend --python render_real_pose_tactile.py -- \
        --obj pattern_33 --frames /path/list.json --out /media/hdd2/ihsuan/gs_blender/real_filtered_phys

--frames: JSON list of sample indices (ints) of real_filtered/<obj>/session_000/sensor_0000.
Output: <out>/<obj>/session_000/sensor_0000/samples/XXXX.png  (same look/post-fx as renders_v3).
The object pose is taken verbatim from the real pose json (location / rotation_euler / scale),
i.e. exactly the placement used to render the real GT depth.
"""
import argparse
import json
import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
os.environ.setdefault('GELSIGHT_FIXED_PARAMS', os.path.join(_dir, 'bo_results', 'tactile_v2', 'best_params.json'))
import math  # noqa: E402
import time  # noqa: E402

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
import scripting as S  # noqa: E402  (module-level = config + function defs only)


def enable_gpu():
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        n = 0
        for d in prefs.devices:
            d.use = d.type == 'CUDA'
            n += d.use
        bpy.context.scene.cycles.device = 'GPU'
        print(f'[phys] Cycles on GPU ({n} CUDA devices)', flush=True)
    except Exception as e:  # noqa: BLE001
        print(f'[phys] GPU enable failed: {e}', flush=True)

REAL_ROOT = os.environ.get('PHYS_REAL_ROOT', '/media/hdd2/ihsuan/gs_blender/real_filtered')   # needs <obj>/session_000/sensor_0000/raw_data/*_pose.json + *_gt.npy


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--out', default=os.environ.get('PHYS_OUT', '/media/hdd2/ihsuan/gs_blender/real_filtered_phys'))
    ap.add_argument('--session', default='session_000')
    args = ap.parse_args(argv)

    frames = json.load(open(args.frames))
    unit_in = os.path.join(REAL_ROOT, args.obj, args.session, 'sensor_0000')
    unit_out = os.path.join(args.out, args.obj, args.session, 'sensor_0000')
    os.makedirs(os.path.join(unit_out, 'samples'), exist_ok=True)

    # import the mesh (.obj) exactly like scripting.py's main
    mesh_dir = os.path.join(_dir, 'meshes')
    obj_file = args.obj + '.obj'
    assert os.path.exists(os.path.join(mesh_dir, obj_file)), obj_file
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=os.path.join(mesh_dir, obj_file), directory=mesh_dir, files=[{"name": obj_file}])
    new = [o.name for o in bpy.data.objects if o.name not in before]
    bl_name = new[0] if new else args.obj[:-4]
    obj = bpy.data.objects[bl_name]
    obj.hide_render = True
    obj.scale = (0.001, 0.001, 0.001)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    # mesh-frame bbox centre (mm) and lowest z (mm): the real GT (build_real_filtered / gt_depth_mm)
    # rotates the mesh about its bbox centre and presses its lowest point `press` into the gel.
    co = [v.co for v in obj.data.vertices]
    bb = Vector(((min(c.x for c in co) + max(c.x for c in co)) / 2, (min(c.y for c in co) + max(c.y for c in co)) / 2, 0.0))
    zmin = min(c.z for c in co)
    print(f'[phys] imported {obj_file} as {bl_name}; bbox centre {tuple(round(v, 2) for v in bb[:2])} mm, zmin {zmin:.2f} mm', flush=True)
    if not os.environ.get('GELSIGHT_FORCE_CPU'):
        enable_gpu()

    sensor = S.create_sensor()
    sensor.randomize()          # loads GELSIGHT_FIXED_PARAMS (calibrated sensor)
    sensor.apply()

    done = 0
    for idx in frames:
        tag = f'{int(idx):04d}'
        out_png = os.path.join(unit_out, 'samples', tag + '.png')
        if os.path.exists(out_png) and os.path.getsize(out_png) > 1000:
            continue
        pose = json.load(open(os.path.join(unit_in, 'raw_data', tag + '_pose.json')))
        th = pose['rotation_euler'][2]
        press = pose['sample_z']
        cx, cy = pose['location'][0], pose['location'][1]          # = (sample_x, -sample_y): bbox centre in world
        c, sn = math.cos(th), math.sin(th)
        bx, by = bb.x * 0.001, bb.y * 0.001
        t0 = time.time()
        obj.rotation_euler = (0.0, 0.0, th)
        obj.scale = (0.001, 0.001, 0.001)
        obj.location = (cx - (c * bx - sn * by), cy - (sn * bx + c * by), -press - zmin * 0.001)
        obj.hide_render = True
        bpy.data.objects['GelSurface'].modifiers['Shrinkwrap'].target = obj
        bpy.context.scene.frame_set(0)
        bpy.context.scene.render.filepath = out_png[:-4]
        bpy.ops.render.render(write_still=True)
        gt = os.path.join(unit_in, 'raw_data', tag + '_gt.npy')
        S._tactile_post_fx(out_png, dmap_path=gt if os.path.exists(gt) else None)
        done += 1
        if done <= 3 or done % 20 == 0:
            print(f'[phys] {tag}: {time.time() - t0:.1f}s', flush=True)
        if done % 20 == 0:
            print(f'[phys] {args.obj}: {done}/{len(frames)}', flush=True)
    print(f'[phys] {args.obj}: done {done} rendered, {len(frames) - done} existed', flush=True)


main()
