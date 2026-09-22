"""
render_tactile_trajectory.py  (run inside Blender)
===================================================
Two GIFs showing the tactile simulation setup:

1. press_trajectory.gif — Object presses into the gel from z=0 (no contact)
   to z=1.2mm (full contact). Camera is the fixed tactile camera. Shows
   the transition from background calibration image to full contact pattern.

2. scene_orbit.gif — An observer camera orbits around the sensor setup
   (gel, 6 light panels, object) so you can see the physical arrangement.
   This is NOT a tactile render; it's a "behind the scenes" view.

Usage:
    GELSIGHT_FIXED_PARAMS=bo_results/tactile_v2/best_params.json \
    /home/shared/blender-4.2.0-linux-x64/blender -t 8 --background \
      gelsight_sampler.blend --python render_tactile_trajectory.py
"""
import os, sys, json, math, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import bpy
from mathutils import Vector, Euler, Matrix

OBJ_NAME = 'pattern_04_3_lines_angle_1'
SESSION = 'session_000'
SAMPLE_IDX = 100
OUT_DIR = os.path.join(ROOT, 'render_tactile_trajectory')
GIF_MS = 100


def import_and_place():
    """Import mesh, apply default material, place at the sample's pose."""
    mesh_path = os.path.join(ROOT, 'meshes', f'{OBJ_NAME}.obj')
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=mesh_path, directory=os.path.dirname(mesh_path),
                          files=[{'name': os.path.basename(mesh_path)}])
    name = [o.name for o in bpy.data.objects if o.name not in before][0]
    bpy.data.objects[name].hide_render = True

    pose_path = os.path.join(ROOT, 'renders_v3', OBJ_NAME, SESSION,
                             'sensor_0000', 'raw_data', f'{SAMPLE_IDX:04d}_pose.json')
    with open(pose_path) as f:
        pose = json.load(f)
    return name, pose


def setup_sensor():
    """Apply the BO-tuned sensor parameters (lights, gel, camera)."""
    # scripting.py's create_sensor uses GELSIGHT_FIXED_PARAMS
    # We import it here so it runs inside blender
    import scripting
    sensor = scripting.create_sensor()
    sensor.apply()
    return sensor


# ═══════════════════════════════════════════════════════════════
# Part 1: Press trajectory (tactile camera, varying press depth)
# ═══════════════════════════════════════════════════════════════
def render_press_trajectory(obj_name, pose, sensor):
    out = os.path.join(OUT_DIR, 'press')
    os.makedirs(out, exist_ok=True)

    obj = bpy.data.objects[obj_name]
    obj.rotation_euler = pose['rotation_euler']
    obj.scale = pose['scale']

    # Base location at z=0 (no contact): object's lowest point touches gel
    session_path = os.path.join(ROOT, 'renders_v3', OBJ_NAME, SESSION, 'session.json')
    with open(session_path) as f:
        sess = json.load(f)
    z_anchor = sess.get('z_anchor', 0.0)

    # The original location has press depth baked in as pose['sample_z']
    # We want to sweep from 0 (just touching) to 1.2mm (full press)
    base_x, base_y = pose['location'][0], pose['location'][1]

    N = 30
    depths = np.concatenate([
        np.linspace(0.0, 0.0012, 20),      # slow press in
        np.linspace(0.0012, 0.0012, 5),     # hold at max
        np.linspace(0.0012, 0.0, 5),        # pull out
    ])
    N = len(depths)
    paths = []
    t0 = time.time()

    for i, dz in enumerate(depths):
        obj.location = (base_x, base_y, -dz - z_anchor)
        obj.hide_render = False
        bpy.data.objects['GelSurface'].modifiers["Shrinkwrap"].target = obj
        bpy.context.view_layer.update()
        bpy.context.scene.frame_set(0)

        fp = os.path.join(out, f'frame_{i:03d}')
        bpy.context.scene.render.filepath = fp
        bpy.ops.render.render(write_still=True)

        # Apply tactile post-fx (same as training renders)
        import scripting as sc
        sc._tactile_post_fx(fp + '.png', dmap_path=None)

        paths.append(fp + '.png')
        print(f'  press [{i+1}/{N}] depth={dz*1000:.2f}mm  {time.time()-t0:.0f}s', flush=True)

    obj.hide_render = True
    return paths


# ═══════════════════════════════════════════════════════════════
# Part 2: Scene orbit (observer camera, shows setup)
# ═══════════════════════════════════════════════════════════════
def render_scene_orbit(obj_name, pose):
    out = os.path.join(OUT_DIR, 'orbit')
    os.makedirs(out, exist_ok=True)

    obj = bpy.data.objects[obj_name]
    obj.rotation_euler = pose['rotation_euler']
    obj.scale = pose['scale']

    session_path = os.path.join(ROOT, 'renders_v3', OBJ_NAME, SESSION, 'session.json')
    with open(session_path) as f:
        sess = json.load(f)
    z_anchor = sess.get('z_anchor', 0.0)
    obj.location = (pose['location'][0], pose['location'][1],
                    -pose['sample_z'] - z_anchor)
    obj.hide_render = False
    bpy.data.objects['GelSurface'].modifiers["Shrinkwrap"].target = obj
    bpy.context.view_layer.update()

    # Create observer camera
    cam_data = bpy.data.cameras.new('ObserverCam')
    cam_data.angle = math.radians(50)
    cam_obj = bpy.data.objects.new('ObserverCam', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    # Save original camera
    orig_cam = bpy.context.scene.camera
    bpy.context.scene.camera = cam_obj

    # Make everything visible for the orbit (gel, lights, object)
    for name in ['GelSurface', 'InterfaceSurface']:
        if name in bpy.data.objects:
            bpy.data.objects[name].hide_render = False

    # Disable compositor (avoid depth tint)
    orig_nodes = bpy.context.scene.use_nodes
    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.film_transparent = False

    # Orbit parameters
    N = 36
    radius = 0.06       # 60mm from center
    center = (0, 0, -0.005)  # slightly below gel plane
    paths = []
    t0 = time.time()

    for i in range(N):
        angle = 2 * math.pi * i / N
        cx = center[0] + radius * math.cos(angle)
        cy = center[1] + radius * math.sin(angle)
        cz = center[2] + 0.025  # observer is 25mm above center

        cam_obj.location = (cx, cy, cz)
        # Point camera at center
        direction = Vector(center) - Vector((cx, cy, cz))
        rot = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot.to_euler()
        bpy.context.view_layer.update()
        bpy.context.scene.frame_set(0)

        fp = os.path.join(out, f'frame_{i:03d}')
        bpy.context.scene.render.filepath = fp
        bpy.ops.render.render(write_still=True)
        paths.append(fp + '.png')
        print(f'  orbit [{i+1}/{N}] angle={math.degrees(angle):.0f}deg  {time.time()-t0:.0f}s',
              flush=True)

    # Restore
    bpy.context.scene.camera = orig_cam
    bpy.context.scene.use_nodes = orig_nodes
    obj.hide_render = True
    return paths


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    obj_name, pose = import_and_place()
    sensor = setup_sensor()

    print('=== Part 1: Press trajectory ===', flush=True)
    press_paths = render_press_trajectory(obj_name, pose, sensor)

    print('\n=== Part 2: Scene orbit ===', flush=True)
    orbit_paths = render_scene_orbit(obj_name, pose)

    # Write marker file for GIF assembly (PIL not available in Blender)
    import json as _json
    _json.dump({'press': press_paths, 'orbit': orbit_paths},
               open(os.path.join(OUT_DIR, 'frame_paths.json'), 'w'))

    print(f'\nAll frames rendered in {time.time()-t0:.0f}s')
    print(f'Run the GIF assembler outside Blender to create the GIFs.')

    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        sys.exit(0)


main()
