"""
render_rgb_trajectory.py  (run inside Blender)
===============================================
Render a sequence of RGB frames simulating a camera descending from far
above down to the gel contact level, then assemble into a GIF.

The object is placed at a fixed pose; only cam_z changes between frames.
All rendering uses the same RGB pipeline as the real training renders
(render_rgb_sample + _gel_fx post-processing).

Usage:
    GELSIGHT_RGB_PARAMS=bo_results/rgb/best_rgb_params.json \
    /home/shared/blender-4.2.0-linux-x64/blender -t 8 --background \
      gelsight_sampler.blend --python render_rgb_trajectory.py

Output: render_rgb_trajectory/trajectory.gif  (+ individual PNGs)
"""
import os, sys, json, math, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import bpy
from mathutils import Vector

# Reuse the RGB render helpers from rerender_rgb_only.py
import rerender_rgb_only as lib

# ── Config ──
OBJ_NAME = 'pattern_04_3_lines_angle_1'   # Y-junction, visually distinctive
SESSION = 'session_000'
SAMPLE_IDX = 100                            # pick a sample with decent contact
OUT_DIR = os.path.join(ROOT, 'render_rgb_trajectory')
N_FRAMES = 40                               # number of frames in the GIF
CAM_Z_START = -0.30                         # 300 mm above gel (far away, wide view)
CAM_Z_END = -0.012                          # 12 mm (close to tactile camera height ~15mm)
GIF_DURATION_MS = 80                        # per-frame duration in the GIF
FOV = lib.RGB_FOV                           # 55 degrees


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # ── Import mesh and apply material ──
    mesh_path = os.path.join(ROOT, 'meshes', f'{OBJ_NAME}.obj')
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=mesh_path, directory=os.path.dirname(mesh_path),
                          files=[{'name': os.path.basename(mesh_path)}])
    blender_name = [o.name for o in bpy.data.objects if o.name not in before][0]

    color = lib.OBJ_BLACK_COLOR if OBJ_NAME in lib.BLACK_OBJS else lib.OBJ_BLUE_COLOR
    mat = bpy.data.materials.new(f'solid_{blender_name}')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = lib._rgb_bo.get('obj_roughness', 0.25)
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    obj_bl = bpy.data.objects[blender_name]
    obj_bl.data.materials.clear()
    obj_bl.data.materials.append(mat)
    obj_bl.hide_render = True

    # ── Setup platform ──
    lib._setup_platform()

    # ── Place object at a fixed pose from pose.json ──
    sensor_dir = os.path.join(ROOT, 'renders_v3', OBJ_NAME, SESSION, 'sensor_0000')
    pose_path = os.path.join(sensor_dir, 'raw_data', f'{SAMPLE_IDX:04d}_pose.json')
    with open(pose_path) as f:
        pose = json.load(f)

    obj_bl.location = pose['location']
    obj_bl.rotation_euler = pose['rotation_euler']
    obj_bl.scale = pose['scale']
    bpy.context.view_layer.update()
    lib._set_platform_rotation(pose['rotation_euler'][2])

    print(f'Object: {OBJ_NAME}, color: {"BLACK" if color == lib.OBJ_BLACK_COLOR else "BLUE"}')
    print(f'Pose: loc={[round(x*1000,1) for x in pose["location"]]}mm  '
          f'rz={math.degrees(pose["rotation_euler"][2]):.1f}deg')
    print(f'Rendering {N_FRAMES} frames, cam_z from {CAM_Z_START*1000:.0f}mm '
          f'to {CAM_Z_END*1000:.0f}mm')

    # ── Render frames at different camera heights ──
    cam_zs = np.linspace(CAM_Z_START, CAM_Z_END, N_FRAMES)
    frame_paths = []

    for i, cz in enumerate(cam_zs):
        out_path = os.path.join(OUT_DIR, f'frame_{i:03d}')
        lib.render_rgb_sample(blender_name, FOV, out_path, cam_z=cz)
        frame_paths.append(out_path + '.png')
        elapsed = time.time() - t0
        print(f'  [{i+1}/{N_FRAMES}] cam_z={cz*1000:.1f}mm  {elapsed:.0f}s', flush=True)

    # ── Assemble GIF ──
    from PIL import Image as PILImage
    frames = [PILImage.open(p) for p in frame_paths]
    gif_path = os.path.join(OUT_DIR, 'trajectory.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=GIF_DURATION_MS, loop=0)
    print(f'\nGIF saved: {gif_path} ({len(frames)} frames, '
          f'{len(frames)*GIF_DURATION_MS/1000:.1f}s)')
    print(f'Total time: {time.time()-t0:.0f}s')

    # Cleanup
    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        sys.exit(0)


main()
