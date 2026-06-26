'''scripting.py'''
CONTINUE = False

# sampling config

NUM_SENSORS = 1
NUM_CALIBRATION = 1
NUM_OBJ_SAMPLES = 30

OBJ_SIZE_MIN = 0.01
OBJ_SIZE_MAX = 0.05

X_MIN = -0.007
X_MAX = 0.007

Y_MIN = -0.007
Y_MAX = 0.007

CALIB_DEPTH_MIN = 0.0014
CALIB_DEPTH_MAX = 0.0018

OBJ_DEPTH_MIN = 0.0006
OBJ_DEPTH_MAX = 0.0018

# sensor parameters config

FOV_MIN = 20
FOV_MAX = 60

LENGTH_MIN = 0.0075
LENGTH_MAX = 0.0125

SMOOTHNESS_MIN = 30
SMOOTHNESS_MAX = 50

# GelSurface 材質參數
GEL_ROUGHNESS_MIN = 0.4   #0=鏡面反射, 1=完全漫反射
GEL_ROUGHNESS_MAX = 0.6
GEL_FAC_MIN = 0.2   # 0=全透明, 1=全反光
GEL_FAC_MAX = 0.3

AO_DISTANCE = 0.01  # AO 搜尋半徑 (m)；3mm 對應 gel 包住 cross tube 凹谷尺度
DARK_BASE_GAIN = 0.9  # gel 整體反射倍率：1.0=不壓暗、0.0=全黑；contact 想更黑就調小
CONTACT_Z_THRESH  = -0.001  # 世界 z 小於這個值 → 最暗（contact）
BG_Z_THRESH       =  0.0005  # 世界 z 大於這個值 → 最亮（背景）
# Contact 最暗區的 sRGB 色 (0~255)，預設 (75, 38, 55) 暗紫紅
CONTACT_DARK_COLOR = (60/255, 60/255, 60/255, 1.0)

# 反射率
ROUGH_MIN = 0.4
ROUGH_MAX = 0.5

# SCALE: 顏色過度平滑程度，數值越大越平滑
SCALE_MIN = 0.4
SCALE_MAX = 0.6

RED_STR_MIN = 45.0   # BO best: 57.09
RED_STR_MAX = 70.0

RED_COL_MIN = [0.9, 0.0, 0.1]
RED_COL_MAX = [1.0, 0.05, 0.2]

GREEN_STR_MIN = 10.0   # BO best: 23.46
GREEN_STR_MAX = 35.0

GREEN_COL_MIN = [0.0, 0.9, 0.0]
GREEN_COL_MAX = [0.05, 1.0, 0.05]

BLUE_STR_MIN = 50.0   # BO best: 62.38
BLUE_STR_MAX = 75.0

BLUE_COL_MIN = [0.0, 0.0, 0.9]
BLUE_COL_MAX = [0.05, 0.05, 1.0]

# 儲存燈光初始位置（第一次 apply 時自動讀取）
_LIGHT_INITIAL_MATRICES = {}

import bpy
from random import uniform as ru
from math import pi, tan
import os
import shutil
import sys
import numpy as np
import random
import json
import math
from mathutils import Euler

_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_PARAMS_PATH = os.environ.get(
    'GELSIGHT_FIXED_PARAMS',
    os.path.join(_PROJ_DIR, 'bo_results_tactile', 'best_params.json'),
)
RGB_PARAMS_PATH = os.environ.get(
    'GELSIGHT_RGB_PARAMS',
    os.path.join(_PROJ_DIR, 'bo_results', 'best_rgb_params.json'),
)
BG_RENDER_PATH    = os.environ.get('GELSIGHT_BG_RENDER', None)

# Load RGB BO params (falls back to hardcoded defaults if file missing)
_rgb_bo = {}
if RGB_PARAMS_PATH and os.path.exists(RGB_PARAMS_PATH):
    with open(RGB_PARAMS_PATH) as _f:
        _rgb_bo = json.load(_f)
    print(f'[scripting] RGB params loaded from {RGB_PARAMS_PATH}')


def _flip_png(png_path):
    """Rotate PNG 180° to correct for Blender upward-facing camera."""
    img_bpy = bpy.data.images.load(png_path)
    W, H = img_bpy.size[0], img_bpy.size[1]
    px = np.array(img_bpy.pixels[:], dtype=np.float32).reshape(H, W, 4)
    bpy.data.images.remove(img_bpy)
    px = px[::-1, ::-1, :]   # 180° rotation
    out_img = bpy.data.images.new('_flip_tmp', W, H, alpha=False)
    out_img.pixels = px.flatten().tolist()
    out_img.filepath_raw = png_path
    out_img.file_format = 'PNG'
    out_img.save()
    bpy.data.images.remove(out_img)

def _barrel(img, k1):
    H, W = img.shape[:2]
    cx2, cy2 = W / 2.0, H / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx - cx2) / cx2; yn = (yy - cy2) / cy2
    r2 = xn**2 + yn**2; fac = 1.0 + k1 * r2
    xs = np.clip(xn * fac * cx2 + cx2, 0, W - 1)
    ys = np.clip(yn * fac * cy2 + cy2, 0, H - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, W - 1)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, H - 1)
    wx = (xs - x0)[:, :, None]; wy = (ys - y0)[:, :, None]
    return np.clip(img[y0,x0]*(1-wx)*(1-wy)+img[y0,x1]*wx*(1-wy)+
                   img[y1,x0]*(1-wx)*wy+img[y1,x1]*wx*wy, 0, 1)

def _gaussian_blur_ch(img, sigma):
    size = max(3, int(6 * sigma + 1) | 1)
    x = np.arange(size) - size // 2
    k = np.exp(-x**2 / (2 * sigma**2)); k /= k.sum()
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        h = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1, img[:,:,c])
        out[:,:,c] = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 0, h)
    return np.clip(out, 0.0, 1.0)

def _gel_fx(png_path):
    img_bpy = bpy.data.images.load(png_path)
    W, H = img_bpy.size[0], img_bpy.size[1]
    px = np.array(img_bpy.pixels[:], dtype=np.float32).reshape(H, W, 4)
    bpy.data.images.remove(img_bpy)
    rgb = _barrel(px[:,:,:3], RGB_BARREL_K1)
    blurred = _gaussian_blur_ch(rgb, RGB_BLUR_SIGMA)
    cy3, cx3 = H / 2.0, W / 2.0
    yy3, xx3 = np.mgrid[0:H, 0:W]
    dist3 = np.sqrt(((yy3-cy3)/cy3)**2 + ((xx3-cx3)/cx3)**2)
    weight = np.clip(dist3 ** RGB_BLUR_FALLOFF, 0, 1)[:,:,None]
    rgb = rgb * (1 - weight) + blurred * weight
    mask = np.clip(1.0 - dist3**2 * RGB_VIGNETTE, 0, 1)[:,:,None]
    rgb = np.clip(rgb * mask, 0, 1)
    out_img = bpy.data.images.new('_gel_tmp', W, H, alpha=False)
    out_px = np.ones((H, W, 4), dtype=np.float32); out_px[:,:,:3] = rgb
    out_img.pixels = out_px.flatten().tolist()
    out_img.filepath_raw = png_path
    out_img.file_format = 'PNG'
    out_img.save()
    bpy.data.images.remove(out_img)

_platform_objs_global = []

def _setup_platform():
    global _platform_objs_global
    plat_path = os.path.join(dir, 'meshes', '202000 6152_200.obj')
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=plat_path, directory=os.path.dirname(plat_path),
                          files=[{'name': os.path.basename(plat_path)}])
    new_objs = [o for o in bpy.data.objects if o.name not in before]
    teal_mat = bpy.data.materials.new('platform_teal_rgb')
    teal_mat.use_nodes = True
    p_bsdf = teal_mat.node_tree.nodes.get('Principled BSDF')
    if p_bsdf:
        p_bsdf.inputs['Base Color'].default_value         = (
            _rgb_bo.get('plat_r', 26/255),
            _rgb_bo.get('plat_g', 115/255),
            _rgb_bo.get('plat_b', 106/255), 1.0)
        p_bsdf.inputs['Roughness'].default_value          = _rgb_bo.get('plat_roughness', 0.25)
        p_bsdf.inputs['Specular IOR Level'].default_value = 0.6
        p_bsdf.inputs['Metallic'].default_value           = _rgb_bo.get('plat_metallic',  0.85)
    for po in new_objs:
        po.data.materials.clear()
        po.data.materials.append(teal_mat)
        po.scale          = (0.001, -0.001, 0.001)
        po.rotation_euler = (math.pi / 2, 0.0, -math.pi / 2)
        po.location       = (-0.08, 0.055, 0.04)
        po.hide_render    = False
    bpy.context.view_layer.update()
    _platform_objs_global = new_objs
    print(f'Platform ready: {[o.name for o in new_objs]}')

def render_rgb_sample(obj_name, fov_deg, out_path):
    cam = bpy.data.objects['Camera']
    cam_data = cam.data
    orig_cam_z = cam.location[2]
    orig_fov   = cam_data.angle
    orig_dof   = cam_data.dof.use_dof

    cam.location[2]         = RGB_CAM_Z
    cam_data.angle          = math.radians(fov_deg)
    cam_data.dof.use_dof    = False

    world = bpy.data.worlds.get('World')
    orig_bg = None
    if world and world.node_tree:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            orig_bg = (tuple(bg_node.inputs['Color'].default_value),
                       bg_node.inputs['Strength'].default_value)
            bg_node.inputs['Color'].default_value    = (0.25, 0.25, 0.25, 1.0)
            bg_node.inputs['Strength'].default_value = RGB_WORLD_STR

    vis_save = {}
    for name in RGB_HIDE_NAMES:
        if name in bpy.data.objects:
            vis_save[name] = bpy.data.objects[name].hide_render
            bpy.data.objects[name].hide_render = True
    bpy.data.objects[obj_name].hide_render = False

    scene = bpy.context.scene
    orig_nodes = scene.use_nodes
    orig_trans = scene.render.film_transparent
    orig_mode  = scene.render.image_settings.color_mode
    orig_fp    = scene.render.filepath

    scene.use_nodes = False
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.filepath = out_path
    scene.frame_set(0)
    bpy.ops.render.render(write_still=True)

    _gel_fx(out_path + '.png')

    scene.use_nodes = orig_nodes
    scene.render.film_transparent = orig_trans
    scene.render.image_settings.color_mode = orig_mode
    scene.render.filepath = orig_fp
    bpy.data.objects[obj_name].hide_render = True
    for name, vis in vis_save.items():
        bpy.data.objects[name].hide_render = vis
    cam.location[2]      = orig_cam_z
    cam_data.angle       = orig_fov
    cam_data.dof.use_dof = orig_dof
    if orig_bg and world and world.node_tree:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value    = orig_bg[0]
            bg_node.inputs['Strength'].default_value = orig_bg[1]

def find_highest_at_xy(object, cx, cy, sensor_width) -> float:
    obj = bpy.data.objects[object]
    mw = obj.matrix_world
    half = sensor_width / 2

    highest_z = None
    highest_co = None

    for v in obj.data.vertices:
        world_co = mw @ v.co
        if (abs(world_co.x - cx) < half and
                abs(world_co.y - cy) < half):
            if highest_z is None or world_co.z > highest_z:
                highest_z = world_co.z
                highest_co = world_co

    if highest_co is None:
        return find_lowest(object)

    return highest_co

def move_object_at_xy(object, location, rotation, sensor_width, z_anchor=None) -> None:
    obj = bpy.data.objects[object]
    obj.rotation_euler = rotation

    cx, cy, press_depth = location[0], location[1], location[2]

    if z_anchor is None:
        # 舊版 fallback：找 cell 內最低點當 anchor
        obj.location = (0, 0, 0)
        bpy.context.scene.frame_set(0)
        mw = obj.matrix_world
        half = sensor_width / 2
        lowest_z = None
        for v in obj.data.vertices:
            world_co = mw @ v.co
            if (abs(world_co.x - cx) < half and
                    abs(world_co.y - cy) < half):
                if lowest_z is None or world_co.z < lowest_z:
                    lowest_z = world_co.z
        z_anchor = lowest_z if lowest_z is not None else 0.0

    # cell 中心 (cx, cy) → 世界 (0, 0)
    # 全域 z_anchor (cross tip 最低點) → 世界 z = -press_depth
    obj.location = (-cx, -cy, -press_depth - z_anchor)
    bpy.context.scene.frame_set(0)

    obj.hide_render = True
    bpy.data.objects['GelSurface'].modifiers["Shrinkwrap"].target = obj
    
def move_object(object, location, rotation) -> None:
    bpy.data.objects[object].rotation_euler = rotation
    bpy.context.scene.frame_set(0)

    glbl_co = bpy.data.objects[object].location
    low_co = find_lowest(object)

    x = glbl_co[0] - low_co[0] + location[0]
    y = glbl_co[1] - low_co[1] + location[1]
    z = glbl_co[2] - low_co[2] - location[2]
    bpy.data.objects[object].location = (x, y, z)

    bpy.data.objects[object].hide_render = True
    bpy.data.objects['GelSurface'].modifiers["Shrinkwrap"].target = bpy.data.objects[object]


def find_lowest(object) -> float:
    obj = bpy.data.objects[object]
    mw = obj.matrix_world
    glbl_co = [mw @ v.co for v in obj.data.vertices]
    minZ = min([co.z for co in glbl_co])

    lowest = []
    for v in obj.data.vertices:
        if (mw @ v.co).z == minZ:
            lowest.append(mw @ v.co)
    return random.choice(lowest)


def set_emittor(emittor, strength, color) -> None:
    emission_node = bpy.data.materials[emittor].node_tree.nodes['Emission']
    emission_node.inputs['Color'].default_value = color
    emission_node.inputs['Strength'].default_value = strength


def set_smoothness(val) -> None:
    bpy.data.objects['GelSurface'].modifiers['CorrectiveSmooth'].iterations = val
    bpy.data.objects['GelSurface'].modifiers["Shrinkwrap"].offset = 1e-03 * (val * val) / 50000
    # 改用法向量投影，讓膠面沿感測器方向貼合物體
    bpy.data.objects['GelSurface'].modifiers["Shrinkwrap"].wrap_method = 'PROJECT'
    bpy.data.objects['GelSurface'].modifiers["Shrinkwrap"].use_project_z = True
    bpy.data.objects['GelSurface'].modifiers["Shrinkwrap"].use_negative_direction = True
    bpy.data.objects['GelSurface'].modifiers["Shrinkwrap"].use_positive_direction = False


def set_scale(val) -> None:
    co_z = -0.0065 + 0.0011 / val
    bpy.data.objects['LightSurfaceBL'].scale[1] = val
    bpy.data.objects['LightSurfaceTR'].scale[1] = val
    bpy.data.objects['LightSurfaceTL'].scale[1] = val
    bpy.data.objects['LightSurfaceBR'].scale[1] = val
    bpy.data.objects['LightSurfaceBL'].location[2] = co_z
    bpy.data.objects['LightSurfaceTR'].location[2] = co_z
    bpy.data.objects['LightSurfaceTL'].location[2] = co_z
    bpy.data.objects['LightSurfaceBR'].location[2] = co_z


def set_light_type(type) -> None:
    if type == 'point':
        bpy.data.objects['LightSurfaceBL'].scale[0] = 0.1
        bpy.data.objects['LightSurfaceTR'].scale[0] = 0.1
        bpy.data.objects['LightSurfaceTL'].scale[2] = 0.125
        bpy.data.objects['LightSurfaceBR'].scale[2] = 0.125
    if type == 'long':
        bpy.data.objects['LightSurfaceBL'].scale[0] = 1
        bpy.data.objects['LightSurfaceTR'].scale[0] = 1
        bpy.data.objects['LightSurfaceTL'].scale[2] = 1
        bpy.data.objects['LightSurfaceBR'].scale[2] = 1

def set_green_lights(strength):
    for name in ['LightSurfaceRGreen', 'LightSurfaceLGreen']:
        if name in bpy.data.objects:
            bpy.data.objects[name].hide_render = False

    green_color = (
        ru(GREEN_COL_MIN[0], GREEN_COL_MAX[0]),
        ru(GREEN_COL_MIN[1], GREEN_COL_MAX[1]),
        ru(GREEN_COL_MIN[2], GREEN_COL_MAX[2]),
        1.0
    )
    for mat_name in ['RGreenEmittor', 'LGreenEmittor']:
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
            mat.node_tree.nodes['Emission'].inputs['Color'].default_value = green_color
            mat.node_tree.nodes['Emission'].inputs['Strength'].default_value = strength

def set_cam(fov, length) -> None:
    height = (length / tan((fov / 360) * pi))
    bpy.data.objects['Camera'].location[2] = -height
    bpy.data.objects['Camera'].data.angle = (fov / 180) * pi
    bpy.data.scenes["Scene"].node_tree.nodes["Map Range"].inputs[2].default_value = height
    bpy.data.scenes["Scene"].node_tree.nodes["Map Range"].inputs[1].default_value = height - 0.002


class create_sensor():
    def __init__(self,
                 randomize=True,
                 smoothness=None,
                 top_str=None,
                 top_col=(None, None, None, 1),
                 bot_str=None,
                 bot_col=(None, None, None, 1),
                 lef_str=None,
                 lef_col=(None, None, None, 1),
                 rig_str=None,
                 rig_col=(None, None, None, 1),
                 scale=None,
                 light_type=None,
                 angle=None,
                 fov=None,
                 roughness=None,
                 length=None,
                 write_dir=None,
                 read_dir=None):

        self.smoothness = smoothness
        self.scale = scale
        self.light_type = light_type
        self.angle = angle
        self.emittors = [[top_str, top_col], [bot_str, bot_col], [lef_str, lef_col], [rig_str, rig_col]]
        self.fov = fov
        self.roughness = roughness
        self.length = length
        self.lg_str   = None   # RGreenEmittor  (image: 右)
        self.lg_color = None
        self.rg_str   = None   # LGreenEmittor (image: 左)
        self.rg_color = None
        self.gel_roughness = None
        self.gel_fac = None
        
        if read_dir != None:
            f = open(read_dir, 'r')
            content = f.readlines()
            self.smoothness = int(content[0])
            self.scale = float(content[1])
            self.light_type = content[2].strip('\n')
            self.angle = content[3].strip('\n')
            self.emittors[0][0] = float(content[4])
            self.emittors[0][1] = (float(content[5]), float(content[6]), float(content[7]), 1)
            self.emittors[1][0] = float(content[8])
            self.emittors[1][1] = (float(content[9]), float(content[10]), float(content[11]), 1)
            self.emittors[2][0] = float(content[12])
            self.emittors[2][1] = (float(content[13]), float(content[14]), float(content[15]), 1)
            self.emittors[3][0] = float(content[16])
            self.emittors[3][1] = (float(content[17]), float(content[18]), float(content[19]), 1)
            self.fov = float(content[20])
            self.roughness = float(content[21])
            self.length = float(content[22])
        else:
            if randomize == True:
                self.randomize()
            if write_dir != None:
                f = open(write_dir, "w+")
                f.write(f'{self.smoothness}\n')
                f.write(f'{self.scale}\n')
                f.write(f'{self.light_type}\n')
                f.write(f'{self.angle}\n')
                f.write(f'{self.emittors[0][0]}\n')
                f.write(f'{self.emittors[0][1][0]}\n')
                f.write(f'{self.emittors[0][1][1]}\n')
                f.write(f'{self.emittors[0][1][2]}\n')
                f.write(f'{self.emittors[1][0]}\n')
                f.write(f'{self.emittors[1][1][0]}\n')
                f.write(f'{self.emittors[1][1][1]}\n')
                f.write(f'{self.emittors[1][1][2]}\n')
                f.write(f'{self.emittors[2][0]}\n')
                f.write(f'{self.emittors[2][1][0]}\n')
                f.write(f'{self.emittors[2][1][1]}\n')
                f.write(f'{self.emittors[2][1][2]}\n')
                f.write(f'{self.emittors[3][0]}\n')
                f.write(f'{self.emittors[3][1][0]}\n')
                f.write(f'{self.emittors[3][1][1]}\n')
                f.write(f'{self.emittors[3][1][2]}\n')
                f.write(f'{self.fov}\n')
                f.write(f'{self.roughness}\n')
                f.write(f'{self.length}\n')
                f.close()
    def _apply_fixed(self, p):
        """Lighting from BO; gel/camera randomized within physical ranges."""
        # Gel & camera — randomized (BO can't optimize these from background-only renders)
        self.smoothness    = random.randrange(30, 45)
        self.scale         = 0.4918   # matches scripting_bo.py hardcoded value
        self.light_rot_z   = -math.pi # matches scripting_bo.py rot_z=-3.14159
        self.fov           = ru(30.0, 50.0)
        self.roughness     = ru(ROUGH_MIN, ROUGH_MAX)
        self.gel_roughness = float(p.get('gel_roughness', 0.45))
        self.gel_fac       = float(p.get('gel_fac', 0.25))
        self.length        = ru(LENGTH_MIN, LENGTH_MAX)
        self.angle         = 'str'
        self.light_type    = 'long'

        # 6 lights fully independent — from BO best params
        # BLEmittor (image: 右上)
        self.emittors[0] = [
            float(p.get('top_str',  80.0)),
            (float(p.get('top_r',  0.937)), float(p.get('top_g',  0.545)), float(p.get('top_b',  1.0)),   1.0),
        ]
        # TREmittor (image: 左下)
        self.emittors[1] = [
            float(p.get('bot_str',  80.0)),
            (float(p.get('bot_r',  0.443)), float(p.get('bot_g',  0.278)), float(p.get('bot_b',  0.357)), 1.0),
        ]
        # TLEmittor (image: 右下)
        self.emittors[2] = [
            float(p.get('left_str', 75.0)),
            (float(p.get('left_r',  1.0)),  float(p.get('left_g',  0.0)),  float(p.get('left_b',  0.486)), 1.0),
        ]
        # BREmittor (image: 左上)
        self.emittors[3] = [
            float(p.get('right_str', 75.0)),
            (float(p.get('right_r', 0.992)), float(p.get('right_g', 0.133)), float(p.get('right_b', 0.373)), 1.0),
        ]
        # RGreenEmittor (image: 右)
        self.lg_str   = float(p.get('lg_str', 40.0))
        self.lg_color = (float(p.get('lg_r', 0.412)), float(p.get('lg_g', 0.890)), float(p.get('lg_b', 0.475)))
        # LGreenEmittor (image: 左)
        self.rg_str   = float(p.get('rg_str', 40.0))
        self.rg_color = (float(p.get('rg_r', 0.675)), float(p.get('rg_g', 0.725)), float(p.get('rg_b', 0.302)))

    '''ORIGINAL: def randomize
    def randomize(self):
        self.smoothness = random.randrange(SMOOTHNESS_MIN, SMOOTHNESS_MAX)
        self.scale = ru(SCALE_MIN, SCALE_MAX)
        self.fov = ru(FOV_MIN, FOV_MAX)
        self.roughness = ru(ROUGH_MIN, ROUGH_MAX)
        if self.length is None:
            self.length = ru(LENGTH_MIN, LENGTH_MAX)

        if random.random() < 0.35:
            self.angle = 'diag'
        else:
            self.angle = 'str'

        if random.random() < 0.35:
            self.light_type = 'point'
        else:
            self.light_type = 'long'

        emittors = ['RED', 'GREEN', 'BLUE', 'BLOCK']
        emittors = random.sample(emittors, 4)

        for idx, emittor in enumerate(emittors):
            if emittor == 'RED':
                self.emittors[idx][0] = ru(RED_STR_MIN, RED_STR_MAX)
                self.emittors[idx][1] = (ru(RED_COL_MIN[0], RED_COL_MAX[0]),
                                         ru(RED_COL_MIN[1], RED_COL_MAX[1]),
                                         ru(RED_COL_MIN[2], RED_COL_MAX[2]), 1)
            if emittor == 'GREEN':
                self.emittors[idx][0] = ru(GREEN_STR_MIN, GREEN_STR_MAX)
                self.emittors[idx][1] = (ru(GREEN_COL_MIN[0], GREEN_COL_MAX[0]),
                                         ru(GREEN_COL_MIN[1], GREEN_COL_MAX[1]),
                                         ru(GREEN_COL_MIN[2], GREEN_COL_MAX[2]), 1)
            if emittor == 'BLUE':
                self.emittors[idx][0] = ru(BLUE_STR_MIN, BLUE_STR_MAX)
                self.emittors[idx][1] = (ru(BLUE_COL_MIN[0], BLUE_COL_MAX[0]),
                                         ru(BLUE_COL_MIN[1], BLUE_COL_MAX[1]),
                                         ru(BLUE_COL_MIN[2], BLUE_COL_MAX[2]), 1)
            if emittor == 'BLOCK':
                self.emittors[idx][0] = 0
                self.emittors[idx][1] = (0, 0, 0, 1)
    '''
    
    def randomize(self):
        if FIXED_PARAMS_PATH and os.path.exists(FIXED_PARAMS_PATH):
            with open(FIXED_PARAMS_PATH) as _f:
                self._apply_fixed(json.load(_f))
            return

        self.smoothness = random.randrange(SMOOTHNESS_MIN, SMOOTHNESS_MAX)
        self.scale = ru(SCALE_MIN, SCALE_MAX)
        self.fov = ru(FOV_MIN, FOV_MAX)
        self.roughness = ru(ROUGH_MIN, ROUGH_MAX)
        self.gel_roughness = ru(GEL_ROUGHNESS_MIN, GEL_ROUGHNESS_MAX)
        self.gel_fac = ru(GEL_FAC_MIN, GEL_FAC_MAX)

        if self.length is None:
            self.length = ru(LENGTH_MIN, LENGTH_MAX)

        self.angle = 'str'
        self.light_type = 'long'

        # Top = 紫色（紅+藍混合）
        self.emittors[0][0] = ru(BLUE_STR_MIN, BLUE_STR_MAX)
        self.emittors[0][1] = (
            ru(RED_COL_MIN[0], RED_COL_MAX[0]),   # R: 0.9~1.0
            0.0,                                   # G: 0
            ru(BLUE_COL_MIN[2], BLUE_COL_MAX[2]), # B: 0.9~1.0
            1)

        # Bottom = 紫色
        self.emittors[1][0] = ru(BLUE_STR_MIN, BLUE_STR_MAX)
        self.emittors[1][1] = (
            ru(RED_COL_MIN[0], RED_COL_MAX[0]),
            0.0,
            ru(BLUE_COL_MIN[2], BLUE_COL_MAX[2]),
            1)

        # Left = 紅色
        self.emittors[2][0] = ru(RED_STR_MIN, RED_STR_MAX)
        self.emittors[2][1] = (
            ru(RED_COL_MIN[0], RED_COL_MAX[0]),
            ru(RED_COL_MIN[1], RED_COL_MAX[1]),
            ru(RED_COL_MIN[2], RED_COL_MAX[2]),
            1)

        # Right = 紅色
        self.emittors[3][0] = ru(RED_STR_MIN, RED_STR_MAX)
        self.emittors[3][1] = (
            ru(RED_COL_MIN[0], RED_COL_MAX[0]),
            ru(RED_COL_MIN[1], RED_COL_MAX[1]),
            ru(RED_COL_MIN[2], RED_COL_MAX[2]),
            1)

        # 綠燈各自獨立（random mode 用相同隨機綠色）
        green_color = (
            ru(GREEN_COL_MIN[0], GREEN_COL_MAX[0]),
            ru(GREEN_COL_MIN[1], GREEN_COL_MAX[1]),
            ru(GREEN_COL_MIN[2], GREEN_COL_MAX[2]),
        )
        self.lg_str   = ru(GREEN_STR_MIN, GREEN_STR_MAX)
        self.lg_color = green_color
        self.rg_str   = ru(GREEN_STR_MIN, GREEN_STR_MAX)
        self.rg_color = green_color
    
    def apply(self):
        global _LIGHT_INITIAL_MATRICES

        # 最先儲存初始位置（在 set_scale 之前）
        light_names = ['LightSurfaceBL', 'LightSurfaceTR',
                    'LightSurfaceTL', 'LightSurfaceBR']
        if not _LIGHT_INITIAL_MATRICES:
            for name in light_names:
                _LIGHT_INITIAL_MATRICES[name] = bpy.data.objects[name].matrix_world.copy()
            print('已儲存燈光初始位置')

        set_smoothness(self.smoothness)
        set_scale(self.scale)
        set_light_type(self.light_type)
        set_cam(self.fov, self.length)
        for obj_name, mat_name, str_val, color in [
            ('LightSurfaceRGreen',  'RGreenEmittor',  self.lg_str, self.lg_color),
            ('LightSurfaceLGreen', 'LGreenEmittor', self.rg_str, self.rg_color),
        ]:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                obj.hide_render = False
            mat = bpy.data.materials.get(mat_name)
            if mat:
                node = mat.node_tree.nodes.get('Emission')
                if node:
                    node.inputs['Color'].default_value    = (*color, 1.0)
                    node.inputs['Strength'].default_value = str_val

        # 設定 GelSurface 材質
        mat = bpy.data.materials['aluminum-specular-mat']
        nt = mat.node_tree
        nodes = nt.nodes
        glossy = nodes.get('Glossy BSDF')
        if glossy:
            glossy.inputs['Roughness'].default_value = self.gel_roughness
        mix_node = nodes.get('Mix Shader')
        if mix_node:
            mix_node.inputs['Fac'].default_value = self.gel_fac

        # === Dark contact: 用世界 Z 座標判斷是否為按壓區 ===
        # Gel 平的區域 z ≈ +0.06mm（背景）, contact 區 z ≈ -1mm（被壓進去）。
        # Map Range 算 factor (0=contact, 1=背景)，再用 Mix Color 在兩個顏色間混合：
        #   factor = 0 → CONTACT_DARK_COLOR (暗紫紅)
        #   factor = 1 → 白色 × DARK_BASE_GAIN (背景亮)
        if glossy is not None and not glossy.inputs['Color'].is_linked:
            geom = nodes.new('ShaderNodeNewGeometry')
            sep  = nodes.new('ShaderNodeSeparateXYZ')
            nt.links.new(geom.outputs['Position'], sep.inputs[0])

            mr = nodes.new('ShaderNodeMapRange')
            mr.clamp = True
            mr.interpolation_type = 'SMOOTHSTEP'
            mr.inputs['From Min'].default_value = CONTACT_Z_THRESH
            mr.inputs['From Max'].default_value = BG_Z_THRESH
            mr.inputs['To Min'].default_value   = 0.0
            mr.inputs['To Max'].default_value   = 1.0
            nt.links.new(sep.outputs['Z'], mr.inputs['Value'])

            mix = nodes.new('ShaderNodeMix')
            mix.data_type = 'RGBA'
            # RGBA 模式: inputs[0]=Factor, inputs[6]=A 色, inputs[7]=B 色; outputs[2]=Result
            mix.inputs[6].default_value = CONTACT_DARK_COLOR
            mix.inputs[7].default_value = (DARK_BASE_GAIN, DARK_BASE_GAIN, DARK_BASE_GAIN, 1.0)
            nt.links.new(mr.outputs['Result'], mix.inputs[0])
            nt.links.new(mix.outputs[2], glossy.inputs['Color'])

        bpy.context.scene.frame_set(0)

        # 從初始位置開始旋轉（不累積）
        rot_z = getattr(self, 'light_rot_z',
                        pi / 4 if self.angle == 'diag' else 0.0)
        rot_mat = Euler((0, 0, rot_z)).to_matrix().to_4x4()
        co_z = -0.0065 + 0.0011 / self.scale

        for name in light_names:
            bpy.data.objects[name].matrix_world = rot_mat @ _LIGHT_INITIAL_MATRICES[name]
            bpy.data.objects[name].location[2]  = co_z        # must set AFTER matrix_world
            bpy.data.objects[name].scale[1]     = self.scale  # must set AFTER matrix_world

        if self.light_type == 'long':
            set_emittor('BLEmittor', self.emittors[0][0], self.emittors[0][1])
            set_emittor('TREmittor', self.emittors[1][0], self.emittors[1][1])
            set_emittor('TLEmittor', self.emittors[2][0], self.emittors[2][1])
            set_emittor('BREmittor', self.emittors[3][0], self.emittors[3][1])
        else:
            set_emittor('BLEmittor', self.emittors[0][0] * 5, self.emittors[0][1])
            set_emittor('TREmittor', self.emittors[1][0] * 5, self.emittors[1][1])
            set_emittor('TLEmittor', self.emittors[2][0] * 5, self.emittors[2][1])
            set_emittor('BREmittor', self.emittors[3][0] * 5, self.emittors[3][1])

def get_depth(dir) -> None:
    import tempfile, glob

    scene = bpy.context.scene
    tree = scene.node_tree

    rl_node = next(n for n in tree.nodes if n.type == 'R_LAYERS')
    fo_node = tree.nodes.new('CompositorNodeOutputFile')
    fo_node.format.file_format = 'OPEN_EXR'
    fo_node.format.color_depth = '32'
    tmp_dir = tempfile.gettempdir()
    fo_node.base_path = tmp_dir
    fo_node.file_slots[0].path = 'gs_depth_tmp_'

    tree.links.new(rl_node.outputs['Depth'], fo_node.inputs[0])
    bpy.ops.render.render(write_still=False)

    exr_files = sorted(glob.glob(os.path.join(tmp_dir, 'gs_depth_tmp_*.exr')))
    exr_path = exr_files[-1]
    exr_img = bpy.data.images.load(exr_path)
    w, h = exr_img.size
    dmap = np.array(exr_img.pixels[:], dtype=np.float32).reshape(h, w, exr_img.channels)
    dmap = dmap[:, :, 0]

    dmap = np.rot90(dmap, k=2)
    dmap = np.fliplr(dmap)

    cam_z = abs(bpy.data.objects['Camera'].location[2])
    background_mask = dmap > (cam_z * 0.99)
    dmap[background_mask] = 0.0
    dmap = cam_z - dmap
    dmap[background_mask] = 0.0

    np.save(dir, dmap)

    tree.nodes.remove(fo_node)
    bpy.data.images.remove(exr_img)
    os.remove(exr_path)


def get_gt_depth(dir, obj_name) -> None:
    import tempfile, glob

    gel_objects = ['GelSurface', 'InterfaceSurface', 'EpoxySurface']
    gel_visibility = {}
    for name in gel_objects:
        if name in bpy.data.objects:
            gel_visibility[name] = bpy.data.objects[name].hide_render
            bpy.data.objects[name].hide_render = True

    bpy.data.objects[obj_name].hide_render = False

    scene = bpy.context.scene
    tree = scene.node_tree

    rl_node = next(n for n in tree.nodes if n.type == 'R_LAYERS')
    fo_node = tree.nodes.new('CompositorNodeOutputFile')
    fo_node.format.file_format = 'OPEN_EXR'
    fo_node.format.color_depth = '32'
    tmp_dir = tempfile.gettempdir()
    fo_node.base_path = tmp_dir
    fo_node.file_slots[0].path = 'gs_gt_depth_tmp_'

    tree.links.new(rl_node.outputs['Depth'], fo_node.inputs[0])
    bpy.ops.render.render(write_still=False)

    exr_files = sorted(glob.glob(os.path.join(tmp_dir, 'gs_gt_depth_tmp_*.exr')))
    exr_path = exr_files[-1]
    exr_img = bpy.data.images.load(exr_path)
    w, h = exr_img.size
    dmap = np.array(exr_img.pixels[:], dtype=np.float32).reshape(h, w, exr_img.channels)
    dmap = dmap[:, :, 0]

    dmap = np.rot90(dmap, k=2)
    dmap = np.fliplr(dmap)

    cam_z = abs(bpy.data.objects['Camera'].location[2])
    background_mask = dmap > (cam_z * 0.99)
    dmap[background_mask] = 0.0
    dmap = cam_z - dmap
    dmap[background_mask] = 0.0

    np.save(dir, dmap)

    tree.nodes.remove(fo_node)
    bpy.data.images.remove(exr_img)
    os.remove(exr_path)

    for name, vis in gel_visibility.items():
        bpy.data.objects[name].hide_render = vis

    bpy.data.objects[obj_name].hide_render = True


def get_rgb_image(dir, obj_name, cam_z_override=None) -> None:
    """Render RGB image of the object from the GelSight camera viewpoint.
    Gel surface is hidden; all emittors are set to neutral white for clean appearance.
    cam_z_override: if given, temporarily move camera to this height (metres, positive value).
                    e.g. cam_z_override=0.05 pulls the camera back for a wider view.
                    If None, keeps the same height as the tactile render.
    """
    emittor_names = [
        'BLEmittor', 'TREmittor', 'TLEmittor', 'BREmittor',
        'RGreenEmittor', 'LGreenEmittor',
    ]
    WHITE = (1.0, 1.0, 1.0, 1.0)
    RGB_STRENGTH = 5.0

    # Save current emittor state
    saved_emittors = {}
    for mat_name in emittor_names:
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            continue
        node = mat.node_tree.nodes.get('Emission')
        if node is None:
            continue
        saved_emittors[mat_name] = (
            tuple(node.inputs['Color'].default_value),
            node.inputs['Strength'].default_value,
        )
        node.inputs['Color'].default_value = WHITE
        node.inputs['Strength'].default_value = RGB_STRENGTH

    # Optionally adjust camera height
    cam = bpy.data.objects['Camera']
    orig_cam_z = cam.location[2]
    if cam_z_override is not None:
        cam.location[2] = -abs(cam_z_override)

    # Hide gel + light panels, show object
    hidden_objects = [
        'GelSurface', 'InterfaceSurface', 'EpoxySurface',
        'LightSurfaceBL', 'LightSurfaceTR',
        'LightSurfaceTL', 'LightSurfaceBR',
        'LightSurfaceRGreen', 'LightSurfaceLGreen',
    ]
    gel_visibility = {}
    for name in hidden_objects:
        if name in bpy.data.objects:
            gel_visibility[name] = bpy.data.objects[name].hide_render
            bpy.data.objects[name].hide_render = True
    bpy.data.objects[obj_name].hide_render = False

    # Gray world background
    world = bpy.data.worlds.get('World')
    orig_bg_color = None
    if world and world.node_tree:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            orig_bg_color = tuple(bg_node.inputs['Color'].default_value)
            bg_node.inputs['Color'].default_value    = (0.45, 0.45, 0.45, 1.0)
            bg_node.inputs['Strength'].default_value = 1.0

    # Disable compositor (avoids depth-tint from GelSight compositor nodes)
    orig_use_nodes = bpy.context.scene.use_nodes
    bpy.context.scene.use_nodes = False

    # Render RGB PNG
    orig_filepath = bpy.context.scene.render.filepath
    bpy.context.scene.render.filepath = dir
    bpy.context.scene.frame_set(0)
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.render.filepath = orig_filepath

    # Restore compositor, camera, and world background
    bpy.context.scene.use_nodes = orig_use_nodes
    cam.location[2] = orig_cam_z
    if orig_bg_color is not None and world and world.node_tree:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value = orig_bg_color

    # Restore gel visibility
    for name, vis in gel_visibility.items():
        bpy.data.objects[name].hide_render = vis
    bpy.data.objects[obj_name].hide_render = True

    # Restore emittor colours / strengths
    for mat_name, (color, strength) in saved_emittors.items():
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            continue
        node = mat.node_tree.nodes.get('Emission')
        if node is None:
            continue
        node.inputs['Color'].default_value = color
        node.inputs['Strength'].default_value = strength


dir = os.path.dirname(bpy.data.filepath)
sys.path.append(dir)
render_dir = os.environ.get('GELSIGHT_RENDER_DIR', os.path.join(dir, 'renders'))

# ── RGB render parameters ──────────────────────────────────────
RGB_CAM_Z       = -0.085
RGB_DOF_FOCUS   = 0.085
RGB_DOF_FSTOP   = _rgb_bo.get('dof_fstop',      1.2)
RGB_WORLD_STR   = _rgb_bo.get('world_strength',  2.0)
RGB_BARREL_K1   = _rgb_bo.get('barrel_k1',       0.07)
RGB_VIGNETTE    = _rgb_bo.get('vignette',         0.25)
RGB_BLUR_SIGMA  = _rgb_bo.get('blur_sigma',       1.25)
RGB_BLUR_FALLOFF = 1.5
RGB_HIDE_NAMES  = ['GelSurface', 'InterfaceSurface', 'EpoxySurface',
                   'LightSurfaceBL', 'LightSurfaceTR',
                   'LightSurfaceTL', 'LightSurfaceBR',
                   'LightSurfaceRGreen', 'LightSurfaceLGreen']

# ── Saturation enhancement for tactile renders ─────────────────
TACTILE_SAT_BOOST = 1.4   # 1.0 = no change, 1.4 = 40% more saturated

def _boost_saturation(png_path, factor):
    img_bpy = bpy.data.images.load(png_path)
    W, H = img_bpy.size[0], img_bpy.size[1]
    px = np.array(img_bpy.pixels[:], dtype=np.float32).reshape(H, W, 4)
    bpy.data.images.remove(img_bpy)
    rgb = px[:, :, :3]
    # RGB → HSV
    cmax = rgb.max(axis=2); cmin = rgb.min(axis=2); delta = cmax - cmin
    v = cmax
    s = np.where(cmax > 0, delta / cmax, 0.0)
    h = np.zeros_like(v)
    m = delta > 0
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    mr = m & (cmax == r); h[mr] = (60 * ((g[mr]-b[mr]) / delta[mr])) % 360
    mg = m & (cmax == g); h[mg] = 60 * ((b[mg]-r[mg]) / delta[mg] + 2)
    mb = m & (cmax == b); h[mb] = 60 * ((r[mb]-g[mb]) / delta[mb] + 4)
    # Boost S
    s = np.clip(s * factor, 0.0, 1.0)
    # HSV → RGB
    h6 = h / 60.0; i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s); q = v * (1 - f*s); t = v * (1 - (1-f)*s)
    rgb_out = np.zeros_like(rgb)
    for ii, (r0,g0,b0) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        mask = i == ii
        rgb_out[:,:,0][mask] = r0[mask]
        rgb_out[:,:,1][mask] = g0[mask]
        rgb_out[:,:,2][mask] = b0[mask]
    out_px = np.ones((H, W, 4), dtype=np.float32)
    out_px[:,:,:3] = np.clip(rgb_out, 0, 1)
    out_img = bpy.data.images.new('_sat_tmp', W, H, alpha=False)
    out_img.pixels = out_px.flatten().tolist()
    out_img.filepath_raw = png_path
    out_img.file_format = 'PNG'
    out_img.save()
    bpy.data.images.remove(out_img)
mesh_dir = os.path.join(dir, 'meshes')

if __name__ == '__main__':

    # ── Background-only render mode (for Bayesian optimization) ──────────────
    if BG_RENDER_PATH:
        _sensor = create_sensor()
        _sensor.apply()
        move_object('IndenterSurface', (0, 0, -1), (0, 0, 0))
        bpy.context.scene.render.filepath = BG_RENDER_PATH
        bpy.context.scene.frame_set(0)
        bpy.ops.render.render(write_still=True)
        print(f'BG render saved to: {BG_RENDER_PATH}.png')
        try:
            bpy.ops.wm.quit_blender()
        except Exception:
            import sys as _sys; _sys.exit(0)
    # ─────────────────────────────────────────────────────────────────────────

    sensors = []
    if not CONTINUE:
        session_backup = None
        _session_path = os.path.join(render_dir, 'session.json')
        if os.path.exists(_session_path):
            with open(_session_path) as f:
                session_backup = f.read()

        if os.path.exists(render_dir):
            for d in os.listdir(render_dir):
                if 'sensor' in d:
                    shutil.rmtree(os.path.join(render_dir, d), ignore_errors=True)
        else:
            os.mkdir(render_dir)

        if session_backup is not None:
            with open(_session_path, 'w') as f:
                f.write(session_backup)

        for idx in range(NUM_SENSORS):
            idx_formatted = '{0:04}'.format(idx)
            sensor_dir = os.path.join(render_dir, f'sensor_{idx_formatted}')
            os.mkdir(sensor_dir)
            os.mkdir(os.path.join(sensor_dir, 'calibration'))
            os.mkdir(os.path.join(sensor_dir, 'samples'))
            os.mkdir(os.path.join(sensor_dir, 'raw_data'))
            os.mkdir(os.path.join(sensor_dir, 'rgb'))
            sensor_txt_dir = os.path.join(sensor_dir, 'parameters.txt')
            sensors.append(create_sensor(write_dir=sensor_txt_dir))

    else:
        if os.path.exists(render_dir):
            sensor_dirs = [d for d in os.listdir(render_dir) if 'sensor' in d]
            sensor_dirs.sort()

            if len(sensor_dirs) == 0:
                for idx in range(NUM_SENSORS):
                    idx_formatted = '{0:04}'.format(idx)
                    sensor_dir = os.path.join(render_dir, f'sensor_{idx_formatted}')
                    os.makedirs(os.path.join(sensor_dir, 'calibration'), exist_ok=True)
                    os.makedirs(os.path.join(sensor_dir, 'samples'), exist_ok=True)
                    os.makedirs(os.path.join(sensor_dir, 'raw_data'), exist_ok=True)
                    os.makedirs(os.path.join(sensor_dir, 'rgb'), exist_ok=True)
                    sensor_txt_dir = os.path.join(sensor_dir, 'parameters.txt')
                    sensors.append(create_sensor(write_dir=sensor_txt_dir))
            else:
                for sensor_dir in sensor_dirs:
                    sensor_txt_dir = os.path.join(render_dir, sensor_dir, 'parameters.txt')
                    sensors.append(create_sensor(read_dir=sensor_txt_dir))
                    full_sensor_dir = os.path.join(render_dir, sensor_dir)
                    os.makedirs(os.path.join(full_sensor_dir, 'calibration'), exist_ok=True)
                    os.makedirs(os.path.join(full_sensor_dir, 'samples'), exist_ok=True)
                    os.makedirs(os.path.join(full_sensor_dir, 'raw_data'), exist_ok=True)
                    os.makedirs(os.path.join(full_sensor_dir, 'rgb'), exist_ok=True)
        else:
            os.mkdir(render_dir)
            for idx in range(NUM_SENSORS):
                idx_formatted = '{0:04}'.format(idx)
                sensor_dir = os.path.join(render_dir, f'sensor_{idx_formatted}')
                os.mkdir(sensor_dir)
                os.mkdir(os.path.join(sensor_dir, 'calibration'))
                os.mkdir(os.path.join(sensor_dir, 'samples'))
                os.mkdir(os.path.join(sensor_dir, 'raw_data'))
                os.mkdir(os.path.join(sensor_dir, 'rgb'))
                sensor_txt_dir = os.path.join(sensor_dir, 'parameters.txt')
                sensors.append(create_sensor(write_dir=sensor_txt_dir))

    # generate calibration
    calibration_objects = ['IndenterSurface', 'Cube']
    for sensor_idx, sensor in enumerate(sensors):
        sensor_idx_formatted = '{0:04}'.format(sensor_idx)
        sensor_dir = os.path.join(render_dir, f'sensor_{sensor_idx_formatted}')

        if CONTINUE:
            if len(os.listdir(os.path.join(sensor_dir, 'calibration'))) < (NUM_CALIBRATION * len(calibration_objects) + 1):
                shutil.rmtree(os.path.join(sensor_dir, 'calibration'))
                os.mkdir(os.path.join(sensor_dir, 'calibration'))
            else:
                continue

        sensor.apply()

        overall_calib_idx = 0
        calib_idx_formatted = '{0:04}'.format(overall_calib_idx)
        calib_out = os.path.join(sensor_dir, 'calibration', calib_idx_formatted)
        bpy.context.scene.render.filepath = calib_out
        move_object('IndenterSurface', (0, 0, -1), (0, 0, 0))
        bpy.context.scene.frame_set(0)
        bpy.ops.render.render(write_still=True)
        _boost_saturation(calib_out + '.png', TACTILE_SAT_BOOST)
        overall_calib_idx += 1

        qt = (sensor.length * 2) / 3
        CALIB_X = [qt, 0, -qt, qt, 0, -qt, qt, 0, -qt]
        CALIB_Y = [qt, qt, qt, 0, 0, 0, -qt, -qt, -qt]

        for obj_idx, calib_obj in enumerate(calibration_objects):
            for calib_idx in range(NUM_CALIBRATION):
                calib_idx_formatted = '{0:04}'.format(overall_calib_idx)
                calib_out2 = os.path.join(sensor_dir, 'calibration', calib_idx_formatted)
                bpy.context.scene.render.filepath = calib_out2

                x = ru(-0.001, 0.001) + CALIB_X[calib_idx]
                y = ru(-0.001, 0.001) + CALIB_Y[calib_idx]
                z = ru(CALIB_DEPTH_MIN, CALIB_DEPTH_MAX)
                a_x = pi / 4
                a_y = pi / 4
                a_z = ru(0, 2 * pi)

                move_object(calib_obj, (x, y, z), (a_x, a_y, a_z))
                bpy.context.scene.frame_set(0)
                bpy.ops.render.render(write_still=True)
                _boost_saturation(calib_out2 + '.png', TACTILE_SAT_BOOST)
                overall_calib_idx += 1

    # setup platform for RGB renders
    _setup_platform()

    # import meshes
    obj_dir = [f for f in sorted(os.listdir(mesh_dir)) if f.lower().endswith('.obj')]
    obj_name_map = {}  # filename-stem → actual Blender object name
    for obj_idx, obj_file in enumerate(obj_dir):
        _before = set(bpy.data.objects.keys())
        bpy.ops.wm.obj_import(filepath=os.path.join(mesh_dir, obj_file), directory=mesh_dir,
                              files=[{"name": obj_file}])
        _new = [o.name for o in bpy.data.objects if o.name not in _before]
        stem = obj_file[:-4]
        imported_name = _new[0] if _new else stem
        obj_dir[obj_idx] = imported_name
        obj_name_map[stem] = imported_name
        bpy.data.objects[imported_name].hide_render = True

        # Assign blue plastic material for RGB renders
        mat = bpy.data.materials.new(name=f'plastic_blue_{imported_name}')
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (
                _rgb_bo.get('obj_r', 0.03),
                _rgb_bo.get('obj_g', 0.18),
                _rgb_bo.get('obj_b', 0.75), 1.0)
            bsdf.inputs['Roughness'].default_value = _rgb_bo.get('obj_roughness', 0.35)
            bsdf.inputs['Specular IOR Level'].default_value = 0.5
            bsdf.inputs['Metallic'].default_value = 0.0
        obj_blender = bpy.data.objects[imported_name]
        obj_blender.data.materials.clear()
        obj_blender.data.materials.append(mat)

    # remove incomplete render batch
    if CONTINUE:
        sensor_idx_formatted = '{0:04}'.format(0)
        samples_dir = os.path.join(render_dir, f'sensor_{sensor_idx_formatted}', 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        first_sensor_samples = os.listdir(samples_dir)

        sensor_idx_formatted = '{0:04}'.format(NUM_SENSORS - 1)
        last_samples_dir = os.path.join(render_dir, f'sensor_{sensor_idx_formatted}', 'samples')
        os.makedirs(last_samples_dir, exist_ok=True)
        last_sensor_samples = os.listdir(last_samples_dir)

        overall_idx = min(len(first_sensor_samples), len(last_sensor_samples))
        sample_idx_formatted = '{0:04}'.format(overall_idx)

        if len(first_sensor_samples) != len(last_sensor_samples):
            for sensor_idx, sensor in enumerate(sensors):
                sensor_idx_formatted = '{0:04}'.format(sensor_idx)
                sample_dir = os.path.join(render_dir, f'sensor_{sensor_idx_formatted}', 'samples',
                                          f'{sample_idx_formatted}.png')
                dmap_dir = os.path.join(render_dir, f'sensor_{sensor_idx_formatted}', 'raw_data',
                                        f'{sample_idx_formatted}.npy')
                if os.path.exists(sample_dir):
                    os.remove(sample_dir)
                if os.path.exists(dmap_dir):
                    os.remove(dmap_dir)
    else:
        overall_idx = 0

    # load session
    session_path = os.path.join(render_dir, 'session.json')

    if os.path.exists(session_path):
        with open(session_path) as f:
            session = json.load(f)

        fixed_rotation = tuple(session.get('base_rotation', session.get('fixed_rotation')))
        rotation_step  = session.get('rotation_step', 0.15)
        fixed_scale    = session['fixed_scale']
        obj            = obj_name_map.get(session['obj'], session['obj'])

        X_MIN           = session.get('X_MIN',           X_MIN)
        X_MAX           = session.get('X_MAX',           X_MAX)
        Y_MIN           = session.get('Y_MIN',           Y_MIN)
        Y_MAX           = session.get('Y_MAX',           Y_MAX)
        NUM_OBJ_SAMPLES = session.get('NUM_OBJ_SAMPLES', NUM_OBJ_SAMPLES)
        OBJ_DEPTH_MIN   = session.get('OBJ_DEPTH_MIN',   OBJ_DEPTH_MIN)
        OBJ_DEPTH_MAX   = session.get('OBJ_DEPTH_MAX',   OBJ_DEPTH_MAX)

        print(f'繼續上次 session: {obj}')
        print(f'  base_rotation={[round(x, 3) for x in fixed_rotation]}')
        print(f'  rotation_step={rotation_step:.3f}')
        print(f'  scale={fixed_scale:.4f}')
        print(f'  samples={NUM_OBJ_SAMPLES}, X=[{X_MIN},{X_MAX}], Y=[{Y_MIN},{Y_MAX}]')
    else:
        fixed_rotation = (ru(0, 2 * pi), ru(0, 2 * pi), ru(0, 2 * pi))
        rotation_step  = 0.15
        obj            = 'screw'
        fixed_scale    = max(bpy.data.objects[obj].dimensions) / ru(OBJ_SIZE_MIN, OBJ_SIZE_MAX)

        session = {
            'obj': obj,
            'base_rotation': list(fixed_rotation),
            'rotation_step': rotation_step,
            'fixed_scale': fixed_scale,
        }
        with open(session_path, 'w') as f:
            json.dump(session, f, indent=2)
        print(f'新 session: {obj}, fixed_scale={fixed_scale:.4f}')

    # generate samples
    valid_cells = session.get('valid_cells', None)
    sensor_width = session.get('_sensor_length_mm', 10.0) / 1000.0 * 2
    z_anchor = session.get('z_anchor', None)
    if z_anchor is not None:
        print(f'  使用全域 z_anchor = {z_anchor*1000:.3f}mm')

    if valid_cells is not None:
        NUM_OBJ_SAMPLES = len(valid_cells)
        sensor_step = session.get('_sensor_step', 0.006)
        jitter = sensor_step * 0.15
        print(f'valid_cells 模式：共 {NUM_OBJ_SAMPLES} 個有效格子')
    else:
        grid_n = math.ceil(math.sqrt(NUM_OBJ_SAMPLES))
        sensor_step = (X_MAX - X_MIN) / grid_n * 0.8
        jitter = sensor_step * 0.15
        print(f'方形 grid 模式：{NUM_OBJ_SAMPLES} 個 sample')

    for sample_idx in range(overall_idx, NUM_OBJ_SAMPLES):
        bpy.data.objects[obj].scale = (1 / fixed_scale, 1 / fixed_scale, 1 / fixed_scale)

        if valid_cells is not None:
            cell = valid_cells[sample_idx]

            # ✅ 相容 dict 格式（新版）和 list 格式（舊版）
            if isinstance(cell, dict):
                cx  = cell['cx']
                cy  = cell['cy']
                # 同一 session 統一使用 base_rotation，不做 per-cell 法向量補正
                a_x = fixed_rotation[0]
                a_y = fixed_rotation[1]
                a_z = fixed_rotation[2]
                # ✅ 格子專屬按壓深度
                cell_depth_min = cell.get('depth_min', OBJ_DEPTH_MIN)
                cell_depth_max = cell.get('depth_max', OBJ_DEPTH_MAX)
            else:
                # 舊版 list: [gx, gy, cx, cy, rx, ry, rz, depth_min, depth_max]
                cx  = cell[2]
                cy  = cell[3]
                a_x = fixed_rotation[0]
                a_y = fixed_rotation[1]
                a_z = fixed_rotation[2]
                cell_depth_min = cell[7] if len(cell) > 7 else OBJ_DEPTH_MIN
                cell_depth_max = cell[8] if len(cell) > 8 else OBJ_DEPTH_MAX

            x = cx + ru(-jitter, jitter)
            y = cy + ru(-jitter, jitter)
            z = ru(cell_depth_min, cell_depth_max)  # ✅ 用格子專屬深度

        else:
            grid_x = (sample_idx % grid_n) * sensor_step + X_MIN + sensor_step / 2
            grid_y = (sample_idx // grid_n) * sensor_step + Y_MIN + sensor_step / 2
            x = grid_x + ru(-jitter, jitter)
            y = grid_y + ru(-jitter, jitter)
            delta = rotation_step * sample_idx
            a_x = fixed_rotation[0] + delta * 0.3
            a_y = fixed_rotation[1] + delta * 0.1
            a_z = fixed_rotation[2] + delta * 0.05
            z = ru(OBJ_DEPTH_MIN, OBJ_DEPTH_MAX)

        move_object_at_xy(obj, (x, y, z), (a_x, a_y, a_z), sensor_width, z_anchor=z_anchor)

        overall_idx_formatted = '{0:04}'.format(overall_idx)

        for sensor_idx, sensor in enumerate(sensors):
            sensor_idx_formatted = '{0:04}'.format(sensor_idx)
            sensor_dir = os.path.join(render_dir, f'sensor_{sensor_idx_formatted}')
            sensor.apply()

            tactile_path = os.path.join(sensor_dir, 'samples', overall_idx_formatted)
            bpy.context.scene.render.filepath = tactile_path
            bpy.context.scene.frame_set(0)
            bpy.ops.render.render(write_still=True)
            _boost_saturation(tactile_path + '.png', TACTILE_SAT_BOOST)

            obj_data = bpy.data.objects[obj]
            world_matrix = obj_data.matrix_world
            world_matrix_inv = world_matrix.inverted()

            pose = {
                'obj_name': obj,
                'sample_x': x,
                'sample_y': y,
                'sample_z': z,
                'location': list(obj_data.location),
                'rotation_euler': list(obj_data.rotation_euler),
                'scale': list(obj_data.scale),
                'fixed_scale': fixed_scale,
                'world_matrix': [list(row) for row in world_matrix],
                'world_matrix_inv': [list(row) for row in world_matrix_inv],
                'camera_fov': sensor.fov,
                'camera_length': sensor.length,
                'camera_location': list(bpy.data.objects['Camera'].location),
            }
            pose_path = os.path.join(sensor_dir, 'raw_data', f'{overall_idx_formatted}_pose.json')
            with open(pose_path, 'w') as f:
                json.dump(pose, f, indent=2)

            get_depth(os.path.join(sensor_dir, 'raw_data', f'{overall_idx_formatted}.npy'))
            get_gt_depth(os.path.join(sensor_dir, 'raw_data', f'{overall_idx_formatted}_gt.npy'), obj)

            # RGB render with platform background + gel FX
            rgb_dir = os.path.join(sensor_dir, 'rgb')
            os.makedirs(rgb_dir, exist_ok=True)
            render_rgb_sample(obj, sensor.fov,
                              os.path.join(rgb_dir, overall_idx_formatted))

        overall_idx += 1

    # remove meshes
    for obj_name in obj_dir:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[obj_name].select_set(True)
        bpy.ops.object.delete()

    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        import sys as _sys
        _sys.exit(0)


'''
mat = bpy.data.materials['aluminum-specular-mat']
nodes = mat.node_tree.nodes
links = mat.node_tree.links

nodes.clear()

# Glossy 反光
glossy = nodes.new('ShaderNodeBsdfGlossy')
glossy.inputs['Roughness'].default_value = 0.02
glossy.location = (-200, 100)

# Transparent 透明
transparent = nodes.new('ShaderNodeBsdfTransparent')
transparent.location = (-200, -100)

# Mix：調整 Fac 控制透明比例
mix = nodes.new('ShaderNodeMixShader')
mix.inputs['Fac'].default_value = 0.7  # 0=全透明, 1=全反光
mix.location = (0, 0)

output = nodes.new('ShaderNodeOutputMaterial')
output.location = (200, 0)

links.new(transparent.outputs[0], mix.inputs[1])
links.new(glossy.outputs[0], mix.inputs[2])
links.new(mix.outputs[0], output.inputs[0])

mat.blend_method = 'BLEND'
print('完成')
'''