"""
scripting_test_lights.py
每盞燈給不同顏色，render 一張背景圖，確認各燈對應 image 哪個區域。

顏色對應：
  BLEmittor        → 紅色   (1, 0, 0)
  TREmittor     → 綠色   (0, 1, 0)
  TLEmittor       → 藍色   (0, 0, 1)
  BREmittor      → 黃色   (1, 1, 0)
  RGreenEmittor  → 青色   (0, 1, 1)
  LGreenEmittor → 洋紅   (1, 0, 1)

Usage:
  & "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" `
    --background gelsight_sampler.blend `
    --python scripting_test_lights.py
"""

import bpy, os, sys
from math import pi, tan

SCRIPT_DIR  = os.path.dirname(bpy.data.filepath)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'bo_results', 'light_position_test.png')
STR = 80.0   # 統一強度（高，讓顏色清楚可見）

# ── 設定 4 個主燈顏色 ──────────────────────────────────────────
def set_emittor(mat_name, strength, color_rgb):
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        print(f'WARNING: material {mat_name} not found')
        return
    node = mat.node_tree.nodes.get('Emission')
    if node:
        node.inputs['Color'].default_value    = (*color_rgb, 1.0)
        node.inputs['Strength'].default_value = strength

set_emittor('BLEmittor',    STR, (1, 0, 0))   # 紅
set_emittor('TREmittor', STR, (0, 1, 0))   # 綠
set_emittor('TLEmittor',   STR, (0, 0, 1))   # 藍
set_emittor('BREmittor',  STR, (1, 1, 0))   # 黃

# ── 設定 2 個 Green side 燈 ────────────────────────────────────
for mat_name, color in [('RGreenEmittor',  (0, 1, 1)),   # 青
                         ('LGreenEmittor', (1, 0, 1))]:  # 洋紅
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        print(f'WARNING: material {mat_name} not found')
        continue
    node = mat.node_tree.nodes.get('Emission')
    if node:
        node.inputs['Color'].default_value    = (*color, 1.0)
        node.inputs['Strength'].default_value = STR

# 確保 green lights 可見
for name in ['LightSurfaceRGreen', 'LightSurfaceLGreen']:
    obj = bpy.data.objects.get(name)
    if obj:
        obj.hide_render = False

# ── Gel 材質調成漫反射（diffuse），讓顏色均勻散射到鏡頭 ──────────
mat = bpy.data.materials.get('aluminum-specular-mat')
if mat:
    nodes = mat.node_tree.nodes
    glossy = nodes.get('Glossy BSDF')
    if glossy:
        glossy.inputs['Roughness'].default_value = 0.5   # 接近完全漫反射
    mix_node = nodes.get('Mix Shader')
    if mix_node:
        mix_node.inputs['Fac'].default_value = 0.5      # 幾乎不透明

# ── 把物體移出畫面（只看背景）──────────────────────────────────
indenter = bpy.data.objects.get('IndenterSurface')
if indenter:
    indenter.location = (0, 0, -1)
    bpy.data.objects['GelSurface'].modifiers["Shrinkwrap"].target = indenter

# ── Render ─────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
bpy.context.scene.render.filepath = OUTPUT_PATH
bpy.context.scene.frame_set(0)
bpy.ops.render.render(write_still=True)
print(f'\n✅ 輸出: {OUTPUT_PATH}')
print('顏色對應:')
print('  紅   = BLEmittor')
print('  綠   = TREmittor')
print('  藍   = TLEmittor')
print('  黃   = BREmittor')
print('  青   = RGreenEmittor (side)')
print('  洋紅 = LGreenEmittor (side)')

try:
    bpy.ops.wm.quit_blender()
except Exception:
    sys.exit(0)
