# ============================================================
# GelSight Dataset Render — Google Colab Setup
# Runtime: GPU → A100  (Runtime > Change runtime type)
# ============================================================

# %% [Cell 1] Check GPU
import subprocess
subprocess.run(['nvidia-smi'], check=False)

# %% [Cell 2] Mount Google Drive (output will be saved here)
from google.colab import drive
drive.mount('/content/drive')

DRIVE_OUT = '/content/drive/MyDrive/gs_blender_renders'   # change if needed
import os; os.makedirs(DRIVE_OUT, exist_ok=True)
print(f'Output → {DRIVE_OUT}')

# %% [Cell 3] Install Blender 4.2 (Linux)
BLENDER_TAR = 'blender-4.2.0-linux-x64.tar.xz'
BLENDER_URL = f'https://mirrors.dotsrc.org/blender/release/Blender4.2/{BLENDER_TAR}'
BLENDER_DIR = '/opt/blender-4.2.0-linux-x64'
BLENDER_BIN = f'{BLENDER_DIR}/blender'

if not os.path.exists(BLENDER_BIN):
    subprocess.run(['wget', '-q', BLENDER_URL, '-O', f'/tmp/{BLENDER_TAR}'], check=True)
    subprocess.run(['tar', '-xf', f'/tmp/{BLENDER_TAR}', '-C', '/opt/'], check=True)

subprocess.run([BLENDER_BIN, '--version'], check=True)
print(f'Blender ready: {BLENDER_BIN}')

# %% [Cell 4] Clone repo
REPO = 'https://github.com/cynthiahuang1004/gs_blender.git'
PROJECT = '/content/gs_blender'

if not os.path.exists(PROJECT):
    subprocess.run(['git', 'clone', REPO, PROJECT], check=True)
else:
    subprocess.run(['git', '-C', PROJECT, 'pull'], check=True)

os.chdir(PROJECT)
print(f'Project ready: {PROJECT}')

# %% [Cell 5] Patch render_dataset.py — Linux Blender path + Drive output
from pathlib import Path

rd = Path('render_dataset.py').read_text()

# Fix Blender path
rd = rd.replace(
    r"Path(r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe')",
    f"Path('{BLENDER_BIN}')"
)

# Fix output root → Google Drive
rd = rd.replace(
    "RENDERS_ROOT = SCRIPT_DIR / 'renders'",
    f"RENDERS_ROOT = Path('{DRIVE_OUT}')"
)

Path('render_dataset.py').write_text(rd)
print('render_dataset.py patched')

# %% [Cell 6] Patch scripting.py — enable CUDA for A100
CUDA_PATCH = '''
# ── Colab: force CUDA (A100 does not support OptiX) ───────────
import bpy as _bpy_colab, os as _os_colab
if _os_colab.environ.get('COLAB_GPU'):
    try:
        _prefs = _bpy_colab.context.preferences.addons['cycles'].preferences
        _prefs.compute_device_type = 'CUDA'
        _prefs.get_devices()
        for _d in _prefs.devices:
            _d.use = (_d.type == 'CUDA')
        _bpy_colab.context.scene.cycles.device = 'GPU'
        print('[colab] CUDA GPU enabled')
    except Exception as _e:
        print(f'[colab] CUDA warning: {_e}')
'''

sc = Path('scripting.py').read_text()
sc = sc.replace('import bpy\n', 'import bpy\n' + CUDA_PATCH, 1)
Path('scripting.py').write_text(sc)
print('scripting.py patched for CUDA')

# %% [Cell 7] Copy session.json files from repo to Drive output
import shutil, json

# session.json files are committed to the repo under renders/
repo_renders = Path(PROJECT) / 'renders'
for obj_dir in sorted(repo_renders.iterdir()):
    if not obj_dir.is_dir():
        continue
    for sess_dir in sorted(obj_dir.iterdir()):
        if not sess_dir.is_dir() or not sess_dir.name.startswith('session_'):
            continue
        sj = sess_dir / 'session.json'
        if not sj.exists():
            continue
        dst = Path(DRIVE_OUT) / obj_dir.name / sess_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sj, dst / 'session.json')

sessions = list(Path(DRIVE_OUT).glob('*/session_*/session.json'))
print(f'Session files ready: {len(sessions)} sessions')

# %% [Cell 8] Run — 4 workers on A100
# Adjust --gpus based on how many workers you want (4 recommended for A100)
N_WORKERS = 4
gpus_arg = ','.join(['0'] * N_WORKERS)

import os
env = os.environ.copy()
env['COLAB_GPU'] = '1'   # trigger CUDA patch in scripting.py

proc = subprocess.run(
    ['python', 'render_dataset.py', '--gpus', gpus_arg, '--reverse'],
    env=env,
    cwd=PROJECT,
)
print(f'Done  rc={proc.returncode}')

# %% [Cell 9] (Optional) Run only specific objects
# subprocess.run(['python', 'render_dataset.py', '--gpus', gpus_arg,
#                 '--obj', 'button', 'edge'], env=env, cwd=PROJECT)
