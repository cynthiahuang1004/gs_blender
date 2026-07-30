# === Colab Cell: SCP renders from server to Google Drive ===
# Paste this into a Colab cell and run it.
# It will ask for your SSH password.

from google.colab import drive
drive.mount('/content/drive')

import os

SERVER = 'shared@141.212.82.43'
REMOTE_DIR = '/media/hdd2/ihsuan/gs_blender/renders'
LOCAL_DIR = '/content/drive/MyDrive/HDR_Lab/gs_blender_renders'

OBJECTS = [
    'edge', 'hex_key', 'marble',
    'pattern_01_2_lines_angle_2', 'pattern_01_2_lines_angle_3',
    'pattern_04_3_lines_angle_2',
    'pattern_32', 'pattern_33', 'pattern_35', 'pattern_37',
    'peg2', 'peg3', 'ping_pong',
]

os.makedirs(LOCAL_DIR, exist_ok=True)

# Install sshpass for non-interactive password auth
os.system('apt-get install -y sshpass > /dev/null 2>&1')

# Ask for password once
import getpass
password = getpass.getpass('SSH password for shared@141.212.82.43: ')

for i, obj in enumerate(OBJECTS):
    dst = os.path.join(LOCAL_DIR, obj)
    os.makedirs(dst, exist_ok=True)
    print(f'[{i+1}/{len(OBJECTS)}] Downloading {obj}...')
    cmd = (f'sshpass -p "{password}" scp -r -o StrictHostKeyChecking=no '
           f'{SERVER}:{REMOTE_DIR}/{obj}/ {dst}/')
    ret = os.system(cmd)
    if ret != 0:
        print(f'  ERROR: scp failed for {obj} (exit {ret})')
    else:
        print(f'  Done: {obj}')

print(f'\nAll done! Files in {LOCAL_DIR}')
