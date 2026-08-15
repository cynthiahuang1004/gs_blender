"""
merge_extended_renders.py
=========================
Merge cloud-rendered extension samples (0200-0799) into renders_v3.

Steps:
  1. Extract each <obj>.tar from renders_v3_tar_600/ into a staging dir
  2. Verify every target session has 600 complete samples (idx 200-799):
       samples/*.png, rgb/*.png, raw_data/*.npy, *_gt.npy, *_pose.json
     (rgb_contact is NOT rendered for the extension — not required)
  3. Copy sample files into renders_v3/<obj>/<sess>/sensor_0000/...
       - session.json / .bak200 in the tar are IGNORED (local ones are the
         800-cell canonical versions)
       - never overwrites: aborts if any destination file already exists
  4. Print summary; run post_process_fast.py separately afterwards.

Usage:
    python merge_extended_renders.py            # verify + merge
    python merge_extended_renders.py --dry-run  # verify only
"""
import os, sys, json, tarfile, shutil, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAR_DIR = os.path.join(SCRIPT_DIR, 'renders_v3_tar_600')
STAGING = os.path.join(SCRIPT_DIR, 'renders_v3_600_staging')
DEST = os.path.join(SCRIPT_DIR, 'renders_v3')

TARGETS = {
    'pattern_01_2_lines_angle_1_2': ['session_000', 'session_001', 'session_011'],
    'pattern_01_2_lines_angle_2':   ['session_002', 'session_003', 'session_004'],
    'pattern_01_2_lines_angle_3':   ['session_000', 'session_001', 'session_011'],
    'pattern_04_3_lines_angle_1':   ['session_000', 'session_001', 'session_011'],
    'pattern_04_3_lines_angle_2':   ['session_008', 'session_009', 'session_010'],
    'pattern_06_5_lines_angle_1':   ['session_000', 'session_001', 'session_011'],
    'pattern_31_rod':               ['session_002', 'session_003', 'session_004'],
    'pattern_32':                   ['session_002', 'session_003', 'session_004'],
    'pattern_33':                   ['session_002', 'session_003', 'session_004'],
    'pattern_35':                   ['session_002', 'session_003', 'session_004'],
    'pattern_36':                   ['session_002', 'session_003', 'session_004'],
    'pattern_37':                   ['session_002', 'session_003', 'session_004'],
}
IDX_START, IDX_END = 200, 800   # [200, 800)

# (subdir, filename template) — rgb_contact intentionally excluded
FILE_SPECS = [
    ('samples',  '{idx}.png'),
    ('rgb',      '{idx}.png'),
    ('raw_data', '{idx}.npy'),
    ('raw_data', '{idx}_gt.npy'),
    ('raw_data', '{idx}_pose.json'),
]


def file_ok(path):
    try:
        size = os.path.getsize(path)
        if size == 0:
            return False
        if path.endswith('.png'):
            with open(path, 'rb') as f:
                f.seek(max(0, size - 12))
                return b'IEND' in f.read()
        if path.endswith('.npy'):
            np.load(path, mmap_mode='r')
            return True
        if path.endswith('.json'):
            with open(path) as f:
                json.load(f)
            return True
        return True
    except Exception:
        return False


def extract_all():
    os.makedirs(STAGING, exist_ok=True)
    for obj in TARGETS:
        tar_path = os.path.join(TAR_DIR, f'{obj}.tar')
        marker = os.path.join(STAGING, f'.{obj}.extracted')
        if os.path.exists(marker):
            print(f'  {obj}: already extracted')
            continue
        if not os.path.exists(tar_path):
            print(f'  {obj}: TAR MISSING'); return False
        print(f'  {obj}: extracting ...', flush=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(STAGING)
        open(marker, 'w').close()
    return True


def verify():
    problems = []
    n_ok = 0
    for obj, sessions in TARGETS.items():
        for sess in sessions:
            sensor = os.path.join(STAGING, obj, sess, 'sensor_0000')
            for i in range(IDX_START, IDX_END):
                idx = f'{i:04d}'
                for sub, tmpl in FILE_SPECS:
                    p = os.path.join(sensor, sub, tmpl.format(idx=idx))
                    if not os.path.exists(p):
                        problems.append(f'MISSING {obj}/{sess}/{sub}/{tmpl.format(idx=idx)}')
                    elif not file_ok(p):
                        problems.append(f'CORRUPT {obj}/{sess}/{sub}/{tmpl.format(idx=idx)}')
                    else:
                        n_ok += 1
    return n_ok, problems


def check_dest_conflicts():
    conflicts = []
    for obj, sessions in TARGETS.items():
        for sess in sessions:
            sensor = os.path.join(DEST, obj, sess, 'sensor_0000')
            for i in range(IDX_START, IDX_END):
                idx = f'{i:04d}'
                for sub, tmpl in FILE_SPECS:
                    p = os.path.join(sensor, sub, tmpl.format(idx=idx))
                    if os.path.exists(p):
                        conflicts.append(p)
    return conflicts


def merge():
    n = 0
    for obj, sessions in TARGETS.items():
        for sess in sessions:
            src_sensor = os.path.join(STAGING, obj, sess, 'sensor_0000')
            dst_sensor = os.path.join(DEST, obj, sess, 'sensor_0000')
            for i in range(IDX_START, IDX_END):
                idx = f'{i:04d}'
                for sub, tmpl in FILE_SPECS:
                    src = os.path.join(src_sensor, sub, tmpl.format(idx=idx))
                    dst = os.path.join(dst_sensor, sub, tmpl.format(idx=idx))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if os.path.exists(dst):
                        raise RuntimeError(f'refusing to overwrite {dst}')
                    shutil.copy2(src, dst)
                    n += 1
        print(f'  {obj}: merged', flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    print('[1/4] Extracting tars to staging ...')
    if not extract_all():
        sys.exit(1)

    print('[2/4] Verifying staged samples ...')
    n_ok, problems = verify()
    expected = len(TARGETS) * 3 * (IDX_END - IDX_START) * len(FILE_SPECS)
    print(f'  {n_ok}/{expected} files OK, {len(problems)} problems')
    for p in problems[:30]:
        print('   ', p)
    if problems:
        print('ABORT: fix problems above before merging.'); sys.exit(1)

    print('[3/4] Checking destination for conflicts ...')
    conflicts = check_dest_conflicts()
    if conflicts:
        print(f'  {len(conflicts)} destination files already exist, e.g.:')
        for c in conflicts[:10]:
            print('   ', c)
        print('ABORT: will not overwrite.'); sys.exit(1)
    print('  no conflicts')

    if args.dry_run:
        print('[4/4] dry-run: skipping merge'); return

    print('[4/4] Merging into renders_v3 ...')
    n = merge()
    print(f'\nDone: {n} files merged.')
    print('Next: python3 post_process_fast.py --workers 16')


if __name__ == '__main__':
    main()
