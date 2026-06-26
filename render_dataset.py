"""
render_dataset.py
=================
Batch dataset generator.

Directory layout
----------------
renders/
  <obj>/
    session_000/
      session.json        ← already copied from renders_version1
      sensor_0000/        ← created by scripting.py during render
        calibration/
        raw_data/         ← pose JSON + depth .npy
        samples/          ← tactile PNG
        rgb/              ← RGB PNG
    session_001/
    ...

Usage
-----
    python render_dataset.py              # render everything not yet done
    python render_dataset.py --dry-run   # preview what would run
    python render_dataset.py --obj button edge   # only these objects
"""

import os, json, subprocess, time, argparse
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
BLENDER      = Path(r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe')
BLEND_FILE   = SCRIPT_DIR / 'gelsight_sampler.blend'
SCRIPTING    = SCRIPT_DIR / 'scripting.py'
RENDERS_ROOT = SCRIPT_DIR / 'renders'
FIXED_PARAMS = SCRIPT_DIR / 'bo_results_tactile' / 'best_params.json'
RGB_PARAMS   = SCRIPT_DIR / 'bo_results' / 'best_rgb_params.json'

BLENDER_TIMEOUT = 7200  # 2 hours per session (generous)


def _n_expected(session_dir: Path) -> int:
    sj = session_dir / 'session.json'
    if not sj.exists():
        return 0
    with open(sj) as f:
        s = json.load(f)
    vc = s.get('valid_cells')
    if vc is not None:
        return len(vc)
    return s.get('NUM_OBJ_SAMPLES', 0)


def is_complete(session_dir: Path) -> bool:
    n = _n_expected(session_dir)
    if n == 0:
        return False
    samples = session_dir / 'sensor_0000' / 'samples'
    if not samples.exists():
        return False
    done = sum(1 for p in samples.iterdir() if p.suffix == '.png')
    return done >= n


def run_session(session_dir: Path) -> bool:
    env = os.environ.copy()
    env['GELSIGHT_RENDER_DIR']   = str(session_dir)
    env['GELSIGHT_FIXED_PARAMS'] = str(FIXED_PARAMS)
    env['GELSIGHT_RGB_PARAMS']   = str(RGB_PARAMS)

    result = subprocess.run(
        [str(BLENDER), '--background', str(BLEND_FILE),
         '--python', str(SCRIPTING)],
        env=env, cwd=str(SCRIPT_DIR),
        capture_output=True, timeout=BLENDER_TIMEOUT,
    )
    if result.returncode != 0:
        tail = (result.stderr or b'')[-600:].decode('utf-8', errors='replace')
        print(f'\n    FAILED rc={result.returncode}\n{tail}')
        return False
    return True


def collect_sessions(obj_filter=None):
    sessions = []
    for obj_dir in sorted(RENDERS_ROOT.iterdir()):
        if not obj_dir.is_dir():
            continue
        if obj_filter and obj_dir.name not in obj_filter:
            continue
        for session_dir in sorted(obj_dir.iterdir()):
            if session_dir.is_dir() and session_dir.name.startswith('session_'):
                sessions.append(session_dir)
    return sessions


def main():
    parser = argparse.ArgumentParser(description='Batch GelSight dataset renderer')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run without calling Blender')
    parser.add_argument('--obj', nargs='+', metavar='OBJ',
                        help='Only render these object(s) (e.g. button edge)')
    args = parser.parse_args()

    sessions = collect_sessions(obj_filter=set(args.obj) if args.obj else None)
    total    = len(sessions)
    n_done   = sum(1 for s in sessions if is_complete(s))
    n_todo   = total - n_done

    print('=' * 60)
    print(f'GelSight Dataset Batch Render')
    print(f'  renders root : {RENDERS_ROOT}')
    print(f'  fixed params : {FIXED_PARAMS}')
    print(f'  sessions     : {total}  (done={n_done}  todo={n_todo})')
    print('=' * 60)

    t_start = time.time()
    fail_list = []

    for i, session_dir in enumerate(sessions):
        obj_name  = session_dir.parent.name
        sess_name = session_dir.name
        n_exp     = _n_expected(session_dir)
        prefix    = f'[{i+1:3d}/{total}] {obj_name}/{sess_name}  ({n_exp} samples)'

        if is_complete(session_dir):
            print(f'{prefix}  ✓ skip')
            continue

        print(f'{prefix}  ...', end='', flush=True)

        if args.dry_run:
            print('  (dry-run)')
            continue

        t0 = time.time()
        ok = run_session(session_dir)
        elapsed = time.time() - t0

        if ok:
            samples_done = sum(
                1 for p in (session_dir / 'sensor_0000' / 'samples').iterdir()
                if p.suffix == '.png'
            ) if (session_dir / 'sensor_0000' / 'samples').exists() else 0
            print(f'  OK  {samples_done}/{n_exp} samples  ({elapsed:.0f}s)')
        else:
            print(f'  FAIL  ({elapsed:.0f}s)')
            fail_list.append(str(session_dir))

    total_elapsed = time.time() - t_start
    print('=' * 60)
    print(f'Done in {total_elapsed/60:.1f} min  |  failed: {len(fail_list)}')
    if fail_list:
        print('Failed sessions:')
        for p in fail_list:
            print(f'  {p}')


if __name__ == '__main__':
    main()
