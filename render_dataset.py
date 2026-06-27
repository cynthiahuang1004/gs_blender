"""
render_dataset.py
=================
Batch dataset generator with multi-GPU support.

Directory layout
----------------
renders/
  <obj>/
    session_000/
      session.json        <- already copied from renders_version1
      sensor_0000/        <- created by scripting.py during render
        calibration/
        raw_data/         <- pose JSON + depth .npy
        samples/          <- tactile PNG
        rgb/              <- RGB PNG
    session_001/
    ...

Usage
-----
    python render_dataset.py              # render everything not yet done (1 GPU)
    python render_dataset.py --gpus 0,1,2,3   # use 4 GPUs in parallel
    python render_dataset.py --dry-run   # preview what would run
    python render_dataset.py --obj button edge   # only these objects
"""

import os, json, subprocess, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def run_session(session_dir: Path, gpu_id: int = 0) -> bool:
    env = os.environ.copy()
    env['GELSIGHT_RENDER_DIR']   = str(session_dir)
    env['GELSIGHT_FIXED_PARAMS'] = str(FIXED_PARAMS)
    env['GELSIGHT_RGB_PARAMS']   = str(RGB_PARAMS)
    env['CUDA_VISIBLE_DEVICES']  = str(gpu_id)
    env['PYTHONUNBUFFERED']      = '1'

    log_path = session_dir / 'render.log'
    with open(log_path, 'w') as log_f:
        result = subprocess.run(
            [str(BLENDER), '--background', str(BLEND_FILE),
             '--python', str(SCRIPTING)],
            env=env, cwd=str(SCRIPT_DIR),
            stdout=log_f, stderr=subprocess.STDOUT,
            timeout=BLENDER_TIMEOUT,
        )
    if result.returncode != 0:
        tail = log_path.read_text(errors='replace')[-600:]
        print(f'\n    FAILED rc={result.returncode}\n{tail}')
        return False
    return True


def _worker(args):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    session_dir, gpu_id, index, total = args
    obj_name  = session_dir.parent.name
    sess_name = session_dir.name
    n_exp     = _n_expected(session_dir)
    prefix    = f'[{index:3d}/{total}] GPU{gpu_id} {obj_name}/{sess_name} ({n_exp} samples)'

    print(f'{prefix}  ...', flush=True)
    t0 = time.time()
    ok = run_session(session_dir, gpu_id=gpu_id)
    elapsed = time.time() - t0

    if ok:
        samples_path = session_dir / 'sensor_0000' / 'samples'
        samples_done = sum(
            1 for p in samples_path.iterdir() if p.suffix == '.png'
        ) if samples_path.exists() else 0
        print(f'{prefix}  OK  {samples_done}/{n_exp} samples  ({elapsed:.0f}s)')
    else:
        print(f'{prefix}  FAIL  ({elapsed:.0f}s)')

    return (str(session_dir), ok)


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
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated GPU IDs (e.g. 0,1,2,3)')
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(',')]
    n_workers = len(gpu_ids)

    sessions = collect_sessions(obj_filter=set(args.obj) if args.obj else None)
    total    = len(sessions)
    n_done   = sum(1 for s in sessions if is_complete(s))
    todo     = [s for s in sessions if not is_complete(s)]
    n_todo   = len(todo)

    print('=' * 60)
    print(f'GelSight Dataset Batch Render')
    print(f'  renders root : {RENDERS_ROOT}')
    print(f'  fixed params : {FIXED_PARAMS}')
    print(f'  sessions     : {total}  (done={n_done}  todo={n_todo})')
    print(f'  GPUs         : {gpu_ids}  ({n_workers} workers)')
    print('=' * 60)

    if args.dry_run:
        for i, s in enumerate(todo):
            gpu_id = gpu_ids[i % n_workers]
            print(f'  [{i+1:3d}/{n_todo}] GPU{gpu_id} {s.parent.name}/{s.name}  (dry-run)')
        return

    t_start = time.time()
    fail_list = []

    work_items = []
    for i, session_dir in enumerate(todo):
        gpu_id = gpu_ids[i % n_workers]
        work_items.append((session_dir, gpu_id, i + 1, n_todo))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, item): item for item in work_items}
        for future in as_completed(futures):
            session_path, ok = future.result()
            if not ok:
                fail_list.append(session_path)

    total_elapsed = time.time() - t_start
    print('=' * 60)
    print(f'Done in {total_elapsed/60:.1f} min  |  failed: {len(fail_list)}')
    if fail_list:
        print('Failed sessions:')
        for p in fail_list:
            print(f'  {p}')


if __name__ == '__main__':
    main()
