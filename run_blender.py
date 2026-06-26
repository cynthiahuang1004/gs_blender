import os, subprocess, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
render_dir = os.path.join(script_dir, 'renders')
BLENDER_PATH = r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' 

session_dirs = sorted([
    d for d in os.listdir(render_dir)
    if d.startswith('session_') and os.path.isdir(os.path.join(render_dir, d))
])

print(f'找到 {len(session_dirs)} 個 session：{session_dirs}')

for session_name in session_dirs:
    session_render_dir = os.path.join(render_dir, session_name)
    print(f'\n========== 跑 {session_name} ==========')

    env = os.environ.copy()
    env['GELSIGHT_RENDER_DIR'] = session_render_dir

    for attempt in range(3):
        print(f'  attempt {attempt+1}/3')
        result = subprocess.run(
            [BLENDER_PATH, '--background', 'gelsight_sampler.blend',
            '--python', 'scripting.py'],
            cwd=script_dir,
            env=env
        )
        if result.returncode == 0:
            print(f'  {session_name} 完成')
            break
        else:
            print(f'  失敗，rc={result.returncode}')
    else:
        print(f'  {session_name} 失敗 3 次，跳過')

print('\n全部 session 完成')