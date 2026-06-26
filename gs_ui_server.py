"""
gs_ui_server.py — GelSight Parameter Studio
============================================
Interactive web UI for GelSight sensor calibration.

Install dependencies:
    pip install fastapi uvicorn pillow numpy

Run:
    python gs_ui_server.py
Then open: http://localhost:7860
"""

import os, sys, json, asyncio, subprocess, shutil, base64, io
import queue, threading, tempfile, time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
BLENDER_PATH  = Path(r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe')
BLEND_FILE    = SCRIPT_DIR / 'gelsight_sampler.blend'
RENDER_SCRIPT = SCRIPT_DIR / 'gs_ui_render.py'
FINAL_PARAMS  = SCRIPT_DIR / 'bo_results' / 'final_params.json'
MESH_DIR      = SCRIPT_DIR / 'meshes'
BO_SCRIPT     = SCRIPT_DIR / 'bo_optimize.py'

# ── Default parameters ─────────────────────────────────────────
# All values match scripting_bo.py / scripting.py current calibration
DEFAULT_PARAMS: dict = {
    # 4 main emittors (BL=top, TR=bot, TL=left, BR=right)
    "top_str":   80.0,  "top_r":   0.3,  "top_g":  0.65, "top_b":  0.3,
    "bot_str":   40.0,  "bot_r":   0.1,  "bot_g":  0.5,  "bot_b":  0.9,
    "left_str":  30.0,  "left_r":  0.9,  "left_g": 0.05, "left_b": 0.05,
    "right_str": 120.0, "right_r": 1.0,  "right_g": 0.0, "right_b": 0.0,
    # Green lights
    "lg_str":    60.0,  "lg_r":   0.3,   "lg_g":  0.65,  "lg_b":  0.3,
    "rg_str":   120.0,  "rg_r":   0.3,   "rg_g":  0.7,   "rg_b":  0.3,
    # Gel / camera
    "gel_roughness": 0.4455, "gel_fac": 0.2971,
    "smoothness": 30,  "scale": 0.4918,
    "fov": 60.0,       "length": 0.008751,
    "light_rot_z": -3.14159,
    # Post-FX (applied server-side, not passed to Blender)
    "barrel_k1":  0.07, "vignette":   0.25,
    "blur_sigma": 1.25, "blur_falloff": 1.5,
    "sat_boost":  1.4,
}

_FX_KEYS = {"barrel_k1", "vignette", "blur_sigma", "blur_falloff", "sat_boost"}

# Load saved params if available
_current_params: dict = DEFAULT_PARAMS.copy()
if FINAL_PARAMS.exists():
    with open(FINAL_PARAMS) as _f:
        _current_params.update(json.load(_f))

# ── Post-processing helpers ────────────────────────────────────

def _barrel(rgb: np.ndarray, k1: float) -> np.ndarray:
    H, W = rgb.shape[:2]
    cx, cy = W / 2., H / 2.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx - cx) / cx;  yn = (yy - cy) / cy
    fac = 1. + k1 * (xn**2 + yn**2)
    xs = np.clip(xn * fac * cx + cx, 0, W - 1)
    ys = np.clip(yn * fac * cy + cy, 0, H - 1)
    x0 = np.floor(xs).astype(int);  x1 = np.minimum(x0 + 1, W - 1)
    y0 = np.floor(ys).astype(int);  y1 = np.minimum(y0 + 1, H - 1)
    wx = (xs - x0)[:, :, None];     wy = (ys - y0)[:, :, None]
    return np.clip(
        rgb[y0, x0] * (1 - wx) * (1 - wy) + rgb[y0, x1] * wx * (1 - wy) +
        rgb[y1, x0] * (1 - wx) * wy        + rgb[y1, x1] * wx * wy, 0, 1)


def _gblur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.05:
        return rgb.copy()
    sz = max(3, int(6 * sigma + 1) | 1)
    x = np.arange(sz) - sz // 2
    k = np.exp(-x**2 / (2 * sigma**2));  k /= k.sum()
    out = np.empty_like(rgb)
    for c in range(rgb.shape[2]):
        h = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1, rgb[:, :, c])
        out[:, :, c] = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 0, h)
    return np.clip(out, 0, 1)


def _apply_gel_fx(arr: np.ndarray, P: dict) -> np.ndarray:
    """arr: uint8 H×W×3. Returns uint8."""
    rgb = arr.astype(np.float32) / 255.
    H, W = rgb.shape[:2]
    rgb = _barrel(rgb, float(P.get('barrel_k1', 0.07)))
    blurred = _gblur(rgb, float(P.get('blur_sigma', 1.25)))
    cy, cx = H / 2., W / 2.
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    w = np.clip(dist ** float(P.get('blur_falloff', 1.5)), 0, 1)[:, :, None]
    rgb = rgb * (1 - w) + blurred * w
    mask = np.clip(1. - dist ** 2 * float(P.get('vignette', 0.25)), 0, 1)[:, :, None]
    rgb = np.clip(rgb * mask, 0, 1)
    return (rgb * 255).astype(np.uint8)


def _apply_sat(arr: np.ndarray, factor: float) -> np.ndarray:
    """arr: uint8 H×W×3. Returns uint8."""
    rgb = arr.astype(np.float32) / 255.
    cmax = rgb.max(2);  cmin = rgb.min(2);  delta = cmax - cmin
    v = cmax;  s = np.where(cmax > 0, delta / cmax, 0.)
    h = np.zeros_like(v);  m = delta > 0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mr = m & (cmax == r);  h[mr] = (60 * ((g[mr] - b[mr]) / delta[mr])) % 360
    mg = m & (cmax == g);  h[mg] = 60 * ((b[mg] - r[mg]) / delta[mg] + 2)
    mb = m & (cmax == b);  h[mb] = 60 * ((r[mb] - g[mb]) / delta[mb] + 4)
    s = np.clip(s * factor, 0, 1)
    h6 = h / 60.;  i = np.floor(h6).astype(int) % 6;  f = h6 - np.floor(h6)
    p = v * (1 - s);  q = v * (1 - f * s);  t = v * (1 - (1 - f) * s)
    out = np.zeros_like(rgb)
    for ii, (r0, g0, b0) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        mk = i == ii
        out[:, :, 0][mk] = r0[mk];  out[:, :, 1][mk] = g0[mk];  out[:, :, 2][mk] = b0[mk]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _arr_to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO();  img.save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()

# ── Blender render (blocking, called in thread pool) ───────────

def _run_render(mode: str, P: dict, obj_path: str, depth_mm: float) -> tuple:
    with tempfile.TemporaryDirectory() as tmp:
        params_file = os.path.join(tmp, 'params.json')
        out_path    = os.path.join(tmp, 'render')
        blend_copy  = os.path.join(tmp, 'scene.blend')

        with open(params_file, 'w') as f:
            json.dump({k: v for k, v in P.items() if k not in _FX_KEYS}, f)

        shutil.copy(str(BLEND_FILE), blend_copy)

        env = os.environ.copy()
        env.update({
            'GS_UI_MODE':   mode,
            'GS_UI_PARAMS': params_file,
            'GS_UI_OBJ':    obj_path,
            'GS_UI_DEPTH':  str(depth_mm),
            'GS_UI_OUT':    out_path,
        })

        t0 = time.time()
        proc = subprocess.run(
            [str(BLENDER_PATH), '--background', blend_copy,
             '--python', str(RENDER_SCRIPT)],
            env=env, capture_output=True, timeout=300,
        )
        elapsed = time.time() - t0

        png = None
        for cand in (out_path + '.png', out_path + '0001.png'):
            if os.path.exists(cand):
                png = cand;  break

        if proc.returncode != 0 or png is None:
            stderr = proc.stderr.decode('utf-8', errors='replace')
            stdout = proc.stdout.decode('utf-8', errors='replace')
            combined = (stdout + stderr)[-1200:]
            raise RuntimeError(f'Blender rc={proc.returncode}\n{combined}')

        arr = np.array(Image.open(png).convert('RGB'))
        arr = _apply_sat(arr, float(P.get('sat_boost', 1.4)))
        arr = _apply_gel_fx(arr, P)
        return arr, elapsed

# ── FastAPI ────────────────────────────────────────────────────
app = FastAPI(title='GelSight Parameter Studio')

_render_busy = False
_bo_proc: Optional[subprocess.Popen] = None
_bo_queue: queue.Queue = queue.Queue()


class RenderReq(BaseModel):
    mode: str      = 'bg'
    params: dict   = {}
    obj_path: str  = ''
    depth_mm: float = 1.0


@app.post('/render')
async def render_endpoint(req: RenderReq):
    global _render_busy
    if _render_busy:
        raise HTTPException(503, 'Render already in progress')
    _render_busy = True
    try:
        P         = {**_current_params, **req.params}
        obj_path  = str(MESH_DIR / req.obj_path) if req.obj_path else ''
        arr, elapsed = await asyncio.get_event_loop().run_in_executor(
            None, _run_render, req.mode, P, obj_path, req.depth_mm)
        return JSONResponse({'image': _arr_to_b64(arr), 'elapsed': round(elapsed, 1)})
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        _render_busy = False


@app.get('/params')
async def get_params():
    return JSONResponse(_current_params)


class SaveReq(BaseModel):
    params: dict


@app.post('/params/save')
async def save_params(req: SaveReq):
    global _current_params
    _current_params.update(req.params)
    # Save only Blender-relevant keys to final_params.json (used by bo_optimize)
    to_save = {k: v for k, v in _current_params.items() if k not in _FX_KEYS}
    FINAL_PARAMS.parent.mkdir(exist_ok=True)
    with open(FINAL_PARAMS, 'w') as f:
        json.dump(to_save, f, indent=2)
    return JSONResponse({'status': 'saved', 'path': str(FINAL_PARAMS)})


@app.get('/meshes')
async def list_meshes():
    if not MESH_DIR.exists():
        return JSONResponse([])
    return JSONResponse(sorted(f.name for f in MESH_DIR.glob('*.obj')))


@app.post('/bo/start')
async def bo_start():
    global _bo_proc, _bo_queue
    if _bo_proc and _bo_proc.poll() is None:
        raise HTTPException(409, 'BO already running')
    while not _bo_queue.empty():
        try: _bo_queue.get_nowait()
        except: pass
    _bo_proc = subprocess.Popen(
        [sys.executable, str(BO_SCRIPT)],
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    def _reader():
        for line in _bo_proc.stdout:
            _bo_queue.put(line.rstrip())
        _bo_queue.put('__DONE__')
    threading.Thread(target=_reader, daemon=True).start()
    return JSONResponse({'status': 'started'})


@app.post('/bo/stop')
async def bo_stop():
    global _bo_proc
    if _bo_proc and _bo_proc.poll() is None:
        _bo_proc.terminate()
    return JSONResponse({'status': 'stopped'})


@app.get('/bo/stream')
async def bo_stream():
    async def gen():
        while True:
            try:
                line = _bo_queue.get(timeout=25)
                yield f"data: {line}\n\n"
                if line == '__DONE__':
                    break
            except queue.Empty:
                yield "data: __PING__\n\n"
    return StreamingResponse(gen(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache',
                                      'X-Accel-Buffering': 'no'})


@app.get('/')
async def index():
    return FileResponse(str(SCRIPT_DIR / 'gs_ui.html'))


if __name__ == '__main__':
    print('GelSight Parameter Studio')
    print('Open: http://localhost:7860')
    uvicorn.run(app, host='127.0.0.1', port=7860, log_level='warning')
