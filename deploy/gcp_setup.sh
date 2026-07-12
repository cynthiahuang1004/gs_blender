#!/bin/bash
# GCP VM one-shot setup: install Blender + clone repo
set -e

BLENDER_TAR="blender-4.2.0-linux-x64.tar.xz"
BLENDER_URL="https://mirrors.dotsrc.org/blender/release/Blender4.2/${BLENDER_TAR}"
BLENDER_BIN="/opt/blender-4.2.0-linux-x64/blender"
REPO="https://github.com/cynthiahuang1004/gs_blender.git"
PROJECT="$HOME/gs_blender"

# ── Blender ────────────────────────────────────────────────────
if [ ! -f "$BLENDER_BIN" ]; then
    echo "[1/3] Downloading Blender..."
    wget -q "$BLENDER_URL" -O "/tmp/${BLENDER_TAR}"
    echo "[1/3] Extracting..."
    sudo tar -xf "/tmp/${BLENDER_TAR}" -C /opt/
    rm "/tmp/${BLENDER_TAR}"
fi
echo "[1/3] Blender ready: $BLENDER_BIN"

# ── Repo ───────────────────────────────────────────────────────
if [ ! -d "$PROJECT" ]; then
    echo "[2/3] Cloning repo..."
    git clone "$REPO" "$PROJECT"
else
    echo "[2/3] Updating repo..."
    git -C "$PROJECT" fetch origin
    git -C "$PROJECT" reset --hard origin/main
fi

# ── Patch Blender path ─────────────────────────────────────────
echo "[3/3] Patching render_dataset.py..."
sed -i "s|Path(r'C:\\\\Program Files\\\\Blender Foundation\\\\Blender 4.5\\\\blender.exe')|Path('${BLENDER_BIN}')|g" \
    "$PROJECT/render_dataset.py"

echo ""
echo "Setup complete! Run:"
echo "  cd $PROJECT"
echo "  tmux new -s render"
echo "  python render_dataset.py --gpus 0,0,0,0 --reverse --exclude ping_pong"
