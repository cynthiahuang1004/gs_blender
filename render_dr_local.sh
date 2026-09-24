#!/bin/bash
# Domain-randomisation renders of all 12 objects (one random sensor per session), run locally.
#   tar xzf dr_inputs.tar.gz            -> ./dr_inputs/renders_v3/<obj>/session_XXX/sensor_0000/raw_data/*
#   SIM_ROOT=$PWD/dr_inputs/renders_v3 DR_OUT=$PWD/renders_v3_dr BLENDER=/path/to/blender ./render_dr_local.sh [obj ...]
cd "$(dirname "$0")"; BLENDER=${BLENDER:-blender}; export SIM_ROOT=${SIM_ROOT:-$PWD/dr_inputs/renders_v3}; OUT=${DR_OUT:-$PWD/renders_v3_dr}
objs=("$@"); [ ${#objs[@]} -eq 0 ] && objs=($(ls "$SIM_ROOT"))
for obj in "${objs[@]}"; do
  echo "$(date +%m-%d\ %H:%M) start $obj"
  "$BLENDER" -t ${PHYS_THREADS:-8} --background gelsight_sampler.blend --python render_random_sensor.py -- \
    --obj "$obj" --sessions $(seq 0 11) --out "$OUT" --every ${DR_EVERY:-3} 2>&1 | grep -E "\[dr\]|Traceback|Error"
done
echo "$(date +%m-%d\ %H:%M) all done"
