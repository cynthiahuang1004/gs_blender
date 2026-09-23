#!/bin/bash
# Render the physics (Blender) tactile images at the real GT poses, for the physics-guided GAN.
# Run from the gs_blender repo root on any machine with Blender 4.2 + a GPU.
#
#   PHYS_REAL_ROOT=/path/to/phys_inputs/real_filtered   # unpacked phys_inputs.tar (pose json + GT npy)
#   PHYS_OUT=/path/to/real_filtered_phys                 # output; send this folder back to the server
#   BLENDER=/path/to/blender                             # default: `blender` on PATH
#   ./render_phys_local.sh [obj ...]                     # default: all 12 objects in phys_frames/
#
# Output: $PHYS_OUT/<obj>/session_000/sensor_0000/samples/XXXX.png. Existing files are skipped, so it is
# safe to re-run or to split objects across machines. ~460 frames (K<=25 subsets + val/3) come first in
# every list, the rest later.
cd "$(dirname "$0")"
BLENDER=${BLENDER:-blender}
objs=("$@"); [ ${#objs[@]} -eq 0 ] && objs=($(ls phys_frames | sed 's/.json$//'))
for obj in "${objs[@]}"; do
  echo "$(date +%m-%d\ %H:%M) start $obj"
  GELSIGHT_FIXED_PARAMS=bo_results/tactile_v2/best_params.json \
    "$BLENDER" -t ${PHYS_THREADS:-8} --background gelsight_sampler.blend --python render_real_pose_tactile.py -- \
    --obj "$obj" --frames "phys_frames/$obj.json" ${PHYS_OUT:+--out "$PHYS_OUT"} 2>&1 | grep -E "\[phys\]|Traceback|Error"
done
echo "$(date +%m-%d\ %H:%M) all done"
