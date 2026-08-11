#!/bin/bash
# Aggregate walrus-2 + reasoning_effort sweep results, one row per effort.
set -u
cd /home/khkramer/src/aiewf-eval

R="runs/aiwf_medium_context"

declare -A RUNS_minimal=(
  [1]="${R}/20260505T154504_gpt-realtime-alpha-dolphin-14_9887111c"
  [2]="${R}/20260505T155526_gpt-realtime-alpha-dolphin-14_6d18362b"
  [3]="${R}/20260505T160529_gpt-realtime-alpha-dolphin-14_3650a479"
  [4]="${R}/20260505T161726_gpt-realtime-alpha-dolphin-14_70fdf56d"
  [5]="${R}/20260505T162932_gpt-realtime-alpha-dolphin-14_7fa5f810"
  [6]="${R}/20260505T163957_gpt-realtime-alpha-dolphin-14_f244b8ab"
  [7]="${R}/20260505T165038_gpt-realtime-alpha-dolphin-14_946717ef"
  [8]="${R}/20260505T165605_gpt-realtime-alpha-dolphin-14_daa54040"
  [9]="${R}/20260505T170612_gpt-realtime-alpha-dolphin-14_58c02667"
  [10]="${R}/20260505T171623_gpt-realtime-alpha-dolphin-14_fab7bd5b"
)

declare -A RUNS_low=(
  [1]="${R}/20260505T154509_gpt-realtime-alpha-dolphin-14_dedbabc5"
  [2]="${R}/20260505T155552_gpt-realtime-alpha-dolphin-14_84cd1b3c"
  [3]="${R}/20260505T160642_gpt-realtime-alpha-dolphin-14_d7dbb4a7"
  [4]="${R}/20260505T161737_gpt-realtime-alpha-dolphin-14_d98892b1"
  [5]="${R}/20260505T162847_gpt-realtime-alpha-dolphin-14_273eb039"
  [6]="${R}/20260505T164012_gpt-realtime-alpha-dolphin-14_de7ee6f2"
  [7]="${R}/20260505T164957_gpt-realtime-alpha-dolphin-14_f93687bb"
  [8]="${R}/20260505T170136_gpt-realtime-alpha-dolphin-14_96c9ecf0"
  [9]="${R}/20260505T171120_gpt-realtime-alpha-dolphin-14_f177315c"
  [10]="${R}/20260505T172148_gpt-realtime-alpha-dolphin-14_b5095db5"
)

declare -A RUNS_medium=(
  [1]="${R}/20260505T154513_gpt-realtime-alpha-dolphin-14_e27fb371"
  [2]="${R}/20260505T155603_gpt-realtime-alpha-dolphin-14_d7ba6e8f"
  [3]="${R}/20260505T160717_gpt-realtime-alpha-dolphin-14_2e9a70b3"
  [4]="${R}/20260505T161815_gpt-realtime-alpha-dolphin-14_6e084099"
  [5]="${R}/20260505T162940_gpt-realtime-alpha-dolphin-14_3d87db5a"
  [6]="${R}/20260505T164024_gpt-realtime-alpha-dolphin-14_134beb01"
  [7]="${R}/20260505T165104_gpt-realtime-alpha-dolphin-14_e5f10594"
  [8]="${R}/20260505T170212_gpt-realtime-alpha-dolphin-14_573b97cb"
  [9]="${R}/20260505T171323_gpt-realtime-alpha-dolphin-14_d7f9e71e"
  [10]="${R}/20260505T172352_gpt-realtime-alpha-dolphin-14_ba33bac6"
)

declare -A RUNS_high=(
  [1]="${R}/20260505T154517_gpt-realtime-alpha-dolphin-14_4ec639fb"
  [2]="${R}/20260505T155633_gpt-realtime-alpha-dolphin-14_288b3872"
  [3]="${R}/20260505T160720_gpt-realtime-alpha-dolphin-14_47883b15"
  [4]="${R}/20260505T161825_gpt-realtime-alpha-dolphin-14_6246a1f7"
  [5]="${R}/20260505T162855_gpt-realtime-alpha-dolphin-14_a3bc79f9"
  [6]="${R}/20260505T163859_gpt-realtime-alpha-dolphin-14_d4c9afd8"
  [7]="${R}/20260505T164935_gpt-realtime-alpha-dolphin-14_401fe183"
  [8]="${R}/20260505T170027_gpt-realtime-alpha-dolphin-14_8a0911ab"
  [9]="${R}/20260505T171147_gpt-realtime-alpha-dolphin-14_e5dae5c9"
  [10]="${R}/20260505T172203_gpt-realtime-alpha-dolphin-14_0fff35b7"
)

for effort in minimal low medium high; do
  echo "=== dolphin-14 + walrus-2 (${effort}): 10 runs ==="
  declare -n arr="RUNS_${effort}"
  uv run python /home/khkramer/src/aiewf-eval/scripts/benchmark_summary.py "${arr[@]}" 2>&1 | tail -7
  echo
done
