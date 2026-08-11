#!/bin/bash
# Aggregate dolphin-14 sweep results, one row per effort level.
# Builds README-format speech-to-speech table.
set -u
cd /home/khkramer/src/aiewf-eval

R="runs/aiwf_medium_context"

declare -A RUNS_minimal=(
  [1]="${R}/20260505T112648_gpt-realtime-alpha-dolphin-14_571a7576"
  [2]="${R}/20260505T115602_gpt-realtime-alpha-dolphin-14_b9aabed7"
  [3]="${R}/20260505T120645_gpt-realtime-alpha-dolphin-14_ec8c7cf4"
  [4]="${R}/20260505T121649_gpt-realtime-alpha-dolphin-14_f3442077"
  [5]="${R}/20260505T122646_gpt-realtime-alpha-dolphin-14_54c3d2f0"
  [6]="${R}/20260505T123738_gpt-realtime-alpha-dolphin-14_e6cf55a6"
  [7]="${R}/20260505T124851_gpt-realtime-alpha-dolphin-14_abbd75ae"
  [8]="${R}/20260505T125525_gpt-realtime-alpha-dolphin-14_2881feaa"
  [9]="${R}/20260505T130628_gpt-realtime-alpha-dolphin-14_d37300b9"
  [10]="${R}/20260505T131630_gpt-realtime-alpha-dolphin-14_6161e5d9"
)

declare -A RUNS_low=(
  [1]="${R}/20260505T112652_gpt-realtime-alpha-dolphin-14_ebbc5909"
  [2]="${R}/20260505T115605_gpt-realtime-alpha-dolphin-14_f784bde5"
  [3]="${R}/20260505T120629_gpt-realtime-alpha-dolphin-14_38308062"
  [4]="${R}/20260505T121706_gpt-realtime-alpha-dolphin-14_0fffab2e"
  [5]="${R}/20260505T122737_gpt-realtime-alpha-dolphin-14_f6654fb2"
  [6]="${R}/20260505T123806_gpt-realtime-alpha-dolphin-14_c55bb910"
  [7]="${R}/20260505T124840_gpt-realtime-alpha-dolphin-14_60368df5"
  [8]="${R}/20260505T125749_gpt-realtime-alpha-dolphin-14_f826deac"
  [9]="${R}/20260505T130827_gpt-realtime-alpha-dolphin-14_b06be3f6"
  [10]="${R}/20260505T131939_gpt-realtime-alpha-dolphin-14_ec31eae9"
)

declare -A RUNS_medium=(
  [1]="${R}/20260505T113703_gpt-realtime-alpha-dolphin-14_34a6788f"
  [2]="${R}/20260505T115609_gpt-realtime-alpha-dolphin-14_1dd817e4"
  [3]="${R}/20260505T120703_gpt-realtime-alpha-dolphin-14_dea7fbc5"
  [4]="${R}/20260505T121652_gpt-realtime-alpha-dolphin-14_02bc6778"
  [5]="${R}/20260505T122727_gpt-realtime-alpha-dolphin-14_dcdb43b0"
  [6]="${R}/20260505T123721_gpt-realtime-alpha-dolphin-14_4b8c4b4f"
  [7]="${R}/20260505T124755_gpt-realtime-alpha-dolphin-14_a65624b5"
  [8]="${R}/20260505T125837_gpt-realtime-alpha-dolphin-14_2fed7c09"
  [9]="${R}/20260505T130843_gpt-realtime-alpha-dolphin-14_14bb68dc"
  [10]="${R}/20260505T131908_gpt-realtime-alpha-dolphin-14_eb207582"
)

# high series: excludes 20260505T125042_..._b44c7823 (WebSocket keepalive failure)
declare -A RUNS_high=(
  [1]="${R}/20260505T113707_gpt-realtime-alpha-dolphin-14_08965365"
  [2]="${R}/20260505T115612_gpt-realtime-alpha-dolphin-14_550387e1"
  [3]="${R}/20260505T120706_gpt-realtime-alpha-dolphin-14_d6531135"
  [4]="${R}/20260505T121812_gpt-realtime-alpha-dolphin-14_30660acd"
  [5]="${R}/20260505T122820_gpt-realtime-alpha-dolphin-14_b3fa4661"
  [6]="${R}/20260505T123843_gpt-realtime-alpha-dolphin-14_47271d3f"
  [7]="${R}/20260505T132008_gpt-realtime-alpha-dolphin-14_643a8dcc"
  [8]="${R}/20260505T132656_gpt-realtime-alpha-dolphin-14_60574794"
  [9]="${R}/20260505T132746_gpt-realtime-alpha-dolphin-14_65ca10da"
  [10]="${R}/20260505T132953_gpt-realtime-alpha-dolphin-14_a5ee9673"
)

for effort in minimal low medium high; do
  echo "=== dolphin-14 (${effort}): 10 runs ==="
  declare -n arr="RUNS_${effort}"
  uv run python /home/khkramer/src/aiewf-eval/scripts/benchmark_summary.py "${arr[@]}" 2>&1 | tail -7
  echo
done
