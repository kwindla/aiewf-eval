# DeepSeek V4 on Baseten Model API

Serving route for the `deepseek-v4-flash-0731` and `deepseek-v4-pro-0813`
leaderboard rows: Baseten's OpenAI-compatible Model API with the exact
model IDs `deepseek-ai/DeepSeek-V4-Flash-0731` and
`deepseek-ai/DeepSeek-V4-Pro-0813`, native `reasoning_effort`,
temperature 1.0, model-default top-p, 8,192-token output cap.

Campaigns:

- `aiewf-medium-3arm-n30-20260817/` — Flash low / Flash high / Pro low,
  30 conversations per arm, fixed 900-turn denominator. See
  `analysis/REPORT.md`; `manifest.tsv` maps every published number to its
  judged run directory.
