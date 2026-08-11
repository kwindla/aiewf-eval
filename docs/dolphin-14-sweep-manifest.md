# gpt-realtime-alpha-dolphin-14 reasoning effort sweep manifest

10 runs each at minimal / low / medium / high. Run dir prefixes per effort.

## Existing single-run sweep (run #1 of each)
- minimal: `runs/aiwf_medium_context/20260505T112648_gpt-realtime-alpha-dolphin-14_571a7576/`
- low:     `runs/aiwf_medium_context/20260505T112652_gpt-realtime-alpha-dolphin-14_ebbc5909/`
- medium:  `runs/aiwf_medium_context/20260505T113703_gpt-realtime-alpha-dolphin-14_34a6788f/`
- high:    `runs/aiwf_medium_context/20260505T113707_gpt-realtime-alpha-dolphin-14_08965365/`

## 10-run aggregate batches (runs #2..#10)
Each batch starts 4 simultaneous runs (one per effort level), then waits.
Batches are filed below as they start, with the run dir per effort.
