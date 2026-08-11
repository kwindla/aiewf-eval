# Targeted KV replay analysis

## Greedy mechanism probe

| Arm | Turn | Prefix kind | Snapshots | Success |
|---|---:|---|---:|---:|
| bf16 | 12 | golden_mechanism | 1 | 100.0% |
| bf16 | 12 | real_prefix_bank | 12 | 41.7% |
| bf16 | 15 | golden_mechanism | 1 | 0.0% |
| bf16 | 15 | real_prefix_bank | 12 | 58.3% |
| fp8 | 12 | golden_mechanism | 1 | 0.0% |
| fp8 | 12 | real_prefix_bank | 12 | 33.3% |
| fp8 | 15 | golden_mechanism | 1 | 0.0% |
| fp8 | 15 | real_prefix_bank | 12 | 58.3% |

## Teacher-forced canonical tool sequence

Positive log-probability differences favor BF16 KV for the exact expected tool-call suffix. The decision margin compares `<|tool_call>` with each arm's best first-token alternative.

| Cache | Turn | Prefix kind | Matched snapshots | BF16−FP8 mean logp/token | BF16−FP8 first-decision margin |
|---|---:|---|---:|---:|---:|
| cold | 12 | golden_mechanism | 1 | +0.18572 | +6.43266 |
| cold | 12 | real_prefix_bank | 12 | +0.03166 | +0.65612 |
| cold | 15 | golden_mechanism | 1 | -0.03966 | -1.15146 |
| cold | 15 | real_prefix_bank | 12 | +0.00204 | +0.21346 |
| warm | 12 | golden_mechanism | 1 | +0.19910 | +8.36454 |
| warm | 12 | real_prefix_bank | 12 | +0.01587 | +1.40994 |
| warm | 15 | golden_mechanism | 1 | -0.06667 | -1.83374 |
| warm | 15 | real_prefix_bank | 12 | -0.01150 | -0.10555 |
