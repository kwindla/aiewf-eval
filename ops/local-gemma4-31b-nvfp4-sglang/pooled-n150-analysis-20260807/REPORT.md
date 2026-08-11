# Gemma 4 31B pooled N=150 deployment comparison

The frozen N=30 and extension N=120 cohorts are pooled within each arm, for
150 conversations and 4,500 fixed-denominator turns per configuration. The
local FP8-KV/BF16-KV contrast is the primary KV-cache comparison; BaseTen is an
end-to-end deployment reference.

| Configuration | Strict pass | Whole-conversation bootstrap 95% CI | Tool error | Instruction error | KB error | TTFAT P50 | TTFAT P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BaseTen BF16 weights/KV + MTP | 4346/4500 (96.58%) | 96.13–97.02% | 3.33% | 3.40% | 0.02% | 490ms | 718ms |
| Local FP8 KV | 4297/4500 (95.49%) | 94.96–96.00% | 4.47% | 4.51% | 0.00% | 105ms | 309ms |
| Local BF16 KV | 4327/4500 (96.16%) | 95.62–96.64% | 3.78% | 3.82% | 0.00% | 128ms | 336ms |

Local BF16 KV minus local FP8 KV is **+0.67 percentage points**
(independent whole-conversation bootstrap 95% CI **-0.07 to
+1.40**; 50,000 replicates). Local FP8 KV minus
BaseTen is **-1.09 points** (95% CI
**-1.78 to -0.42**), and local BF16 KV
minus BaseTen is **-0.42 points** (95% CI
**-1.09 to +0.24**; all differences use
independent whole-conversation resampling with
50,000 replicates).

| Turn | BaseTen errors | Local FP8-KV errors | Local BF16-KV errors | BF16 − FP8 errors |
|---:|---:|---:|---:|---:|
| 12 | 105/150 | 98/150 | 70/150 | -28 |
| 13 | 1/150 | 0/150 | 0/150 | +0 |
| 14 | 2/150 | 0/150 | 1/150 | +1 |
| 15 | 38/150 | 86/150 | 77/150 | -9 |
| 16 | 2/150 | 2/150 | 2/150 | +0 |
| 17 | 4/150 | 6/150 | 9/150 | +3 |
| 24 | 2/150 | 11/150 | 14/150 | +3 |

Missing future turns after a model-caused early exit count as failures. Latency
uses observed scripted turns only. Each confidence interval resamples whole
conversations, not individual turns. Each difference interval resamples its two
150-conversation arms independently.

Both arms use the same NVFP4 checkpoint, one RTX 5090, the same SGLang image,
sampling settings, batch-one execution, and no MTP. Besides KV precision, the
BF16 arm uses compact asymmetric KV-pool limits required to fit the GPU.

The BaseTen arm uses BF16 weights/KV, NEXTN MTP, two H100s, and a newer SGLang
version. Its comparison with either local arm therefore includes several
deployment differences and should not be interpreted as a quantization-only
effect.
