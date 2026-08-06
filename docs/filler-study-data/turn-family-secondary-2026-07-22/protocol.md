# Turn-family secondary analysis

Taxonomy and analysis plan frozen at 2026-07-22T06:51:35-07:00, before any
turn-family effects were computed. The analyst had already seen the published
overall and per-turn study results, so this is a post-hoc, exploratory secondary
analysis—not a prospective confirmatory test. To constrain outcome-driven turn
selection, a separate reviewer derived the taxonomy using only the benchmark
turns and system contract and did not inspect campaign artifacts.

## Question and scope

For the eleven focused 96-dot comparisons with 30 attempts per arm, where within
the scripted conversation does the overall pass-rate effect occur? The analysis
reports every one of the 30 fixed scripted positions and a benchmark-semantic
five-family decomposition. The primary whole-conversation estimand and its
confidence interval remain unchanged.

The five mutually exclusive families in `turn-families.json` exhaust turns 0–29:

1. grounded event information and reference resolution;
2. recommendation protocol and personalized synthesis;
3. tool preparation, slot elicitation, and disambiguation;
4. nonterminal tool commitment and confirmation;
5. interaction-boundary and closing pragmatics.

## Estimands and uncertainty

For each model and turn, the turn effect is the dot-arm strict-pass rate minus the
no-filler strict-pass rate over the fixed 30-attempt denominator. Every turn also
partitions failures into missing/post-abort turns and observed judged failures.
The report's missing-turn panel reverses the raw missing-rate sign—no-filler minus
dots—so positive/blue consistently means benefit. At every turn, the pass effect
equals the aligned missing contribution plus the aligned observed-failure
contribution.

For each model, arm, and family, a conversation's family score is its joint-pass
fraction over the fixed turns in that family. Missing and post-abort future turns
remain failures, exactly as in the primary fixed-denominator analysis.

The fixed-denominator within-family effect is the dot-arm mean family score minus the
no-filler mean family score, in percentage points. The contribution to the
30-turn overall effect is that within-family effect multiplied by the
family's turn count divided by 30. Contributions sum exactly to the published
overall point estimate.

Pointwise 95% intervals come from 100,000 independent-arm bootstrap resamples of
whole conversations. The same arm-specific resample indices are used across all
30 turns and all five families within a model, preserving their covariance. On
every bootstrap draw, the mean of the turn effects reconstructs the overall
effect and the weighted family contributions do the same. Intervals are
descriptive and unadjusted across 330 model-turn and 55 model-family cells; no
p-values or simultaneous-coverage claim is reported.

Family-specific intervals need not be narrower than the primary interval. The
conditional effect is expressed on a smaller subset of turns, which magnifies
both its point estimate and its uncertainty. The contribution scale is directly
comparable with the primary 30-turn effect.

## Frozen inputs

| file | SHA-256 |
|---|---|
| `benchmarks/_shared/turns.py` | `c88da69f8ade0e04e943b7493629ff96481d2779c001be7f77f0de82fbdc456b` |
| `benchmarks/aiwf_medium_context/prompts/system.py` | `6003f0f482c757a9bec6ed01e2993c7192112984e2037cf79d830bd46d76e9a6` |
| `dot-stability-n30-2026-07-20/analyze.py` | `3d9094da5c9858554baf9760eec9bbba786e71f72de64e362cfde9ce814dfe70` |
| `dot-stability-n30-2026-07-20/aggregates.json` | `573e53779774f61c8cc9641d553c02c2368c56a2785fddc87071cdb5c22a1d99` |
| `gemini-minimal-dots-2026-07-21/analyze.py` | `aa7b2ed23cb5cb5ec612626f3aa788f85d6f8b5af286c8eed991ae165ab8d2ee` |
| `gemini-minimal-dots-2026-07-21/aggregates.json` | `41be324032aaecffd03b3e43ffa35242a3e9b19c82c404093138f5905a7ecff2` |
| `turn-families.json` | `058ea3ada0d087ddd2afad5a4d02b9120e5a14c2a28e6ab952fdadc67f3e946e` |
| `source-manifest.tsv` | `5a0a222e8f0d4f5e4297ff618407d2dab2416feea5850c8cce4362e741b1fffb` |

The analysis reloads the exact primary conversation manifests through the two
campaign analyzers and verifies 30 unique conversations per arm for all eleven
models. `source-manifest.tsv` freezes all 660 run paths plus every transcript and
judgment SHA-256; the analyzer refuses a pool substitution or content change even
when aggregate totals remain the same. It does not rejudge or regenerate any
conversation.

## Completed results

The all-turn view confirms that the decomposition is heterogeneous rather than a
universal “problem-turn” effect:

- nonterminal tool-commitment estimates are positive for GPT-5.4 (+32.0 points),
  GPT-5.5 (+10.0), GPT-5.6-sol (+20.0), and Gemma 4 31B (+12.0), but negative for
  Qwen3-8B (-24.0) and GLM-5.2 (-12.0);
- Qwen3-8B simultaneously has a +26.1-point tool-preparation estimate, so its
  +0.9-point overall effect masks opposing family effects;
- Gemini 3.6 Flash's -7.8-point overall estimate is driven primarily by a
  -33.3-point tool-preparation estimate, contributing -6.7 points on the original
  30-turn scale;
- Gemini 3.5 Flash has a +10.0-point recommendation estimate, offset by other
  families; its primary overall interval still spans zero;
- Terra and Inkling family losses partly reflect early exits and their resulting
  missing future turns, which are shown separately from observed judged failures.

All family intervals are pointwise and unadjusted. These patterns are descriptive
decompositions, not treatment-by-family interaction tests or simultaneous
cross-screen discoveries. The report therefore shows all 30 turns in chronological
order, uses a fixed symmetric ±50-point color scale, and keeps the exact family
estimates underneath the contribution view. Long missing-turn suffix bands should
be read as one completion hazard propagated across later turns, not as independent
semantic effects.

## Completed-artifact integrity

| file | SHA-256 |
|---|---|
| `turn-families.json` | `058ea3ada0d087ddd2afad5a4d02b9120e5a14c2a28e6ab952fdadc67f3e946e` |
| `source-manifest.tsv` | `5a0a222e8f0d4f5e4297ff618407d2dab2416feea5850c8cce4362e741b1fffb` |
| `build_source_manifest.py` | `4f383a877bfcec3209845b57f9d31cb50d173e76e0239172901a694840c00f6b` |
| `analyze.py` | `016b3f5dc9b6e372c7fc1309d380530ff66b0a95109c5cbf82b83e5faa6d68ac` |
| `aggregates.json` | `bbdc69b54fc74436010a455eb602e782dde66a04606eb950638af78ee31d1040` |
| `verify_outputs.py` | `67b494653d2b541c1e3f1652e4bed64d71899ca9d4e456d1399ff1630985cfaa` |
| `scripts/build_filler_report.py` | `7700aa7096ab7712c1fbc727e206f9d9fc2159016f7830ef47934c9947b41b0c` |
| `docs/filler-token-latent-scratchpad-study.md` | `1c968ef7950134016199b30bf377a221b33b9a8a9a4ee0739fcbead66f1183f1` |
| `docs/filler-token-latent-scratchpad-study.html` | `8efc1948ca05981df992d1837809153636d6d46e6154e868022cd6047bf274f7` |
