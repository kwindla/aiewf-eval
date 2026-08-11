"""Scatter: P50 latency (TTFT Med) vs Pass Rate for OPEN-WEIGHTS models scoring above 90%.

Data = the open-weights rows of README.md's aiwf_medium_context text-mode table (closed APIs —
Claude / Gemini / GPT-5.x/4.x/4o / Nova — excluded). Run:
    uv run --with matplotlib python scripts/plot_open_weights_latency_passrate.py
Output: docs/open-weights-latency-vs-passrate-v3.png
"""

import matplotlib.pyplot as plt

# (label, ttft_med_ms, pass_rate_pct) — open-weights models, current README numbers.
MODELS = [
    ("nemotron-3-ultra (128)", 541, 100.0),
    ("qwen3.5-27b (thinking)", 1443, 99.0),
    ("glm-5 (thinking)", 841, 98.7),
    ("nemotron-3-ultra (96)", 529, 98.3),
    ("kimi-k2.6 Cerebras (thinking)", 452, 98.0),
    ("nemotron-3-super-120b (512)", 687, 97.0),
    ("zai-org/glm-5.1", 845, 95.7),
    ("qwen3.5-4b (thinking)", 778, 95.0),
    ("google/gemma-4-31b-it", 358, 95.0),
    ("qwen3.5-9b (thinking)", 904, 94.7),
    ("qwen3.5-27b", 1494, 94.3),
    ("kimi-k2.6 Cerebras (instant)", 256, 94.0),
    ("nemotron-3-nano-30b (512)", 940, 90.6),
    ("qwen3.5-9b", 908, 89.3),
    ("qwen3.5-4b", 773, 88.7),
    ("gpt-oss-120b (groq)", 98, 86.3),
    ("glm-4.7-flash", 940, 84.7),
]

# Only models scoring above 92% pass rate, and exclude Kimi-instant + Nemotron Nano.
EXCLUDE = {"kimi-k2.6 Cerebras (instant)", "nemotron-3-nano-30b (512)"}
MODELS = [m for m in MODELS if m[2] > 92.0 and m[0] not in EXCLUDE]

fig, ax = plt.subplots(figsize=(13, 8.5))

xs = [m[1] for m in MODELS]
ys = [m[2] for m in MODELS]
ax.scatter(xs, ys, s=80, color="#1f5fa8", zorder=4, edgecolor="white", linewidth=0.9)

# manual label offsets (dx ms, dy %) to reduce overlap
offsets = {
    "nemotron-3-ultra (128)": (16, 0.10),
    "qwen3.5-27b (thinking)": (-18, 0.0),
    "glm-5 (thinking)": (14, 0.20),
    "nemotron-3-ultra (96)": (16, 0.22),
    "kimi-k2.6 Cerebras (thinking)": (14, -0.22),
    "nemotron-3-super-120b (512)": (16, 0.15),
    "zai-org/glm-5.1": (10, 0.40),
    "qwen3.5-4b (thinking)": (14, 0.25),
    "google/gemma-4-31b-it": (16, 0.0),
    "qwen3.5-9b (thinking)": (14, -0.30),
    "qwen3.5-27b": (-18, 0.0),
}
for label, x, y in MODELS:
    dx, dy = offsets.get(label, (15, 0.10))
    ax.annotate(label, (x, y), xytext=(x + dx, y + dy),
                ha="right" if dx < 0 else "left", va="center",
                fontsize=9, color="#222")

ax.set_xlabel("P50 latency — TTFT median (ms)", fontsize=12)
ax.set_ylabel("Pass rate (%)", fontsize=12)
ax.set_title("Open-weights models (>92% pass rate): latency vs. accuracy\n"
             "aiwf_medium_context benchmark", fontsize=13)
ax.set_xlim(280, 1560)
ax.set_ylim(92, 100.7)
ax.set_yticks(range(92, 101))
ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
out = "docs/open-weights-latency-vs-passrate-v3.png"
fig.savefig(out, dpi=160)
print(f"Wrote {out} with {len(MODELS)} models")
