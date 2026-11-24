"""Fig. 2A: Spearman correlation of each plot-similarity measure with the mean human rating.

Run from the repository root:  python nlp/scripts/make_fig2a.py
Writes results/nlp/fig2a_validation.pdf and .png. Colours match Person A's Fig. 2B.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import build_similarity as bs  # noqa: E402
import paths  # noqa: E402

OUT = paths.ROOT / "results" / "nlp"
C_TFIDF, C_EMBED, C_GREY = "#2a78d6", "#1baf7a", "#8a8984"   # A's C_OBS, C_ALT, C_REW
MEASURES = [("tfidf", "TF-IDF", C_TFIDF), ("mpnet", "Sentence embeddings", C_EMBED),
            ("minilm", "MiniLM", C_GREY), ("lda", "LDA topics", C_GREY)]
RATERS = ["rater_1", "rater_2", "rater_3"]
N_BOOT, SEED = 2000, 0

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "font.size": 9})

ids = bs.node_ids()
idx = {f: i for i, f in enumerate(ids)}
sheet = pd.read_csv(paths.PAIR_RATINGS, sep="\t", dtype=str, keep_default_na=False)
y = sheet[RATERS].astype(int).mean(axis=1).to_numpy()
qi = sheet["query"].map(idx).to_numpy()
ci = sheet["candidate"].map(idx).to_numpy()
boot = np.random.default_rng(SEED).integers(0, len(sheet), size=(N_BOOT, len(sheet)))

rows = []
for key, label, col in MEASURES:
    s = bs.load_shared(key)[qi, ci]
    rho = spearmanr(s, y)[0]
    lo, hi = np.nanpercentile([spearmanr(s[b], y[b])[0] for b in boot], [2.5, 97.5])
    rows.append((label, col, rho, lo, hi))
    print(f"{label:20s} rho = {rho:.2f} [{lo:.2f}, {hi:.2f}]")

fig, ax = plt.subplots(figsize=(4.0, 1.9))
for k, (label, col, rho, lo, hi) in enumerate(rows):
    yk = len(rows) - 1 - k
    ax.plot([lo, hi], [yk, yk], color=col, lw=2, alpha=0.6, solid_capstyle="round")
    ax.plot(rho, yk, "o", color=col, ms=7, mec="white", mew=1.5, zorder=5)
    ax.text(hi + 0.02, yk, f"{rho:.2f}", va="center", fontsize=8, color=col)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
for t, (_, col, *_r) in zip(ax.get_yticklabels(), rows[::-1]):
    t.set_color(col if col != C_GREY else "#52514e")
ax.axvline(0, color="#52514e", lw=0.8)
ax.set_xlim(-0.05, 0.85)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Spearman $\\rho$ with mean human rating")
ax.text(-0.5, 1.02, "A", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig2a_validation.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig2a_validation.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig2a_validation.pdf")
