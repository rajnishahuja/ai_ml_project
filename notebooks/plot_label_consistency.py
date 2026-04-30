"""
plot_label_consistency.py
=========================
Visualise label distribution and cross-source consistency across
clause types and labeling sources:
  - AGREED (Qwen v1 + Gemini v1 unanimous)
  - MANUAL_REVIEW (human labels)
  - GEMINI_PRO_REVIEW (Gemini 2.5 Pro)
  - SOFT_LABEL v2 — resolved (Qwen v2 + Gemini v2 consensus)
  - SOFT_LABEL v2 — unresolved (still in dispute)
  Plus per-labeler breakdown: Qwen v1, Gemini v1, Qwen v2, Gemini v2

Output: notebooks/label_consistency.png
"""

import csv, json, os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

os.chdir(Path(__file__).parent.parent)

# ── colours ────────────────────────────────────────────────────────────────
C = {"LOW": "#4caf50", "MEDIUM": "#ff9800", "HIGH": "#f44336"}
ORDER = ["LOW", "MEDIUM", "HIGH"]

# ── load master CSV ─────────────────────────────────────────────────────────
with open("data/review/master_label_review.csv", encoding="utf-8") as f:
    master = list(csv.DictReader(f))
master_by_rn = {int(r["row_num"]): r for r in master}

# ── load v2 labeler outputs ─────────────────────────────────────────────────
def load_labeler(path):
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["row_num"]: r for r in data if r["label"] != "ERROR"}

gemini_v2 = load_labeler("data/synthetic/soft_label_relabel_v2_gemini.json")
qwen_v2   = load_labeler("data/synthetic/soft_label_relabel_v2_qwen.json")
analysis  = json.loads(Path("data/synthetic/relabel_v2_analysis.json").read_text(encoding="utf-8"))

# Build 2-tier consensus for SOFT_LABEL rows (reuse analysis output)
cons_by_rn = {r["row_num"]: r for r in analysis}

# ── helper ──────────────────────────────────────────────────────────────────
def pct_series(counter, labels=ORDER):
    total = sum(counter.values())
    return [100 * counter.get(l, 0) / total if total else 0 for l in labels]

# ── build source-level label distributions ──────────────────────────────────
sources = {
    "AGREED\n(Qwen v1+\nGemini v1)":    Counter(),
    "MANUAL\nREVIEW\n(human)":           Counter(),
    "GEMINI PRO\nREVIEW":                Counter(),
    "SOFT v2\nResolved\n(consensus)":    Counter(),
    "SOFT v2\nUnresolved\n(dispute)":    Counter(),
    "Qwen v1\n(all trainable)":          Counter(),
    "Gemini v1\n(all trainable)":        Counter(),
    "Qwen v2\n(SOFT rows)":              Counter(),
    "Gemini v2\n(SOFT rows)":            Counter(),
}

for r in master:
    cat   = r["category"]
    fl    = r.get("final_label", "").strip().upper()
    ql    = r.get("qwen_label",  "").strip().upper()
    gl    = r.get("gemini_label","").strip().upper()

    if cat == "AGREED" and fl in ORDER:
        sources["AGREED\n(Qwen v1+\nGemini v1)"][fl] += 1
    if cat == "MANUAL_REVIEW" and fl in ORDER:
        sources["MANUAL\nREVIEW\n(human)"][fl] += 1
    if cat == "GEMINI_PRO_REVIEW" and fl in ORDER:
        sources["GEMINI PRO\nREVIEW"][fl] += 1
    if ql in ORDER:
        sources["Qwen v1\n(all trainable)"][ql] += 1
    if gl in ORDER:
        sources["Gemini v1\n(all trainable)"][gl] += 1

for row in analysis:
    rn  = row["row_num"]
    if row["resolved"] and row["consensus_label"] in ORDER:
        sources["SOFT v2\nResolved\n(consensus)"][row["consensus_label"]] += 1
    elif not row["resolved"]:
        sources["SOFT v2\nUnresolved\n(dispute)"]["MEDIUM"] += 0   # placeholder — just count
        # use Qwen v2 label as reference for unresolved (disputed)
        lbl = row.get("qwen_label") or row.get("gemini_label")
        if lbl in ORDER:
            sources["SOFT v2\nUnresolved\n(dispute)"][lbl] += 1
    ql2 = row.get("qwen_label")
    gl2 = row.get("gemini_label")
    if ql2 in ORDER:
        sources["Qwen v2\n(SOFT rows)"][ql2] += 1
    if gl2 in ORDER:
        sources["Gemini v2\n(SOFT rows)"][gl2] += 1

# ── build clause-type breakdown ──────────────────────────────────────────────
# For each clause type: collect labels from each source with enough rows (>= 5)
TRAINABLE_CATS = {"AGREED", "MANUAL_REVIEW", "GEMINI_PRO_REVIEW"}

ct_agreed   = defaultdict(Counter)
ct_manual   = defaultdict(Counter)
ct_gempro   = defaultdict(Counter)
ct_soft_res = defaultdict(Counter)
ct_qwen_v2  = defaultdict(Counter)
ct_gem_v2   = defaultdict(Counter)

for r in master:
    cat = r["category"]
    fl  = r.get("final_label", "").strip().upper()
    ct  = r.get("clause_type", "Unknown")
    if cat == "AGREED" and fl in ORDER:
        ct_agreed[ct][fl] += 1
    if cat == "MANUAL_REVIEW" and fl in ORDER:
        ct_manual[ct][fl] += 1
    if cat == "GEMINI_PRO_REVIEW" and fl in ORDER:
        ct_gempro[ct][fl] += 1

for row in analysis:
    ct = master_by_rn[row["row_num"]].get("clause_type", "Unknown")
    if row["resolved"] and row["consensus_label"] in ORDER:
        ct_soft_res[ct][row["consensus_label"]] += 1
    ql2 = row.get("qwen_label")
    gl2 = row.get("gemini_label")
    if ql2 in ORDER:
        ct_qwen_v2[ct][ql2] += 1
    if gl2 in ORDER:
        ct_gem_v2[ct][gl2] += 1

# Pick clause types with >= 10 hard-labeled rows (AGREED source)
top_cts = sorted(
    [(ct, sum(ct_agreed[ct].values())) for ct in ct_agreed if sum(ct_agreed[ct].values()) >= 10],
    key=lambda x: -x[1]
)[:20]
top_ct_names = [x[0] for x in top_cts]

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 36))
fig.patch.set_facecolor("#1a1a2e")
gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35,
                      left=0.08, right=0.97, top=0.96, bottom=0.04)

title_kw  = dict(color="white", fontsize=13, fontweight="bold", pad=10)
label_kw  = dict(color="#cccccc", fontsize=9)
tick_kw   = dict(colors="#aaaaaa", labelsize=8.5)

# ──────────────────────────────────────────────────────────────────────────────
# Panel 1: Source-level overall label distribution (stacked bar)
# ──────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor("#16213e")

src_names = list(sources.keys())
bars = {lbl: [] for lbl in ORDER}
totals = []
for name in src_names:
    cnt = sources[name]
    total = sum(cnt.values())
    totals.append(total)
    for lbl in ORDER:
        bars[lbl].append(100 * cnt.get(lbl, 0) / total if total else 0)

x = np.arange(len(src_names))
w = 0.65
bottom = np.zeros(len(src_names))
for lbl in ORDER:
    vals = np.array(bars[lbl])
    ax1.bar(x, vals, w, bottom=bottom, color=C[lbl], label=lbl, alpha=0.88)
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v > 4:
            ax1.text(xi, b + v/2, f"{v:.0f}%", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")
    bottom += vals

# annotate total N
for xi, n in enumerate(totals):
    ax1.text(xi, 102, f"n={n}", ha="center", va="bottom",
             fontsize=8, color="#bbbbbb")

ax1.set_xticks(x)
ax1.set_xticklabels(src_names, fontsize=9, color="#cccccc")
ax1.set_ylabel("% of rows", **label_kw)
ax1.set_ylim(0, 115)
ax1.set_title("Label Distribution by Source", **title_kw)
ax1.tick_params(axis="y", **tick_kw)
ax1.tick_params(axis="x", colors="#cccccc")
ax1.spines[:].set_color("#333355")
ax1.legend(handles=[mpatches.Patch(color=C[l], label=l) for l in ORDER],
           loc="upper right", fontsize=9, framealpha=0.3,
           labelcolor="white", facecolor="#222244")
ax1.grid(axis="y", color="#333355", alpha=0.5, linewidth=0.5)

# ──────────────────────────────────────────────────────────────────────────────
# Panel 2: Per-clause-type HIGH% comparison across 3 sources
# ──────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
ax2.set_facecolor("#16213e")

src_map = [
    ("AGREED",  ct_agreed,  "#4fc3f7", "AGREED (hard)"),
    ("MANUAL",  ct_manual,  "#ff8a65", "MANUAL REVIEW (human)"),
    ("QWENv2",  ct_qwen_v2, "#ce93d8", "Qwen v2 (SOFT rows)"),
    ("GEMv2",   ct_gem_v2,  "#80cbc4", "Gemini v2 (SOFT rows)"),
    ("SOFTres", ct_soft_res,"#fff176", "SOFT v2 Consensus"),
]

n_cts = len(top_ct_names)
n_src = len(src_map)
group_w = 0.8
bar_w   = group_w / n_src
xs = np.arange(n_cts)

for si, (key, ct_dict, color, label) in enumerate(src_map):
    high_pcts = []
    for ct in top_ct_names:
        cnt   = ct_dict.get(ct, Counter())
        total = sum(cnt.values())
        high_pcts.append(100 * cnt.get("HIGH", 0) / total if total else np.nan)
    offsets = xs + (si - n_src/2 + 0.5) * bar_w
    ax2.bar(offsets, high_pcts, bar_w * 0.9, color=color, alpha=0.82, label=label)

short_names = [ct.replace(" ", "\n") if len(ct) > 18 else ct for ct in top_ct_names]
ax2.set_xticks(xs)
ax2.set_xticklabels(short_names, fontsize=7.5, color="#cccccc", rotation=30, ha="right")
ax2.set_ylabel("% HIGH labels", **label_kw)
ax2.set_title("HIGH Label Rate by Clause Type — Cross-Source Comparison", **title_kw)
ax2.tick_params(axis="y", **tick_kw)
ax2.spines[:].set_color("#333355")
ax2.legend(fontsize=8.5, framealpha=0.3, labelcolor="white", facecolor="#222244",
           loc="upper right")
ax2.grid(axis="y", color="#333355", alpha=0.5, linewidth=0.5)
ax2.set_ylim(0, 100)

# ──────────────────────────────────────────────────────────────────────────────
# Panel 3: Heatmap — HIGH% by clause type × source
# ──────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
ax3.set_facecolor("#16213e")

heat_sources = [
    ("AGREED",  ct_agreed),
    ("Manual",  ct_manual),
    ("GemPro",  ct_gempro),
    ("Qwen v2", ct_qwen_v2),
    ("Gem v2",  ct_gem_v2),
    ("Soft res",ct_soft_res),
]
heat_matrix = []
for ct in top_ct_names:
    row_vals = []
    for _, ct_dict in heat_sources:
        cnt   = ct_dict.get(ct, Counter())
        total = sum(cnt.values())
        row_vals.append(100 * cnt.get("HIGH", 0) / total if total >= 3 else np.nan)
    heat_matrix.append(row_vals)

df_heat = pd.DataFrame(heat_matrix,
                       index=[ct[:30] for ct in top_ct_names],
                       columns=[h[0] for h in heat_sources])

sns.heatmap(df_heat, ax=ax3, cmap="YlOrRd", annot=True, fmt=".0f",
            linewidths=0.4, linecolor="#333355",
            cbar_kws={"label": "% HIGH", "shrink": 0.8},
            annot_kws={"size": 7.5})
ax3.set_title("HIGH% Heatmap: Clause Type × Source", **title_kw)
ax3.tick_params(axis="x", colors="#cccccc", labelsize=8.5, rotation=30)
ax3.tick_params(axis="y", colors="#cccccc", labelsize=7.5)
ax3.figure.axes[-1].tick_params(colors="#aaaaaa", labelsize=7)
ax3.figure.axes[-1].yaxis.label.set_color("#aaaaaa")

# ──────────────────────────────────────────────────────────────────────────────
# Panel 4: MEDIUM% heatmap
# ──────────────────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
ax4.set_facecolor("#16213e")

heat_med = []
for ct in top_ct_names:
    row_vals = []
    for _, ct_dict in heat_sources:
        cnt   = ct_dict.get(ct, Counter())
        total = sum(cnt.values())
        row_vals.append(100 * cnt.get("MEDIUM", 0) / total if total >= 3 else np.nan)
    heat_med.append(row_vals)

df_med = pd.DataFrame(heat_med,
                      index=[ct[:30] for ct in top_ct_names],
                      columns=[h[0] for h in heat_sources])

sns.heatmap(df_med, ax=ax4, cmap="Blues", annot=True, fmt=".0f",
            linewidths=0.4, linecolor="#333355",
            cbar_kws={"label": "% MEDIUM", "shrink": 0.8},
            annot_kws={"size": 7.5})
ax4.set_title("MEDIUM% Heatmap: Clause Type × Source", **title_kw)
ax4.tick_params(axis="x", colors="#cccccc", labelsize=8.5, rotation=30)
ax4.tick_params(axis="y", colors="#cccccc", labelsize=7.5)
ax4.figure.axes[-1].tick_params(colors="#aaaaaa", labelsize=7)
ax4.figure.axes[-1].yaxis.label.set_color("#aaaaaa")

# ──────────────────────────────────────────────────────────────────────────────
# Panel 5: Clause-type stacked bars — AGREED source (full distribution)
# ──────────────────────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
ax5.set_facecolor("#16213e")

bottom5 = np.zeros(n_cts)
for lbl in ORDER:
    vals = np.array([100 * ct_agreed[ct].get(lbl, 0) / max(sum(ct_agreed[ct].values()), 1)
                     for ct in top_ct_names])
    ax5.barh(np.arange(n_cts), vals, left=bottom5, color=C[lbl], alpha=0.85, label=lbl)
    bottom5 += vals

ax5.set_yticks(np.arange(n_cts))
ax5.set_yticklabels([ct[:32] for ct in top_ct_names], fontsize=8, color="#cccccc")
ax5.set_xlabel("% of rows", **label_kw)
ax5.set_title("AGREED (Hard Labels) — Full Distribution", **title_kw)
ax5.tick_params(axis="x", **tick_kw)
ax5.spines[:].set_color("#333355")
ax5.set_xlim(0, 105)
ax5.legend(handles=[mpatches.Patch(color=C[l], label=l) for l in ORDER],
           fontsize=8, framealpha=0.3, labelcolor="white", facecolor="#222244",
           loc="lower right")
ax5.grid(axis="x", color="#333355", alpha=0.5, linewidth=0.5)

# ──────────────────────────────────────────────────────────────────────────────
# Panel 6: Qwen v2 vs Gemini v2 agreement rate by clause type
# ──────────────────────────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
ax6.set_facecolor("#16213e")

agree_rates = []
total_counts = []
for ct in top_ct_names:
    qc = ct_qwen_v2.get(ct, Counter())
    gc = ct_gem_v2.get(ct, Counter())
    # rows labeled by both: use analysis
    ct_rows = [r for r in analysis if master_by_rn[r["row_num"]].get("clause_type") == ct]
    if not ct_rows:
        agree_rates.append(np.nan)
        total_counts.append(0)
        continue
    agreed_n = sum(1 for r in ct_rows if r["resolved"])
    total_n  = len(ct_rows)
    agree_rates.append(100 * agreed_n / total_n)
    total_counts.append(total_n)

colors_bar = ["#81c784" if r >= 70 else "#ffb74d" if r >= 50 else "#e57373"
              for r in [x if not np.isnan(x) else 0 for x in agree_rates]]

bars6 = ax6.barh(np.arange(n_cts), agree_rates, color=colors_bar, alpha=0.85)
ax6.axvline(64, color="#ffffff", linewidth=1.2, linestyle="--", alpha=0.5, label="Overall avg 64%")
for xi, (rate, n) in enumerate(zip(agree_rates, total_counts)):
    if not np.isnan(rate):
        ax6.text(min(rate + 1, 99), xi, f"{rate:.0f}% (n={n})",
                 va="center", fontsize=7, color="#cccccc")

ax6.set_yticks(np.arange(n_cts))
ax6.set_yticklabels([ct[:32] for ct in top_ct_names], fontsize=8, color="#cccccc")
ax6.set_xlabel("Agreement rate (%)", **label_kw)
ax6.set_title("Qwen v2 + Gemini v2 Agreement Rate by Clause Type", **title_kw)
ax6.tick_params(axis="x", **tick_kw)
ax6.spines[:].set_color("#333355")
ax6.set_xlim(0, 115)
ax6.legend(fontsize=8, framealpha=0.3, labelcolor="white", facecolor="#222244")
ax6.grid(axis="x", color="#333355", alpha=0.5, linewidth=0.5)

# ── save ─────────────────────────────────────────────────────────────────────
OUT = Path("notebooks/label_consistency.png")
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")
