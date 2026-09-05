#!/usr/bin/env python3
"""Paper figures — every datapoint traces to results/ or a ledger record.

Outputs PDF (vector) into paper/figs/. Style: clean, grayscale-safe,
single-column friendly (3.3in) except fig1 (full width 7in).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "figure.dpi": 150,
})

C_CAT = "#B03A2E"    # catalog
C_BASE = "#1F618D"   # baseline / dense twin
C_DEAD = "#7B241C"
C_LIVE = "#1E8449"
C_GRAY = "#666666"


# ---------------------------------------------------------------------------
# FIG 1 (headline): the silent channel collapse — loss looks fine, channel dead
# Data: results/nanochat/PHASE5-CHANNEL-COLLAPSE.md (wire analysis, launch 7)
#       + arm-catalog logs (val bpb) + launch-8 telemetry (live channel).
# ---------------------------------------------------------------------------
def fig1_collapse():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))

    # (a) verdict mix across steps, launch 7 (dead) — measured wire dumps
    steps_dead = [0, 100, 200, 300, 500]
    commit_dead = [100.0, 0.066, 0.0, 0.0, 0.0]   # step0=structural (zero deltas commit)
    ax = axes[0]
    ax.plot(steps_dead, commit_dead, "o-", color=C_DEAD, label="commit rate")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("training step")
    ax.set_ylabel("commit rate (%)")
    ax.set_title("(a) channel death (d20, launch 7)", fontsize=8)
    ax.annotate("last commit\nstep ~150", xy=(200, 1), xytext=(230, 30),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_ylim(-4, 104)

    # (b) …while validation bpb improves normally (same run)
    ax = axes[1]
    bpb_steps = [0, 250, 500, 750]
    bpb = [3.169, 1.562, 1.368, 1.319]
    ax.plot(bpb_steps, bpb, "s-", color=C_GRAY)
    ax.set_xlabel("training step")
    ax.set_ylabel("val bits-per-byte")
    ax.set_title("(b) loss looks healthy (same run)", fontsize=8)
    ax.annotate("model routes around\nits own dead channel", xy=(500, 1.37),
                xytext=(180, 2.35), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.6))

    # (c) with the proposal clip: channel alive (launch 8 telemetry, same cfg)
    ax = axes[2]
    steps_live = [1, 2, 3, 4, 50, 100, 150, 200, 250, 300, 350, 400, 450,
                  500, 550, 600, 650, 700, 750]
    commit_live = [32.8, 34.3, 34.9, 36.2, 39.5, 28.4, 28.0, 34.2, 40.4,
                   31.0, 24.0, 23.0, 22.7, 22.7, 23.4, 22.9, 23.6, 20.5, 21.1]
    ax.plot(steps_live, commit_live, "-", color=C_LIVE, label="with clip (live)")
    ax.plot(steps_dead[1:], commit_dead[1:], "o--", color=C_DEAD,
            label="no clip (dead)", ms=3)
    ax.set_xlabel("training step")
    ax.set_ylabel("commit rate (%)")
    ax.set_title("(c) proposal clip keeps it alive", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_ylim(-4, 55)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_collapse.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG 2: preregistration catches our own headline — LR sweep reversal
# Data: results/train/NVIDIA-H200/lrsweep-441b9d2f6a4f/NOTES.md
# ---------------------------------------------------------------------------
def fig2_lrsweep():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    lrs = [1e-4, 3e-4, 6e-4, 1.2e-3]
    cat = [187.84, 144.13, 167.80, 153.44]
    base = [58.65, 49.33, 304.70, 303.93]
    ax.plot(lrs, base, "s-", color=C_BASE, label="dense twin")
    ax.plot(lrs, cat, "o-", color=C_CAT, label="catalog")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("val PPL (wikitext-103, 20k steps)")
    ax.axvline(6e-4, color=C_GRAY, lw=0.5, ls=":")
    ax.annotate('single-LR "PASS"\n(−44.9%)', xy=(6e-4, 200), xytext=(1.35e-4, 350),
                fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.annotate("best-vs-best:\ntwin 2.9× better", xy=(3e-4, 49.33),
                xytext=(4.6e-4, 62), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_lrsweep.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG 3: seed robustness (the property that DID survive the sweep)
# Data: results/train/NVIDIA-H200/seeds-441b9d2f6a4f/NOTES.md
# ---------------------------------------------------------------------------
def fig3_seeds():
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    seeds = ["1337", "2026", "4242"]
    cat = [167.80, 167.68, 166.15]
    base = [304.70, 291.29, 313.87]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, base, w, color=C_BASE, label="dense twin (@6e-4)")
    ax.bar(x + w / 2, cat, w, color=C_CAT, label="catalog (@6e-4)")
    for i, (b, c) in enumerate(zip(base, cat)):
        ax.text(i - w / 2, b + 4, f"{b:.0f}", ha="center", fontsize=6)
        ax.text(i + w / 2, c + 4, f"{c:.0f}", ha="center", fontsize=6)
    ax.set_xticks(x, seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("val PPL")
    ax.set_title("seed spread: twin 7.5%, catalog 1.0%", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_seeds.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG 4: fault injection — 36/36 with correct failure class
# Data: results/inject/inject-summary.json
# ---------------------------------------------------------------------------
def fig4_inject():
    import json
    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                           "inject", "inject-summary.json")) as f:
        d = json.load(f)["summary"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    names = list(d["by_fault"].keys())
    counts = [len(v) for v in d["by_fault"].values()]
    klass = [v[0] for v in d["by_fault"].values()]
    y = np.arange(len(names))
    ax.barh(y, counts, color=C_BASE, height=0.6)
    for i, (c, k) in enumerate(zip(counts, klass)):
        ax.text(c - 0.15, i, k.replace("_", " "), ha="right", va="center",
                fontsize=6, color="white")
    ax.set_yticks(y, [n.replace("_", " ") for n in names], fontsize=6.5)
    ax.set_xlabel("injections caught (of injected)")
    ax.set_title(f"fault injection: {d['caught']}/{d['injected']} caught (100%)",
                 fontsize=8)
    ax.set_xlim(0, 6.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_inject.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG 5: the seam architecture (schematic — no external data)
# ---------------------------------------------------------------------------
def fig5_architecture():
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F4F6F7", ec="k", fontsize=7, lw=0.8):
        r = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw,
                          zorder=2)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, zorder=3)

    def arrow(x0, y0, x1, y1, text="", above=True, color="k"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=color))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + (0.05 if above else -0.09),
                    text, ha="center", fontsize=6, color=color)

    # model column
    box(0.02, 0.55, 0.18, 0.32, "transformer block\n(attn / mlp)\nPROPOSES $\\delta$")
    box(0.02, 0.10, 0.18, 0.30, "HyphaeWrite\nthe only writer of $h$\npack$\\to$allocate$\\to$apply",
        fontsize=6.5)
    arrow(0.11, 0.55, 0.11, 0.42)

    # verdicts
    box(0.26, 0.10, 0.16, 0.77,
        "verdict stream\n16B facts\nCOMMIT\nOVERFLOW\n(contention)\nABORT (rule)",
        fc="#FDF2E9")
    arrow(0.20, 0.25, 0.26, 0.35)

    # seam
    box(0.48, 0.42, 0.18, 0.45, "seam (rank 0)\nT2 chain over wire\nHyphae ledger\n(durable, embedded)",
        fc="#EBF5FB")
    arrow(0.42, 0.55, 0.48, 0.60, "spool")
    box(0.48, 0.10, 0.18, 0.24, "CPU verifier\nreplays spills\nbit-for-bit", fc="#E9F7EF")
    arrow(0.57, 0.42, 0.57, 0.34)

    # optimizer gate
    box(0.73, 0.55, 0.25, 0.32, "optimizer.step()\nGATED by CommitReceipt\nno receipt $\\Rightarrow$ all ranks die",
        fc="#F9EBEA", fontsize=6.5)
    arrow(0.66, 0.68, 0.73, 0.68, "receipt\nhead")
    box(0.73, 0.10, 0.25, 0.30, "STEP chain\n$c_{prev} \\to c_{next}$\ncodebook moves only\ninside the ledger", fontsize=6.5)
    arrow(0.855, 0.55, 0.855, 0.42)

    fig.savefig(os.path.join(OUT, "fig5_architecture.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG 6: two-phase selection (policy 7) — Trainium port schematic
# ---------------------------------------------------------------------------
def fig6_twophase():
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, title, sub, fc):
        r = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="k", lw=0.7)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top",
                fontsize=7.2, weight="bold")
        ax.text(x + w / 2, y + 0.045, sub, ha="center", va="bottom", fontsize=6.8)

    def arrow(y0, y1, text):
        ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.9))
        ax.text(0.53, (y0 + y1) / 2, text, fontsize=6.8, ha="left", va="center")

    box(0.02, 0.74, 0.96, 0.24, "A — device proposes (fast, NOT verified)",
        "bf16 matmul scores $\\to$ top-$M$ candidate features", "#FDF2E9")
    box(0.02, 0.40, 0.96, 0.24, "B — reference disposes (exact, verified)",
        "recompute $M$ scores, pinned math, policy-6 verdicts", "#E9F7EF")
    box(0.02, 0.06, 0.96, 0.24, "C — device applies (bit contract)",
        "$h' = \\mathrm{bf16}(\\mathrm{fp32}(\\mathrm{bf16}(h)) + \\mathrm{mag}\\cdot\\hat{n}_f)$",
        "#EBF5FB")
    arrow(0.74, 0.645, "$M$ ids")
    arrow(0.40, 0.305, "facts")
    fig.savefig(os.path.join(OUT, "fig6_twophase.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_collapse()
    fig2_lrsweep()
    fig3_seeds()
    fig4_inject()
    fig5_architecture()
    fig6_twophase()
    print("figures written to", OUT)


# ---------------------------------------------------------------------------
# FIG 7: phase-5 result — CORE per task, live catalog vs dense twin
# Data: results/models/d20-p5/eval/base_eval_{catalog,vanilla}.csv
# ---------------------------------------------------------------------------
def fig7_core():
    import csv
    base = os.path.join(os.path.dirname(__file__), "..", "results", "models", "d20-p5", "eval")
    def load(name):
        rows = {}
        with open(os.path.join(base, f"base_eval_{name}.csv")) as f:
            for r in csv.reader(f):
                if len(r) < 3 or r[0].strip() in ("Task",): continue
                rows[r[0].strip()] = float(r[2]) if r[2].strip() else None
        return rows
    cat, van = load("catalog"), load("vanilla")
    tasks = [t for t in van if t != "CORE" and van[t] is not None]
    tasks.sort(key=lambda t: van[t], reverse=True)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(tasks)); w = 0.4
    ax.bar(x - w/2, [van[t] for t in tasks], w, color=C_BASE, label=f"dense twin (CORE {van['CORE']:.3f})")
    ax.bar(x + w/2, [cat[t] for t in tasks], w, color=C_CAT, label=f"catalog, live channel (CORE {cat['CORE']:.3f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x, [t.replace("bigbench_", "bb_").replace("_zeroshot", "_0s") for t in tasks], fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("centered accuracy")
    ax.set_title("nanochat d20, 4980 steps, same data/silicon: CORE per task", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_core.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG 8: third silicon — Trainium2 d10, catalog (policy 7) vs dense twin
# Data: results/trn2-d10/curves.csv (from catalog7.log / vanilla6.log; ledgers in S3)

def fig8_trainium():
    import csv
    path = os.path.join(os.path.dirname(__file__), "..", "results", "trn2-d10", "curves.csv")
    rows = list(csv.DictReader(open(path)))
    f = lambda k: [(int(r["step"]), float(r[k])) for r in rows if r[k] != ""]
    lv, lc = f("loss_vanilla"), f("loss_catalog")
    ent, com = f("usage_entropy_bits"), f("commit_pct")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax.plot(*zip(*lv), color=C_BASE, marker="o", ms=2.5, label="dense twin (84.5k tok/s)")
    ax.plot(*zip(*lc), color=C_CAT, marker="s", ms=2.5, label="catalog, policy 7 (2.8k tok/s)")
    ax.set_xlabel("step"); ax.set_ylabel("train loss")
    ax.set_title("(a) Trainium2 d10 (60M), 400 steps, same seed/data", fontsize=8)
    ax.annotate("4.67", xy=lc[-1], xytext=(-28, 6), textcoords="offset points", fontsize=7, color=C_CAT)
    ax.annotate("3.41", xy=lv[-1], xytext=(-28, -10), textcoords="offset points", fontsize=7, color=C_BASE)
    ax.legend(fontsize=7, loc="upper right")
    ax2.plot(*zip(*ent), color=C_LIVE, marker="o", ms=2.5, label="codebook usage entropy (bits)")
    ax2.axhline(8.0, color=C_GRAY, lw=0.8, ls="--")
    ax2.text(203, 7.25, "guard threshold 8 b", fontsize=6.5, color=C_GRAY)
    ax2.axvspan(0, 200, color=C_GRAY, alpha=0.08, lw=0)
    ax2.text(100, 0.35, "declared warm-up (200 steps)", fontsize=6.5, color=C_GRAY, ha="center")
    ax2.axvline(149, color=C_DEAD, lw=0.8, ls=":")
    ax2.text(153, 3.1, "guard kill @149\n(manifest 1: 100-step warm-up)", fontsize=6, color=C_DEAD)
    ax2.set_ylabel("bits", color=C_LIVE); ax2.set_ylim(0, 10.5); ax2.set_xlabel("step")
    ax3 = ax2.twinx()
    ax3.plot(*zip(*com), color=C_CAT, marker="s", ms=2.5, lw=0.9, alpha=0.8, label="commit rate (%)")
    ax3.set_ylabel("commit %", color=C_CAT); ax3.set_ylim(0, 100)
    ax2.set_title("(b) the channel under the executable guard", fontsize=8)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_trainium.pdf"), bbox_inches="tight")
    plt.close(fig)


ALL = [fig1_collapse, fig2_lrsweep, fig3_seeds, fig4_inject, fig5_architecture,
       fig6_twophase, fig7_core, fig8_trainium]


def export_png(dpi=200):
    """README copies: paper/figs/png/<name>.png (raster, GitHub renders them)."""
    import matplotlib.backends.backend_pdf  # noqa: F401  (ensure backend)
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
    png_dir = os.path.join(OUT, "png"); os.makedirs(png_dir, exist_ok=True)
    for name in sorted(os.listdir(OUT)):
        if not name.endswith(".pdf"):
            continue
        src = os.path.join(OUT, name); dst = os.path.join(png_dir, name[:-4] + ".png")
        if fitz is not None:
            doc = fitz.open(src); pix = doc[0].get_pixmap(dpi=dpi); pix.save(dst)
        else:
            import subprocess
            subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", src, dst[:-4]], check=True)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["all"]
    if "all" in which:
        for fn in ALL:
            fn()
        export_png()
    else:
        for w in which:
            if w != "png":
                globals()[w]()
        if "png" in which:
            export_png()
