#!/usr/bin/env python3
"""First look at what Hyphae has seen (phase 5, d20) — wire-dump analysis.

Flow (on the droplet, against the volume SNAPSHOT — never the live ledger):
  hytorch-ledger-query /mnt/hyvol/nano-hyphae dump-wire nano-d20-catalog 500 /tmp/w500
  python3 ledger_analysis.py /tmp/w500 [/tmp/w400 ...]

Per wired step: verdict mix, reasons, feature usage (entropy, head mass,
distinct), attn-vs-mlp split by unit (layer%2), magnitude percentiles.
These are the raw observables behind H1/H3 of PHASE4-MECHANISM.md — computed
on live training data for the first time.
"""

import glob
import json
import os
import sys
from collections import Counter

import numpy as np

WIRE_DTYPE = np.dtype([
    ("feature", "<u2"), ("slot", "u1"), ("device", "u1"), ("mag_bf16", "<u2"),
    ("layer", "<u2"), ("pos", "<u4"), ("cand", "<u2"), ("verdict", "u1"),
    ("reason", "u1"),
])


def bf16_abs(bits):
    b = (bits.astype(np.uint32) & 0x7FFF) << 16
    return np.abs(b.view(np.float32))


def analyze_dir(d):
    files = sorted(glob.glob(os.path.join(d, "frame-*.bin")))
    if not files:
        return None
    recs = np.concatenate([np.frombuffer(open(f, "rb").read(), dtype=WIRE_DTYPE)
                           for f in files])
    v = recs["verdict"]
    commits = recs[v == 0]
    out = {
        "dir": d,
        "n_frames": len(files),
        "n_records": int(len(recs)),
        "verdicts": {
            "commit": int((v == 0).sum()),
            "overflow": int((v == 1).sum()),
            "abort": int((v == 2).sum()),
        },
        "reasons_nonzero": {int(r): int(c) for r, c in
                            zip(*np.unique(recs["reason"][v != 0], return_counts=True))},
    }
    if len(commits):
        feats, counts = np.unique(commits["feature"], return_counts=True)
        p = counts / counts.sum()
        order = np.argsort(counts)[::-1]
        n_head = max(1, int(32768 * 0.01))          # top-1% of the CATALOG (327)
        head_ids = feats[order[:n_head]]
        head_mask = np.isin(commits["feature"], head_ids)
        mags = bf16_abs(commits["mag_bf16"])
        unit = commits["layer"] % 2                  # frame layer id = 2L+unit
        e_attn = float(mags[unit == 0].sum())
        e_mlp = float(mags[unit == 1].sum())
        out.update({
            "distinct_features": int(len(feats)),
            "feature_entropy_bits": round(float(-(p * np.log2(p)).sum()), 3),
            "max_possible_entropy": round(float(np.log2(len(feats))), 3),
            "top10": [[int(feats[i]), int(counts[i])] for i in order[:10]],
            "head1pct_commit_frac": round(float(head_mask.mean()), 4),
            "head1pct_energy_frac": round(float(mags[head_mask].sum() / mags.sum()), 4),
            "attn_mlp_commit_ratio": round(float((unit == 0).sum() / max(1, (unit == 1).sum())), 3),
            "attn_mlp_energy_ratio": round(e_attn / max(1e-9, e_mlp), 3),
            "mag_p50": round(float(np.percentile(mags, 50)), 5),
            "mag_p90": round(float(np.percentile(mags, 90)), 5),
            "mag_p99": round(float(np.percentile(mags, 99)), 5),
            "mag_max": round(float(mags.max()), 4),
            "mag_at_cap_frac": round(float((mags >= 63.5).mean()), 6),
        })
        # per-layer commit distribution (are some layers writing much more?)
        L = commits["layer"] // 2
        lay, layc = np.unique(L, return_counts=True)
        out["commits_by_layer"] = {int(l): int(c) for l, c in zip(lay, layc)}
    return out


def main():
    results = [r for r in (analyze_dir(d) for d in sys.argv[1:]) if r]
    for r in results:
        print(json.dumps(r), flush=True)
    if len(results) >= 2:
        print(json.dumps({"trend": {
            "dirs": [r["dir"] for r in results],
            "entropy": [r.get("feature_entropy_bits") for r in results],
            "distinct": [r.get("distinct_features") for r in results],
            "head_energy": [r.get("head1pct_energy_frac") for r in results],
            "overflow": [r["verdicts"]["overflow"] for r in results],
            "attn_mlp_energy": [r.get("attn_mlp_energy_ratio") for r in results],
        }}, indent=1))


if __name__ == "__main__":
    main()
