"""Knowledge-boundary task builder (PHASE4-MECHANISM.md experiment).

Mechanical confabulation labels WITHOUT an LLM judge: entities binned by
their frequency IN THE MODEL'S OWN TRAINING SHARDS (which we have, hashed).

  E_high : entity appears >= high_min times   -> model should know it
  E_low  : 2..low_max times                   -> the borderline (Karpathy zone)
  E_zero : 0 times (verified by scan)         -> pure OOD; anything specific
                                                 the model asserts is invented

Prompts are identical in FORM across bins (cloze over真 sentences for
E_high/E_low, template questions for all three), so the only varying factor
is the knowledge boundary. Labels for generation:
  - E_high/E_low cloze: correct = exact next-token match vs the corpus span.
  - E_zero: any confident completion is a confabulation BY CONSTRUCTION.

Usage (on the droplet, where the shards live):
  python -m hytorch.knowledge_boundary --shards /var/lib/hytorch/arm-catalog/base_data_climbmix \
      --tokenizer-dir /var/lib/hytorch/nanochat-cache/tokenizer --out kb.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

# Simple capitalized-span entity heuristic: 2-3 capitalized words (names,
# places, orgs). Deliberately dumb and transparent — auditable, no NER model.
ENT_RE = re.compile(r"\b([A-Z][a-z]{2,})(?: ([A-Z][a-z]{2,})){1,2}\b")
STOP = {"The", "This", "That", "There", "These", "Those", "When", "Where",
        "What", "Which", "While", "After", "Before", "During", "However",
        "Although", "Because", "Between", "About", "Against", "Among"}


def scan_shards(shard_dir: str, max_files: int = 0):
    """Count entity spans + keep one context sentence per entity."""
    import pyarrow.parquet as pq

    counts: Counter = Counter()
    context: dict[str, str] = {}
    files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".parquet"))
    if max_files:
        files = files[:max_files]
    for fn in files:
        table = pq.read_table(os.path.join(shard_dir, fn), columns=["text"])
        for batch in table.to_batches():
            for text in batch.column("text").to_pylist():
                if not text:
                    continue
                for m in ENT_RE.finditer(text):
                    ent = m.group(0)
                    if ent.split()[0] in STOP:
                        continue
                    counts[ent] += 1
                    if ent not in context and counts[ent] >= 2:
                        # keep a sentence containing the entity for cloze
                        s = max(m.start() - 200, 0)
                        snippet = text[s:m.start() + len(ent) + 100]
                        context[ent] = snippet
    return counts, context


# Real-world entities engineered to be ABSENT from a fineweb-edu subset with
# high probability; presence is VERIFIED by the scan before use (any that
# appear get dropped). Fictional-but-plausible names → E_zero by construction.
ZERO_CANDIDATES = [
    "Tormund Vexley", "Ilaria Brancusi Voss", "Kaspar Windemuth",
    "Odalys Ferreira Quint", "Bartholomew Krezner", "Yevgenia Talmadge",
    "Rustam Oyelaran", "Priya Vanterpool", "Silvano Achterberg",
    "Marisol Uzcategui Renn", "Declan Furuhashi", "Anneliese Vogt Okafor",
    "Cormac Zielinski Rhee", "Beatrix Falconer Adeyemi", "Ozren Maltby",
    "Xiomara Pellworth", "Gideon Rasmussen Choi", "Perpetua Lindqvist",
    "Thaddeus Okonkwo Blair", "Ingrid Palamara",
]

TEMPLATES = [
    "Question: Who is {e}? Answer: {e} is",
    "{e} is best known for",
    "The most notable achievement of {e} was",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--high-min", type=int, default=100)
    ap.add_argument("--low-max", type=int, default=10)
    ap.add_argument("--per-bin", type=int, default=120)
    args = ap.parse_args()

    counts, context = scan_shards(args.shards, args.max_files)
    print(f"scanned: {len(counts):,} distinct entity spans")

    high = [(e, c) for e, c in counts.most_common() if c >= args.high_min][:args.per_bin]
    low = [(e, c) for e, c in counts.items()
           if 2 <= c <= args.low_max and e in context][:args.per_bin]
    zero = [(e, 0) for e in ZERO_CANDIDATES if counts.get(e, 0) == 0]
    dropped = [e for e in ZERO_CANDIDATES if counts.get(e, 0) > 0]
    if dropped:
        print(f"E_zero candidates present in corpus, dropped: {dropped}")

    task = {"bins": {}, "templates": TEMPLATES,
            "params": {"high_min": args.high_min, "low_max": args.low_max}}
    for name, ents in (("E_high", high), ("E_low", low), ("E_zero", zero)):
        task["bins"][name] = [
            {"entity": e, "train_count": c, "context": context.get(e, "")[:400]}
            for e, c in ents
        ]
        print(f"{name}: {len(ents)} entities")

    with open(args.out, "w") as f:
        json.dump(task, f, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
