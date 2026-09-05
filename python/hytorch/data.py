"""wikitext-103 → memmap token bins (gpt2 BPE via tiktoken).

Produces train.bin / val.bin of uint16 tokens plus data-manifest.json with
sha256 of each bin — the dataset becomes a citable fact of the run.

Usage:
  python -m hytorch.data --out /data/wikitext103
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request

import numpy as np

BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1"
FILES = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "val": ["validation-00000-of-00001.parquet"],
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    print(f"  fetch {os.path.basename(dest)}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.rename(tmp, dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-only", action="store_true", help="skip train (quick eval sets)")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256
    os.makedirs(args.out, exist_ok=True)
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    manifest = {"dataset": "wikitext-103-raw-v1", "tokenizer": "gpt2-bpe(tiktoken)",
                "eot": eot, "splits": {}}

    splits = ["val"] if args.val_only else ["train", "val"]
    for split in splits:
        parts = []
        for fname in FILES[split]:
            dest = os.path.join(raw_dir, fname)
            download(f"{BASE}/{fname}", dest)
            parts.append(dest)

        bin_path = os.path.join(args.out, f"{split}.bin")
        if not os.path.exists(bin_path):
            print(f"  tokenize {split}…")
            token_chunks: list[np.ndarray] = []
            n_docs = 0
            for p in parts:
                table = pq.read_table(p, columns=["text"])
                for batch in table.to_batches():
                    for text in batch.column("text").to_pylist():
                        if not text:
                            continue
                        ids = enc.encode_ordinary(text)
                        if not ids:
                            continue
                        ids.append(eot)
                        token_chunks.append(np.asarray(ids, dtype=np.uint16))
                        n_docs += 1
            tokens = np.concatenate(token_chunks)
            mm = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(len(tokens),))
            mm[:] = tokens
            mm.flush()
            del mm
            print(f"  {split}: {n_docs} docs, {len(tokens):,} tokens")

        manifest["splits"][split] = {
            "bin": os.path.basename(bin_path),
            "tokens": int(os.path.getsize(bin_path) // 2),
            "sha256": sha256_file(bin_path),
        }

    with open(os.path.join(args.out, "data-manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


class BinDataset:
    """Deterministic random-window sampler over a token bin (seeded)."""

    def __init__(self, bin_path: str, seq_len: int, batch: int, seed: int):
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.batch = batch
        self.rng = np.random.default_rng(seed)

    def sample(self):
        import torch

        ix = self.rng.integers(0, len(self.tokens) - self.seq_len - 1, size=self.batch)
        x = np.stack([self.tokens[i:i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.tokens[i + 1:i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def eval_batches(self, n_batches: int):
        """Sequential non-overlapping windows for a stable PPL eval."""
        import torch

        out = []
        stride = self.seq_len + 1
        for b in range(n_batches):
            xs, ys = [], []
            for j in range(self.batch):
                i = (b * self.batch + j) * stride
                if i + stride >= len(self.tokens):
                    break
                xs.append(self.tokens[i:i + self.seq_len].astype(np.int64))
                ys.append(self.tokens[i + 1:i + 1 + self.seq_len].astype(np.int64))
            if not xs:
                break
            out.append((torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))))
        return out


if __name__ == "__main__":
    raise SystemExit(main())
