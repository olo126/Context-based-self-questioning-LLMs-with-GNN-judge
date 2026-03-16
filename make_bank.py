#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def iter_cleaned(ds, require_min_chars: int, max_chars: int, source_name: str):
    for ex in ds:
        txt = ex.get("text")
        if not isinstance(txt, str):
            continue

        txt = clean_text(txt)
        if len(txt) < require_min_chars:
            continue

        if max_chars and len(txt) > max_chars:
            txt = txt[: max_chars] + "…"

        yield {
            "text": txt,
            "source_dataset": source_name,
            "source": ex.get("source", None),
            "url": ex.get("url", None),
        }


def take_n_records(ds, n: int, require_min_chars: int, max_chars: int, source_name: str):
    out = []
    for rec in iter_cleaned(ds, require_min_chars, max_chars, source_name):
        out.append(rec)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output bank jsonl path")
    ap.add_argument("--fineweb-docs", type=int, default=2500, help="How many FineWeb docs to save")
    ap.add_argument("--openwebmath-docs", type=int, default=2500, help="How many OpenWebMath docs to save")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--split", default="train", help="Split name (default: train)")
    ap.add_argument("--buffer-size", type=int, default=50_000,
                    help="Streaming shuffle buffer_size")
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="Max characters to store per doc (truncate for speed). 0 = no truncation")
    ap.add_argument("--require-min-chars", type=int, default=300,
                    help="Skip docs shorter than this after cleaning")
    ap.add_argument("--cache-dir", default=None,
                    help="Optional HF datasets cache dir")
    ap.add_argument("--fineweb-config", default="default",
                    help="FineWeb subset/config name (default: default)")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fineweb = load_dataset(
        "HuggingFaceFW/fineweb",
        args.fineweb_config,
        split=args.split,
        streaming=True,
        cache_dir=args.cache_dir,
    ).shuffle(seed=args.seed, buffer_size=args.buffer_size)

    openwebmath = load_dataset(
        "open-web-math/open-web-math",
        split=args.split,
        streaming=True,
        cache_dir=args.cache_dir,
    ).shuffle(seed=args.seed + 1, buffer_size=args.buffer_size)

    fineweb_recs = take_n_records(
        fineweb,
        args.fineweb_docs,
        args.require_min_chars,
        args.max_chars,
        "fineweb",
    )
    openwebmath_recs = take_n_records(
        openwebmath,
        args.openwebmath_docs,
        args.require_min_chars,
        args.max_chars,
        "openwebmath",
    )

    all_recs = fineweb_recs + openwebmath_recs
    random.shuffle(all_recs)

    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(all_recs):
            rec["id"] = i
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(all_recs)} docs to: {out_path}")
    print(f"  FineWeb: {len(fineweb_recs)}")
    print(f"  OpenWebMath: {len(openwebmath_recs)}")


if __name__ == "__main__":
    main()
