"""
Download Amazon Electronics reviews to data/raw.parquet.

    python -m scripts.setup_data              # try real data (HuggingFace, then UCSD)
    python -m scripts.setup_data --synthetic  # synthetic data for debug pass only

The pipeline's data stage reads data/raw.parquet and skips downloading.
Real data is required for Pass 1 and Pass 2 dissertation results.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import pathlib
import random
import ssl
import sys

# Bypass SSL for Python stdlib
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

# Bypass SSL for requests / urllib3
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["CURL_CA_BUNDLE"] = ""
os.environ.pop("HF_HUB_OFFLINE", None)

import requests   # noqa: E402
import urllib3    # noqa: E402

urllib3.disable_warnings()

# Patch every requests.Session so HuggingFace Hub calls skip cert check too.
_orig_send = requests.Session.send


def _send_no_verify(self, *args, **kwargs):
    kwargs["verify"] = False
    return _orig_send(self, *args, **kwargs)


requests.Session.send = _send_no_verify  # type: ignore[method-assign]

import pyarrow as pa         # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

N   = 5_000
CAT = "Electronics"
SRC = "McAuley-Lab/Amazon-Reviews-2023"
OUT = pathlib.Path("data/raw.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = pa.schema([
    ("user",   pa.string()),
    ("item",   pa.string()),
    ("rating", pa.float64()),
    ("review", pa.string()),
])


def _to_row(r: dict) -> dict:
    return {
        "user":   str(r.get("user_id",     r.get("reviewerID", ""))),
        "item":   str(r.get("parent_asin", r.get("asin", ""))),
        "rating": float(r.get("rating",    r.get("overall", 0.0))),
        "review": str(r.get("text",        r.get("reviewText", "")) or r.get("title", "")),
    }


# ── Attempt 1: HuggingFace datasets streaming ────────────────────────────────

def _try_huggingface() -> list[dict]:
    try:
        import datasets as hf
    except ImportError:
        print("  datasets library not available")
        return []
    for ext, fmt in [(".jsonl.gz", "json"), (".parquet", "parquet"), (".jsonl", "json")]:
        try:
            print(f"  trying HuggingFace datasets ({ext}) …")
            url = f"hf://datasets/{SRC}/raw_review_{CAT}/*{ext}"
            ds = hf.load_dataset(fmt, data_files={"full": url}, split="full", streaming=True)
            rows: list[dict] = []
            for rec in ds:
                rows.append(_to_row(rec))
                if len(rows) >= N:
                    break
            if rows:
                print(f"  got {len(rows):,} rows via HuggingFace ({ext})")
                return rows
        except Exception as e:
            print(f"  HuggingFace {ext} failed: {e}")
    return []


# ── Attempt 2: UCSD direct download (McAuley group hosts the original data) ──

def _try_ucsd() -> list[dict]:
    base = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories"
    url  = f"{base}/{CAT}.jsonl.gz"
    print(f"  trying UCSD direct download …")
    print(f"  {url}")
    try:
        resp = requests.get(url, stream=True, verify=False, timeout=60)
        resp.raise_for_status()
        rows: list[dict] = []
        buf = b""
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            buf += chunk
            try:
                with gzip.open(io.BytesIO(buf), "rt", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        rows.append(_to_row(json.loads(line)))
                        if len(rows) >= N:
                            break
            except (gzip.BadGzipFile, EOFError):
                pass  # need more chunks to decompress
            if len(rows) >= N:
                break
        if rows:
            print(f"  got {len(rows):,} rows via UCSD")
            return rows
    except Exception as e:
        print(f"  UCSD download failed: {e}")
    return []


# ── Synthetic data (debug pass only) ─────────────────────────────────────────

REVIEW_TEMPLATES = [
    "Great product, works exactly as described. Very happy with my purchase.",
    "Good quality for the price. Would recommend to others looking for this.",
    "Arrived quickly and in perfect condition. Exactly what I needed.",
    "Excellent build quality. Has been working perfectly for months.",
    "Does what it says. Good value for money overall.",
    "Not bad but could be better. Average quality for the price.",
    "Disappointed with the quality. Stopped working after a few weeks.",
    "Amazing product! Exceeded my expectations in every way.",
    "Solid product. Easy to set up and use. No complaints.",
    "Decent product but the instructions could be clearer.",
]


def _make_synthetic() -> list[dict]:
    print("  generating synthetic Electronics reviews (for debug only) …")
    rng = random.Random(42)
    n_items = 200
    n_warm_users = 80   # users with >= 10 reviews (warm)
    n_cold_users = 120  # users with 1-2 reviews (cold)
    items = [f"ITEM{i:06d}" for i in range(n_items)]
    warm_users = [f"WARM{i:06d}" for i in range(n_warm_users)]
    cold_users = [f"COLD{i:06d}" for i in range(n_cold_users)]
    rows: list[dict] = []
    # warm users: 10-20 reviews each, no duplicates per user
    for user in warm_users:
        n = rng.randint(10, 20)
        chosen = rng.sample(items, min(n, n_items))
        for item in chosen:
            rating = rng.choices([1, 2, 3, 4, 5], weights=[2, 3, 10, 25, 60])[0]
            rows.append({
                "user": user, "item": item,
                "rating": float(rating),
                "review": rng.choice(REVIEW_TEMPLATES),
            })
    # cold users: 1-2 reviews each
    for user in cold_users:
        n = rng.randint(1, 2)
        chosen = rng.sample(items, n)
        for item in chosen:
            rating = rng.choices([1, 2, 3, 4, 5], weights=[2, 3, 10, 25, 60])[0]
            rows.append({
                "user": user, "item": item,
                "rating": float(rating),
                "review": rng.choice(REVIEW_TEMPLATES),
            })
    rng.shuffle(rows)
    rows = rows[:N]
    print(f"  generated {len(rows):,} synthetic rows  "
          f"({n_warm_users} warm users, {n_cold_users} cold users, {n_items} items)")
    print("  WARNING: synthetic data — pipeline metrics will be meaningless.")
    print("           Use this only to verify the pipeline runs end-to-end.")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for debug pass only (no download needed)")
    args = parser.parse_args()

    if args.synthetic:
        rows = _make_synthetic()
    else:
        print(f"Downloading {N:,} {CAT} reviews …")
        rows = _try_huggingface()
        if not rows:
            rows = _try_ucsd()
        if not rows:
            print()
            print("Both download sources failed.  Options:")
            print()
            print("  A) Manual download (real data for dissertation):")
            print(f"     1. Open https://huggingface.co/datasets/{SRC}")
            print("     2. Click 'Files and versions'")
            print(f"     3. Download any file from  raw_review_{CAT}/")
            print(f"     4. Save it to  data/raw/{CAT}.jsonl.gz")
            print("     5. Re-run: python -m recsys.run --config configs/default.yaml")
            print()
            print("  B) Synthetic data (debug pass only, metrics are not real):")
            print("     python -m scripts.setup_data --synthetic")
            sys.exit(1)

    pq.write_table(pa.Table.from_pylist(rows[:N], schema=SCHEMA), str(OUT))
    print(f"Saved {min(len(rows), N):,} rows to {OUT}")
    print("Run: python -m recsys.run --config configs/default.yaml")


if __name__ == "__main__":
    main()
