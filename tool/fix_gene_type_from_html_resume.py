#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Gene_type by scraping NCBI Gene HTML (<dt>Gene type</dt><dd>...</dd>)
- Reads previous CSV (must contain column "GeneID"; "Gene_type" will be created if missing)
- Parallel + resumable (state file records processed GeneIDs)
- By default overwrite the input CSV atomically (safe temp -> replace). You can choose --out to write a new file.

Usage (overwrite input in place):
  python fix_gene_type_from_html_resume.py input.csv --workers 6 --window 4000

Or write to a new file:
  python fix_gene_type_from_html_resume.py input.csv --out input.gtype_fixed.csv --workers 6

Dependencies: requests
  pip install requests
"""

import csv
import json
import time
import argparse
import re
import threading
from html import unescape
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from tqdm import tqdm

GENE_PAGE = "https://www.ncbi.nlm.nih.gov/gene"
STATE_SUFFIX = ".gtype.state.json"
WRITE_LOCK = threading.Lock()

# --- HTML parser for <dt>Gene type</dt><dd>...</dd> ---
_DT_DD_GTYPE = re.compile(r'<dt>\s*Gene\s*type\s*</dt>\s*<dd\b[^>]*>(.*?)</dd>', re.I | re.S)
_A_TAG_RE    = re.compile(r'<a\b[^>]*>.*?</a>', re.I | re.S)
_TAG_RE      = re.compile(r'<[^>]+>')

def _extract_gene_type_from_html(html: str) -> str:
    m = _DT_DD_GTYPE.search(html)
    if not m:
        return ""
    dd_html = m.group(1)
    # remove <a>See more</a> and any tags
    dd_html = _A_TAG_RE.sub("", dd_html)
    text = _TAG_RE.sub("", dd_html)
    # unescape & normalize whitespace
    text = unescape(text)
    text = " ".join(text.split()).strip().strip('"“”‘’')
    # normalize "protein-coding" -> "protein coding" (optional)
    text = text.replace("-coding", " coding")
    return text

def fetch_gene_type_from_html(session: requests.Session, gene_id: str, lang: str = "en-US,en;q=0.8", timeout: int = 60) -> str:
    url = f"{GENE_PAGE}/{gene_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GeneTypeFixer/1.0)",
        "Accept-Language": lang,
    }
    # default summary page
    r = session.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    txt = _extract_gene_type_from_html(r.text)
    if txt:
        return txt
    # try Full
    r2 = session.get(url, params={"report": "Full"}, headers=headers, timeout=timeout)
    r2.raise_for_status()
    return _extract_gene_type_from_html(r2.text)

def load_state(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done": []}

def save_state(path: Path, state: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def read_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [r for r in reader]
    if "GeneID" not in fieldnames:
        raise ValueError("Input CSV must include column 'GeneID'.")
    if "Gene_type" not in fieldnames:
        fieldnames.append("Gene_type")
        # ensure column exists on each row (empty for now)
        for r in rows:
            r.setdefault("Gene_type", "")
    return fieldnames, rows

def write_rows_atomic(csv_path: Path, fieldnames, rows):
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tmp.replace(csv_path)

def worker_fetch(ids_slice, lang):
    sess = requests.Session()
    out = {}
    for gid in tqdm(ids_slice):
        tries = 0
        while True:
            tries += 1
            try:
                out[gid] = fetch_gene_type_from_html(sess, gid, lang=lang)
                break
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else None
                if code in (403, 429, 500, 502, 503, 504):
                    time.sleep(1.0)
                    continue
                out[gid] = ""
                break
            except Exception:
                if tries >= 2:
                    out[gid] = ""
                    break
                time.sleep(0.8)
                continue
        # polite pacing per id (overall still parallel)
        time.sleep(0.05)
    return out

def main():
    ap = argparse.ArgumentParser(description="Fix Gene_type by scraping NCBI Gene HTML (dt/dd).")
    ap.add_argument("csv", help="Input CSV (previous step output). Will be overwritten unless --out is provided.")
    ap.add_argument("--out", default="", help="Write to a new CSV instead of overwriting input.")
    ap.add_argument("--workers", type=int, default=4, help="Parallel threads")
    ap.add_argument("--window", type=int, default=2000, help="Rows per window (split among workers)")
    ap.add_argument("--accept_language", default="en-US,en;q=0.8", help="Accept-Language to ensure 'Gene type' label")
    args = ap.parse_args()

    csv_in = Path(args.csv)
    csv_out = Path(args.out) if args.out else csv_in  # overwrite by default
    state_path = csv_in.with_suffix(STATE_SUFFIX)     # state bound to input name

    fieldnames, all_rows = read_rows(csv_in)
    state = load_state(state_path)
    done = set(state.get("done", []))

    # build index map to preserve input order
    pos = {str(r.get("GeneID","")).strip(): i for i, r in enumerate(all_rows)}
    N = len(all_rows)
    idx = 0

    while idx < N:
        # gather a window of rows to process (skip those done)
        window_rows = []
        start = idx
        while idx < N and len(window_rows) < max(1, args.window):
            r = all_rows[idx]
            gid = str(r.get("GeneID","")).strip()
            if gid and gid not in done:
                window_rows.append(r)
            idx += 1

        if not window_rows:
            continue

        # shard by workers
        workers = max(1, args.workers)
        if workers <= 1 or len(window_rows) <= 1:
            shards = [window_rows]
        else:
            step = (len(window_rows) + workers - 1) // workers
            shards = [window_rows[i:i+step] for i in range(0, len(window_rows), step)]

        # fetch in parallel
        out_maps = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for shard in shards:
                ids = [str(r["GeneID"]).strip() for r in shard if str(r.get("GeneID","")).strip()]
                futs.append(ex.submit(worker_fetch, ids, args.accept_language))
            for fu in as_completed(futs):
                out_maps.append(fu.result())

        # merge results
        gtype_map = {}
        for m in out_maps:
            gtype_map.update(m)

        # apply back to master rows (in place)
        for r in window_rows:
            gid = str(r["GeneID"]).strip()
            new_val = gtype_map.get(gid, "")
            if new_val:
                r["Gene_type"] = new_val
            # mark done regardless; if为空，说明页面缺失或被限流后两次重试失败
            done.add(gid)

        # write progress atomically
        # (write the entire file each window to keep it simple & robust)
        with WRITE_LOCK:
            # sort to original order just in case
            all_rows.sort(key=lambda x: pos.get(str(x.get("GeneID","")).strip(), 1 << 30))
            write_rows_atomic(csv_out, fieldnames, all_rows)
            save_state(state_path, {"done": sorted(done)})

        print(f"[{start}-{idx-1}] updated {len(window_rows)} rows; done={len(done)}/{N}")

    print(f"Gene_type fixed. Output -> {csv_out}")
    print(f"State -> {state_path}")

if __name__ == "__main__":
    main()