#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augment CSV with Expression_Summary (NCBI Gene):
- Try EFetch Gene XML -> parse Gene-commentary(heading='Expression')
- Fallback to HTML Summary page: <dt>Expression</dt><dd> ... See more</dd>
- Parallel + Resumable (state file)
- Polite pacing; small retries; autotune XML batch size on 414/413/5xx

Input CSV: must包含列 "GeneID"（Entrez Gene numeric id）
Output: <input>.with_expression.csv
State:  <input>.expr.state.json

Usage:
  python augment_with_expression_resume_parallel_dtdd.py \
      path/to/human_gene_header_plus_summary.csv \
      --email you@example.com \
      --api_key YOUR_NCBI_API_KEY \
      --workers 6 \
      --window 4000 \
      --expr_batch 200
  # 只用 HTML（跳过 XML）更快更稳定：
  # --force_html

Notes:
- 强烈建议提供 --email（E-utilities最佳实践）
- 有 API key 时可适当提高并行度
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
from tqdm import tqdm

try:
    import requests
except Exception as e:
    raise SystemExit("This script requires 'requests'. Install: pip install requests") from e

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GENE_PAGE = "https://www.ncbi.nlm.nih.gov/gene"

DEFAULT_OUT_SUFFIX = ".with_expression.csv"
STATE_SUFFIX = ".expr.state.json"

WRITE_LOCK = threading.Lock()


# -------------------- Utilities --------------------

def _sleep(api_key: str | None):
    # Gentle pacing; with parallelism, keep per-thread sleep small.
    time.sleep(0.25 if api_key else 0.5)

def load_state(state_path: Path):
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done": []}

def save_state(state_path: Path, state: dict):
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(state_path)

def iter_input_rows(csv_in: Path):
    with csv_in.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader]
    if not fieldnames or "GeneID" not in fieldnames:
        raise ValueError("Input CSV must include a 'GeneID' column (Entrez numeric ID).")
    return fieldnames, rows

def init_output(csv_out: Path, fieldnames):
    if not csv_out.exists() or csv_out.stat().st_size == 0:
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

def append_rows(csv_out: Path, fieldnames, rows: list[dict]):
    if not rows:
        return
    with WRITE_LOCK:
        with csv_out.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for r in rows:
                writer.writerow(r)


# -------------------- Expression via EFetch Gene XML --------------------

def fetch_expression_xml_batch(session: requests.Session, ids, email: str, api_key: str | None, timeout=120):
    """POST efetch gene xml for a batch of ids."""
    params = {"db": "gene", "retmode": "xml"}
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    data = {"id": ",".join(ids)}
    r = session.post(f"{EUTILS}/efetch.fcgi", params=params, data=data, timeout=timeout)
    r.raise_for_status()
    return r.text

def _all_gene_commentary_with_heading_expression(eg):
    # yield all Gene-commentary whose heading == 'Expression' (case-insensitive)
    for gc in eg.findall(".//Gene-commentary"):
        head = gc.findtext("Gene-commentary_heading") or ""
        if head.strip().lower() == "expression":
            yield gc

def _extract_text_from_gc(gc):
    # prefer direct text
    txt = (gc.findtext("Gene-commentary_text") or "").strip()
    if txt:
        return txt
    # explore nested text nodes as fallback
    for sub in gc.findall(".//Gene-commentary_text"):
        if sub.text and sub.text.strip():
            return sub.text.strip()
    # last resort: collect a couple of text elements
    parts = []
    for elem in gc.iter():
        if elem.tag.endswith("Gene-commentary_text") and elem.text and elem.text.strip():
            parts.append(elem.text.strip())
            if len(parts) >= 2:
                break
    return " ".join(parts).strip()

def parse_expression_from_gene_xml(xml_text: str):
    """Return dict[GeneID] -> Expression_Summary (string)."""
    import xml.etree.ElementTree as ET
    out = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for eg in root.findall(".//Entrezgene"):
        gid = (eg.findtext(".//Gene-track/Gene-track_geneid") or "").strip()
        chunks = []
        for gc in _all_gene_commentary_with_heading_expression(eg):
            t = _extract_text_from_gc(gc)
            if t:
                chunks.append(t)
        # Keep first chunk or join distinct chunks; but usually first is the summary-like sentence
        expr = "; ".join(dict.fromkeys(chunks)) if chunks else ""
        if gid:
            out[gid] = expr
    return out

def fetch_expression_for_ids_xml(session: requests.Session, ids_slice, email: str, api_key: str | None, batch: int = 200):
    """Autotune batch on 414/413/5xx. Returns dict[gid]->expr."""
    result = {}
    i = 0
    min_batch = 50
    cur = max(min_batch, batch)
    while i < len(ids_slice):
        B = min(cur, len(ids_slice) - i)
        chunk = ids_slice[i:i+B]
        tries = 0
        while True:
            tries += 1
            try:
                xml = fetch_expression_xml_batch(session, chunk, email, api_key)
                part = parse_expression_from_gene_xml(xml)
                for gid in chunk:
                    result.setdefault(gid, part.get(gid, ""))  # default empty
                _sleep(api_key)
                break
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status in (400, 413, 414) or (status and 500 <= status < 600):
                    # shrink and retry this window
                    newB = max(min_batch, B // 2)
                    if newB == B and B > min_batch:
                        newB = B - 50
                    if newB < min_batch:
                        newB = min_batch
                    cur = newB
                    time.sleep(0.8)
                    continue
                else:
                    # soft retry then give up
                    if tries >= 3:
                        for gid in chunk:
                            result.setdefault(gid, "")
                        break
                    time.sleep(0.8)
                    continue
            except Exception:
                if tries >= 3:
                    for gid in chunk:
                        result.setdefault(gid, "")
                    break
                time.sleep(0.8)
                continue
        i += B
    return result


# -------------------- Expression via HTML <dt>Expression</dt><dd>...</dd> --------------------

_DT_DD_EXPR = re.compile(r'<dt>\s*Expression\s*</dt>\s*<dd\b[^>]*>(.*?)</dd>', re.I | re.S)
_A_TAG_RE   = re.compile(r'<a\b[^>]*>.*?</a>', re.I | re.S)
_TAG_RE     = re.compile(r'<[^>]+>')

def _extract_expression_from_html(html: str) -> str:
    m = _DT_DD_EXPR.search(html)
    if not m:
        return ""
    dd_html = m.group(1)
    dd_html = _A_TAG_RE.sub("", dd_html)  # strip <a>See more</a>
    text = _TAG_RE.sub("", dd_html)       # strip any leftover tags
    text = unescape(text)
    text = " ".join(text.split())         # normalize whitespace
    # strip wrapping quotes if any
    if len(text) >= 2 and text[0] in "\"'“”‘’" and text[-1] in "\"'“”‘’":
        text = text[1:-1].strip()
    return text.strip()

def fetch_expression_from_html_dt_dd(session: requests.Session, gene_id: str, lang: str = "en-US,en;q=0.8", timeout: int = 60) -> str:
    url = f"{GENE_PAGE}/{gene_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GeneExpressionBot/1.0)",
        "Accept-Language": lang,
    }
    # try summary page (default)
    r = session.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    txt = _extract_expression_from_html(r.text)
    if txt:
        return txt
    # fallback: Full report (sometimes duplicated)
    r2 = session.get(url, params={"report": "Full"}, headers=headers, timeout=timeout)
    r2.raise_for_status()
    return _extract_expression_from_html(r2.text)

def fetch_expression_for_ids_html(session: requests.Session, ids_slice, lang: str = "en-US,en;q=0.8"):
    out = {}
    for gid in tqdm(ids_slice):
        tries = 0
        while True:
            tries += 1
            try:
                out[gid] = fetch_expression_from_html_dt_dd(session, gid, lang=lang)
                break
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status in (403, 429, 500, 502, 503, 504):
                    time.sleep(1.0)
                    continue
                else:
                    out[gid] = "ERROR"
                    break
            except Exception:
                if tries >= 10:
                    out[gid] = "ERROR"
                    break
                time.sleep(0.8)
                continue
    return out


# -------------------- Pipeline --------------------

def main():
    ap = argparse.ArgumentParser(description="Augment Expression_Summary (XML first, HTML dt/dd fallback).")
    ap.add_argument("csv", help="Input CSV from your earlier pipeline (must include GeneID)")
    ap.add_argument("--email", default="", help="Email for NCBI E-utilities")
    ap.add_argument("--api_key", default="", help="API key (NCBI E-utilities / Datasets)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel threads")
    ap.add_argument("--window", type=int, default=2000, help="Rows per window (split among workers)")
    ap.add_argument("--expr_batch", type=int, default=200, help="EFetch XML batch per worker (autotunes down on 414/413/5xx)")
    ap.add_argument("--force_html", action="store_true", help="Skip XML and use HTML dt/dd directly")
    ap.add_argument("--accept_language", default="en-US,en;q=0.8", help="Accept-Language for HTML (ensure 'Expression' label)")
    args = ap.parse_args()

    csv_in = Path(args.csv)
    email = args.email
    api_key = args.api_key or None
    workers = max(1, args.workers)
    window = max(1, args.window)
    expr_batch = max(50, args.expr_batch)
    lang = args.accept_language

    state_path = csv_in.with_suffix(STATE_SUFFIX)
    csv_out = csv_in.with_suffix(DEFAULT_OUT_SUFFIX)

    fieldnames, all_rows = iter_input_rows(csv_in)
    if "Expression_Summary" not in fieldnames:
        fieldnames.append("Expression_Summary")

    state = load_state(state_path)
    done = set(state.get("done", []))

    init_output(csv_out, fieldnames)

    # keep input order in output
    pos = {str(r.get("GeneID","")).strip(): i for i, r in enumerate(all_rows)}

    idx, N = 0, len(all_rows)
    while idx < N:
        # collect a window of not-yet-done rows
        window_rows = []
        start_idx = idx
        while idx < N and len(window_rows) < window:
            gid = str(all_rows[idx].get("GeneID","")).strip()
            if gid and gid not in done:
                window_rows.append(all_rows[idx])
            idx += 1
        if not window_rows:
            continue

        # split into shards
        if workers <= 1 or len(window_rows) <= 1:
            shards = [window_rows]
        else:
            step = (len(window_rows) + workers - 1) // workers
            shards = [window_rows[i:i+step] for i in range(0, len(window_rows), step)]

        out_maps = []

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = []
            for shard in shards:
                ids = [str(r["GeneID"]).strip() for r in shard if str(r.get("GeneID","")).strip()]
                sess = requests.Session()
                if args.force_html:
                    futures.append(ex.submit(fetch_expression_for_ids_html, sess, ids, lang))
                else:
                    futures.append(ex.submit(fetch_expression_for_ids_xml, sess, ids, email, api_key, expr_batch))
            for fu in as_completed(futures):
                out_maps.append(fu.result())

        expr_map = {}
        for m in out_maps:
            expr_map.update(m)

        # If used XML first, top off missing with HTML dt/dd fallback
        if not args.force_html:
            missing = [gid for gid in expr_map if not expr_map[gid]]
            if missing:
                # fetch HTML fallback in small batches (polite)
                for i2 in range(0, len(missing), 50):
                    sub = missing[i2:i2+50]
                    m2 = fetch_expression_for_ids_html(requests.Session(), sub, lang=lang)
                    expr_map.update(m2)

        # write this window (preserve original order)
        merged = []
        for r in window_rows:
            gid = str(r["GeneID"]).strip()
            newr = dict(r)
            newr["Expression_Summary"] = expr_map.get(gid, "")
            merged.append(newr)
        merged.sort(key=lambda r: pos.get(str(r["GeneID"]).strip(), 1 << 30))

        append_rows(csv_out, fieldnames, merged)

        # update state
        for r in merged:
            done.add(str(r["GeneID"]).strip())
        state["done"] = sorted(done)
        save_state(state_path, state)

        print(f"[Window {start_idx}-{idx-1}] wrote {len(merged)} rows; done={len(done)}/{N}")

    print(f"Done. Output -> {csv_out}")
    print(f"State  -> {state_path}")


if __name__ == "__main__":
    main()