# NCBI Gene Data Collection Tools

This directory contains scripts to fetch and augment gene annotation data from the NCBI Gene database.

## Overview

These tools are essential for the scLLM-DSC pipeline, as they retrieve biological annotations that are later encoded by LLMs to provide semantic knowledge for cell clustering.

## Scripts

### 1. `main_augment_expression.py`

Fetches gene expression summaries from NCBI Gene database.

#### Features
- **Dual Strategy**: Tries XML API first (faster, batch-capable), falls back to HTML scraping
- **Parallel Processing**: Multi-threaded workers for faster data collection
- **Resumable**: Uses state file (`.expr.state.json`) to track progress and resume after interruption
- **Auto-tuning**: Automatically adjusts batch size on HTTP 414/413/5xx errors
- **Rate Limiting**: Respects NCBI's usage guidelines with polite pacing

#### Usage

```bash
python main_augment_expression.py \
    path/to/input.csv \
    --email your@email.com \
    --api_key YOUR_NCBI_API_KEY \
    --workers 6 \
    --window 4000 \
    --expr_batch 200
```

#### Arguments

- `csv`: Input CSV file path (must contain `GeneID` column with Entrez Gene IDs)
- `--email`: Your email address (required by NCBI E-utilities best practices)
- `--api_key`: NCBI API key (optional but recommended for higher rate limits)
- `--workers`: Number of parallel threads (default: 4)
- `--window`: Number of rows to process per window (default: 2000)
- `--expr_batch`: Batch size for XML fetching (default: 200, auto-tuned on errors)
- `--force_html`: Skip XML and use HTML scraping only (faster but more fragile)
- `--accept_language`: Accept-Language header for HTML requests (default: "en-US,en;q=0.8")

#### Output

- `input.with_expression.csv`: Augmented CSV with new `Expression_Summary` column
- `input.expr.state.json`: Progress state file for resumability

#### Example

```bash
# Start fresh
python main_augment_expression.py \
    ../reference_data/human_genes.csv \
    --email researcher@university.edu \
    --api_key abcd1234efgh5678 \
    --workers 8 \
    --window 5000

# If interrupted, just rerun the same command - it will resume from state file
```

#### Tips

- **With API Key**: You can use up to 10 requests/second. Try `--workers 8-10`
- **Without API Key**: Limited to 3 requests/second. Use `--workers 4-6`
- **Faster but Less Robust**: Use `--force_html` to skip XML entirely
- **Network Issues**: The script auto-retries and saves progress every window

---

### 2. `fix_gene_type_from_html_resume.py`

Scrapes gene type information from NCBI Gene HTML pages.

#### Features
- **HTML Parsing**: Extracts `<dt>Gene type</dt><dd>...</dd>` from NCBI Gene pages
- **Parallel Scraping**: Multi-threaded workers
- **Resumable**: Uses state file (`.gtype.state.json`) for progress tracking
- **Atomic Writes**: Safe temporary file handling prevents data corruption
- **In-place Update**: Overwrites input file by default (or writes to new file with `--out`)

#### Usage

```bash
# Overwrite input file (default)
python fix_gene_type_from_html_resume.py \
    input.csv \
    --workers 6 \
    --window 4000

# Write to new file
python fix_gene_type_from_html_resume.py \
    input.csv \
    --out input.fixed.csv \
    --workers 6
```

#### Arguments

- `csv`: Input CSV file (must contain `GeneID` column)
- `--out`: Output CSV file path (default: overwrites input)
- `--workers`: Number of parallel threads (default: 4)
- `--window`: Rows per processing window (default: 2000)
- `--accept_language`: Accept-Language header (default: "en-US,en;q=0.8")

#### Output

- Modified CSV with `Gene_type` column added/updated
- `.gtype.state.json`: Progress state file

#### Example

```bash
python fix_gene_type_from_html_resume.py \
    ../reference_data/human_genes.with_expression.csv \
    --workers 8 \
    --window 5000
```

---

### 3. `run_express.sh` & `run.sh`

Convenience shell scripts to run the above tools with preset parameters.

#### `run_express.sh`

Runs expression summary augmentation:

```bash
#!/bin/bash
./run_express.sh path/to/input.csv your@email.com YOUR_API_KEY
```

#### `run.sh`

Runs gene type fixing:

```bash
#!/bin/bash
./run.sh path/to/input.csv
```

---

## Complete Workflow

### Step 1: Prepare Initial Gene List

Start with a CSV containing at minimum:
- `GeneID`: Entrez Gene ID (numeric)
- `Official_Symbol`: Gene symbol (e.g., "TP53")

Example:
```csv
GeneID,Official_Symbol
7157,TP53
672,BRCA1
1956,EGFR
```

### Step 2: Augment Expression Summaries

```bash
python main_augment_expression.py \
    gene_list.csv \
    --email researcher@university.edu \
    --api_key YOUR_NCBI_API_KEY \
    --workers 8 \
    --window 5000
```

Output: `gene_list.with_expression.csv` with `Expression_Summary` column

### Step 3: Fix Gene Types

```bash
python fix_gene_type_from_html_resume.py \
    gene_list.with_expression.csv \
    --workers 8 \
    --window 5000
```

Output: `gene_list.with_expression.csv` updated with `Gene_type` column

### Step 4: Use in scLLM-DSC

Place the final CSV in `../reference_data/`:
```bash
cp gene_list.with_expression.csv ../reference_data/human.csv
```

Now run the main pipeline:
```bash
cd ..
python main_Gene.py --dataset your_dataset.h5 --reference_file human.csv
python main_AE.py --dataname your_dataset --num_class 10 --species human
```

---

## NCBI API Guidelines

### Rate Limits

- **Without API Key**: 3 requests/second
- **With API Key**: 10 requests/second

### Getting an API Key

1. Create NCBI account: https://www.ncbi.nlm.nih.gov/account/
2. Go to Settings → API Key Management
3. Generate new API key
4. Use with `--api_key` parameter

### Best Practices

1. **Always provide email**: Required by NCBI E-utilities usage guidelines
2. **Use API key**: For faster, more reliable access
3. **Be polite**: Don't hammer their servers - these scripts include rate limiting
4. **Off-peak hours**: Run large jobs during off-peak hours when possible
5. **Cache results**: State files allow resuming - don't re-fetch completed data

---

## Troubleshooting

### Issue: Rate limiting (HTTP 429)

**Solution**: Reduce workers or add delays
```bash
python main_augment_expression.py input.csv --workers 2 --email your@email.com
```

### Issue: Connection timeout

**Solution**: Script auto-retries. If persistent, check network connection.

### Issue: Empty Expression_Summary

**Cause**: Gene may not have expression annotation in NCBI, or page format changed

**Check manually**: Visit `https://www.ncbi.nlm.nih.gov/gene/GENEID`

### Issue: Script interrupted mid-run

**Solution**: Just rerun the same command - it will resume from state file
```bash
# Will pick up where it left off
python main_augment_expression.py input.csv --email your@email.com
```

### Issue: HTTP 414 (URI Too Long) errors

**Solution**: Script auto-reduces batch size. If persistent, manually set smaller batch:
```bash
python main_augment_expression.py input.csv --expr_batch 100 --email your@email.com
```

### Issue: HTML parsing returns empty strings

**Cause**: NCBI page structure may have changed

**Solution**: 
1. Check a sample gene page manually
2. Update regex patterns in the script if needed
3. Report issue with specific GeneID that fails

---

## State File Format

### Expression State (`.expr.state.json`)

```json
{
  "done": ["7157", "672", "1956"]
}
```

### Gene Type State (`.gtype.state.json`)

```json
{
  "done": ["7157", "672", "1956"]
}
```

State files track which GeneIDs have been processed. Delete to start fresh.

---

## Performance Tips

### For Small Gene Lists (<1000 genes)

```bash
python main_augment_expression.py input.csv --workers 4 --email your@email.com
```

### For Medium Lists (1,000-10,000 genes)

```bash
python main_augment_expression.py input.csv \
    --workers 8 \
    --window 5000 \
    --api_key YOUR_KEY \
    --email your@email.com
```

### For Large Lists (>10,000 genes)

```bash
# Use XML batch mode with API key
python main_augment_expression.py input.csv \
    --workers 10 \
    --window 10000 \
    --expr_batch 300 \
    --api_key YOUR_KEY \
    --email your@email.com
```

### For Unreliable Networks

```bash
# Smaller windows so progress saves more frequently
python main_augment_expression.py input.csv \
    --workers 4 \
    --window 1000 \
    --email your@email.com
```

---

## Output CSV Columns

After running both scripts, your CSV will contain:

| Column | Description | Source |
|--------|-------------|--------|
| `GeneID` | Entrez Gene ID | Input |
| `Official_Symbol` | Gene symbol | Input |
| `Expression_Summary` | Expression annotation text | `main_augment_expression.py` |
| `Gene_type` | Gene type (e.g., "protein coding") | `fix_gene_type_from_html_resume.py` |

Additional columns from your input CSV are preserved.

---

## Dependencies

```bash
pip install requests tqdm
```

Both scripts use only standard library plus `requests` and `tqdm`.

---

## License

MIT License - see repository LICENSE file.

---

## Contributing

Found a bug or NCBI changed their page structure? Please open an issue or PR!

---

## Credits

These tools were developed as part of the scLLM-DSC project to enable LLM-based semantic encoding of gene annotations for single-cell clustering.
