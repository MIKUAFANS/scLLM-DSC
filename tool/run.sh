#!/bin/bash

# Script to fix gene types from NCBI HTML
# Usage: ./run.sh [input_csv]

INPUT_CSV=${1:-"../reference_data/human_genes.with_expression.csv"}

echo "=========================================="
echo "Fixing Gene Types from NCBI HTML"
echo "Input: $INPUT_CSV"
echo "=========================================="

python fix_gene_type_from_html_resume.py \
    "$INPUT_CSV" \
    --workers 6 \
    --window 4000

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Gene type fixing completed!"
    echo "Output: $INPUT_CSV (updated in place)"
    echo "=========================================="
else
    echo "Error during gene type fixing"
    exit 1
fi
