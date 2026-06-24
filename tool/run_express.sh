#!/bin/bash

# Script to augment gene expression data from NCBI
# Usage: ./run_express.sh [input_csv] [email] [api_key]

INPUT_CSV=${1:-"../reference_data/human_genes.csv"}
EMAIL=${2:-"your@email.com"}
API_KEY=${3:-""}

echo "=========================================="
echo "Augmenting Expression Summary from NCBI"
echo "Input: $INPUT_CSV"
echo "Email: $EMAIL"
echo "=========================================="

# Build command
CMD="python main_augment_expression.py \"$INPUT_CSV\" --email \"$EMAIL\" --workers 6 --window 4000 --expr_batch 200"

# Add API key if provided
if [ -n "$API_KEY" ]; then
    CMD="$CMD --api_key \"$API_KEY\""
    echo "Using API key for higher rate limits"
fi

# Run the command
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Expression augmentation completed!"
    echo "Output: ${INPUT_CSV%.csv}.with_expression.csv"
    echo "=========================================="
else
    echo "Error during expression augmentation"
    exit 1
fi
