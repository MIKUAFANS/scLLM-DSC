#!/bin/bash

# Example script to run scLLM-DSC pipeline
# Modify parameters as needed for your dataset

DATASET="Sonya_HumanLiver_counts_top5000"
NUM_CLASS=11
SPECIES="human"
GPU=0

echo "=========================================="
echo "Running scLLM-DSC Pipeline"
echo "Dataset: $DATASET"
echo "Number of classes: $NUM_CLASS"
echo "Species: $SPECIES"
echo "=========================================="

# Step 1: Generate gene embeddings (if not exists)
echo "Step 1: Checking gene embeddings..."
if [ ! -f "output/${DATASET}/weighted_cell_embeddings.npz" ]; then
    echo "Gene embeddings not found. Running main_Gene.py..."
    python main_Gene.py \
        --dataset ${DATASET}.h5 \
        --reference_file ${SPECIES}.csv \
        --dataset_path ./datasets/ \
        --save_path ./output/

    if [ $? -ne 0 ]; then
        echo "Error generating gene embeddings. Exiting."
        exit 1
    fi
else
    echo "Gene embeddings found. Skipping generation."
fi

# Step 2: Run clustering
echo ""
echo "Step 2: Running clustering..."
python main_AE.py \
    --dataname ${DATASET} \
    --num_class ${NUM_CLASS} \
    --species ${SPECIES} \
    --gpu ${GPU} \
    --epochs 200 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --factor_ncut 0.15 \
    --factor_mse 0.3 \
    --factor_KL 0.32 \
    --factor_cl 0.3 \
    --omega 0.5

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Pipeline completed successfully!"
    echo "Results saved in: result/${DATASET}/"
    echo "Logs saved in: log/${DATASET}/"
    echo "=========================================="
else
    echo "Error during clustering. Check logs for details."
    exit 1
fi
