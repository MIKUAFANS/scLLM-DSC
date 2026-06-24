# Quick Start Guide

Get started with scLLM-DSC in 5 minutes.

## Prerequisites

- Python 3.12+
- CUDA-capable GPU (optional but recommended)
- OpenAI API key

## 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/scLLM-DSC.git
cd scLLM-DSC

# Install dependencies (choose one method)
# Option A: Using Pixi (recommended)
curl -fsSL https://pixi.sh/install.sh | bash
pixi install
pixi shell

# Option B: Using pip
pip install -r requirements.txt
```

## 2. Configuration

```bash
# Copy and edit environment variables
cp .env.example .env
nano .env  # Add your OpenAI API key
```

## 3. Prepare Data

### Option A: Use Example Dataset

Download a small example dataset:

```bash
mkdir -p datasets reference_data

# Download example (replace with actual URLs)
# wget -O datasets/example.h5 https://example.com/example.h5
# wget -O reference_data/human.csv https://example.com/human.csv
```

### Option B: Use Your Own Data

Place your scRNA-seq data in `datasets/`:

```bash
# Your data should be in .h5 or .h5ad format
cp /path/to/your_data.h5 datasets/
```

Generate reference data from NCBI:

```bash
cd tool
python main_augment_expression.py \
    initial_gene_list.csv \
    --email your@email.com \
    --workers 4
cd ..
```

## 4. Run Pipeline

```bash
# Run with example dataset
bash run_example.sh

# Or run manually
python main_AE.py \
    --dataname Sonya_HumanLiver_counts_top5000 \
    --num_class 11 \
    --species human \
    --gpu 0
```

## 5. Check Results

```bash
# View results
cat result/Sonya_HumanLiver_counts_top5000/Sonya_HumanLiver_counts_top5000_results.csv

# View logs
tail -f log/Sonya_HumanLiver_counts_top5000/Sonya_HumanLiver_counts_top5000.log
```

## Expected Output

```
Results (averaged over 5 seeds):
- ACC: 0.85-0.90
- NMI: 0.80-0.85
- ARI: 0.75-0.82
```

## Common Issues

**Issue**: CUDA out of memory

```bash
# Solution: Use smaller dimensions
python main_AE.py --dims_encoder [128,16] --dataname your_dataset
```

**Issue**: Gene embeddings not found

```bash
# Solution: Run gene embedding generation first
python main_Gene.py --dataset your_dataset.h5 --reference_file human.csv
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [INSTALL.md](INSTALL.md) for installation troubleshooting
- See [tool/README.md](tool/README.md) for NCBI data collection
- Explore hyperparameter tuning in the main README

## Dataset Requirements

Your dataset should have:

- Expression matrix: `X` (cells × genes)
- Cell labels: `Y` (for evaluation)
- Gene names: `gene_names` or `var_names`

Supported formats:

- `.h5` (HDF5) with proper keys
- `.h5ad` (AnnData/Scanpy format)

## Directory Structure After Setup

```
scLLM-DSC/
├── datasets/              # Your scRNA-seq data
│   └── example.h5
├── reference_data/        # NCBI gene annotations
│   └── human.csv
├── output/               # Generated embeddings
├── embeddings/           # Final unified embeddings
├── result/               # Clustering results
└── log/                  # Training logs
```

## Getting Help

- Documentation: [README.md](README.md)
- Installation: [INSTALL.md](INSTALL.md)
- Issues: https://github.com/MIKUAFANS/scLLM-DSC/issues
- Email: xuping0098@gmail.com

## Example Workflow

```bash
# Complete workflow for a new dataset
cd scLLM-DSC

# 1. Prepare reference data (one-time setup)
cd tool
./run_express.sh ../reference_data/gene_list.csv your@email.com YOUR_API_KEY
./run.sh ../reference_data/gene_list.with_expression.csv
cd ..

# 2. Generate gene embeddings
python main_Gene.py \
    --dataset my_dataset.h5 \
    --reference_file human.csv

# 3. Run clustering
python main_AE.py \
    --dataname my_dataset \
    --num_class 10 \
    --species human

# 4. Analyze results
python -c "import pandas as pd; print(pd.read_csv('result/my_dataset/my_dataset_results.csv'))"
```

That's it! You're ready to cluster single-cell data with biological semantics.
