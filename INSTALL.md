# Installation Guide

This guide provides detailed installation instructions for scLLM-DSC.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3090 or better)
- **Memory**: At least 16GB RAM (32GB recommended for large datasets)
- **Storage**: At least 10GB free space for dependencies and data

### Software Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.12 or higher
- **CUDA**: 13.0 or higher (for GPU acceleration)
- **Git**: For cloning the repository

## Installation Methods

### Method 1: Using Pixi (Recommended)

[Pixi](https://pixi.sh/) is a fast, cross-platform package manager that simplifies environment management.

#### Step 1: Install Pixi

```bash
# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex
```

Restart your terminal after installation.

#### Step 2: Clone and Setup

```bash
git clone https://github.com/yourusername/scLLM-DSC.git
cd scLLM-DSC

# Install all dependencies
pixi install

# Activate the environment
pixi shell
```

That's it! All dependencies are now installed and configured.

### Method 2: Using Conda/Mamba

If you prefer Conda/Mamba:

```bash
git clone https://github.com/yourusername/scLLM-DSC.git
cd scLLM-DSC

# Create environment
conda create -n scllm-dsc python=3.12
conda activate scllm-dsc

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install scanpy h5py numpy pandas scipy scikit-learn
pip install openai sentence-transformers loguru tqdm
pip install torchmetrics einops accelerate huggingface_hub
pip install python-dotenv joblib optuna
```

### Method 3: Using pip + virtualenv

For standard Python virtual environments:

```bash
git clone https://github.com/yourusername/scLLM-DSC.git
cd scLLM-DSC

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## Verify Installation

Test that everything is installed correctly:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import scanpy; print(f'Scanpy: {scanpy.__version__}')"
python -c "import openai; print('OpenAI client imported successfully')"
```

Expected output:
```
PyTorch: 2.9.x+cu118
CUDA available: True
Scanpy: 1.11.x
OpenAI client imported successfully
```

## Configuration

### 1. OpenAI API Setup

Create a `.env` file with your API credentials:

```bash
cp .env.example .env
nano .env  # Edit with your API key
```

Content of `.env`:
```bash
BASE_URL=https://api.openai.com/v1
API_KEY=sk-your-actual-api-key-here
```

### 2. Directory Structure

Create necessary directories:

```bash
mkdir -p datasets reference_data output embeddings result log
```

### 3. Download Reference Data

You need gene reference files for your species. See the main README and `tool/README.md` for instructions on how to generate these from NCBI.

Or download pre-generated files (if available):
```bash
# Example - replace with actual download URLs
wget -O reference_data/human.csv https://example.com/human_gene_ref.csv
wget -O reference_data/mouse.csv https://example.com/mouse_gene_ref.csv
```

## Troubleshooting Installation

### Issue: CUDA not available

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Check NVIDIA driver installation:
   ```bash
   nvidia-smi
   ```
2. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: Out of memory errors

**Solutions**:
1. Use smaller batch sizes
2. Reduce embedding dimensions:
   ```bash
   python main_AE.py --dims_encoder [128,16] --dims_decoder [16,128]
   ```
3. Use CPU (slower but works):
   ```bash
   python main_AE.py --gpu -1  # Use CPU
   ```

### Issue: Pixi command not found

**Solution**: Restart terminal after Pixi installation, or manually add to PATH:
```bash
export PATH="$HOME/.pixi/bin:$PATH"
```

### Issue: OpenAI API errors

**Symptoms**: 
- `openai.error.AuthenticationError`
- `openai.error.RateLimitError`

**Solutions**:
1. Check `.env` file is properly configured
2. Verify API key is valid
3. Check API quota/billing
4. For rate limits, reduce request frequency or use API key with higher limits

### Issue: Missing gene reference files

**Symptom**: `FileNotFoundError: reference_data/human.csv`

**Solution**: Follow instructions in `tool/README.md` to generate reference files from NCBI:
```bash
cd tool
python main_augment_expression.py \
    initial_gene_list.csv \
    --email your@email.com \
    --api_key YOUR_NCBI_KEY
```

### Issue: Dataset format errors

**Symptom**: `KeyError: 'gene_names'` or `KeyError: 'X'`

**Solution**: Ensure your dataset is in one of the supported formats:
- `.h5` with keys: `X`, `Y`, `gene_names` (or `var_names`)
- `.h5ad` (AnnData format) with proper structure

Convert to supported format:
```python
import scanpy as sc
import h5py

# Convert AnnData to H5
adata = sc.read_h5ad('your_data.h5ad')
with h5py.File('your_data.h5', 'w') as f:
    f.create_dataset('X', data=adata.X.toarray())
    f.create_dataset('Y', data=adata.obs['cell_type'].cat.codes.values)
    f.create_dataset('gene_names', data=adata.var_names.values.astype('S'))
```

## Platform-Specific Notes

### Linux

Most straightforward platform. Follow standard instructions.

### macOS

- Apple Silicon (M1/M2/M3): PyTorch has MPS (Metal Performance Shaders) support:
  ```bash
  pip install torch torchvision torchaudio
  # Use --gpu 0 but it will use MPS instead of CUDA
  ```
- Intel Macs: Follow Linux instructions

### Windows

- **Recommended**: Use WSL2 (Windows Subsystem for Linux)
  1. Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install
  2. Follow Linux installation instructions inside WSL2
  
- **Native Windows**: 
  - Use Anaconda/Miniconda
  - Some dependencies may require Visual Studio Build Tools

## Docker Installation (Advanced)

For containerized deployment:

```bash
# Build image
docker build -t scllm-dsc .

# Run container
docker run --gpus all -v $(pwd)/data:/app/data scllm-dsc \
    python main_AE.py --dataname your_dataset
```

(Dockerfile to be provided)

## Uninstallation

### Pixi
```bash
rm -rf .pixi
pixi clean
```

### Conda
```bash
conda env remove -n scllm-dsc
```

### pip + virtualenv
```bash
rm -rf venv
```

## Getting Help

If you encounter issues not covered here:

1. Check the [main README](README.md) for usage instructions
2. Search [existing issues](https://github.com/MIKUAFANS/scLLM-DSC/issues)
3. Open a new issue with:
   - Your OS and Python version
   - Full error message and traceback
   - Steps to reproduce the problem

## Next Steps

After successful installation:

1. Read the main [README.md](README.md) for usage instructions
2. Try the example script: `bash run_example.sh`
3. See [tool/README.md](tool/README.md) for NCBI data collection
4. Explore the IJCAI paper for methodology details
