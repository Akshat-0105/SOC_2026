# Setup Guide

This guide will help you set up your development environment for the Multi-Agent RL project.

## Prerequisites Check

Before starting, ensure you have:
- Python 3.9 or higher
- Git
- At least 8GB RAM (16GB recommended)
- GPU with CUDA support (optional but recommended)

## Step 1: Python Environment

### Option A: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n marl-env python=3.10
conda activate marl-env

# Install PyTorch with CUDA support (if you have GPU)
# Visit https://pytorch.org/get-started/locally/ for your specific configuration
# Example for CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt
```

### Option B: Using venv

```bash
# Create virtual environment
python -m venv marl-env

# Activate environment
# On Windows:
marl-env\Scripts\activate
# On Linux/Mac:
source marl-env/bin/activate

# Install PyTorch
# Visit https://pytorch.org/get-started/locally/ for the right command
pip install torch torchvision torchaudio

# Install other requirements
pip install -r requirements.txt
```

## Step 2: Verify Installation

Run this verification script:

```python
# verify_setup.py
import sys
import torch
import gymnasium as gym
import numpy as np
import pandas as pd
from pettingzoo.mpe import simple_spread_v3

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

print("\nTesting Gymnasium...")
env = gym.make("CartPole-v1")
print("✓ Gymnasium working")

print("\nTesting PettingZoo...")
env = simple_spread_v3.env()
print("✓ PettingZoo working")

print("\nTesting NumPy...")
arr = np.random.randn(10)
print("✓ NumPy working")

print("\nTesting Pandas...")
df = pd.DataFrame({'a': [1, 2, 3]})
print("✓ Pandas working")

print("\n✅ All core libraries installed successfully!")
```

Save this as `verify_setup.py` and run:
```bash
python verify_setup.py
```

## Step 3: Jupyter Setup (Optional but Recommended)

```bash
# Install Jupyter extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Launch Jupyter
jupyter notebook
```

## Step 4: Weights & Biases (Optional)

For experiment tracking:

```bash
# Install wandb (already in requirements.txt)
wandb login
# Follow the prompts to authenticate
```

## Step 5: IDE Setup

### VS Code (Recommended)

Install these extensions:
- Python
- Jupyter
- Pylance
- Python Indent
- autoDocstring

### PyCharm

Configure the interpreter to use your conda/venv environment.

## Troubleshooting

### GPU Not Detected

If CUDA is installed but PyTorch doesn't detect it:
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
# Visit pytorch.org for the right installation command
```

### Import Errors

```bash
# Ensure environment is activated
conda activate marl-env  # or source marl-env/bin/activate

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Memory Issues

If you encounter out-of-memory errors:
- Reduce batch sizes in training
- Use gradient accumulation
- Enable mixed precision training (FP16)

### Windows-Specific Issues

If you're on Windows and encounter path issues:
- Use forward slashes in paths: `data/env/` instead of `data\env\`
- Use raw strings: `r"C:\path\to\file"`

## Docker Setup (Advanced)

If you prefer Docker:

```bash
cd setup/docker
docker build -t marl-project .
docker run -it --gpus all -v $(pwd):/workspace marl-project
```

See `docker/README.md` for details.

## Next Steps

Once setup is complete:
1. Verify everything works with `python verify_setup.py`
2. Head to [Week 1](../week1/README.md) to start learning
3. Join the Discord/Slack for discussions (if available)

## Getting Help

If you encounter issues:
1. Check the [FAQ](../shared/FAQ.md)
2. Search for the error message
3. Open an issue on GitHub
4. Ask your mentor

---

**Environment ready?** Let's start with [Week 1](../week1/README.md)!
