# COMP/ENGN 6528 Mini-Project: Implicit Neural Representation (INR)

This project explores the ability of deep neural networks to learn and represent a single high-resolution image through a coordinate-based Multi-Layer Perceptron (MLP). It specifically analyzes the impact of **Positional Encoding** in overcoming the spectral bias of neural networks.

## Project Structure

- `model.py`: Defines the `CoordinateMLP` and `PositionalEncoding` layers.
- `dataset.py`: A PyTorch Dataset that maps image pixel coordinates to RGB values.
- `train.py`: The main entry point for training and evaluation.
- `requirements.txt`: List of required Python packages.
- `results/`: Directory where training logs, loss curves, and reconstructed images are saved.

## Installation

To set up the environment, run:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Main Experiment (With Positional Encoding)
This will train the model using Fourier features to capture high-frequency details.
```bash
python train.py --image ../jcsmr-1.jpg --outdir results
```

### 2. Baseline Experiment (Without Positional Encoding)
This will train a standard MLP to demonstrate the "Spectral Bias" (blurry results).
```bash
python train.py --image ../jcsmr-1.jpg --outdir results --baseline
```

## Key Parameters
You can adjust the following in `train.py` or via command line arguments:
- `--epochs`: Number of training iterations (default: 500 for quick test).
- `--image`: Path to the target image.
- `num_frequencies`: Number of frequency bands for Positional Encoding (default: 10).

## Results
After training, check the `results/` folder:
- `*_final.png`: The reconstructed image.
- `*_loss.png`: The MSE loss curve during training.
- Console output will show the final **PSNR** (Peak Signal-to-Noise Ratio).
