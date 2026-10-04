# AI Based Digital Evidence

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/github/license/thourihan/DeepfakeDetection.svg)
![GitHub release](https://img.shields.io/github/v/release/thourihan/DeepfakeDetection.svg)
![Last Commit](https://img.shields.io/github/last-commit/thourihan/DeepfakeDetection.svg)
![Stars](https://img.shields.io/github/stars/thourihan/DeepfakeDetection?style=social)
![Framework](https://img.shields.io/badge/framework-PyTorch-red.svg)

## Project Overview
This project detects deepfake images using modern CNN/ViT-family backbones. We use EfficientNet, FasterViT, and EfficientFormerV2-S1 to balance speed and accuracy, and to diversify architectures to reduce brittleness to dataset-specific artifacts. Models are integrated via `timm`, with reproducible training/evaluation and Grad-CAM visualizations.

Check out the research paper [here](https://docs.google.com/document/d/1Duh0sKFxBPB-_-t1U8HRtR3py7OVKb715McTbgyHh7Q/edit?usp=sharing)!

## Features
- **Multiple Backbones (via timm):** EfficientNet, FasterViT, and EfficientFormerV2-S1 for complementary accuracy/latency trade-offs.
- **Gradio Interface:** Simple web UI for uploading images and viewing model predictions.
- **Grad-CAM Visualization:** Heatmaps highlighting the regions driving model decisions (available across models).
- **Training & Evaluation Scripts:** Scripts for training and evaluation with Grad-CAM outputs. 

## Sample Predictions and Heatmap Visualizations
Here are a couple of examples of the model's output, along with heatmap visualizations for each.

### Deepfake Detection Example
**Mark Zuckerberg Deepfake**  
Predictions:  
EfficientNet: fake (62.11% confidence)  
FasterViT: fake (97.33% confidence)

<table>
  <tr>
    <td>
      <p><strong>Input Image</strong></p>
      <img src="docs/images/mark-zuckerberg-deepfake.webp" alt="Mark Zuckerberg Deepfake" width="361" height="224"/>
    </td>
    <td>
      <p><strong>Heatmap Visualization</strong></p>
      <img src="docs/images/mark-zuckerberg-deepfake-heatmap.png" alt="Heatmap of Mark Zuckerberg Deepfake" width="224" height="224"/>
    </td>
  </tr>
</table>

### Real Image Detection Example
**Donald Trump Real**  
Predictions:  
EfficientNet: real (97.83% confidence)  
FasterViT: real (98.55% confidence)

<table>
  <tr>
    <td>
      <p><strong>Input Image</strong></p>
      <img src="docs/images/donald-trump-real.jpg" alt="Donald Trump Real" width="301" height="224"/>
    </td>
    <td>
      <p><strong>Heatmap Visualization</strong></p>
      <img src="docs/images/donald-trump-real-heatmap.png" alt="Heatmap of Donald Trump Real" width="224" height="224"/>
    </td>
  </tr>
</table>

## Installation & Setup

Follow these steps to set up and run the project from your terminal:

### 1. Navigate to the project directory
Open your terminal (PowerShell, Command Prompt, or terminal emulator) and navigate to the project directory:
```bash
# For Windows (if stored on drive D:):
d:
cd "d:\DEEPFAKE DET"

# For Unix/macOS:
cd "/path/to/DEEPFAKE DET"
```

### 2. Set up a Python Virtual Environment
It is highly recommended to isolate your dependencies using a virtual environment:
```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment:
# Windows (PowerShell):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1

# Windows (CMD):
venv\Scripts\activate.bat

# Unix/macOS:
source venv/bin/activate
```

### 3. Install dependencies
Install all required libraries defined in [requirements.txt](file:///d:/DEEPFAKE%20DET/requirements.txt):
```bash
pip install -r requirements.txt
```

---

## Orchestrating Runs & Running Scripts

There are three main entrypoints, all routed through `orchestrator.py` and driven by YAML configs under `config/`:

- **`train.py`**: Trains the models listed under `models:` in `config/train.yaml` (or a custom config).
- **`inference.py`**: Runs batch evaluation or single-image/video forensics.
- **`web_ui.py`**: Launches an interactive browser-based Gradio UI.

### Running Commands

#### 1. Start the Interactive Web UI
This launches a browser-based forensic dashboard where you can upload images/videos, examine Grad-CAM heatmaps, and view ensembled authenticity ratings:
```bash
python web_ui.py --config config/inference.yaml
```
Once started, open the link in your browser: `http://127.0.0.1:7860`

#### 2. Run Forensic Analysis on a Single Image/Video
To run single-file analysis and print the report directly in your terminal, pass the `--input` flag pointing to any media file:
```bash
# Analyze a real image
python inference.py --input docs/images/donald-trump-real.jpg

# Analyze a deepfake image
python inference.py --input docs/images/mark-zuckerberg-deepfake.webp
```

#### 3. Run Batch Evaluation
To evaluate models defined in `config/inference.yaml` against a test dataset:
```bash
python inference.py
```

#### 4. Train Models
To start training models listed in `config/train.yaml`:
```bash
python train.py
```

*Note: You can pass `--config <path_to_config>` to any of these scripts to use a custom YAML configuration.*

Each run directory contains `checkpoints/` (latest & best checkpoints), `logs/` (console outputs), and `plots/` (confusion matrix and ROC curve when labels are available). The setup targets frame-level deepfake vs. real classification but works for multiclass `ImageFolder` datasets as well.


### Per-model transform toggles

Every transform in the training and evaluation pipelines can be toggled on or off per backbone. Add a `transforms:` block under each `models.<name>` entry in the YAML config and enable the transforms you need for training/inference for each model.

---

## CIFAKE Benchmark & Ablation Study Results

The pipeline incorporates rigorous ML research standards evaluated on the public **CIFAKE (Real and AI-Generated Synthetic Images)** benchmark.

### 1. Dataset & Split Partitioning
- **Dataset**: [CIFAKE Benchmark](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (60,000 samples: 30,000 Real CIFAR-10 + 30,000 Stable Diffusion v1.4 synthetic).
- **Split Scheme**: Strict 3-way disjoint partition to eliminate data leakage:
  - **Training Split**: 4,000 samples (2,000 FAKE + 2,000 REAL)
  - **Validation Split**: 1,000 samples (500 FAKE + 500 REAL) — Used strictly for operating threshold ($\tau^*$) calibration and ensemble weight optimization.
  - **Held-Out Test Split**: 2,000 samples (1,000 FAKE + 1,000 REAL) — Frozen blind evaluation.

### 2. Standard Evaluation Metrics Table
*Evaluated on held-out test split with operating thresholds ($\tau^*$) calibrated via Youden's J statistic on the validation set:*

| Model Architecture | Operating Threshold ($\tau^*$) | Accuracy | Balanced Accuracy | Precision | Recall | Specificity | F1 Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weighted Ensemble** | **0.5599** | **89.90%** | **89.90%** | **93.09%** | **86.20%** | **93.60%** | **0.8951** | **0.9699** | **0.9702** |
| **FasterViT-2-224** | **0.4656** | **89.20%** | **89.20%** | **88.81%** | **89.70%** | **88.70%** | **0.8925** | **0.9596** | **0.9577** |
| **EfficientNet-B3** | 0.4381 | 85.80% | 85.80% | 84.76% | 87.30% | 84.30% | 0.8601 | 0.9308 | 0.9312 |
| **EfficientFormerV2-S1** | 0.4617 | 82.55% | 82.55% | 81.33% | 84.50% | 80.60% | 0.8288 | 0.9111 | 0.9100 |

### 3. Systematic Ablation Study
The ablation study systematically analyzes:
1. **Architecture Ablation**: Solo backbones vs. Pairwise vs. Uniform Ensemble vs. Weighted Ensemble ([ablation_architecture.csv](file:///d:/ai%20based%20digital%20evidence/results/ablation/ablation_architecture.csv)).
2. **Ensemble Weights Strategy**: Uniform ($1:1:1$) vs. Optimal Validation-Tuned vs. FasterViT-Heavy vs. EfficientNet-Heavy ([ablation_ensemble_weights.csv](file:///d:/ai%20based%20digital%20evidence/results/ablation/ablation_ensemble_weights.csv)).
3. **Threshold Calibration**: Default ($\tau = 0.50$) vs. Calibrated Operating Threshold ($\tau^*$) ([ablation_threshold.csv](file:///d:/ai%20based%20digital%20evidence/results/ablation/ablation_threshold.csv)).

### 4. Generated Diagnostic Plot Images (300 DPI)
All visual diagnostic plots are generated and saved to `results/plots/`:
- **ROC-AUC Curve**: `results/plots/roc_auc_curve.png`
- **PR-AUC Curve**: `results/plots/pr_auc_curve.png`
- **Confusion Matrices (2x4 comparative grid)**: `results/plots/confusion_matrices.png`
- **Comprehensive Metrics Comparison Bar Chart**: `results/plots/metrics_comparison.png`

To run the master pipeline end-to-end:
```bash
python retrain_and_evaluate.py
```