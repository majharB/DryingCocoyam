# Explainable AI-Driven Sparse Hyperspectral Modeling for Cocoyam Moisture Forecasting

This repository contains the full codebase for the study on non-destructive, real-time moisture prediction during cocoyam drying using hyperspectral imaging (HSI), CNN-LSTM deep learning models, and the shifted-window integrated-gradients (SWING) explainability framework. The pipeline covers data preprocessing, model training, explainability-driven wavelength selection, ablation studies, and post-training INT8 quantization for edge deployment.

## Overview

Cocoyam tubers were dried under controlled conditions across sixteen independent drying batches, with hyperspectral images (112 bands) and process metadata (drying temperature, air velocity, elapsed drying time) collected throughout drying. This codebase implements the full analysis pipeline used to:

- Train CNN-LSTM models on full-spectrum HSI and metadata to predict moisture content
- Apply SWING attribution to identify physically relevant wavelengths and metadata features
- Construct and evaluate reduced-input models (single-wavelength, 3-band, 8-band, 4-band) guided by attribution results
- Run an input-modality ablation study (HSI-only, metadata-only, HSI+metadata)
- Quantize the best-performing compact model (INT8) for embedded deployment
- Evaluate all models under leave-one-batch-out (LOBO) cross-validation


## Installation

```bash
git clone https://github.com/<your-org>/cocoyam-drying-hsi.git
cd cocoyam-drying-hsi
pip install -r requirements.txt
```

Requires Python 3.9+, TensorFlow or PyTorch (specify version used), NumPy, SciPy, scikit-learn, and Captum (for integrated gradients).

## Usage

**1. Train full-spectrum model**
```bash
python src/train/train_cnnlstm.py
```

## Key Results

| Model | RMSE (%) | R² | RPD | Notes |
|---|---|---|---|---|
| Full spectrum (112 bands) | 3.6 | 0.93 | 4.3 | Baseline |
| 3-wavelength (1398, 1405, 1413 nm) | 3.6 | 0.93 | 4.3 | Matches baseline |
| Single wavelength (1398 nm) | 3.7 | 0.93 | 4.4 | Statistically equivalent |
| Single wavelength, INT8 quantized | 3.9 | 0.92 | 4.2 | 72.8% smaller, 87.8% faster |

## 📝 Citation

BibTeX
```
@article{alsatouf_2026_cocoyam,
title={Explainable AI-Driven Sparse Hyperspectral Modeling for Cocoyam Moisture Forecasting},
author={Alsatouf, A. and Dey, R. and Ndisya, J. M. and Arefi, A. and Sturm, B. and H"ohne, M. M.-C. and Babor, M.},
journal={Manuscript submitted for publication.},
year={2026},
doi={XXXX}
}
```
APA
```
Alsatouf, A., Dey, R., Ndisya, J. M., Arefi, A., Sturm, B., Höhne, M. M.-C., & Babor, M. (2026). Explainable AI-driven sparse hyperspectral modeling for cocoyam moisture forecasting. Manuscript submitted for publication. https://doi.org/XXXX
```


## License

Specify license (e.g., MIT, Apache 2.0) here.

## Contact

For questions, please open an issue or contact the corresponding author.
