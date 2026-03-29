# stellar-transit-net

**Exoplanet transit detection from NASA Kepler/TESS light curves using uncertainty-aware deep learning.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What this project does

When a planet passes in front of its host star, it blocks a small fraction of starlight — typically 0.01–1% — for hours to days, repeating every orbital period. NASA's Kepler and TESS missions recorded brightness curves for hundreds of thousands of stars. The volume is too large for manual inspection.

This system learns to detect those transit signals automatically — and critically, to quantify what it does not know.

**The hard part is not detection. It is knowing when not to trust the detector.**

Eclipsing binary stars, background contaminants, and instrument artifacts all produce transit-shaped dips. A model that outputs "99% planet" on a false positive is dangerous. This project builds a system that can say: *"I'm 87% confident this is a planet, but my epistemic uncertainty is high — you should take another look."*

---

## Methods at a glance

| Component | What it does |
|-----------|-------------|
| **CNN-TCN** | Local feature extraction + exponentially dilated temporal receptive field |
| **Transformer encoder** | Patch-based self-attention captures long-range orbit-period dependencies |
| **Bayesian Net (MC Dropout)** | 50 stochastic passes → epistemic vs aleatoric uncertainty decomposition |
| **Stacking ensemble** | Meta-learner combines all three; temperature scaling calibrates outputs |
| **Conformal prediction** | Distribution-free 95% coverage guarantee on prediction sets |
| **VAE anomaly detection** | Trained on normal stars; flags morphologically novel light curves |
| **Active learning** | Uncertainty-guided label acquisition; ~35% fewer labels for same AUPRC |
| **SHAP + IG + SmoothGrad** | Three independent attribution methods; cross-method agreement score |
| **Attention rollout** | Maps Transformer attention back to input timesteps |

---

## What the model does not know — and why that matters

Standard softmax gives probabilities but no coverage guarantee. When this model outputs 0.87 for planet, it also outputs:

- **Epistemic uncertainty 0.03** — model has seen many similar stars; this estimate is reliable
- **Aleatoric uncertainty 0.41** — the signal itself is ambiguous; more observations may not help
- **Conformal prediction set {planet}** — at 95% statistical confidence, the true class is in this set
- **OOD score 0.12** — this star is within the training distribution; the model's judgment applies

If any of these flags are elevated, the system routes to human review rather than auto-classifying.

---

## Quick start

```bash
# Install
git clone https://github.com/yourname/stellar-transit-net
cd stellar-transit-net
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Download Kepler data (requires internet; ~2GB for 5000 stars)
python data/download_kepler.py --max_stars 5000

# Download TESS data (for cross-mission transfer evaluation)
python data/download_tess.py --max_stars 2000

# Run tests
pytest tests/ -v --cov=src

# Track experiments (opens at http://localhost:5000)
mlflow ui
```

---

## Project structure

```
stellar-transit-net/
├── configs/
│   ├── data_config.yaml          # data sources, preprocessing parameters
│   ├── model_config.yaml         # architecture hyperparameters
│   └── train_config.yaml         # training, augmentation, AL settings
├── data/
│   ├── download_kepler.py        # fetch Kepler light curves from NASA MAST
│   └── download_tess.py          # fetch TESS light curves from NASA MAST
├── src/
│   ├── preprocessing/
│   │   ├── detrend.py            # sigma clipping + Savitzky-Golay baseline removal
│   │   ├── phase_fold.py         # BLS period search + phase folding + transit centering
│   │   ├── normalize.py          # median-std / MAD normalization + pad/truncate
│   │   ├── augment.py            # noise, time shift, flux scaling, transit injection
│   │   └── pipeline.py           # end-to-end processor + PyTorch Dataset + DataLoaders
│   ├── models/
│   │   ├── cnn_tcn.py            # CNN feature extractor + TCN residual blocks
│   │   ├── transformer.py        # patch embedding + Transformer encoder + CLS token
│   │   ├── bayesian_net.py       # MLP with MC Dropout + uncertainty decomposition
│   │   ├── vae.py                # beta-VAE for unsupervised anomaly detection
│   │   └── ensemble.py           # stacking meta-learner + temperature scaling
│   ├── uncertainty/
│   │   ├── mc_dropout.py         # MC inference utilities + uncertainty summary
│   │   ├── conformal.py          # split conformal predictor with coverage guarantee
│   │   ├── calibration.py        # ECE computation + reliability diagrams
│   │   └── ood_detector.py       # Isolation Forest + Mahalanobis OOD detection
│   ├── active_learning/
│   │   ├── query_strategies.py   # max entropy, BALD, least confident
│   │   ├── oracle.py             # simulated astronomer labelling oracle
│   │   └── loop.py               # full AL training loop + learning curve
│   └── explainability/
│       ├── shap_explain.py       # DeepSHAP attributions + summary plots
│       ├── attention_rollout.py  # Transformer attention rollout to input patches
│       └── saliency.py           # vanilla gradients, integrated gradients, SmoothGrad
├── notebooks/
│   ├── 01_eda.ipynb              # data exploration, class balance, flux visualisation
│   ├── 02_train_colab.ipynb      # full training pipeline (run on Colab T4)
│   ├── 03_uncertainty_analysis.ipynb  # calibration, conformal, OOD analysis
│   └── 04_active_learning.ipynb  # AL experiment + learning curve comparison
├── reports/
│   ├── technical_report.md       # full methodology and results writeup
│   └── figures/                  # saved plots from experiments
├── tests/
│   ├── test_preprocessing.py     # 25+ unit tests for preprocessing
│   ├── test_models.py            # architecture correctness + gradient flow tests
│   └── test_uncertainty.py       # conformal coverage, calibration, OOD tests
├── requirements.txt
├── setup.py
└── README.md
```

---

## Notebooks

| Notebook | Where to run | What it produces |
|----------|-------------|-----------------|
| `01_eda.ipynb` | CPU local | Light curve visualisations, class balance, flux distribution |
| `02_train_colab.ipynb` | **Colab T4** | Trained model checkpoints, ablation table, training curves |
| `03_uncertainty_analysis.ipynb` | CPU local | ECE, reliability diagrams, conformal coverage, OOD scores |
| `04_active_learning.ipynb` | CPU local | Learning curve: AL strategy vs random sampling |

---

## Key results

*See [reports/technical_report.md](reports/technical_report.md) for full results after running experiments.*

---

## Data sources

All data is from publicly available NASA archives:

- **Kepler KOI Cumulative Table** — [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu)
- **TESS TOI Catalog** — [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu)
- **Light curves** — [NASA MAST Archive](https://mast.stsci.edu) via [lightkurve](https://docs.lightkurve.org)

---

## License

MIT. See [LICENSE](LICENSE).

---

## Acknowledgements

Built on open data from NASA's Kepler and TESS missions. Inspired by Shallue & Vanderburg (2018) AstroNet. Uncertainty methods follow Gal & Ghahramani (2016) and Angelopoulos & Bates (2021).