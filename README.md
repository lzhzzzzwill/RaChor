# RaChor

**Raman Chorus (RaChor)** connects Raman fingerprints with paired Sn and Bi responses through shared spectral representation and global–regional hierarchical stacking. It combines selected Raman channels, spectral-shape descriptors, and complementary regression models, then uses target-specific TabPFN meta-learners to produce the final predictions.

RaChor organizes prediction into four connected components:

- **Spectral preprocessing:** retain the 400–1800 cm⁻¹ fingerprint interval, apply standard normal variate (SNV) normalization and a Savitzky–Golay first derivative, and standardize the spectral channels.
- **Shared representation:** identify a common candidate-band pool for Sn and Bi using MultiTaskLassoCV, apply target-specific feature limits, and add spectral center, width, intensity sum, and four cosine positional descriptors.
- **Global and regional prediction:** combine PLS, KRR, ElasticNet, random Fourier feature ridge regression, and random-subspace ridge models globally; construct regional stacks over the low-, middle-, and high-frequency intervals.
- **Target-specific fusion:** use separate TabPFN meta-learners to combine the global prediction with retained regional predictions for each metal.

```mermaid
flowchart LR
    A[Raman spectra] --> B[400–1800 cm⁻¹ · SNV · SG · Standardization]
    B --> C[Shared bands and spectral descriptors]
    C --> D[Global stack]
    C --> E[Regional stacks]
    D --> F[Target-specific TabPFN fusion]
    E --> F
    F --> G[Sn and Bi predictions]
```

## Script architecture

```text
RaChor/
├── main.py                         # Cross-validation and full-data model export
├── config.py                       # Folds, seeds, feature limits, regional settings
├── requirements.txt                # RaChor training dependencies
├── export_deployment_artifacts.py   # Export complete prediction bundles
├── data/
│   ├── Raman_spectroscopy_data.xlsx
│   └── Raman_spectroscopy_data_preprocessed.csv
├── data_preprocessing/
│   └── spectral_preprocessing.py
├── features/
│   ├── selector.py
│   └── builder.py
├── models/
│   ├── custom_models.py
│   └── trainer.py
├── pipeline/
│   ├── orchestrator.py
│   └── reporter.py
├── utils/
├── nestedcv_oof_predictions.xlsx    # Saved RaChor evaluation results
└── te_not_in_training_comparison/  # Recorded supplementary-sample checks
```

The `artifacts/` directory is created when exporting a full-data model.

## Core tasks

### Train and evaluate RaChor

The training workflow reads paired metal labels and numeric Raman columns, builds fold-specific representations, tunes the regression models, and reports outer-fold predictions. The default configuration uses five outer folds and five inner folds with a base random seed of 42.

### Export a complete prediction model

The exporter refits the model on the available training data and saves its global models, regional models, final fusion layers, spectral metadata, and fitted preprocessing scaler. These artifacts are separate from models fitted during cross-validation.

## Quick start

Use a Python environment compatible with the pinned packages; Python 3.11 is a suitable starting point. GitHub access is required to clone this private repository.

```bash
git clone https://github.com/lzhzzzzwill/RaChor.git
cd RaChor

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Generate the preprocessed spectral table.
python data_preprocessing/spectral_preprocessing.py

# Cross-validation followed by a full-data model fit and export.
python main.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. TabPFN pretrained checkpoints are obtained through the installed TabPFN package and are not bundled as weights in this repository. If TabPFN is unavailable, the existing factory falls back to Ridge and logs a warning; such runs should be identified separately.

For cross-validation only:

```bash
python main.py --skip-model-export
```

To export the full-data model separately:

```bash
python export_deployment_artifacts.py --artifact-dir artifacts
```

## Data and outputs

The default training table contains 187 samples. The response columns are `118Sn (KED)` and `209Bi (KED)`; spectral columns are recognized by numeric headers within 400–1800 cm⁻¹. Responses retain the numeric scale supplied in the dataset. Other metal-label columns are not used as spectral inputs.

RaChor writes `nestedcv_oof_predictions.xlsx`, including global-stack and final-fusion predictions, fold performance, and a global summary. Historical code and workbook identifiers use `MoE` for final fusion; the implemented method is hierarchical stacking with TabPFN meta-learners.

Metrics are **R², RMSE, and RPD**. RPD is the sample standard deviation (`ddof=1`) divided by RMSE. Pooled OOF metrics are calculated from all outer holdout predictions, not by averaging fold scores.

The existing RaChor workflow standardizes the full spectral dataset before outer cross-validation and performs supervised band selection before internal stacking OOF generation. Sample-level OOF results evaluate the supplied material collection rather than transfer to a new experimental batch or instrument. Recorded supplementary-sample checks are described in [their data notes](te_not_in_training_comparison/README.md).
