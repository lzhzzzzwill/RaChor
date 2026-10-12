# RaChor

**Raman Chorus (RaChor)** is a shared-band global–regional hierarchical stacking framework for Raman-based prediction of paired Sn and Bi responses.

This private repository archives the current training code, visualization code, associated data, and editable architecture diagram as of 2026-10-12. Source files are copied without changing the model implementation. The manuscript title is unchanged; the manuscript itself is outside this code-and-data archive.

## Contents

| Path | Contents |
| --- | --- |
| `07_code/Raman-MP-Metal-Analysis/` | Preprocessing, feature selection/construction, base models, nested evaluation, export script, and training data |
| `07_code/400LR.xlsx` | 200 spectra with metal-response labels |
| `07_code/400LR1.xlsx` | The same 200 spectra without response labels |
| `07_code/400LR_额外9条光谱.xlsx` | Nine spectra with labels differing from matching training records, with a correspondence sheet |
| `08_pictures/figure/` | Current publication plotting scripts, data snapshots, and exported figures |
| `08_pictures/picture_code/` | Historical plotting scripts and their source data, retained for provenance and data refresh |
| `08_pictures/architecture/` | Editable draw.io architecture and its PDF export |
| `投稿/si/04_Supplementary_Data.xlsx` | Supplementary data referenced by the plotting data-refresh workflow |
| `source_manifest.json` | SHA-256 hashes and sizes of the copied source files |

## Training

Use a compatible Python environment (Python 3.11 is a reasonable starting point) and install the recorded dependencies. TabPFN may require access to its pretrained checkpoints; these weights are not bundled here.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r 07_code/Raman-MP-Metal-Analysis/requirements.txt
cd 07_code/Raman-MP-Metal-Analysis
python data_preprocessing/spectral_preprocessing.py
python main.py
```

The preprocessed CSV is already included. Preprocessing regenerates it from the raw training workbook. `main.py` writes `nestedcv_oof_predictions.xlsx` in the training directory. Its `XDG_CACHE_HOME` assignment currently points to an original Linux environment; adjust that path to a writable local cache directory before training elsewhere. No complete training rerun was performed for this archive.

The configured response columns are `118Sn (KED)` and `209Bi (KED)`. MultiTaskLassoCV selects a common candidate-band pool; target-specific caps and separate regression/stacking models produce each response. TabPFN is used as a meta-learner. Historical identifiers such as `MoE` refer to final fusion, not to an explicit gating network.

## Visualization

From the repository root:

```bash
python -m pip install -r 08_pictures/figure/requirements.txt
python 08_pictures/figure/scripts/main.py --skip-appendix
```

Omit `--skip-appendix` to generate individual-sample appendix plots as well. Add `--refresh-data` to rebuild plotting snapshots from the preserved source directories. Plotting uses archived evaluation results; it does not automatically rerun model training or consume a newly generated training workbook. Existing PDFs are included as the current output snapshot.

## Data and implementation notes

- The training workbook contains 187 records. Of the 200 spectra in the 400LR workbooks, 196 exactly match training spectra over all 707 shared bands, covering all 187 distinct training spectra. The nine extra matching records are Y1–Y7, Y9, and Y10; their Sn/Bi labels differ from the matching training records. These are not independent generalization samples.
- Y8, Y118, Y167, and Y171 have no exact match to a training spectrum. Y167 and Y171 lack Sn/Bi labels. Absence of an exact match alone does not establish experimental independence.
- Response values retain the recorded data scale. Confirm their experimental definition and units before interpreting them as mass-normalized adsorption capacities or comparing datasets.
- Initial spectral standardization is fitted on the full input dataset before outer cross-validation. Supervised selection and tuning precede internal OOF generation. Archived performance should be interpreted within that implemented evaluation procedure.
- The current `export_deployment_artifacts.py` fits the final fusion model but omits `moe_meta` from the returned target bundle. This archive preserves that code; fix and validate the export before claiming full-fusion deployment inference. Exported model binaries and the web application are outside this training-and-visualization upload.
- Local caches, credentials, Python bytecode, and temporary files are excluded. The source code and copied data were checked for byte equality against the manifest, and Python files were syntax-checked. Dependency installation, full retraining, and figure regeneration were not performed during upload.

