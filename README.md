# synth-curator-paper

Data and code for the paper **Evaluating Synthetic Skin Lesion Images for Classifier Training: A
Safety-Aware Algorithm** (Kumarasinghe, Bui and Ghanbarzadeh).

The repository holds every result table behind the paper's figures and tables, the scripts that turn
them into the published numbers, and the code of the pipeline stages listed below. It is a record of
one study, so clone it and run the scripts in place. The maintained, installable implementation of
the method is the [`synth-curator`](https://github.com/Amixdu/synth-curator) package
([PyPI](https://pypi.org/project/synth-curator/)).

The 275,472 synthetic images themselves are released separately on Zenodo ([10.5281/zenodo.23851497](https://doi.org/10.5281/zenodo.23851497)). This
repository contains their metadata and scores in `data/dataset_release/metadata.csv`.

```
data/    result tables, per-image predictions and curated image lists
code/    analysis scripts, pipeline code, configs and environment files
LICENSE       MIT license for the code
LICENSE-DATA  CC BY-NC-SA 4.0 license for the data
```

## Reproducing the paper's tables and figures

The scripts in `code/analysis/` read only from `data/` and write to `code/analysis/outputs/`.

```bash
conda env create -f code/environments/analysis.yml
conda activate analysis
cd code/analysis
python make_tables.py     # the 30 tables
python make_figures.py    # the 11 figures
```

Table files are written to `outputs/tables/` and figure PDFs to `outputs/figures/`. The generated
tables are identical to the ones typeset in the paper. Both scripts accept names to run a subset,
for example `python make_tables.py benchmark safety` or `python make_figures.py ranking forest`.
`common.py` holds the constants and data loaders that the two scripts share.

| Group (`make_tables.py`) | Tables |
|---|---|
| `pool` | `pool_composition`, `distributional_pool`, `backbone_arms` |
| `tournament` | `tournament_overall_{swin_b,resnet50}`, `tournament_per_class`, `tournament_per_class_resnet50`, `tournament_paired_{swin_b,resnet50}`, `tournament_fid_per_class` |
| `algorithm` | `algorithm_weights_full` |
| `benchmark` | `benchmark_cohorts_{swin_b,resnet50}`, `benchmark_stats`, `benchmark_stats_resnet50`, `benchmark_per_class`, `benchmark_per_class_resnet50`, `benchmark_per_class_diff`, `benchmark_primary`, `appendix_cohort_metrics_{swin_b,resnet50}`, `cohort_tail_stats` |
| `safety` | `safety_pool_config`, `safety_detection`, `fsms_reference_points`, `fsms_crossmodal`, `appendix_calibration_bands` |
| `resnet` | `resnet_leaders`, `resnet_weights_compare`, `resnet_temperature` |

`make_figures.py` writes `tournament_ranking`, `tournament_heatmap`, `tournament_per_class`,
`weights_heatmap`, `arch_agreement`, `benchmark_swin_b`, `benchmark_forest`, `calibration_curve`,
`safety_vs_tau`, `resnet_tournament_ranking` and `resnet_benchmark`.

## Data

| Folder | Contents | Used for |
|---|---|---|
| `data/metric_tournament/` | `tournament_results.csv` (validation metrics of every classifier in the metric tournament), `raw_predictions/` (per-image predictions of every classifier), `curated/` (the images each distributional metric selected, per seed, as `selections_<metric>.csv.gz`) | Metric tournament, algorithm weights |
| `data/design_choices/` | `temperature_sweep.csv`: validation metrics of the temperature sweeps (Swin-B at 3 values, ResNet-50 at 7 values) | Temperature selection |
| `data/benchmark/` | `final_results.csv` (test metrics and FSMS of every benchmark run), `raw_predictions/` (per-image test predictions), `safety_sweep/` (the tau = 0.90, 0.93 and 0.99 cohorts), `curated/` (the synthetic images each cohort selected, as `selections_<cohort>.csv.gz`), `tail_stats/` (similarity tails of the curated sets) | Clinical augmentation benchmark |
| `data/fsms_validation/` | The calibration and validation tables of the feature-space memorization score (FSMS), including the same-lesion pair table and `lesion_ids.csv` (the ISIC lesion identifier of each real training image, the input of the threshold calibration) | Safety threshold |
| `data/synthetic_pool/` | `config_distributional_metrics.csv`: FID, KID and SWD of the 16 generation configurations | Pool characterization |
| `data/dataset_release/` | `metadata.csv`: the per-image table of the Zenodo release (one row per image, with generation parameters, metric scores and the similarity to the nearest real image). Copied here because the scripts read it | Pool composition, selections, FSMS |

`data/dataset_release/metadata.csv` is a copy of the file in the Zenodo record, and the Zenodo record is the
authoritative source.

### Naming

- **Cohorts.** `random_synth` is the random-sampling baseline, `best_metric` is the best single
  metric of the tournament (FID-Inception), and `algorithm` is the proposed algorithm at
  tau = 0.97. `algorithm_safety090`, `algorithm_safety093` and `algorithm_safety099` are the
  same algorithm at the other thresholds, and the `_n20000` suffix marks the cohorts of 20,000
  images. All other cohorts have 8,012 images.
- **Tournament files.** `edm2_<metric>_seed<k>` names the run of metric `<metric>` at curation
  seed `k`. Instance metrics are deterministic and have a single run. The random control uses
  seeds 42 to 71, extended to 42 to 101 for Swin-B.
- **Benchmark files.** `<cohort>_seed_<k>_<arch>_predictions.csv`, with `<arch>` either `swin_b` or
  `resnet50`. The curation seeds are 0 to 29 in the file names (the pipeline shifts them to 100
  to 129 internally).
- **Selections.** A `selections_<name>.csv.gz` file holds every curated set of one cohort (or one
  tournament metric) in long format: `seed`, `arch` (only for cohorts curated per classifier) and
  `filename`. `filename` is the path of the image inside its generation configuration, for example
  `ckpt_25k_cfg_1.5/class_0/001497.png`, and matches the `filename` column of
  `data/dataset_release/metadata.csv`. Rows keep the order of the original curated sets, and the class is
  the `class_<n>` part of the path. `benchmark/scripts/1_train_augmented.py` reads these files directly
  (`--selections`, `--seed` and `--pool_dir`).
- **Classes** are `AKIEC`, `BCC`, `BKL`, `DF`, `MEL`, `NV` and `VASC`, with labels 0 to 6 in that
  order.

## Code

This repository holds the code for EDM2 training and generation, the memorization-score validation
suite, classifier training and evaluation, and the table and figure scripts. The curation algorithm
(per-class weights, group scoring, temperature sampling, the safety filter) and the per-image
metrics are not part of it. The maintained implementation of the method is the
[`synth-curator`](https://github.com/Amixdu/synth-curator) package.

`code/` is the project root of the scripts. They add this directory to `sys.path`, so run them from
inside `code/` or by their full path. The folders follow the Methods section of the paper:

| Paper section | Code folder | Contents |
|---|---|---|
| Synthetic Pool | `synthetic_pool/` | EDM2 training, generation and evaluation scripts (`edm2/`, see below) and their configs (`configs/`) |
| Tournament Protocol | `classifiers/` | `metric_tournament/`: training of the Swin-B and ResNet-50 classifiers (`train_swin_b.py`, `train_resnet50.py`), evaluation and compilation of `tournament_results.csv`, and the configs. `train_utils.py` and `evaluation.py` are shared with the benchmark |
| Feature-Space Memorization Score | `fsms/` | `metric.py` (the score), and `scripts/`: feature packing, validation, threshold calibration against same-lesion pairs, and the tournament's FSMS |
| Benchmark Protocol | `benchmark/` | `scripts/`: training with synthetic images, FSMS of the cohorts, test-set evaluation, compilation and tail statistics. `configs/`: train and eval configs |
| Results | `analysis/` | The table and figure scripts above and their shared `common.py` |
| | `utils/` | Config loading, real-image standardization and the DINOv2 feature extractor |
| | `environments/` | Conda environment files |

The scripts were run on a GPU cluster, and their configs contain placeholder paths
(`/path/to/scratch`, `/path/to/code`, `/path/to/datasets`). Edit them before use. The image
datasets are not included: ISIC-2018 is available from the
[ISIC Challenge](https://challenge.isic-archive.com/data/) and the synthetic pool from [Zenodo](https://doi.org/10.5281/zenodo.23851497).

### Order of the stages

1. **Synthetic pool** (`synthetic_pool/`, `edm2` environment). Preprocess the real training images,
   train, evaluate snapshots, generate the pool (4 checkpoints x 4 CFG scales), and pad the NV class.
2. **Metrics, features and curation.** Compute the per-image scores
   (`data/dataset_release/metadata.csv`) and the DINOv2 and InceptionV3 features, and curate the
   tournament's sets (`data/metric_tournament/curated/`) and the benchmark cohorts
   (`data/benchmark/curated/`). The code of this stage is not part of this repository.
3. **Metric tournament** (`classifiers/metric_tournament/`, `classifiers` environment). Train a
   classifier on each curated set, evaluate on the real validation set, and compile
   `tournament_results.csv`.
4. **Memorization score and threshold calibration** (`fsms/`, with the `feature_extraction`
   environment for the feature extraction). Extract features of the real test set, pack the
   features, validate FSMS, and calibrate the threshold against same-lesion pairs of real images.
5. **Benchmark** (`benchmark/`). Train with `scripts/1_train_augmented.py`, compute FSMS with
   `scripts/2_compute_fsms.py`, evaluate on the test set with `scripts/3_evaluate_test_set.py`, and
   compile with `scripts/4_compile_results.py`. The configs are in `benchmark/configs/`.

Two parameter sweeps of the study varied one setting over the same configuration. The temperature
sweeps used T = 0.02, 0.05 and 0.1 for Swin-B and T = 0.02, 0.05, 0.1, 0.3, 0.5, 1.0 and 2.0 for
ResNet-50, and their results are in `data/design_choices/temperature_sweep.csv`. The
safety sweep used tau = 0.90, 0.93, 0.97 and 0.99, and its results are in
`data/benchmark/` (`safety_sweep/` for 0.90, 0.93 and 0.99, and `final_results.csv` for 0.97).

### EDM2

`code/synthetic_pool/edm2/` contains only the files written for this study. The EDM2 code itself
comes from [NVlabs/edm2](https://github.com/NVlabs/edm2). Our scripts were built on commit
`4bf8162` ("Autoguidance release"), and every upstream file we used is unchanged from it. To run
them, clone that commit and copy our six files into the same directory:

| File | Role |
|---|---|
| `generate_images_cfg_patch.py` | Modified `generate_images.py` that supports standard single-model classifier-free guidance. Upstream supports only autoguidance with a separate guidance model. |
| `generate_batch.py` | Maps a task index (`SLURM_ARRAY_TASK_ID`) to a checkpoint and CFG scale and generates the per-class volumes |
| `generate_nevus_padding.py` | Generates extra NV images, seeds 1786 to 6500, for each configuration |
| `launcher.py` | Builds the `torchrun` training command from a YAML config |
| `preprocess_edm.py` | Stages the data, audits the VAE and builds the RGB and latent archives |
| `evaluate.py` | Computes FID for each training snapshot |

The configs are in `code/synthetic_pool/configs/` (`train/`, `generate/` and `evaluate/`). The
generation config lists the checkpoint and CFG grid and the number of images per class.

### Environments

| File | Used for |
|---|---|
| `edm2.yml` | Synthetic pool: EDM2 training, generation and evaluation |
| `feature_extraction.yml` | Feature extraction for the memorization-score suite |
| `classifiers.yml` | Classifier training and evaluation (PyTorch 2.2.0, torchvision 0.17.0, scikit-learn 1.4.1, pandas 2.2.1, Python 3.10) |
| `analysis.yml` | The scripts in `code/analysis/` |

The first three are exact exports of the environments used for the study and assume Linux.
`analysis.yml` is a minimal specification with version ranges.

### Inputs that are not in this repository

The scripts read some large inputs that are not included. Obtain them as follows:

| Input | Source |
|---|---|
| ISIC-2018 images | [ISIC Challenge](https://challenge.isic-archive.com/data/) |
| Synthetic pool (275,472 images) | [Zenodo record](https://doi.org/10.5281/zenodo.23851497) |
| DINOv2 features of the real training set, the real test set and the pool (`real_train_features.pt`, `real_test_features.pt`, `synthetic_pool_features.pt`) | `utils/standardize_images.py` (center-crop and resize to 512 x 512) followed by `utils/feature_extraction/extract_dino_features.py`; `fsms/configs/extract_test_features.yml` is an example config. The pool is generated at 512 x 512 and needs no standardization |
| Trained classifier checkpoints | Retrain with `classifiers/metric_tournament/train_*.py` and `benchmark/scripts/1_train_augmented.py`, or use the stored predictions in `data/` |

## Reproducibility notes

- Classifier training uses a fixed seed of 42 with deterministic cuDNN, but PyTorch's full
  deterministic mode was not enabled, so a retrained classifier can differ slightly from the
  stored predictions. The per-image predictions in `data/` are the source of every reported number.
- The curation seeds are shared across the safety thresholds, and cohorts drawn from different
  pools with the same seed number give unrelated draws.

## License

The code in `code/` is released under the MIT license (`LICENSE`), except
`code/synthetic_pool/edm2/generate_images_cfg_patch.py`, which derives from NVIDIA's EDM2 code and
remains under CC BY-NC-SA 4.0 (`code/synthetic_pool/edm2/LICENSE.txt`). The data in `data/` is
released under CC BY-NC-SA 4.0 (`LICENSE-DATA`). The ISIC-2018 data used to train the generator
is released under CC BY-NC, so the data licenses are non-commercial.

## Citation

```
@article{kumarasinghe2026synthetic,
  title  = {Evaluating Synthetic Skin Lesion Images for Classifier Training: A Safety-Aware Algorithm},
  author = {Kumarasinghe, Amindu and Bui, Vinh and Ghanbarzadeh, Reza},
  year   = {2026}
}
```
