# Motor Imagery EEG Classification - Group OLAF

Training pipeline for 4-class Motor Imagery (MI) EEG classification on the
BCI Competition IV Dataset 2a, using a **synthetic / simulated** generator that
reproduces the event-related desynchronization / synchronization (ERD/ERS)
structure described in the project report.

Three classifier families are implemented and benchmarked:

1. **FBCSP + SVM** - Filter Bank Common Spatial Patterns with an RBF-kernel SVM.
2. **varEEGNet** - EEGNet variant with a log-variance compression layer (~1,636 params).
3. **Fusion 3CNNs** - three parallel convolutional branches fused before a dense head (~23k params).

Baselines for completeness: standard **EEGNet** (~3,524 params), **KNN**, **AdaBoost**.

> **Parameter counts.** The realized counts above are close to, but not
> identical to, the published figures cited in the report (EEGNet 3,444;
> varEEGNet 1,524). The small differences come from batch-/layer-norm
> bookkeeping and the exact pooling geometry, which depend on the input length.
> The architectural *mechanism* matches the report: varEEGNet replaces EEGNet's
> average-pool + flatten with a single log-variance layer, and the resulting
> ~53% parameter reduction (3,524 -> 1,636) reproduces the report's claim.

<img width="1040" height="520" alt="vareegnet_loss" src="https://github.com/user-attachments/assets/bfa4dcc7-8804-4809-be26-e60931206c2e" />
<img width="1040" height="520" alt="vareegnet_accuracy" src="https://github.com/user-attachments/assets/ec81c52a-15d0-4cda-87ea-3c3e201debea" />
<img width="1170" height="650" alt="per_subject_accuracy" src="https://github.com/user-attachments/assets/aa4d17ee-253b-4da3-839b-812b3799a57b" />
<img width="1300" height="650" alt="per_fold_accuracy" src="https://github.com/user-attachments/assets/81c0595a-1cde-4d24-b3ac-06f76c4e2a2e" />
<img width="1040" height="650" alt="parameter_efficiency" src="https://github.com/user-attachments/assets/9d60ea64-d47e-4c4e-bfeb-390adf2a3bd1" />
<img width="1170" height="650" alt="model_accuracy_comparison" src="https://github.com/user-attachments/assets/01043f21-677b-4600-86dd-c44d31ece6fb" />
<img width="1170" height="650" alt="kappa_by_model" src="https://github.com/user-attachments/assets/6b42919a-8aaa-42dc-9c0d-77bb3ec1afeb" />
<img width="1040" height="520" alt="fusion3cnns_loss" src="https://github.com/user-attachments/assets/87e7c1a7-fb3a-4970-be01-49214328b066" />
<img width="1040" height="520" alt="fusion3cnns_accuracy" src="https://github.com/user-attachments/assets/aba5a9e7-be2e-405a-8745-075229f19381" />
<img width="1040" height="520" alt="eegnet_loss" src="https://github.com/user-attachments/assets/98e2de45-7a00-4c24-aa60-727e0700d231" />
<img width="1040" height="520" alt="eegnet_accuracy" src="https://github.com/user-attachments/assets/0d0cc29b-de5c-4054-aced-69ef4ef304c1" />



## Dataset

The real BCI-IV 2a dataset is not redistributed here. Instead,
`src/dataset.py` generates a **synthetic** dataset with the same shape
(9 subjects, 22 EEG channels, 250 Hz, 4-second trials, 4 classes, 792 trials)
and physiologically motivated class-dependent band-power modulation over the
motor-cortex channels. This lets the full pipeline run end-to-end and produces
the high in-distribution accuracy reported for the synthetic set. Real-data
behaviour is documented in the report.

## Layout

```
mi_eeg_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml          # full benchmark configuration
│   └── smoke.yaml            # fast end-to-end smoke test
├── src/
│   ├── __init__.py
│   ├── config.py             # dataclass config + YAML loader
│   ├── dataset.py            # synthetic BCI-IV 2a generator
│   ├── preprocessing.py      # bandpass filtering, standardisation
│   ├── fbcsp.py              # Filter Bank CSP feature extractor
│   ├── models.py             # EEGNet, varEEGNet, Fusion 3CNNs (Keras)
│   ├── trainer.py            # deep-model cross-validation loop
│   ├── evaluation.py         # metrics, cross-validation helpers
│   ├── plots.py              # training curves / accuracy plots
│   └── utils.py              # seeding, logging, IO
├── train.py                  # main entry point — runs the full benchmark
└── outputs/                  # results, metrics JSON, figures (created at runtime)
```

## Quick start

```bash
pip install -r requirements.txt
python train.py --config configs/default.yaml
```

Results (per-model accuracy, kappa, per-fold, per-subject) are written to
`outputs/metrics.json` and figures to `outputs/figures/`. A human-readable
summary is printed at the end and saved to `outputs/summary.txt`.

### Runtime and hardware

The classical models (FBCSP + SVM / KNN / AdaBoost) and the full per-subject
analysis run in well under a minute on CPU. The deep models are the expensive
part: on CPU each EEGNet/varEEGNet epoch is a few seconds and Fusion 3CNNs is
slower, so the full default run (5-fold CV, 75 epochs, three deep models) is a
**GPU-recommended** job — expect tens of minutes on CPU.

Two faster paths:

```bash
# Classical only — no TensorFlow, finishes in seconds.
python train.py --config configs/default.yaml --no-deep

# Smoke test — small dataset, few epochs/folds, exercises the whole pipeline
# (all six models + figures) in a couple of minutes. Not reportable numbers.
python train.py --config configs/smoke.yaml
```

The deep models need their full epoch budget to converge on this data; with the
truncated smoke epochs the EEGNet family can sit near chance, which is a budget
artefact, not a bug. Use `configs/default.yaml` (ideally on a GPU) for numbers
you intend to report.

## Reproducibility

All randomness is seeded from `config.seed`. The synthetic generator,
NumPy, Python `random`, and TensorFlow are all seeded. Deep-learning results
on CPU can still vary slightly due to non-deterministic kernels; the seed
makes the synthetic data and CSP fits fully deterministic.

## Notes on scope

- This is a research/benchmark pipeline, not a real-time deployment build.
  Latency and embedded-deployment figures in the report are analytical.
- The synthetic generator is deliberately separable; it validates the
  pipeline and matches the report's synthetic-set numbers. It is **not** a
  substitute for real BCI-IV 2a evaluation.
