# Cross-domain optic disc and cup segmentation with vCDR estimation

Code for the paper:

> Robust Cross-Domain Optic Disc and Cup Segmentation with Clinically Consistent vCDR Estimation for Glaucoma Screening. *Scientific Reports* (under revision).

The method is a two-stage pipeline. Stage I segments the optic disc (OD) on the full fundus image. Stage II segments the optic cup (OC) inside a region of interest (ROI) around the disc. The vertical cup-to-disc ratio (vCDR) is computed from the two masks. Both stages use an EfficientNet-B3 encoder–decoder with a Dilated Residual Bottleneck (DRB) and an Input-Residual Skip (IR-Skip).

## Files

| File | What it does |
|---|---|
| `src/models.py` | All architectures in the paper: baseline, + DRB, + ASPP, + ASPP + IR-Skip, proposed (DRB + IR-Skip) |
| `src/losses.py` | Hybrid loss: Dice + 0.5 × Focal (γ = 2, α = 0.25), and metrics |
| `src/data.py` | Image and mask loading (256 × 256, masks binarised at 127) |
| `make_cup_crops.py` | Stage II training crops: GT disc box + 15 px padding at 512 × 512 |
| `train.py` | Training on ORIGA-Light |
| `finetune.py` | Limited fine-tuning on REFUGE or DRISHTI-GS1 (all layers updated, up to 30 epochs) |
| `vcdr.py` | Fully automatic two-stage inference and vCDR (predicted disc → same crop for cup and disc) |
| `stats.py` | Wilcoxon tests (Holm–Bonferroni) and screening metrics (AUC, sensitivity, specificity, bootstrap 95% CI) |

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+ and TensorFlow 2.15+. A GPU is recommended for training.

## Data

The datasets are public and must be downloaded from their original sources:

- **ORIGA-Light**: 650 images. Source domain. Fixed split (seed 42): 497 train, 88 validation, 65 test.
- **DRISHTI-GS1**: official split, 50 development (fine-tuning: 42 train / 8 validation) and 51 test images.
- **REFUGE**: official split, 800 development (fine-tuning: 680 train / 120 validation) and 400 test images.

All images were resized to 512 × 512. Ground-truth masks are binarised at 50% intensity; for DRISHTI-GS1 this corresponds to the majority of expert annotations.

Each script reads a CSV file with a `filename` column that lists the images of one split. The ORIGA-Light split and the fine-tuning train/validation splits were made with scikit-learn `train_test_split` (stratified, `random_state=42`).

## How to reproduce

```bash
# 1. Stage II training crops (GT disc box + 15 px)
python make_cup_crops.py --image-dir data/full-fundus --disc-dir data/optic-disc \
    --cup-dir data/optic-cup --out-images data/cup_crops/images --out-masks data/cup_crops/masks

# 2. Train disc and cup models on ORIGA-Light (proposed model)
python train.py --variant drb_irskip --image-dir data/full-fundus --mask-dir data/optic-disc \
    --train-csv origa_train.csv --val-csv origa_val.csv --out models/disc_main.h5
python train.py --variant drb_irskip --image-dir data/cup_crops/images --mask-dir data/cup_crops/masks \
    --train-csv origa_train.csv --val-csv origa_val.csv --out models/cup_main.h5

#    Ablations: --variant baseline | drb | aspp | aspp_irskip
#    Loss ablation: --focal-weight 0 (Dice only) or 1 (equal weights)
#    Seeds: --seed 42 | 123 | 2026

# 3. Zero-shot vCDR on a target dataset
python vcdr.py --disc-model models/disc_main.h5 --cup-model models/cup_main.h5 \
    --image-dir data/refuge/images --disc-dir data/refuge/disc_masks --cup-dir data/refuge/cup_masks \
    --names refuge_test.csv --out results/refuge_zs_main.csv

# 4. Limited fine-tuning, then vCDR again (same command as step 3 with the fine-tuned models)
python finetune.py --model models/disc_main.h5 --image-dir data/refuge/images \
    --mask-dir data/refuge/disc_masks --train-csv refuge_ft_train.csv \
    --val-csv refuge_ft_val.csv --out models/disc_refuge_ft.keras

# 5. Statistics
python stats.py --results results
```

Result files from `vcdr.py` must be named `<dataset>_<zs|ft>_<variant>.csv` (for example `refuge_zs_main.csv`, `drishti_zs_drb.csv`) for `stats.py` to find them.

## Notes on the pipeline

- At **training** time the cup ROI comes from the ground-truth disc. At **test** time it comes from the **predicted** disc, so the pipeline is fully automatic.
- Only the largest connected region of the predicted disc is kept before the ROI is drawn (removes small false detections, for example on the image border).
- Cup and disc heights are measured in the same resized crop. Resizing scales all vertical lengths by the same factor, so this factor cancels in the ratio.
- Screening: a positive case is a reference vCDR ≥ 0.65. This measures detection of an enlarged vCDR, not a clinical glaucoma diagnosis.

## Citation

If you use this code, please cite the paper (citation will be added after publication).

## License

MIT License (see `LICENSE`).
