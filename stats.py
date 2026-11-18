# =====================================================================
#  vCDR STATISTICS — run after vcdr_sheets.py (uses its per-image CSVs)
#
#  Part 1: Wilcoxon signed-rank tests, proposed vs each ablation
#          (paired, two-sided, per-image absolute vCDR error),
#          Holm–Bonferroni corrected, with effect size (rank-biserial r)
#          -> Table 6 footnotes, Reviewer 2.2
#  Part 2: Screening at vCDR >= 0.65
#          ROC-AUC, sensitivity, specificity, each with bootstrap 95% CI
#          -> Reviewer 1.1 and 1.2
#
#  Expects CSVs named <dataset>_<zs|ft>_<variant>.csv (from vcdr.py) in --results
#  Output (in --results): stats_wilcoxon.csv, stats_screening.csv
#  Example: python stats.py --results results
# =====================================================================
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results", help="folder with the per-image CSVs from vcdr.py")
OUT_DIR = ap.parse_args().results
CUTOFF  = 0.65      # vCDR threshold (paper Section 5.2.2)
N_BOOT  = 2000      # bootstrap repeats for 95% CI
SEED    = 42

def load(run):
    p = f"{OUT_DIR}/{run}.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df.dropna(subset=["vcdr_true", "vcdr_pred"]).set_index("filename")

def holm(pvals):
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation for paired errors a vs b.
    Positive = b (proposed) has smaller error than a."""
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)

# =====================================================================
# PART 1 — WILCOXON (proposed vs each ablation)
# =====================================================================
FAMILIES = {
    "Architecture": {"baseline": "Baseline", "aspp": "+ ASPP",
                     "drb": "+ DRB", "aspp_irskip": "+ ASPP + IR-Skip"},
    "Loss":         {"loss_dice": "DSC", "loss_equal": "DSC + Focal (1:1)"},
}
DATASETS_TEST = ["refuge", "drishti"]          # external datasets (as in paper)

rows = []
for fam, variants in FAMILIES.items():
    fam_rows = []
    for ds in DATASETS_TEST:
        main = load(f"{ds}_zs_main")
        if main is None:
            print(f"⚠ {ds}_zs_main.csv not found"); continue
        for v, label in variants.items():
            other = load(f"{ds}_zs_{v}")
            if other is None:
                print(f"⚠ {ds}_zs_{v}.csv not found -> skipped"); continue
            common = main.index.intersection(other.index)
            a = other.loc[common, "abs_error"].values          # ablation error
            b = main.loc[common, "abs_error"].values           # proposed error
            stat, p = wilcoxon(a, b, alternative="two-sided")
            fam_rows.append({
                "family": fam, "dataset": ds.upper(), "comparison": f"Proposed vs {label}",
                "n": len(common),
                "MAE ablation": a.mean(), "MAE proposed": b.mean(),
                "proposed better?": "yes" if b.mean() < a.mean() else "NO",
                "W": stat, "p raw": p,
                "effect r": rank_biserial(a, b),
            })
    if fam_rows:
        adj = holm([r["p raw"] for r in fam_rows])      # Holm within each family
        for r, pa in zip(fam_rows, adj):
            r["p Holm"] = pa
            r["sig"] = stars(pa)
        rows += fam_rows

W = pd.DataFrame(rows)
if len(W):
    W.to_csv(f"{OUT_DIR}/stats_wilcoxon.csv", index=False)
    print("=" * 110)
    print("PART 1 — Wilcoxon signed-rank, proposed vs ablation (zero-shot vCDR abs error), Holm within family")
    print("effect r > 0 means the proposed model has lower error; |r| ~0.1 small, ~0.3 medium, ~0.5 large")
    print("=" * 110)
    show = W[["family", "dataset", "comparison", "n", "MAE ablation", "MAE proposed",
              "proposed better?", "p raw", "p Holm", "sig", "effect r"]].copy()
    for c in ["MAE ablation", "MAE proposed", "effect r"]:
        show[c] = show[c].map(lambda x: f"{x:.4f}")
    for c in ["p raw", "p Holm"]:
        show[c] = show[c].map(lambda x: f"{x:.2e}")
    print(show.to_string(index=False))

# =====================================================================
# PART 2 — SCREENING at vCDR >= 0.65 (AUC, sensitivity, specificity + 95% CI)
# =====================================================================
SETTINGS = [("origa_zs_main", "ORIGA-Light (in-domain)"),
            ("drishti_zs_main", "DRISHTI-GS1 (ZS)"), ("drishti_ft_main", "DRISHTI-GS1 (FT)"),
            ("refuge_zs_main", "REFUGE (ZS)"),       ("refuge_ft_main", "REFUGE (FT)")]

def screen(y, score):
    pred = (score >= CUTOFF).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    auc = roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan
    return auc, sens, spec, tp, fn, tn, fp

rng = np.random.default_rng(SEED)
rows = []
for run, label in SETTINGS:
    df = load(run)
    if df is None:
        print(f"⚠ {run}.csv not found"); continue
    y = (df.vcdr_true.values >= CUTOFF).astype(int)        # reference standard: GT vCDR >= 0.65
    s = df.vcdr_pred.values
    auc, sens, spec, tp, fn, tn, fp = screen(y, s)

    boot = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:                           # need both classes
            boot.append(screen(y[idx], s[idx])[:3])
    boot = np.array(boot, dtype=float)
    lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)

    rows.append({
        "Setting": label, "n": n, "positives": int(y.sum()), "negatives": int(n - y.sum()),
        "AUC": auc, "AUC 95% CI": f"{lo[0]:.3f}–{hi[0]:.3f}",
        "Sensitivity": sens, "Sens 95% CI": f"{lo[1]:.3f}–{hi[1]:.3f}",
        "Specificity": spec, "Spec 95% CI": f"{lo[2]:.3f}–{hi[2]:.3f}",
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
    })
    if min(y.sum(), n - y.sum()) < 10:
        print(f"⚠ {label}: only {min(y.sum(), n - y.sum())} images in the smaller class -> wide CI, say so in the paper")

T = pd.DataFrame(rows)
if len(T):
    T.to_csv(f"{OUT_DIR}/stats_screening.csv", index=False)
    print("\n" + "=" * 110)
    print(f"PART 2 — Screening at vCDR >= {CUTOFF}  (positive = reference vCDR >= {CUTOFF}; bootstrap {N_BOOT}x, 95% CI)")
    print("=" * 110)
    show = T.copy()
    for c in ["AUC", "Sensitivity", "Specificity"]:
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    print(show.to_string(index=False))

print(f"\nSaved: {OUT_DIR}/stats_wilcoxon.csv, {OUT_DIR}/stats_screening.csv")
