"""Step 4 - collect results into the manuscript tables and run the statistical tests.

Writes, for one dataset:
  table_test_results.csv      every model on the held-out test set (from ITS confusion matrix)
  table_cv_results.csv        mean +/- SD over repeated stratified group k-fold CV
  table_ablation.csv          proposed model vs. its ablations (test set + CV mean)
  table_efficiency.csv        parameters, GFLOPs, training time, latency (batch 1 and batch B)
  table_significance.csv      exact McNemar test (test set) and corrected resampled t-test (CV),
                              proposed vs. each other model, Holm-adjusted
  confusion_matrix_<model>.png
  tables.md                   all of the above as Markdown, ready to paste

Example:
  python make_tables.py --results results/bell --split splits/bell_split.csv
"""
import argparse, os, json, glob
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

METRICS = ["accuracy", "balanced_accuracy", "precision_macro", "sensitivity_macro", "specificity_macro",
           "f1_macro", "roc_auc", "kappa", "mcc"]


def pct(x):
    return "" if pd.isna(x) else f"{100 * x:.2f}"


def corrected_t(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired CV scores."""
    d = np.asarray(a) - np.asarray(b); k = len(d)
    if k < 2 or np.allclose(d.var(ddof=1), 0):
        return float("nan"), float("nan")
    var = d.var(ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return t, 2 * stats.t.sf(abs(t), k - 1)


def plot_cm(cm_csv, out_png, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cm = pd.read_csv(cm_csv, index_col=0)
    labels = [c.replace("pred_", "") for c in cm.columns]
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 1.5))
    ax.imshow(cm.values, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.values.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right"); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ours", default="proposed")
    a = ap.parse_args()
    df = pd.read_csv(a.split)
    models = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(a.results, "*", "run_config.json"))
                    ) or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(a.results, "*", "cv_metrics.csv")))
    md = []

    # ---- test-set table + efficiency
    test_rows, eff_rows = [], []
    for m in models:
        f = os.path.join(a.results, m, "test_metrics.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        test_rows.append({"model": m, **{k: pct(r[k]) for k in METRICS}, "n_test": r["n"]})
        eff_rows.append({"model": m, "params_M": round(r["params_M"], 2),
                         "GFLOPs": None if r.get("GFLOPs") is None else round(r["GFLOPs"], 2),
                         "train_time_min": round(r["train_time_min"], 2),
                         **{k: round(v, 2) for k, v in r.items() if k.startswith("latency") or k.startswith("glcm")}})
        plot_cm(os.path.join(a.results, m, "test_confusion_matrix.csv"),
                os.path.join(a.results, f"confusion_matrix_{m}.png"), f"{m} - held-out test set")
    if test_rows:
        t = pd.DataFrame(test_rows); t.to_csv(os.path.join(a.results, "table_test_results.csv"), index=False)
        e = pd.DataFrame(eff_rows); e.to_csv(os.path.join(a.results, "table_efficiency.csv"), index=False)
        md += ["## Held-out test set (%)", t.to_markdown(index=False), "", "## Efficiency", e.to_markdown(index=False), ""]

    # ---- CV table
    cv_rows, cv_raw = [], {}
    for m in models:
        f = os.path.join(a.results, m, "cv_metrics.csv")
        if not os.path.exists(f):
            continue
        c = pd.read_csv(f); cv_raw[m] = c
        cv_rows.append({"model": m, "runs": len(c),
                        **{k: f"{100 * c[k].mean():.2f} ± {100 * c[k].std(ddof=1):.2f}" for k in METRICS}})
    if cv_rows:
        cvt = pd.DataFrame(cv_rows); cvt.to_csv(os.path.join(a.results, "table_cv_results.csv"), index=False)
        md += ["## Repeated stratified group k-fold CV (mean ± SD, %)", cvt.to_markdown(index=False), ""]

    # ---- ablation table
    abl = [m for m in models if m == a.ours or m.startswith("abl_")]
    if abl and test_rows:
        t = pd.DataFrame(test_rows).set_index("model")
        rows = []
        for m in abl:
            row = {"variant": m}
            if m in t.index:
                row.update({f"test_{k}": t.loc[m, k] for k in ["accuracy", "balanced_accuracy", "f1_macro"]})
            if m in cv_raw:
                row.update({f"cv_{k}": f"{100 * cv_raw[m][k].mean():.2f} ± {100 * cv_raw[m][k].std(ddof=1):.2f}"
                            for k in ["accuracy", "f1_macro"]})
            rows.append(row)
        ab = pd.DataFrame(rows); ab.to_csv(os.path.join(a.results, "table_ablation.csv"), index=False)
        md += ["## Ablation", ab.to_markdown(index=False), ""]

    # ---- significance
    sig = []
    ours_pred = os.path.join(a.results, a.ours, "test_predictions.csv")
    dev = df[~df.is_test]; k_folds = dev[[c for c in df.columns if c.startswith("cv_r")][0]].nunique()
    n_test_cv = len(dev) / k_folds; n_train_cv = len(dev) - n_test_cv
    for m in models:
        if m == a.ours:
            continue
        row = {"comparison": f"{a.ours} vs {m}"}
        other = os.path.join(a.results, m, "test_predictions.csv")
        if os.path.exists(ours_pred) and os.path.exists(other):
            p1, p2 = pd.read_csv(ours_pred), pd.read_csv(other)
            assert (p1.path.values == p2.path.values).all(), "test sets differ"
            c1, c2 = p1.pred == p1.label, p2.pred == p2.label
            tab = [[int((c1 & c2).sum()), int((c1 & ~c2).sum())], [int((~c1 & c2).sum()), int((~c1 & ~c2).sum())]]
            row.update(mcnemar_b=tab[0][1], mcnemar_c=tab[1][0], mcnemar_p=mcnemar(tab, exact=True).pvalue)
        if a.ours in cv_raw and m in cv_raw:
            x = cv_raw[a.ours].set_index(["repeat", "fold"]).accuracy
            y = cv_raw[m].set_index(["repeat", "fold"]).accuracy
            idx = x.index.intersection(y.index)
            tval, pval = corrected_t(x[idx], y[idx], n_train_cv, n_test_cv)
            row.update(cv_mean_diff_pct=100 * float((x[idx] - y[idx]).mean()), corrected_t=tval, corrected_t_p=pval)
        sig.append(row)
    if sig:
        s = pd.DataFrame(sig)
        for col in ["mcnemar_p", "corrected_t_p"]:
            if col in s and s[col].notna().any():
                ok = s[col].notna()
                s.loc[ok, col + "_holm"] = multipletests(s.loc[ok, col], method="holm")[1]
        s.to_csv(os.path.join(a.results, "table_significance.csv"), index=False)
        md += ["## Statistical tests (proposed vs. others)", s.round(4).to_markdown(index=False), ""]

    for f in ["tuning_summary.json"]:
        p = os.path.join(a.results, f)
        if os.path.exists(p):
            md += ["## Hyperparameter search", "```", open(p).read(), "```"]
    counts = os.path.splitext(a.split)[0].replace("_split", "_counts") + ".csv"
    if os.path.exists(counts):
        md = ["## Dataset and split counts", pd.read_csv(counts).to_markdown(index=False), ""] + md
    open(os.path.join(a.results, "tables.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
