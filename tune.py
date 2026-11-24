"""Step 2 - hyperparameter selection for the proposed model (replaces HLOA).

Random search over the graph/FiLM settings, scored on an inner validation split of the DEVELOPMENT data
only. The held-out test set is never read. The search space, number of trials and the chosen values are
saved so they can be reported in the paper.

Example:
  python tune.py --split splits/bell_split.csv --out results/bell --trials 12
"""
import argparse, os, json, random
import pandas as pd
import torch

from common import load_or_compute_glcm, compute_metrics, save_json
from train_eval import inner_split, train_one

SPACE = {
    "d": [128, 256],
    "heads": [2, 4, 8],
    "layers": [1, 2, 3],
    "k": [4, 8, 12],
    "dropout": [0.2, 0.3, 0.5],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    a = ap.parse_args()
    a.model = "proposed"
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.split); n_classes = df.label.nunique()
    dev = df[~df.is_test].reset_index(drop=True)
    glcm = load_or_compute_glcm(df, os.path.splitext(a.split)[0] + "_glcm.npz")
    tr_all, sel = inner_split(dev, a.seed + 1000)          # selection set = part of development data
    tr, va = inner_split(tr_all, a.seed + 2000)              # early-stopping set

    rng = random.Random(a.seed); seen, rows = set(), []
    while len(rows) < a.trials:
        hp = {k: rng.choice(v) for k, v in SPACE.items()}
        if hp["d"] % hp["heads"] or tuple(hp.values()) in seen:
            continue
        seen.add(tuple(hp.values())); a.hp = hp
        print(f"trial {len(rows) + 1}/{a.trials}: {hp}")
        _, y, p, tt, _ = train_one(tr, va, sel, a, n_classes, glcm, a.seed, device)
        m, _ = compute_metrics(y, p, n_classes)
        rows.append({**hp, "selection_bal_acc": m["balanced_accuracy"], "selection_f1": m["f1_macro"],
                     "train_time_s": tt})
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "tuning_trials.csv"), index=False)
    res = pd.DataFrame(rows).sort_values(["selection_bal_acc", "selection_f1"], ascending=False)
    best = {k: (int(v) if k != "dropout" else float(v)) for k, v in res.iloc[0][list(SPACE)].items()}
    save_json(best, os.path.join(a.out, "best_params.json"))
    save_json(dict(space=SPACE, trials=a.trials, epochs=a.epochs, best=best,
                   n_train=len(tr), n_earlystop=len(va), n_selection=len(sel)),
              os.path.join(a.out, "tuning_summary.json"))
    print("best:", best)


if __name__ == "__main__":
    main()
