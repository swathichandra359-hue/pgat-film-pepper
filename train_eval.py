"""Step 3 - train and evaluate ONE model on ONE dataset split.

--mode cv    : repeated stratified group k-fold CV on the development data (test set never touched)
--mode test  : train on all development data, evaluate ONCE on the held-out test set
--mode both  : both of the above

Within every training run, 15% of the training groups form an inner validation set used only for
early stopping. All models (proposed, ablations, baselines) use the same split file, epochs,
optimiser and augmentation, so their results are directly comparable.

Example:
  python train_eval.py --split splits/bell_split.csv --model proposed --params results/bell/best_params.json \
                       --out results/bell --mode both
"""
import argparse, os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from common import (seed_everything, build_transforms, load_or_compute_glcm, LeafDataset,
                    compute_metrics, save_json)
from models import build_model, uses_glcm


def inner_split(df, seed, val_frac=0.15):
    sg = StratifiedGroupKFold(n_splits=max(2, int(round(1 / val_frac))), shuffle=True, random_state=seed)
    tr, va = next(sg.split(df, df.label, df.group))
    return df.iloc[tr], df.iloc[va]


def make_loaders(tr_df, va_df, ev_df, args, glcm):
    train_tf, eval_tf = build_transforms(args.img_size)
    mu = sd = None
    if glcm is not None:
        G = np.stack([glcm[p] for p in tr_df.path]); mu, sd = G.mean(0), G.std(0) + 1e-6  # train-only stats
    mk = lambda d, tf, sh: DataLoader(LeafDataset(d, tf, glcm, mu, sd), batch_size=args.batch, shuffle=sh,
                                      num_workers=args.workers, pin_memory=True, drop_last=False)
    return mk(tr_df, train_tf, True), mk(va_df, eval_tf, False), mk(ev_df, eval_tf, False)


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); probs, ys = [], []
    for x, g, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True), g.to(device))
        probs.append(torch.softmax(out.float(), 1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def train_one(tr_df, va_df, ev_df, args, n_classes, glcm, seed, device):
    seed_everything(seed)
    tl, vl, el = make_loaders(tr_df, va_df, ev_df, args, glcm)
    model = build_model(args.model, n_classes, pretrained=not args.no_pretrained, img_size=args.img_size,
                        **args.hp).to(device)
    w = None
    if args.class_weight:
        cnt = np.bincount(tr_df.label, minlength=n_classes).astype(np.float32)
        w = torch.tensor(cnt.sum() / (n_classes * np.maximum(cnt, 1)), device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best, best_state, bad, hist = -1, None, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for x, g, y in tl:
            x, g, y = x.to(device, non_blocking=True), g.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x, g), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y)
        sched.step()
        yv, pv = predict(model, vl, device)
        vm, _ = compute_metrics(yv, pv, n_classes)
        score = vm["balanced_accuracy"]
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(tl.dataset), val_bal_acc=score))
        print(f"    ep {ep + 1:3d} loss {tot / len(tl.dataset):.4f} val bal-acc {score:.4f}")
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    y, p = predict(model, el, device)
    return model, y, p, train_time, hist


@torch.no_grad()
def efficiency(model, args, device, n_classes, sample_paths=()):
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    res = dict(params_M=params / 1e6)
    for bs in (1, args.batch):
        x = torch.randn(bs, 3, args.img_size, args.img_size, device=device)
        g = torch.randn(bs, 14, device=device)
        for _ in range(10):
            model(x, g)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter(); reps = 30
        for _ in range(reps):
            model(x, g)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        res[f"latency_ms_per_image_bs{bs}"] = 1000 * dt / bs
    if len(sample_paths):
        from common import glcm_features
        t0 = time.perf_counter()
        for p in sample_paths:
            glcm_features(p)
        res["glcm_ms_per_image_cpu"] = 1000 * (time.perf_counter() - t0) / len(sample_paths)
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(1, 3, args.img_size, args.img_size, device=device); g = torch.randn(1, 14, device=device)
        res["GFLOPs"] = FlopCountAnalysis(model, (x, g)).total() / 1e9
    except Exception:
        res["GFLOPs"] = None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True, help="proposed | abl_* | any timm model name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["cv", "test", "both"], default="both")
    ap.add_argument("--params", default=None, help="JSON with tuned hyperparameters (from tune.py)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true", help="smoke tests only")
    ap.add_argument("--max_folds", type=int, default=None, help="limit CV runs (smoke tests)")
    args = ap.parse_args()
    # tuned hyperparameters apply only to the proposed model and its ablations (graph/FiLM settings);
    # optimiser, learning rate, epochs and augmentation are identical for every model.
    is_ours = args.model == "proposed" or args.model.startswith("abl_")
    args.hp = json.load(open(args.params)) if (args.params and is_ours) else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.split)
    classes = df.sort_values("label").class_name.unique().tolist(); n_classes = len(classes)
    out = os.path.join(args.out, args.model); os.makedirs(out, exist_ok=True)
    glcm = load_or_compute_glcm(df, os.path.splitext(args.split)[0] + "_glcm.npz") if uses_glcm(args.model) else None
    dev = df[~df.is_test].reset_index(drop=True); test = df[df.is_test].reset_index(drop=True)

    if args.mode in ("cv", "both"):
        rows = []; cv_cols = sorted(c for c in df.columns if c.startswith("cv_r"))
        runs = [(c, f) for c in cv_cols for f in sorted(dev[c].unique())][: args.max_folds]
        for i, (col, f) in enumerate(runs):
            print(f"[{args.model}] CV {col} fold {f} ({i + 1}/{len(runs)})")
            tr_all, ev = dev[dev[col] != f], dev[dev[col] == f]
            tr, va = inner_split(tr_all, args.seed + i)
            _, y, p, tt, _ = train_one(tr, va, ev, args, n_classes, glcm, args.seed + i, device)
            m, _ = compute_metrics(y, p, n_classes)
            rows.append(dict(repeat=col, fold=int(f), train_time_s=tt, **m))
            pd.DataFrame(rows).to_csv(os.path.join(out, "cv_metrics.csv"), index=False)
        cv = pd.DataFrame(rows)
        summary = cv.drop(columns=["repeat", "fold"]).agg(["mean", "std"])
        summary.to_csv(os.path.join(out, "cv_summary.csv"))
        print(summary.T)

    if args.mode in ("test", "both"):
        print(f"[{args.model}] final training on development data, single evaluation on held-out test")
        tr, va = inner_split(dev, args.seed)
        model, y, p, tt, hist = train_one(tr, va, test, args, n_classes, glcm, args.seed, device)
        m, cm = compute_metrics(y, p, n_classes)
        m["train_time_min"] = tt / 60
        m.update(efficiency(model, args, device, n_classes, test.path.tolist()[:50] if uses_glcm(args.model) else ()))
        save_json(m, os.path.join(out, "test_metrics.json"))
        pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]) \
            .to_csv(os.path.join(out, "test_confusion_matrix.csv"))
        pred = test[["path", "label", "class_name", "group"]].copy()
        pred["pred"] = p.argmax(1)
        for k, c in enumerate(classes):
            pred[f"prob_{c}"] = p[:, k]
        pred.to_csv(os.path.join(out, "test_predictions.csv"), index=False)
        pd.DataFrame(hist).to_csv(os.path.join(out, "training_history.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(out, "final_model.pt"))
        save_json(dict(model=args.model, hp=args.hp, lr=args.lr, epochs=args.epochs, batch=args.batch,
                       img_size=args.img_size, seed=args.seed, n_train=len(tr), n_val=len(va), n_test=len(test),
                       device=str(device), torch=torch.__version__,
                       gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None),
                  os.path.join(out, "run_config.json"))
        print(json.dumps(m, indent=2)); print(cm)


if __name__ == "__main__":
    main()
