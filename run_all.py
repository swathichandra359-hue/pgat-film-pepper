"""Run the whole protocol for one dataset (Windows, Linux or macOS).

  python run_all.py --data data/bell_pepper --name bell
  python run_all.py --data data/black_pepper --name black --class_weight

Safe to stop and restart: finished steps and finished models are skipped.
Use --quick for a fast check that everything works (tiny settings, a few minutes).
"""
import argparse, os, subprocess, sys

OURS = ["proposed", "abl_backbone_only", "abl_no_film", "abl_no_graph", "abl_grid_only", "abl_knn_only"]
BASELINES = ["resnet50", "vgg16_bn", "densenet121", "inception_v3", "efficientnet_b0",
             "mobilenetv2_100", "mobilenetv3_large_100",
             "vit_small_patch16_224", "deit_small_patch16_224", "swin_tiny_patch4_window7_224"]


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        sys.exit(f"Step failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 epoch, 1 CV fold, 2 tuning trials")
    ap.add_argument("--workers", default="4", help="data-loading workers (use 0 if Windows gives errors)")
    ap.add_argument("--skip_baselines", action="store_true")
    ap.add_argument("--models", nargs="*", default=None, help="run only these models (default: all)")
    ap.add_argument("--no_pretrained", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    split = os.path.join("splits", f"{a.name}_split.csv")
    res = os.path.join("results", a.name + ("_quick" if a.quick else ""))
    extra = ["--workers", a.workers] + (["--class_weight"] if a.class_weight else []) + \
        (["--no_pretrained", "--img_size", "64"] if a.no_pretrained else [])
    quick = ["--epochs", "1", "--max_folds", "1"] if a.quick else []

    if not os.path.exists(split):
        run(["prepare_split.py", "--data_dir", a.data, "--name", a.name, "--out", "splits"])
    else:
        print(f"split exists, reusing {split}")

    models = a.models or (OURS + ([] if a.skip_baselines else BASELINES))
    need_tuning = any(m in OURS for m in models)
    if need_tuning and not os.path.exists(os.path.join(res, "best_params.json")):
        run(["tune.py", "--split", split, "--out", res] + extra +
            (["--trials", "2", "--epochs", "1"] if a.quick else []))

    for m in models:
        if os.path.exists(os.path.join(res, m, "test_metrics.json")):
            print(f"{m}: already finished, skipping")
            continue
        cmd = ["train_eval.py", "--split", split, "--model", m, "--out", res, "--mode", "both"] + extra + quick
        if m in OURS:
            cmd += ["--params", os.path.join(res, "best_params.json")]
        run(cmd)

    run(["make_tables.py", "--results", res, "--split", split])
    if os.path.exists(os.path.join(res, "proposed", "final_model.pt")):
        run(["gradcam.py", "--split", split, "--results", res, "--n", "8"])
    print(f"\nAll done. Open {os.path.join(res, 'tables.md')}")


if __name__ == "__main__":
    main()
