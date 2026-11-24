"""Step 1 - build a leakage-free split for ONE single-species dataset.

* Lists images in <data_dir>/<class_name>/*.{jpg,png,...}
* Groups near-duplicate images (same leaf photographed / augmented copies) with a perceptual hash,
  so that a group never appears in both training and test data (group-wise splitting).
* Holds out ~15% of groups as a test set that is used ONCE, for the final confusion matrix.
* Assigns the remaining development data to repeated stratified group k-fold CV folds.

Output: split CSV + dataset/split count tables for the manuscript.

Example:
  python prepare_split.py --data_dir data/bell_pepper --name bell --out splits
"""
import argparse, os, re
import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(data_dir, classes=None):
    classes = classes or sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    rows = []
    for k, c in enumerate(classes):
        for root, _, files in os.walk(os.path.join(data_dir, c)):
            for f in sorted(files):
                if f.lower().endswith(EXT):
                    rows.append((os.path.abspath(os.path.join(root, f)), k, c))
    return pd.DataFrame(rows, columns=["path", "label", "class_name"]), classes


def near_duplicate_groups(paths, thresh, ids=None):
    """Union-find over pHash Hamming distance <= thresh (plus shared file-name IDs, if given)."""
    hashes = []
    for i, p in enumerate(paths):
        hashes.append(np.asarray(imagehash.phash(Image.open(p).convert("RGB")).hash, dtype=bool).ravel())
        if (i + 1) % 500 == 0:
            print(f"  hashed {i + 1}/{len(paths)}")
    H = np.stack(hashes)
    n = len(H); parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    pairs = 0
    for i in range(n):
        d = (H[i + 1:] != H[i]).sum(1)
        for j in np.nonzero(d <= thresh)[0] + i + 1:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri; pairs += 1
    if ids is not None:
        first = {}
        for i, key in enumerate(ids):
            if key in first:
                ri, rj = find(first[key]), find(i)
                if ri != rj:
                    parent[rj] = ri
            else:
                first[key] = i
    return np.array([find(i) for i in range(n)]), pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--name", required=True, help="dataset tag, e.g. bell or black")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--classes", nargs="*", default=None, help="class folder names (default: all, sorted)")
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hash_thresh", type=int, default=2, help="pHash Hamming distance for near-duplicates")
    ap.add_argument("--group_regex", default=None,
                    help="optional regex with one capture group extracting a plant/leaf ID from the file name; "
                         "used instead of (or merged with) the hash groups when the dataset provides IDs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df, classes = list_images(a.data_dir, a.classes)
    print(f"{len(df)} images, classes: {classes}")
    ids = None
    if a.group_regex:
        def get_id(p):
            m = re.search(a.group_regex, os.path.basename(p))
            return m.group(1) if m else p
        ids = df.path.map(get_id).tolist()
    groups, merged = near_duplicate_groups(df.path.tolist(), a.hash_thresh, ids)
    df["group"] = pd.factorize(groups)[0]
    print(f"{df.group.nunique()} groups after near-duplicate merging ({len(df) - df.group.nunique()} images "
          f"share a group with another image)")

    sizes = df.group.value_counts()
    print(f"group sizes: max {sizes.max()}, groups with >1 image: {(sizes > 1).sum()}")
    if sizes.max() > 0.05 * len(df):
        print(f"WARNING: the largest group holds {sizes.max()} images ({100 * sizes.max() / len(df):.1f}%). "
              f"Distinct leaves are probably being merged - rerun with a smaller --hash_thresh (e.g. 4 or 2).")

    # ---- held-out test set (group-wise, stratified)
    n_split = max(2, int(round(1 / a.test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_split, shuffle=True, random_state=a.seed)
    _, test_idx = next(sgkf.split(df, df.label, df.group))
    df["is_test"] = False
    df.loc[test_idx, "is_test"] = True

    # ---- repeated stratified group k-fold on the development data
    dev = df[~df.is_test]
    for r in range(a.repeats):
        col = f"cv_r{r}"; df[col] = -1
        sg = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed + r)
        for f, (_, vi) in enumerate(sg.split(dev, dev.label, dev.group)):
            df.loc[dev.index[vi], col] = f

    # ---- leakage checks
    assert not set(df[df.is_test].group) & set(df[~df.is_test].group), "group leakage between dev and test"
    for r in range(a.repeats):
        g = df[~df.is_test].groupby("group")[f"cv_r{r}"].nunique()
        assert (g == 1).all(), "group split across CV folds"

    csv = os.path.join(a.out, f"{a.name}_split.csv")
    df.to_csv(csv, index=False)

    # ---- tables for the manuscript (dataset counts and split counts)
    counts = df.groupby("class_name").agg(images=("path", "size"), groups=("group", "nunique"))
    counts["development"] = df[~df.is_test].groupby("class_name").size()
    counts["held_out_test"] = df[df.is_test].groupby("class_name").size()
    counts.loc["Total"] = counts.sum()
    counts.to_csv(os.path.join(a.out, f"{a.name}_counts.csv"))
    print(counts)
    print(f"saved {csv}")


if __name__ == "__main__":
    main()
