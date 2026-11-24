"""Shared utilities: reproducibility, dataset, transforms, GLCM texture features, metrics."""
import os, random, json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(img_size=224):
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    eval_tf = T.Compose([
        T.Resize(int(img_size * 256 / 224)), T.CenterCrop(img_size),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return train_tf, eval_tf


# ---------------------------------------------------------------- GLCM texture features
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS = 32
GLCM_DIM = (len(GLCM_PROPS) + 1) * len(GLCM_DISTANCES)  # +1 for entropy


def glcm_features(path, size=128):
    """14-D texture vector: 6 Haralick properties + entropy, averaged over 4 angles, for d = 1 and 2."""
    g = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    q = (np.asarray(g, dtype=np.float32) / 256.0 * GLCM_LEVELS).astype(np.uint8)
    P = graycomatrix(q, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, levels=GLCM_LEVELS,
                     symmetric=True, normed=True)
    feats = []
    for di in range(len(GLCM_DISTANCES)):
        for p in GLCM_PROPS:
            feats.append(graycoprops(P, p)[di].mean())
        Pd = P[:, :, di, :]
        ent = -(Pd * np.log2(Pd + 1e-12)).sum(axis=(0, 1)).mean()
        feats.append(ent)
    return np.asarray(feats, dtype=np.float32)


def load_or_compute_glcm(df, cache_path):
    """Compute GLCM features once for every image and cache them (keyed by path)."""
    cache = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        cache = dict(zip(z["paths"].tolist(), z["feats"]))
    missing = [p for p in df.path if p not in cache]
    for i, p in enumerate(missing):
        cache[p] = glcm_features(p)
        if (i + 1) % 250 == 0:
            print(f"  GLCM {i + 1}/{len(missing)}")
    if missing:
        np.savez(cache_path, paths=np.array(list(cache.keys()), dtype=object),
                 feats=np.stack(list(cache.values())))
    return cache


class LeafDataset(Dataset):
    def __init__(self, df, transform, glcm_cache=None, glcm_mean=None, glcm_std=None):
        self.paths = df.path.tolist(); self.labels = df.label.astype(int).tolist()
        self.tf = transform; self.glcm = glcm_cache
        self.mu = glcm_mean; self.sd = glcm_std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.tf(Image.open(self.paths[i]).convert("RGB"))
        if self.glcm is not None:
            g = (self.glcm[self.paths[i]] - self.mu) / self.sd
        else:
            g = np.zeros(GLCM_DIM, np.float32)
        return img, torch.from_numpy(g.astype(np.float32)), self.labels[i]


# ---------------------------------------------------------------- metrics
def compute_metrics(y_true, y_prob, n_classes):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(n_classes)),
                                                       average="macro", zero_division=0)
    spec = []
    for k in range(n_classes):
        tn = cm.sum() - cm[k, :].sum() - cm[:, k].sum() + cm[k, k]
        fp = cm[:, k].sum() - cm[k, k]
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    m = dict(accuracy=accuracy_score(y_true, y_pred), balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
             precision_macro=prec, sensitivity_macro=rec, specificity_macro=float(np.mean(spec)), f1_macro=f1,
             roc_auc=auc, kappa=cohen_kappa_score(y_true, y_pred), mcc=matthews_corrcoef(y_true, y_pred),
             n=int(len(y_true)))
    return {k: float(v) if k != "n" else v for k, v in m.items()}, cm


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
