# PGAT-FiLM: pepper leaf disease classification under a leakage-controlled protocol

Code for the paper *Leakage-Controlled Evaluation of Pepper Leaf Disease Classification: A Texture-Conditioned
Patch-Graph Attention Network Compared with CNN and Transformer Baselines on Bell and Black Pepper Images*.

The bell pepper and black pepper datasets are analysed separately.

## Installation

```bash
python -m venv venv && source venv/bin/activate         # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

ImageNet-pretrained weights are downloaded automatically by `timm`.
The experiments in the paper were run on an NVIDIA Tesla T4 (Kaggle Notebooks); `kaggle_pepper.ipynb` reproduces them there.

## Data

```
data/bell_pepper/Pepper__bell___Bacterial_spot/*.JPG     (PlantVillage, Kaggle arjuntejaswi/plant-village)
data/bell_pepper/Pepper__bell___healthy/*.JPG
data/black_pepper/<class folder>/*.jpg                    (Kaggle udi17live/black-pepper-leaf-blight-and-yellow-mottle-virus)
```

All images of each class are used.

## Running

```bash
python check_setup.py data/bell_pepper data/black_pepper
python run_all.py --data data/bell_pepper --name bell
python run_all.py --data data/black_pepper --name black --class_weight
```

Results are written to `results/<name>/`; `tables.md` summarises all tables.

## Pipeline

| Step | Script | Purpose |
|---|---|---|
| 1 | `prepare_split.py` | Groups near-duplicate images by perceptual hash (Hamming distance <= 2), holds out 15% of groups as a test set used once per model, and assigns the remaining groups to 5-fold x 3-repeat stratified group cross-validation. |
| 2 | `tune.py` | Random search (12 trials) over the graph settings of PGAT-FiLM using development data only. |
| 3 | `train_eval.py` | Trains one model: repeated cross-validation, then a final model evaluated on the held-out test set (metrics, confusion matrix, predictions, parameters, GFLOPs, training time, latency). |
| 4 | `make_tables.py` | Test, cross-validation, ablation, efficiency and significance tables (exact McNemar test; Nadeau-Bengio corrected resampled t-test; Holm correction) and confusion-matrix figures. |
| 5 | `gradcam.py` | Grad-CAM heat maps on test images. |

Models: `proposed` (PGAT-FiLM), ablations `abl_backbone_only`, `abl_no_film`, `abl_no_graph`, `abl_grid_only`,
`abl_knn_only`, and baselines `resnet50`, `vgg16_bn`, `densenet121`, `inception_v3`, `efficientnet_b0`,
`mobilenetv2_100`, `mobilenetv3_large_100`, `vit_small_patch16_224`, `deit_small_patch16_224`,
`swin_tiny_patch4_window7_224`.

## Licence

MIT
