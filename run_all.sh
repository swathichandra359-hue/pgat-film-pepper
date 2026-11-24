#!/usr/bin/env bash
# Runs the full revised protocol for ONE dataset.
# Usage:  bash run_all.sh <dataset_dir> <name>
#   e.g.  bash run_all.sh data/bell_pepper bell
#         bash run_all.sh data/black_pepper black
set -euo pipefail
DATA=$1; NAME=$2
SPLITS=splits; RES=results/$NAME
EXTRA=${EXTRA:-}                      # e.g. EXTRA="--class_weight" for imbalanced data

python prepare_split.py --data_dir "$DATA" --name "$NAME" --out $SPLITS
python tune.py --split $SPLITS/${NAME}_split.csv --out $RES $EXTRA

# proposed model + ablations
for M in proposed abl_backbone_only abl_no_film abl_no_graph abl_grid_only abl_knn_only; do
  python train_eval.py --split $SPLITS/${NAME}_split.csv --model $M --params $RES/best_params.json \
                       --out $RES --mode both $EXTRA
done

# baselines: CNNs, lightweight CNNs, vision transformers - same split, same schedule
for M in resnet50 vgg16_bn densenet121 inception_v3 efficientnet_b0 \
         mobilenetv2_100 mobilenetv3_large_100 \
         vit_small_patch16_224 deit_small_patch16_224 swin_tiny_patch4_window7_224; do
  python train_eval.py --split $SPLITS/${NAME}_split.csv --model $M --out $RES --mode both $EXTRA
done

python make_tables.py --results $RES --split $SPLITS/${NAME}_split.csv
python gradcam.py --split $SPLITS/${NAME}_split.csv --results $RES --n 8
echo "Done. Tables: $RES/tables.md"
