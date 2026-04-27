#!/bin/bash
# Training launch command used on 2026-04-23
# Server: ubuntu@63.179.56.99  (8x A100-SXM4-40GB)
# Result: epoch 444 val_loss=0.318, checkpoint saved to
#         pysot/snapshot/all_datasets_ir_siamese/best_model.pth

set -e
cd /home/ubuntu/data/siamrpn_training
source /opt/conda/etc/profile.d/conda.sh
conda activate siamrpn

torchrun --nproc_per_node=8 train_siamrpn_aws.py \
    --cfg  configs/config_ir_siamese.yaml \
    --pretrained pretrained/sot_resnet50.pth \
    --no_early_stop \
    2>&1 | tee train_8gpu_$(date +%Y%m%d_%H%M%S).log

# To attach to the running tmux session:
#   tmux attach -t siamrpn_train
