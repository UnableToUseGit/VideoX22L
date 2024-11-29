cd /share/minghao/Projects/Video-XL/evluation
source activate /share/minghao/Envs/videoxl

EXPNAME=exp1
SETTINGNAME=wo_reload

LOGDIR=./logs/${EXPNAME}/${SETTINGNAME}
RESULTDIR=./results/${EXPNAME}/${SETTINGNAME}
mkdir -p ${LOGDIR}
mkdir -p ${RESULTDIR}

CUDA_VISIBLE_DEVICES=4 python -u mlvu.py \
    --task topic_reasoning \
    --log_level INFO \
    --attn_implementation sdpa \
    --save_dir $RESULTDIR 2>&1 | tee $LOGDIR/topic_reasoning.log &

sleep 120

CUDA_VISIBLE_DEVICES=5 python -u mlvu.py \
    --task needle \
    --log_level INFO \
    --attn_implementation sdpa \
    --save_dir $RESULTDIR 2>&1 | tee $LOGDIR/needle.log &

sleep 120

CUDA_VISIBLE_DEVICES=6 python -u mlvu.py \
    --task plotQA \
    --log_level INFO \
    --attn_implementation sdpa \
    --save_dir $RESULTDIR 2>&1 | tee $LOGDIR/plotQA.log &

sleep 120

CUDA_VISIBLE_DEVICES=7 python -u mlvu.py \
    --task order \
    --log_level INFO \
    --attn_implementation sdpa \
    --save_dir $RESULTDIR 2>&1 | tee $LOGDIR/order.log &

# sleep 90

# CUDA_VISIBLE_DEVICES=4 python -u mlvu.py \
#     --task count \
#     --log_level INFO \
#     --attn_implementation sdpa \
#     --save_dir $RESULTDIR 2>&1 | tee $LOGDIR/count.log &

sleep 120

CUDA_VISIBLE_DEVICES=3 python -u mlvu.py \
    --task ego \
    --log_level INFO \
    --attn_implementation sdpa \
    --save_dir $RESULTDIR 2>&1 | tee $LOGDIR/ego.log &

# sleep 90

# CUDA_VISIBLE_DEVICES=6 python -u mlvu.py \
#     --task anomaly_reco \
#     --log_level INFO \
#     --attn_implementation sdpa \
#     --save_dir $RESULTDIR 2>&1 | tee $LOGDIR/anomaly_reco.log &