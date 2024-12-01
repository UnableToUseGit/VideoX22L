cd /share/minghao/Projects/Video-XL/evluation
source activate /share/minghao/Envs/videoxl

EXPNAME=exp2
SETTINGNAME=w_k3_nopos

LOGDIR=./logs/${EXPNAME}/${SETTINGNAME}
RESULTDIR=./results/${EXPNAME}/${SETTINGNAME}
mkdir -p ${LOGDIR}
mkdir -p ${RESULTDIR}

# CUDA_VISIBLE_DEVICES=0 python -u mlvu.py \
#     --task topic_reasoning \
#     --log_level INFO \
#     --reload_enable \
#     --reload_top_k 3 \
#     --attn_implementation sdpa \
#     --save_dir $RESULTDIR 2>&1 | tee $LOGDIR/topic_reasoning_reload.log &

# sleep 120

CUDA_VISIBLE_DEVICES=6 python -u mlvu.py \
    --task needle \
    --log_level INFO \
    --reload_enable \
    --reload_top_k 3 \
    --attn_implementation sdpa \
    --save_dir $RESULTDIR 2>&1 | tee $LOGDIR/needle_reload.log &

sleep 120

CUDA_VISIBLE_DEVICES=7 python -u mlvu.py \
    --task plotQA \
    --log_level INFO \
    --reload_enable \
    --reload_top_k 3 \
    --attn_implementation sdpa \
    --save_dir $RESULTDIR 2>&1 | tee $LOGDIR/plotQA_reload.log &

# sleep 120

# CUDA_VISIBLE_DEVICES=3 python -u mlvu.py \
#     --task order \
#     --log_level INFO \
#     --reload_enable \
#     --reload_top_k 3 \
#     --attn_implementation sdpa \
#     --save_dir $RESULTDIR 2>&1 | tee $LOGDIR/order_reload.log &

