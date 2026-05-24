#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: Field-aware NS tokenizer ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type fieldaware \
    --user_ns_tokens 11 \
    --user_dense_tokens 1 \
    --item_ns_tokens 4 \
    --item_dense_tokens 0 \
    --num_queries 4 \
    --d_model 128 \
    --seq_semilocal_causal_mask \
    --seq_semilocal_window 128 \
    --ns_groups_json "" \
    --ns_token_time_feats abs \
    --temporal_encoding_mode enhanced \
    --shared_temporal_embeddings \
    --no_calendar_time \
    --no_time_span_buckets \
    --use_temporal_bias \
    --no_time_gap \
    --emb_skip_threshold 1000000 \
    --hash_bucket_size 100000 \
    --din_mode ref \
    --din_hidden_mult 4 \
    --din_dropout 0.01 \
    --din_target_source item_all \
    --num_workers 8 \
    --num_cross_layers 2 \
    --cross_low_rank 64 \
    --use_se_net \
    --use_ns_self_attn \
    --use_ns_output_fusion \
    --precision bf16 \
    --lr_schedule cosine \
    --warmup_steps 500 \
    --use_ema \
    --ema_decay 0.999 \
    --label_smoothing 0.01 \
    --weight_decay 0.02 \
    --loss_type focal \
    --focal_alpha 0.2 \
    --focal_gamma 3.0 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
