configs=(
    # pi0_fast_libero
    # pi0_fast_libero_low_mem_finetune
    pi0_libero
    # pi0_libero_low_mem_finetune
)

for config in ${configs[@]}; do
    uv run scripts/compute_norm_stats.py \
        --config-name $config
done