configs=(
    # pi0_fast_libero
    # pi0_fast_libero_low_mem_finetune
    # pi0_libero
    # pi0_libero_low_mem_finetune
    pi0_fast_maketea_pad_lora_pt_tokenizer
)

export HOME=/cpfs01/shared/optimal/songhaoming
export OPENPI_DATA_HOME=/cpfs01/shared/optimal/vla_next/openpi_hm/data/OPENPI_DATA_HOME
export LEROBOT_HOME=/cpfs01/shared/optimal/vla_next/openpi_hm/data/LEROBOT_HOME
export CUDA_VISIBLE_DEVICES=0
for config in ${configs[@]}; do
    uv run scripts/compute_norm_stats.py \
        --config-name $config
done