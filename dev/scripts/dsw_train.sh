export WANDB_MODE=disabled
CONFIG_NAME=(
    pi0_fast_bridge_low_mem_finetune_hand
    # pi0_fast_bridge_low_mem_finetune
)
# disable jax memory preallocation
# export XLA_PYTHON_CLIENT_PREALLOCATE=false


for config in ${CONFIG_NAME[@]}; do
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $config \
        --exp-name=$config \
        --overwrite
done