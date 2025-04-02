debug=true
if [[ $debug == true ]]; then
    export WANDB_MODE=disabled
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    # export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

CONFIG_NAME=(
    # pi0_fast_bridge_fft_pt_tokenizer
    # pi0_fast_bridge_lora_pt_tokenizer
    # pi0_fast_bridge_pad_lora_pt_tokenizer
    # pi0_fast_bridge_pad_fft_pt_tokenizer
    # pi0_fast_fractal_fft_pt_tokenizer
    # pi0_fast_libero
    pi0_fast_libero_low_mem_finetune
)

for config in ${CONFIG_NAME[@]}; do
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $config \
        --exp-name=$config \
        --overwrite
done