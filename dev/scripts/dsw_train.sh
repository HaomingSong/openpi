debug=true
if [[ $debug == true ]]; then
    wandb_enable=""
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    # export CUDA_VISIBLE_DEVICES=0,1
    export CUDA_VISIBLE_DEVICES=0
    # export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

CONFIG_NAME=(
    # pi0_fast_agibot_368_2
    # pi0_fast_agibot_368_subtask
    # pi0_fast_agibot_1084_2
    # pi0_fast_agibot_1084_subtask
    # pi0_fast_agibot_2246_2
    # pi0_fractal_fft
    # pi0_bridge_fft
    # pi0_bridge_lora
    # pi0_fractal_lora
    # pi0_fast_bridge_fft_pt_tokenizer
    # pi0_fast_fractal_fft_pt_tokenizer
    # pi0_fast_bridge_lora_pt_tokenizer
    # pi0_fast_fractal_lora_pt_tokenizer

    # pi0_kitchen_pot_fft
    # pi0_kitchen_banana_fft
    # pi0_fast_bridge_pad_fft_pt_tokenizer
    # pi0_fast_kitchen_banana_fft_pt_tokenizer
    # pi0_fast_kitchen_pot_fft_pt_tokenizer
    # pi0_fast_kitchen_pot_lora_pt_tokenizer
    # pi0_fast_kitchen_banana_lora_pt_tokenizer

    # pi0_kitchen_banana_raw_fft
    # pi0_kitchen_pot_raw_fft

    # pi0_plush_toy_fft
    # pi0_plush_toy_raw_fft
    # pi0_three_cube_blue_fft
    # pi0_three_cube_blue_raw_fft
    # pi0_three_cube_red_fft
    # pi0_three_cube_red_raw_fft
    # pi0_three_cube_green_fft
    # pi0_three_cube_green_raw_fft
    pi0_fast_agibot_2787_subtask
)

export HOME=/cpfs01/shared/optimal/songhaoming
export OPENPI_DATA_HOME=/cpfs01/shared/optimal/vla_next/openpi_hm/data/OPENPI_DATA_HOME
# export LEROBOT_HOME=/cpfs01/shared/optimal/vla_next/openpi_hm/data/LEROBOT_HOME
export LEROBOT_HOME=/oss/vla_next/DATA/AgiBotWorld-SFT-LeRobot
# export LEROBOT_HOME=/oss/vla_next/DATA/AgiBotWorld-Beta-Franka-LeRobot

wandb_enable=${wandb_enable:+"--wandb_enabled"}


for config in ${CONFIG_NAME[@]}; do
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $config \
        --exp-name=$config \
        --overwrite \
        --entity="academic_cockroach" \
        ${wandb_enable} \
        --batch_size=8
done