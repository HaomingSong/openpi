configs=(
    # pi0_bridge_fft
    # pi0_kitchen_pot_fft
    # pi0_kitchen_banana_fft
    # pi0_fast_bridge_pad_fft_pt_tokenizer
    # pi0_fast_kitchen_banana_fft_pt_tokenizer
    # pi0_fast_kitchen_pot_fft_pt_tokenizer
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

    pi0_fast_agibot_368_2
)

export HOME=/cpfs01/shared/optimal/songhaoming
export OPENPI_DATA_HOME=/cpfs01/shared/optimal/vla_next/openpi_hm/data/OPENPI_DATA_HOME
# export LEROBOT_HOME=/cpfs01/shared/optimal/vla_next/openpi_hm/data/LEROBOT_HOME
# export LEROBOT_HOME=/oss/vla_next/DATA/AgiBotWorld-Beta-Franka-LeRobot
export LEROBOT_HOME=/oss/vla_next/DATA/AgiBotWorld-SFT-LeRobot

export CUDA_VISIBLE_DEVICES=0
for config in ${configs[@]}; do
    uv run scripts/compute_norm_stats.py \
        --config-name $config
done