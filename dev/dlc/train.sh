CONFIG_NAME=(
    # pi0_agibot_pour_water_fft
    # pi0_agibot_pass_water_fft
    # pi0_agibot_fold_shorts_fft
    # pi0_agibot_restock_fft
    # pi0_franka_banana_fft
    # pi0_franka_tea_fft
    # pi0_fast_agibot_pour_water_fft
    # pi0_fast_agibot_pass_water_fft
    # pi0_fast_agibot_2246_2
    # pi0_fast_agibot_1084_2
    # pi0_fast_agibot_1084_subtask
    # pi0_fast_agibot_368_subtask
    # pi0_fast_agibot_368_2
    # pi0_fast_agibot_restock_fft
    # pi0_fast_franka_banana_fft
    # pi0_fast_franka_tea_fft
    pi0_fast_agibot_2787_subtask
)

export HOME=/cpfs01/shared/optimal/songhaoming
export OPENPI_DATA_HOME=/cpfs01/shared/optimal/vla_next/openpi_hm/data/OPENPI_DATA_HOME
export LEROBOT_HOME=/cpfs01/shared/optimal/vla_next/openpi_hm/data/LEROBOT_HOME
# export LEROBOT_HOME=/oss/vla_next/DATA/AgiBotWorld-SFT-LeRobot
# export LEROBOT_HOME=/oss/vla_next/DATA/AgiBotWorld-Beta-Franka-LeRobot
wandb_enable=${wandb_enable:+"--wandb_enabled"}


for config in ${CONFIG_NAME[@]}; do
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $config \
        --exp-name=$config \
        --entity="academic_cockroach" \
        --overwrite \
        ${wandb_enable} \
        --batch_size=8
done