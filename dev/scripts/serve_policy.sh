# uv run scripts/serve_policy.py \
    # --port 8000 \
    # policy:checkpoint \
    # --policy.config=pi0_fast_libero \
    # --policy.dir=checkpoints/pi0_fast_libero/pi0_fast_libero/29999

export HOME=/cpfs01/shared/optimal/songhaoming
export OPENPI_DATA_HOME=/cpfs01/shared/optimal/vla_next/openpi_hm/data/OPENPI_DATA_HOME
export LEROBOT_HOME=/cpfs01/shared/optimal/vla_next/openpi_hm/data/LEROBOT_HOME

export http_proxy=https://songhaoming:cwXa1saSuK3S6zErKdNWVTZUfnxmHp173M0DGqRqUHFJugIRAN9KGuNAFtf6@aliyun-proxy.pjlab.org.cn:13128
export https_proxy=https://songhaoming:cwXa1saSuK3S6zErKdNWVTZUfnxmHp173M0DGqRqUHFJugIRAN9KGuNAFtf6@aliyun-proxy.pjlab.org.cn:13128
export HTTP_PROXY=https://songhaoming:cwXa1saSuK3S6zErKdNWVTZUfnxmHp173M0DGqRqUHFJugIRAN9KGuNAFtf6@aliyun-proxy.pjlab.org.cn:13128
export HTTPS_PROXY=https://songhaoming:cwXa1saSuK3S6zErKdNWVTZUfnxmHp173M0DGqRqUHFJugIRAN9KGuNAFtf6@aliyun-proxy.pjlab.org.cn:13128

# PATH
if [[ ":$PATH:" != *":/cpfs01/shared/optimal/songhaoming/.local/bin:"* ]]; then
    export PATH="/cpfs01/shared/optimal/songhaoming/.local/bin:$PATH"
fi

uv run scripts/serve_policy.py \
    --port 8000 \
    --env LIBERO