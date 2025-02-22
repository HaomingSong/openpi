#!/bin/bash

# Load opencv libs
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx

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

cd /cpfs01/shared/optimal/vla_next/openpi_hm


