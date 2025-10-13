#!/bin/bash
cd /mnt/data/group/wangnan/code/VLMS/AndesVL-V1/infer/vllm/andesvl_vllm_v090
cp chat_utils.py /usr/local/lib/python3.10/site-packages/vllm/entrypoints/chat_utils.py
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
# 获取本机IP地址（适用于Linux）
ip_address=$(ip addr show | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | cut -d'/' -f1 | head -n 1)

# 如果需要兼容macOS，可以使用以下命令：
# ip_address=$(ifconfig | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | head -n 1)

# 检查是否成功获取IP地址
if [ -z "$ip_address" ]; then
    echo "无法获取本机IP地址"
    exit 1
fi

# 打印获取到的IP地址
echo "本机IP地址: $ip_address"

python /mnt/data/group/wangnan/code/VLMS/AndesVL-V1/infer/vllm/andesvl_vllm_v090/test_andesvl_internvit_server.py \
    --trust-remote-code \
    --model /mnt/data/group/wangnan/code/VLMS/AndesVL-V1/infer/hf/full/internvit-qwen3-0509 \
    --served_model_name AndesVL-4B \
    --limit-mm-per-prompt image=10 \
    --tensor-parallel-size 4 \
    --host $ip_address
