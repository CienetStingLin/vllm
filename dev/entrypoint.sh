#!/bin/bash
set -e

source /root/miniconda3/bin/activate vllm

echo "Fetching HuggingFace token from Secret Manager..."
export TOKEN=$(gcloud secrets versions access latest --secret="sting-hg-token")

echo "Logging in to HuggingFace..."
git config --global credential.helper store
huggingface-cli login --token $TOKEN

echo "Starting vLLM server..."
cd /root/vllm
nohup vllm serve "meta-llama/Llama-3.1-8B" \
  --download_dir /tmp \
  --swap-space 16 \
  --disable-log-requests \
  --tensor_parallel_size=4 \
  --max-model-len=2048 &> serve.log &

echo "Running benchmark..."
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model "meta-llama/Llama-3.1-8B" \
  --dataset-name random \
  --random-input-len 1820 \
  --random-output-len 128 \
  --random-prefix-len 0
