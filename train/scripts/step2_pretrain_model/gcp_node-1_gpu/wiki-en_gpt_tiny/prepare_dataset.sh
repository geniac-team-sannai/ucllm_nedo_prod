#!/bin/bash

jsonl_file="./train.jsonl"
output_prefix=${megatron_deepspeed_dir}"/dataset/wiki"
megatron_deepspeed_dir="/home/ext_otomijuf_004_gmail_com/Megatron-DeepSpeed"
tokenizer="llamaste/Llama-2-7b-chat-hf"

python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
    --tokenizer-type HFTokenizer \
    --tokenizer-model ${tokenizer} \
    --input ${jsonl_file} \
    --output-prefix ${output_prefix} \
    --dataset-impl mmap \
    --workers $(grep -c ^processor /proc/cpuinfo) \
    --append-eod
