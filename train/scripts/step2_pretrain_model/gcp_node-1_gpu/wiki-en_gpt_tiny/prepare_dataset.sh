#!/bin/bash
megatron_deepspeed_dir="/home/ext_otomijuf_004_gmail_com/Megatron-DeepSpeed"
mkdir -p ${megatron_deepspeed_dir}"/dataset/sample"

jsonl_file="./train.jsonl"
output_prefix=${megatron_deepspeed_dir}"/dataset/sample/wiki"

# tokenizer="microsoft/phi-2"
tokenizer="${HOME}/ucllm_nedo_prod/train/scripts/step1_train_tokenizer/botchan.model"

python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${tokenizer} \
    --input ${jsonl_file} \
    --output-prefix ${output_prefix} \
    --dataset-impl mmap \
    --workers $(grep -c ^processor /proc/cpuinfo) \
    --append-eod
