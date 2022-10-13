#!/bin/bash
export OPENAI_LOGDIR=logs/DSIFN

CUDA_VISIBLE_DEVICES=$CUDA_ID python scripts/image_sample.py \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --diffusion_steps 1000 \
    --dropout 0.1 \
    --image_size 64 \
    --learn_sigma True \
    --noise_schedule cosine \
    --num_channels 192 \
    --num_head_channels 64 \
    --num_res_blocks 3 \
    --resblock_updown True \
    --use_new_attention_order True \
    --use_fp16 True \
    --use_scale_shift_norm True \
    --batch_size 16 \
    --num_samples 700 \
    --timestep_respacing 250 \
    --model_path models/DSIFN_ema_0.9999_840000.pt 
