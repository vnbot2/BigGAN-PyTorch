#!/bin/bash
python dc_gan.py \
--data_root downloads/moto_mask \
--experiment_name generative_motobike \
--num_epochs 5000 --shuffle --num_workers 16 --batch_size 128 \
--num_D_steps 1 --G_lr 2e-4 --D_lr 8e-4 --D_B2 0.999 --G_B2 0.999 \
--G_ch 32 --D_ch 64 \
--G_attn 32 --D_attn 32 \
--G_nl inplace_relu --D_nl inplace_relu \
--G_shared \
--hier --dim_z 100 --shared_dim 128 \
--SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
--G_ortho 0.0 \
--G_init ortho --D_init ortho \
--G_eval_mode \
--ema --use_ema --ema_start 30000 \
--save_every 1000 --sample_every 500 --log_interval 1 --num_fixed_samples 25 \
 --num_save_copies 2 --seed 1234 \
 --load_in_mem \
 --mask_out