CUDA_VISIBLE_DEVICES=7 python main.py \
--model EDSR --scale 4 \
--lr 1e-4 --epochs 300 --decay 150  --loss 1*L1 \
--data_test Set5+Set14+B100+Urban100 --data_range 1-800/801-810 --dir_data /mnt/disk1/cheeun914/datasets/ \
--n_threads 6 --n_GPUs 1 \
--save edsr_daq_train_2 \
--batch_size 8 --patch_size 192 \
--quantize_a 2 --quantize_w 2 --quantize_quantization 4 \
--n_feats 64 --n_resblocks 16 --res_scale 1.0 \
# --n_feats 256 --n_resblocks 32 --res_scale 0.1 \
#