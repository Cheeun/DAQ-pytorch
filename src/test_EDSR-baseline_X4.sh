CUDA_VISIBLE_DEVICES=0 python main.py \
--model EDSR --scale 4 \
--data_test Set5+Set14+B100+Urban100 --data_range 1-800/801-810 --dir_data /mnt/disk1/cheeun914/datasets/ \
--pre_train ../pretrained_model/pretrained_model_name.pt \
--n_feats 64 --n_resblocks 16 --res_scale 1.0 \
--n_threads 6 --n_GPUs 1 \
--save edsr_baseline_x4_daq_w2a2qq4_test \
--batch_size 8 --patch_size 192 \
--quantize_a 2 --quantize_w 2 \
--quantize_quantization 4 \
--test_only \
# 