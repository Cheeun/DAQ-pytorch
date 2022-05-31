CUDA_VISIBLE_DEVICES=6 python main.py \
--model EDSR --scale 4 \
--data_test Set5+Set14+B100+Urban100 --data_range 1-800/801-810 --dir_data /mnt/disk1/cheeun914/datasets/ \
--pre_train ../pretrained_model/pretrained_model_name.pt \
--n_feats 256 --n_resblocks 32 --res_scale 0.1 \
--n_threads 6 --n_GPUs 1 \
--save edsr_daq_test \
--batch_size 8 --patch_size 192 \
--quantize_a 4 --quantize_w 4 \
--test_only \