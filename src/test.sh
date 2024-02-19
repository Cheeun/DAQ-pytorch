DATA_DIR=/mnt/disk1/cheeun914/datasets/
scale=4
# sh test.sh edsr_baseline 2 2 4 (edsr_baseline w2a2qq4)
# sh test.sh edsr_baseline 3 3 4 (edsr_baseline w3a3qq4)
# sh test.sh edsr_baseline 4 4 4 (edsr_baseline w4a4qq4)
# sh test.sh edsr_full 2 2 8 (edsr_full w2a2qq8)

edsr_full(){
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --test_only \
    --model EDSR --scale $scale \
    --n_feats 256 --n_resblocks 32 --res_scale 0.1 \
    --data_test Set5+Set14+B100+Urban100 --data_range 1-800/801-810 --dir_data $DATA_DIR \
    --pre_train ../pretrained_model/edsr_full-x$scale-daq-w2a2qq8.pt \
    --n_threads 6 --n_GPUs 1 \
    --save edsr_full-x$scale-daq-w2a2qq8-test \
    --quantize_a 2 --quantize_w 2 \
    --quantize_quantization 8 \
    # 
}

edsr_baseline(){
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --test_only \
    --model EDSR --scale $scale \
    --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
    --data_test Set5+Set14+B100+Urban100 --data_range 1-800/801-810 --dir_data /mnt/disk1/cheeun914/datasets/ \
    --pre_train ../pretrained_model/edsr_baseline-x$scale-daq-w2a2qq4.pt \
    --n_threads 6 --n_GPUs 1 \
    --save edsr_baseline-x$scale-daq-w2a2qq4-test \
    --quantize_a 2 --quantize_w 2 \
    --quantize_quantization 4 \
    # 
}

"$@"
