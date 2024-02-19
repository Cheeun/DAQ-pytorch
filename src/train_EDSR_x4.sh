DATA_DIR=/mnt/disk1/cheeun914/datasets/
scale=4

edsr_full(){
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --model EDSR --scale $scale \
    --lr 5e-5 --epochs 300 --decay 150  --loss 1*L1 \
    --data_test Set5+Set14+B100+Urban100 --data_range 1-800/801-810 --dir_data $DATA_DIR \
    --n_threads 6 --n_GPUs 1 \
    --save edsr_full-x$scale-daq-w$2a$1qq$3-train \
    --batch_size 4 --patch_size 192 \
    --finetune --pre_train ../pretrained_model/edsr_full_x4_reproduce.pt \
    --quantize_a $1 --quantize_w $2 --quantize_quantization $3 \
    --n_feats 256 --n_resblocks 32 --res_scale 0.1 \
}

