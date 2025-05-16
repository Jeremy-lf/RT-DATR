# rtdetr-r34 cityscapes_to_foggycity
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_START_PORT=35100
python3.7 \
-m paddle.distributed.launch --gpus=0,1,2,3 --log_dir=log_base tools/train.py \
-c configs/domain_adaption/da_rtdetr_r34_backbone_encoder_instance_dn_cmt_city2foggycity.yml \
--eval \
--enable_ce True 