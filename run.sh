# rtdetr-r34 cityscapes_to_foggycity
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_START_PORT=35100
nohup /root/paddlejob/workspace/env_run/lvfeng/pd2.3_cuda11_py3.7/bin/python3.7 \
-m paddle.distributed.launch --gpus=0,1,2,3 --log_dir=log_base tools/train.py \
-c configs/domain_adaption/da_r50_rtdetr_backbone_encoder_instance_dn_cmt_city2foggycity.yml \
--eval \
--enable_ce True \
> da_rtdetr_r50_cityscapes_to_foggycity_2e4_72e_backbozne_encoder_instance_dn_cmt_loss2.txt 2>&1 &
