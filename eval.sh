

# Source
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python3.7  tools/eval.py \
# -c configs/rtdetr/rtdetr_r34vd_6x_coco.yml \
# -o weights=output/rtdetr_r34vd_6x_coco_cityscapes2foggycityscapes/best_model.pdparams


# target
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7  tools/eval.py \
-c configs/domain_adaption/da_rtdetr_r34_backbone_encoder_instance_dn_cmt_city2foggycity.yml \
-o weights=output/cityscapes2foggycity/best_model.pdparams
