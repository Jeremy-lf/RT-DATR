## [RT-DATR:Real-time Unsupervised Domain Adaptive Detection Transformer with Adversarial Feature Learning](https://arxiv.org/abs/2504.09196)
![image](https://github.com/user-attachments/assets/f3008521-a10f-4089-b6d5-95a21c46ec55)
## Abstract
Despite domain-adaptive object detectors based on CNN and transformers have made significant progress in cross-domain detection tasks, it is regrettable that domain adaptation for real-time transformer-based detectors has not yet been explored. Directly applying existing domain adaptation algorithms has proven to be suboptimal. In this paper, we propose RT-DATR, a simple and efficient real-time domain adaptive detection transformer. Building on RT-DETR as our base detector, we first introduce a local objectlevel feature alignment module to significantly enhance the feature representation of domain invariance during object transfer. Additionally, we introduce a scene semantic feature alignment module designed to boost cross-domain detection performance by aligning scene semantic features. Finally, we introduced a domain query and decoupled it from the object query to further align the instance feature distribution within the decoder layer, reduce the domain gap, and maintain discriminative ability. Experimental results on various benchmarks demonstrate that our method outperforms current state-of-the-art approaches.


## Quick Start
```bash
## Enveriments
pip install -r requeriments.txt

## Train
sh run.sh or \

python export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_START_PORT=35100
python3.7 \
-m paddle.distributed.launch --gpus=0,1,2,3 --log_dir=log_base tools/train.py \
-c configs/domain_adaption/da_r50_rtdetr_backbone_encoder_instance_dn_cmt_city2foggycity.yml \
--eval \
--enable_ce True

## Eval
sh eval.sh or \

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7  tools/eval.py \
-c configs/domain_adaption/da_rtdetr_r34_backbone_encoder_instance_dn_cmt_city2foggycity.yml \
-o weights=output/cityscapes2foggycity/best_model.pdparams

```

## Experiments Results
We evaluated our approach on multiple scene datasets, including weather adaptation (Cityscapes to Foggy Cityscapes), scene adaptation (Cityscapes to BDD100K), artistic-to-real adaptation (Sim10K to Cityscapes) and cross-camera adaptation(KITTI to Cityscapes). 

![image](https://github.com/user-attachments/assets/26e1b8d5-d27e-4256-8e4d-b5718d9cd4be)

![image](https://github.com/user-attachments/assets/999a9fb8-b1a3-4c03-952f-13850fd3e7ea)

<div style="display: flex; gap: 20px;justify-content: center; margin-bottom: 20px;">
  <img src="https://github.com/user-attachments/assets/fd862dbf-aef8-44f1-99ba-8792a8f09ba3" alt="" width="45%">
  <img src="https://github.com/user-attachments/assets/e3eb5b94-4014-4e31-9b08-409b1463e446" alt="" width="45%">
</div>

## Cite
```
@article{lv2025rt,
  title={RT-DATR: Real-time Unsupervised Domain Adaptive Detection Transformer with Adversarial Feature Learning},
  author={Lv, Feng and Xia, Chunlong and Wang, Shuo and Cao, Huo},
  journal={arXiv preprint arXiv:2504.09196},
  year={2025}
}
```

