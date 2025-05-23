# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['DA_RTDETR_Backbone_Encoder_Instance']
# Deformable DETR, DINO use the same architecture as DETR


@register
class DA_RTDETR_Backbone_Encoder_Instance(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process', 'post_process_semi']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck=None,
                 post_process='DETRPostProcess',
                 post_process_semi=None,
                 with_mask=False,
                 exclude_post_process=False):
        super(DA_RTDETR_Backbone_Encoder_Instance, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process
        self.post_process_semi = post_process_semi

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # neck
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None

        # transformer
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck": neck
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Neck
        if self.neck is not None:
            body_feats, domain_outs_enc, domain_labels_enc = self.neck(body_feats, self.inputs)

        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        out_transformer = \
        (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta, domain_outs_dec, domain_labels_dec, masks, outputs, num_denoising, out_instance_query) = \
            self.transformer(body_feats, pad_mask, self.inputs)
        
        # DETR Head
        if self.training:
            out_bboxes, out_logits, gt_bbox, gt_class, \
                dn_out_bboxes, dn_out_logits, dn_meta, _, \
                    domain_outs_dec, domain_labels_dec, masks, outputs, num_denoising, out_instance_query  = self.detr_head(
                        out_transformer, body_feats, self.inputs)

            return out_bboxes, out_logits, gt_bbox, gt_class, \
                   dn_out_bboxes, dn_out_logits, dn_meta, None, \
                       domain_outs_dec, domain_labels_dec, masks, outputs, num_denoising, domain_outs_enc, domain_labels_enc, out_instance_query
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()