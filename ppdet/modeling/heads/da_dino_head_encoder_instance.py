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
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
import pycocotools.mask as mask_util
from ..initializer import linear_init_, constant_
from ..transformers.utils import inverse_sigmoid

__all__ = ['DA_DINOHead_Econder_Instance']

class MLP(nn.Layer):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self._reset_parameters()

    def _reset_parameters(self):
        for l in self.layers:
            linear_init_(l)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiHeadAttentionMap(nn.Layer):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/segmentation.py

        This is a 2D attention module, which only returns the attention softmax (no multiplication by value)
    """

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0,
                 bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Constant()) if bias else False

        self.q_proj = nn.Linear(query_dim, hidden_dim, weight_attr, bias_attr)
        self.k_proj = nn.Conv2D(
            query_dim,
            hidden_dim,
            1,
            weight_attr=weight_attr,
            bias_attr=bias_attr)

        self.normalize_fact = float(hidden_dim / self.num_heads)**-0.5

    def forward(self, q, k, mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        bs, num_queries, n, c, h, w = q.shape[0], q.shape[1], self.num_heads,\
                                      self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1]
        qh = q.reshape([bs, num_queries, n, c])
        kh = k.reshape([bs, n, c, h, w])
        # weights = paddle.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)
        qh = qh.transpose([0, 2, 1, 3]).reshape([-1, num_queries, c])
        kh = kh.reshape([-1, c, h * w])
        weights = paddle.bmm(qh * self.normalize_fact, kh).reshape(
            [bs, n, num_queries, h, w]).transpose([0, 2, 1, 3, 4])

        if mask is not None:
            weights += mask
        # fix a potenial bug: https://github.com/facebookresearch/detr/issues/247
        weights = F.softmax(weights.flatten(3), axis=-1).reshape(weights.shape)
        weights = self.dropout(weights)
        return weights



@register
class DA_DINOHead_Econder_Instance(nn.Layer):
    __inject__ = ['loss']

    def __init__(self, loss='DINOLoss', eval_idx=-1):
        super(DA_DINOHead_Econder_Instance, self).__init__()
        self.loss = loss
        self.eval_idx = eval_idx

    def forward(self, out_transformer, body_feats, inputs=None):
        (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits,
         dn_meta, domain_outs, domain_labels, masks, outputs, num_denoising, out_instance_query) = out_transformer

        if self.training:
            assert inputs is not None
            # assert 'gt_bbox' in inputs and 'gt_class' in inputs

            if dn_meta is not None:
                if isinstance(dn_meta, list):
                    dual_groups = len(dn_meta) - 1
                    dec_out_bboxes = paddle.split(
                        dec_out_bboxes, dual_groups + 1, axis=2)
                    dec_out_logits = paddle.split(
                        dec_out_logits, dual_groups + 1, axis=2)
                    enc_topk_bboxes = paddle.split(
                        enc_topk_bboxes, dual_groups + 1, axis=1)
                    enc_topk_logits = paddle.split(
                        enc_topk_logits, dual_groups + 1, axis=1)

                    dec_out_bboxes_list = []
                    dec_out_logits_list = []
                    dn_out_bboxes_list = []
                    dn_out_logits_list = []
                    loss = {}
                    for g_id in range(dual_groups + 1):
                        if dn_meta[g_id] is not None:
                            dn_out_bboxes_gid, dec_out_bboxes_gid = paddle.split(
                                dec_out_bboxes[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=2)
                            dn_out_logits_gid, dec_out_logits_gid = paddle.split(
                                dec_out_logits[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=2)
                        else:
                            dn_out_bboxes_gid, dn_out_logits_gid = None, None
                            dec_out_bboxes_gid = dec_out_bboxes[g_id]
                            dec_out_logits_gid = dec_out_logits[g_id]
                        out_bboxes_gid = paddle.concat([
                            enc_topk_bboxes[g_id].unsqueeze(0),
                            dec_out_bboxes_gid
                        ])
                        out_logits_gid = paddle.concat([
                            enc_topk_logits[g_id].unsqueeze(0),
                            dec_out_logits_gid
                        ])
                        loss_gid = self.loss(
                            out_bboxes_gid,
                            out_logits_gid,
                            inputs['gt_bbox'],
                            inputs['gt_class'],
                            dn_out_bboxes=dn_out_bboxes_gid,
                            dn_out_logits=dn_out_logits_gid,
                            dn_meta=dn_meta[g_id])
                        # sum loss
                        for key, value in loss_gid.items():
                            loss.update({
                                key: loss.get(key, paddle.zeros([1])) + value
                            })

                    # average across (dual_groups + 1)
                    for key, value in loss.items():
                        loss.update({key: value / (dual_groups + 1)})
                    return loss
                else:
                    if dn_meta is not None:
                        dn_out_bboxes, dec_out_bboxes = paddle.split(
                            dec_out_bboxes, dn_meta['dn_num_split'], axis=2)
                        dn_out_logits, dec_out_logits = paddle.split(
                            dec_out_logits, dn_meta['dn_num_split'], axis=2)
                    else:
                        dn_out_bboxes, dn_out_logits = None, None
            else:
                dn_out_bboxes, dn_out_logits = None, None

            out_bboxes = paddle.concat(
                [enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
            out_logits = paddle.concat(
                [enc_topk_logits.unsqueeze(0), dec_out_logits])

            if 'gt_bbox' in inputs:
                return out_bboxes, out_logits, inputs['gt_bbox'], inputs['gt_class'], \
                    dn_out_bboxes, dn_out_logits, dn_meta, None, \
                        domain_outs, domain_labels, masks, outputs, num_denoising, out_instance_query
            else:
                return out_bboxes, out_logits, None, None, \
                    dn_out_bboxes, dn_out_logits, dn_meta, None, \
                        domain_outs, domain_labels, masks, outputs, None, out_instance_query
        else:
            return (dec_out_bboxes[self.eval_idx],
                    dec_out_logits[self.eval_idx], None)
