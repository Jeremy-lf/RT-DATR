# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import copy
import time
import typing
import numpy as np

import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed import fleet
from ppdet.optimizer import ModelEMA, SimpleModelEMA
from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight, save_model
import ppdet.utils.stats as stats
from ppdet.utils import profiler
from ppdet.modeling.ssod.utils import align_weak_strong_shape
from .trainer import Trainer
from ppdet.utils.logger import setup_logger
from paddle.static import InputSpec
from ppdet.engine.export_utils import _dump_infer_config, _prune_input_spec

from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

import paddle.nn.functional as F

MOT_ARCH = ['JDE', 'FairMOT', 'DeepSORT', 'ByteTrack', 'CenterTrack']

logger = setup_logger('ppdet.engine')

__all__ = ['Trainer_DA_RTDETR_Instance']


class Trainer_DA_RTDETR_Instance(Trainer):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        assert mode.lower() in ['train', 'eval', 'test'], \
                "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()
        self.optimizer = None
        self.is_loaded_weights = False
        self.use_amp = self.cfg.get('amp', False)
        self.amp_level = self.cfg.get('amp_level', 'O1')
        self.custom_white_list = self.cfg.get('custom_white_list', None)
        self.custom_black_list = self.cfg.get('custom_black_list', None)
        # self.use_ema = False
        # build data loader
        capital_mode = self.mode.capitalize()
        self.dataset = self.cfg['{}Dataset'.format(capital_mode)] = create(
            '{}Dataset'.format(capital_mode))()

        if self.mode == 'train':
            self.dataset_unlabel = self.cfg['UnsupTrainDataset'] = create(
                'UnsupTrainDataset')
            self.loader = create('SemiTrainReader')(
                self.dataset, self.dataset_unlabel, cfg.worker_num)

        # build model
        if 'model' not in self.cfg:
            self.model = create(cfg.architecture)
        else:
            self.model = self.cfg.model
            self.is_loaded_weights = True

        # EvalDataset build with BatchSampler to evaluate in single device
        # TODO: multi-device evaluate
        if self.mode == 'eval':
            self._eval_batch_sampler = paddle.io.BatchSampler(
                self.dataset, batch_size=self.cfg.EvalReader['batch_size'])
            # If metric is VOC, need to be set collate_batch=False.
            if cfg.metric == 'VOC':
                cfg['EvalReader']['collate_batch'] = False
            self.loader = create('EvalReader')(self.dataset, cfg.worker_num,
                                               self._eval_batch_sampler)
        # TestDataset build after user set images, skip loader creation here

        # build optimizer in train mode
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            if steps_per_epoch < 1:
                logger.warning(
                    "Samples in dataset are less than batch_size, please set smaller batch_size in TrainReader."
                )
            self.lr = create('LearningRate')(steps_per_epoch)
            self.optimizer = create('OptimizerBuilder')(self.lr, self.model)

            # Unstructured pruner is only enabled in the train mode.
            if self.cfg.get('unstructured_prune'):
                self.pruner = create('UnstructuredPruner')(self.model,
                                                           steps_per_epoch)
        if self.use_amp and self.amp_level == 'O2':
            self.model, self.optimizer = paddle.amp.decorate(
                models=self.model,
                optimizers=self.optimizer,
                level=self.amp_level)

        self.use_ema = ('use_ema' in cfg and cfg['use_ema'])
        if self.use_ema:
            ema_decay = self.cfg.get('ema_decay', 0.9998)
            ema_decay_type = self.cfg.get('ema_decay_type', 'threshold')
            cycle_epoch = self.cfg.get('cycle_epoch', -1)
            ema_black_list = self.cfg.get('ema_black_list', None)
            ema_filter_no_grad = self.cfg.get('ema_filter_no_grad', False)
            self.ema = ModelEMA(
                self.model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch,
                ema_black_list=ema_black_list,
                ema_filter_no_grad=ema_filter_no_grad)
        
        self._nranks = dist.get_world_size()
        self._local_rank = dist.get_rank()

        self.status = {}

        self.start_epoch = 0
        self.start_iter = 0
        self.end_epoch = 0 if 'epoch' not in cfg else cfg.epoch

        # initial default callbacks
        self._init_callbacks()

        # initial default metrics
        self._init_metrics()
        self._reset_metrics()

    def load_weights(self, weights, ARSL_eval=False):
        if self.is_loaded_weights:
            return
        self.start_epoch = 0
        load_pretrain_weight(self.model, weights, ARSL_eval)
        logger.debug("Load weights {} to start training".format(weights))

    def resume_weights(self, weights, exchange=True):
        # support Distill resume weights
        if hasattr(self.model, 'student_model'):
            self.start_epoch = load_weight(self.model.student_model, weights,
                                           self.optimizer, exchange)
        else:
            self.start_epoch = load_weight(self.model, weights, self.optimizer,
                                           self.ema if self.use_ema else None)
        logger.debug("Resume weights of epoch {}".format(self.start_epoch))
        logger.debug("Resume weights of iter {}".format(self.start_iter))

    def train(self, validate=False):
        assert self.mode == 'train', "Model not in 'train' mode"
        Init_mark = False
        if validate:
            self.cfg.EvalDataset = create("EvalDataset")()

        model = self.model
        sync_bn = (getattr(self.cfg, 'norm_type', None) == 'sync_bn' and
                   self.cfg.use_gpu and self._nranks > 1)
        if sync_bn:
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)

        if self.cfg.get('fleet', False):
            model = fleet.distributed_model(model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg[
                'find_unused_parameters'] if 'find_unused_parameters' in self.cfg else False
            model = paddle.DataParallel(
                model, find_unused_parameters=find_unused_parameters)


        self.status.update({
            'epoch_id': self.start_epoch,
            'iter_id': self.start_iter,
            # 'step_id': self.start_step,
            'steps_per_epoch': len(self.loader),
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num)
            self._flops(flops_loader)
        profiler_options = self.cfg.get('profiler_options', None)

        self._compose_callback.on_train_begin(self.status)
        iter_id = self.start_iter
        self.status['iter_id'] = iter_id
        # self.status['eval_interval'] = self.cfg.eval_interval
        # self.status['save_interval'] = self.cfg.save_interval
        for epoch_id in range(self.start_epoch, self.cfg.epoch):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self._compose_callback.on_epoch_begin(self.status)
            self.loader.dataset_label.set_epoch(epoch_id)
            self.loader.dataset_unlabel.set_epoch(epoch_id)
            iter_tic = time.time()
            model.train()
            iter_tic = time.time()
            for step_id in range(len(self.loader)):
                data = next(self.loader)
                data_sup, data_unsup = data
                data_sup['domain'] = 0
                data_unsup['domain'] = 1
                # data_sup['epoch_id'] = epoch_id
                # data_unsup['epoch_id'] = epoch_id
                
                iter_id += 1
                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id
                self.status['iter_id'] = iter_id

                profiler.add_profiler_step(profiler_options)
                self._compose_callback.on_step_begin(self.status)
                
                # DA
                with model.no_sync():
                    if self.cfg.da_method == 'o2net_detr':
                        criterion = create('DETRLoss')
                        outputs_bbox_src, outputs_logit_src, gt_bbox_src, gt_class_src, \
                            domain_outs_src, domain_labels_src, masks_src, hs_src = model(data_sup)
                        outputs_bbox_tgt, outputs_logit_tgt, gt_bbox_tgt, gt_class_tgt, \
                            domain_outs_tgt, domain_labels_tgt, masks_tgt, hs_tgt = model(data_unsup)
                        losses_dict = criterion(outputs_bbox_src, outputs_logit_src, gt_bbox_src, gt_class_src)
                    
                    elif self.cfg.da_method == 'o2net_rtdetr_instance':
                        criterion = create('DINOLoss')
                        outputs_bbox_src, outputs_logit_src, gt_bbox_src, gt_class_src, \
                            dn_out_bboxes, dn_out_logits, dn_meta, gt_score_src, \
                                domain_outs_src, domain_labels_src, masks_src, hs_src, src_num_denoising, outputs_instance_query_src = model(data_sup)
                        
                        outputs_bbox_tgt, outputs_logit_tgt, gt_bbox_tgt, gt_class_tgt, \
                            dn_out_bboxes, dn_out_logits, dn_meta, gt_score_tgt, \
                                domain_outs_tgt, domain_labels_tgt, masks_tgt, hs_tgt, tgt_num_denoising, outputs_instance_query_tgt = model(data_unsup)

                        losses_dict = criterion(outputs_bbox_src, outputs_logit_src, gt_bbox_src, gt_class_src,
                            dn_out_bboxes, dn_out_logits, dn_meta, gt_score_src)
                    
                    # decoder da loss
                    src_domain = paddle.zeros_like(outputs_instance_query_src)
                    tgt_domain = paddle.ones_like(outputs_instance_query_tgt)

                    src_space_query_loss = F.binary_cross_entropy_with_logits(outputs_instance_query_src, \
                        src_domain, reduction='none')
                    tgt_space_query_loss = F.binary_cross_entropy_with_logits(outputs_instance_query_tgt, \
                        tgt_domain, reduction='none')

                    src_prob = F.sigmoid(outputs_instance_query_src)
                    src_p_t = src_prob * src_domain + (1 - src_prob) * (1 - src_domain)
                    src_loss = src_space_query_loss * ((1 - src_p_t) ** 2)
                    
                    tgt_prob = F.sigmoid(outputs_instance_query_tgt)
                    tgt_p_t = tgt_prob * tgt_domain + (1 - tgt_prob) * (1 - tgt_domain)
                    tgt_loss = tgt_space_query_loss * ((1 - tgt_p_t) ** 2)

                    space_loss = src_loss.mean() + tgt_loss.mean()

                    losses_dict['instance_loss'] = space_loss

                    
                    # pseudo-label
                    pseudo = None

                    last_outputs_bbox_tgt = outputs_bbox_tgt[-1]  # 2x300x4
                    last_outputs_logit_tgt = outputs_logit_tgt[-1] # 2x300x3

                    prob = F.sigmoid(last_outputs_logit_tgt) # Bx300xnum_cls
                    topk_values, topk_indexes = paddle.topk(prob.flatten(1), 100, axis=1)
                    scores = topk_values # Bx100
                    index = topk_indexes // last_outputs_logit_tgt.shape[2]  # Bx100
                    labels = topk_indexes % last_outputs_logit_tgt.shape[2]  # labels 2x100

                    batch_ind = paddle.arange(end=scores.shape[0]).unsqueeze(-1).tile([1, 100])
                    index = paddle.stack([batch_ind, index], axis=-1) # Bx100x2
                    boxes = paddle.gather_nd(last_outputs_bbox_tgt, index) # Bx100x4



                    scores_indices = (scores > 0.5) # Pseudo label selection using confidence
                    
                    if scores_indices.sum():
                        pseudo = {'boxes': boxes[scores_indices]}
                                # 'labels': labels[scores_indices]}
                                # 'image_id': targets_tgt[0]['image_id'],
                    #             # 'orig_size': targets_tgt[0]['orig_size'],
                    #             # 'size': targets_tgt[0]['size']}

                    loss_da = 0
                    loss_global_da = 0
                    for l in range(len(domain_outs_src)-1):
                        domain_out_src = domain_outs_src[l]   # torch.Size([2, 2, 64, 64])
                        domain_label_src = domain_labels_src[l]   # torch.Size([1, 64, 64])
                        mask_src = masks_src[l]
                        domain_out_tgt = domain_outs_tgt[l]  # torch.Size([2, 2, 72, 72])
                        domain_label_tgt = domain_labels_tgt[l]  # torch.Size([1, 72, 72])
                        mask_tgt = masks_tgt[l] # 2x64x64

                        domain_prob_src = F.log_softmax(domain_out_src, axis=1)
                        domain_prob_tgt = F.log_softmax(domain_out_tgt, axis=1)

                        domain_label_src = paddle.tile(domain_label_src, [domain_out_src.shape[0], 1, 1])
                        domain_label_tgt = paddle.tile(domain_label_tgt, [domain_out_tgt.shape[0], 1, 1])

                        DA_img_loss_src = F.nll_loss(
                            domain_prob_src, domain_label_src, reduction="none") # 取对应label得元素 -1*domain_prob_src[cls]
                        DA_img_loss_tgt = F.nll_loss(
                            domain_prob_tgt, domain_label_tgt, reduction="none")

                        mask_src = mask_src.astype('bool')
                        mask_tgt = mask_tgt.astype('bool') # full 1->true
                        # mask_src = ~mask_src
                        # mask_tgt = ~mask_tgt # true->false

                        # mask_src_sum = paddle.sum(mask_src)  
                        # mask_tgt_sum = paddle.sum(mask_tgt)
                        # # 确保mask_src_sum不为0，以避免除以零的错误  
                        # if mask_src_sum == 0 or mask_tgt_sum == 0:  
                        #     raise ValueError("mask_src_sum or mask_tgt_sum is zero, which would cause division by zero.")  
                        
                        # 计算加权损失和  
                        weighted_loss_src = paddle.sum(DA_img_loss_src * mask_src)
                        weighted_loss_tgt = paddle.sum(DA_img_loss_tgt * mask_tgt)

                        global_DA_img_loss = weighted_loss_src / mask_src.sum() + weighted_loss_tgt / mask_tgt.sum()

                        # Mask out background regions
                        mask_ins_src = box_to_mask(gt_bbox_src[0], mask_src.shape)
                        mask_ins_src += 1 # for numeric stability 
                        mask_ins_src = mask_ins_src / mask_ins_src.mean()
                        mask_final_src = mask_src * mask_ins_src
                        if pseudo is None:
                            mask_ins_tgt = box_to_mask(None, mask_tgt.shape)
                        else:
                            mask_ins_tgt = box_to_mask(pseudo['boxes'], mask_tgt.shape)

                        mask_ins_tgt += 1 # for numeric stability
                        mask_ins_tgt = mask_ins_tgt / mask_ins_tgt.mean()
                        mask_final_tgt = mask_tgt * mask_ins_tgt

                        if mask_final_tgt.sum() and mask_final_src.sum():
                            DA_img_loss = paddle.sum(DA_img_loss_src * mask_final_src) / mask_final_src.sum() + \
                                paddle.sum(DA_img_loss_tgt * mask_final_tgt) / mask_final_tgt.sum()
                        elif mask_final_src.sum():
                            DA_img_loss = paddle.sum(DA_img_loss_src * mask_final_src) / mask_final_src.sum() + \
                                paddle.sum(DA_img_loss_tgt * mask_tgt) / mask_tgt.sum()
                        elif mask_final_tgt.sum():
                            DA_img_loss = paddle.sum(DA_img_loss_src * mask_src) / mask_src.sum() + \
                                paddle.sum(DA_img_loss_tgt * mask_final_tgt) / mask_final_tgt.sum()
                        else:
                            DA_img_loss = paddle.sum(DA_img_loss_src * mask_src) / mask_src.sum() + \
                                paddle.sum(DA_img_loss_tgt * mask_tgt) / mask_tgt.sum()
                        
                        loss_da += DA_img_loss
                        loss_global_da += global_DA_img_loss
                    
                    # DA total loss
                    losses_dict["loss_da"] = 1.0 * loss_da + loss_global_da
                    # if self.cfg.da_method == 'o2net_rtdetr':
                    #     _, N, _ = hs_src[-1].shape
                    #     denoise_hs_src, _ = paddle.split(hs_src[-1], [N - src_num_denoising, src_num_denoising], axis=1)
                    #     losses_dict["loss_wasserstein"] = swd(denoise_hs_src, hs_tgt[-1])
                    # else:
                    #     losses_dict["loss_wasserstein"] = swd(hs_src[-1], hs_tgt[-1])
        
                    
                    outputs = dict()
                    outputs.update(losses_dict)
                    loss = paddle.add_n([v for k, v in outputs.items() if 'log' not in k])
                    outputs['loss'] = loss

                    # model backward
                    loss.backward()

                fused_allreduce_gradients(
                        list(model.parameters()), None)
                    
                self.optimizer.step()
                curr_lr = self.optimizer.get_lr()
                self.lr.step()
                if self.cfg.get('unstructured_prune'):
                    self.pruner.step()
                self.optimizer.clear_grad()
                
                self.status['learning_rate'] = curr_lr
                if self._nranks < 2 or self._local_rank == 0:
                    self.status['training_staus'].update(outputs)

                self.status['batch_time'].update(time.time() - iter_tic)
                self._compose_callback.on_step_end(self.status)

                if self.use_ema:
                    self.ema.update()
                
                iter_tic = time.time()                

            if self.cfg.get('unstructured_prune'):
                self.pruner.update_params()
            
            is_snapshot = (self._nranks < 2 or (self._local_rank == 0 or self.cfg.metric == "Pose3DEval")) \
                       and ((epoch_id + 1) % self.cfg.snapshot_epoch == 0 or epoch_id == self.end_epoch - 1)

            if is_snapshot and self.use_ema:
                # apply ema weight on model
                weight = copy.deepcopy(self.model.state_dict())
                self.model.set_dict(self.ema.apply())
                self.status['weight'] = weight

            self._compose_callback.on_epoch_end(self.status)

            if validate and is_snapshot:
                if not hasattr(self, '_eval_loader'):
                    # build evaluation dataset and loader
                    self._eval_dataset = self.cfg.EvalDataset
                    self._eval_batch_sampler = \
                        paddle.io.BatchSampler(
                            self._eval_dataset,
                            batch_size=self.cfg.EvalReader['batch_size'])
                    # If metric is VOC, need to be set collate_batch=False.
                    if self.cfg.metric == 'VOC':
                        self.cfg['EvalReader']['collate_batch'] = False
                    if self.cfg.metric == "Pose3DEval":
                        self._eval_loader = create('EvalReader')(
                            self._eval_dataset, self.cfg.worker_num)
                    else:
                        self._eval_loader = create('EvalReader')(
                            self._eval_dataset,
                            self.cfg.worker_num,
                            batch_sampler=self._eval_batch_sampler)
                # if validation in training is enabled, metrics should be re-init
                # Init_mark makes sure this code will only execute once
                if validate and Init_mark == False:
                    Init_mark = True
                    self._init_metrics(validate=validate)
                    self._reset_metrics()

                with paddle.no_grad():
                    self.status['save_best_model'] = True
                    self._eval_with_loader(self._eval_loader)

            if is_snapshot and self.use_ema:
                # reset original weight
                self.model.set_dict(weight)
                self.status.pop('weight')

        self._compose_callback.on_train_end(self.status)

    def _eval_with_loader(self, loader):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num, self._eval_batch_sampler)
            self._flops(flops_loader)
        print("*****teacher evaluate*****")
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            outs = self.model(data)

            # update metrics
            for metric in self._metrics:
                metric.update(data, outs)

            # multi-scale inputs: all inputs have same im_id
            if isinstance(data, typing.Sequence):
                sample_num += data[0]['im_id'].numpy().shape[0]
            else:
                sample_num += data['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        self._compose_callback.on_epoch_end(self.status)
        # reset metric states for metric may performed multiple times
        self._reset_metrics()
        self.status['mode'] = 'train'

    def evaluate(self):
        with paddle.no_grad():
            self._eval_with_loader(self.loader)


def box_to_mask(boxes, size):
    mask = paddle.zeros(size)  # 2x16x34
    if boxes is None:
        return mask
    img_w, img_h = size[-1], size[-2]
    img_w, img_h = paddle.to_tensor([img_w]), paddle.to_tensor([img_h])
    scale_fct = paddle.stack([img_w, img_h, img_w, img_h]).reshape([1, 4])

    # assert len(boxes) == size[0], 'boxes should be [B, N, 4]'
    if len(boxes.shape) != 3:
        boxes = boxes.unsqueeze(0)

    if boxes.shape[1] == 0:
        return mask

    boxes = boxes * scale_fct   # 1x3x4 * 1x4
    boxes = boxes[0] # 3x4
    for box in boxes:
        x, y, w, h = box
        xmin, xmax = x - w / 2, x + w / 2
        ymin, ymax = y - h / 2, y + h / 2
        xmin, xmax, ymin, ymax = max(0, int(xmin)), min(size[-1]-1, int(xmax)), max(0, int(ymin)), min(size[-2]-1, int(ymax))
        try:
            mask[:, ymin: ymax, xmin: xmax] = 1 
        except:
            import pdb; 
            pdb.set_trace()
    
    return mask

def swd(source_features, target_features, M=256):
    batch_size = source_features.shape[0]
    source_features = source_features.reshape([-1, source_features.shape[-1]])
    target_features = target_features.reshape([-1, target_features.shape[-1]])
    
    theta = paddle.rand((M, 256)) # 256 is the feature dim
    # theta = theta / theta.norm(2, dim=1)
    norm = paddle.norm(theta, p=2, axis=1, keepdim=True)
    theta = theta / norm
    source_proj = paddle.matmul(theta, source_features.transpose([1, 0]))
    target_proj = paddle.matmul(theta, target_features.transpose([1, 0]))

    source_proj = paddle.sort(source_proj, axis=1)  
    target_proj = paddle.sort(target_proj, axis=1)

    # loss = paddle.mean(paddle.square(source_proj - target_proj))
    loss = paddle.sum(paddle.square(source_proj - target_proj)) / M / batch_size
    # loss = (source_proj - target_proj).pow(2).sum() / M / batch_size
    return loss