import copy

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
import pdb

@HEADS.register_module()
class TMBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 fc_out_channels=1024,
                 scale=20.,
                 loss_margin=None,
                 reg_class_agnostic=True,
                 split=None,
                 *args,
                 **kwargs):
        # self.loss_margin = loss_margin
        self.split = split
        assert self.split is not None
        super(TMBBoxHead,
              self).__init__(num_shared_fcs=1,
                             reg_class_agnostic=reg_class_agnostic,
                             *args,
                             **kwargs)
        del self.fc_cls
        del self.shared_fcs
        del self.cls_fcs
        del self.reg_fcs
        num_base = 60
        num_novel = 20

        # base branch
        base_shared_fcs = nn.ModuleList()
        base_shared_fcs.append(nn.Linear(49 * 256, fc_out_channels))
        base_shared_fcs.append(nn.Linear(fc_out_channels, fc_out_channels))
        self.base_shared_fcs = base_shared_fcs
        self.base_fc_cls = nn.Linear(self.cls_last_dim, num_base + 1, bias=True)

        # novel branch
        novel_shared_fcs = nn.ModuleList()
        novel_shared_fcs.append(nn.Linear(49 * 256, fc_out_channels))
        novel_shared_fcs.append(nn.Linear(fc_out_channels, fc_out_channels))
        self.novel_shared_fcs = novel_shared_fcs
        self.novel_fc_cls = nn.Linear(self.cls_last_dim,
                                      num_novel,
                                      bias=True)

        # temperature
        self.scale = scale

        # set-specialized margin loss
        self.loss_margin = None
        # build_loss(loss_margin)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        torch.manual_seed(0)
        processing = True
        # During training, processing the checkpoints
        # During testing, directly load the checkpoints
        for param_name in state_dict:
            if 'novel_shared_fcs' in param_name:
                processing = False
                # num_base_classes = 60 + 1
                # base_cpt = 'work_dirs/faster_rcnn_r101_fpn_1x_coco_exclude_split2/latest.pth'
                # base_weights = torch.load(base_cpt, map_location='cpu')
                # if 'state_dict' in base_weights:
                #     base_weights = base_weights['state_dict']
                # # extract head parameters
                # head_params_names = [k for k in state_dict if 'bbox_head' in k]
                # head_params = dict()
                # for k in head_params_names:
                #     # print(k)
                #     head_params[k] = state_dict.pop(k)

                # for n, p in base_weights.items():
                #     if 'bbox_head' not in n:
                #         state_dict[n] = copy.deepcopy(p)

                # # initialize the novel branch with the weight of association
                # # for n, p in head_params.items():
                # #     if 'shared_fcs' in n:
                # #         print(n)
                # #         state_dict[n] = copy.deepcopy(p)

                # # initialize the base branch with the weight of base training
                # for n, p in base_weights.items():
                #     if 'shared_fcs' in n:
                #         print(n)
                #         new_n = n.replace('shared_fcs', 'base_shared_fcs')
                #         state_dict[new_n] = copy.deepcopy(p)
                #     if 'fc_cls' in n:
                #         print(n)
                #         new_n = n.replace('fc', 'base_fc')
                #         state_dict[new_n] = copy.deepcopy(p[:num_base_classes])
                #     if self.reg_class_agnostic and 'fc_reg' in n:
                #         print(n)
                #         state_dict[n] = copy.deepcopy(p)

                # novel_cpt = 'work_dirs/faster_rcnn_r101_fpn_1x_coco_exclude_split2_few_shot_highlr/latest.pth'
                # novel_weights = torch.load(novel_cpt, map_location='cpu')
                # if 'state_dict' in novel_weights:
                #     novel_weights = novel_weights['state_dict']

                # for n, p in novel_weights.items():
                #     if 'novel_fc_cls' in n:
                #         print(n)
                #         state_dict[n] = copy.deepcopy(p)
                #     if 'novel_shared_fcs' in n:
                #         print(n)
                #         state_dict[n] = copy.deepcopy(p)

                # super()._load_from_state_dict(state_dict, prefix, local_metadata,
                #                             strict, missing_keys,
                #                             unexpected_keys, error_msgs)
                break

        if processing:
            # load base training checkpoint
            num_base_classes = 60 + 1
            base_cpt = f'work_dirs/faster_rcnn_r101_fpn_1x_coco_exclude_{self.split}/latest.pth'
            # base_cpt = f'work_dirs/faster_rcnn_r50_fpn_1x_coco_modified_exclude_{self.split}/latest.pth'
            print(base_cpt)
            base_weights = torch.load(base_cpt, map_location='cpu')
            if 'state_dict' in base_weights:
                base_weights = base_weights['state_dict']
            # extract head parameters
            head_params_names = [k for k in state_dict if 'bbox_head' in k]
            head_params = dict()
            for k in head_params_names:
                head_params[k] = state_dict.pop(k)

            for n, p in base_weights.items():
                if 'bbox_head' not in n:
                    state_dict[n] = copy.deepcopy(p)

            # initialize the novel branch with the weight of association
            for n, p in head_params.items():
                if 'shared_fcs' in n:
                    new_n = n.replace('shared_fcs', 'novel_shared_fcs')
                    state_dict[new_n] = copy.deepcopy(p)

            # initialize the base branch with the weight of base training
            for n, p in base_weights.items():
                if 'shared_fcs' in n:
                    new_n = n.replace('shared_fcs', 'base_shared_fcs')
                    state_dict[new_n] = copy.deepcopy(p)
                if 'fc_cls' in n:
                    new_n = n.replace('fc', 'base_fc')
                    state_dict[new_n] = copy.deepcopy(p[:num_base_classes])
                if self.reg_class_agnostic and 'fc_reg' in n:
                    state_dict[n] = copy.deepcopy(p)

            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, x, return_fc_feat=False):
        x = x.flatten(1)

        base_x = x
        novel_x = x
        for fc in self.base_shared_fcs:
            base_x = self.relu(fc(base_x))
        x = base_x
        base_scores = self.base_fc_cls(x)
        bbox_preds = self.fc_reg(x)     


        for fc in self.novel_shared_fcs:
            novel_x = self.relu(fc(novel_x))
        x = novel_x
        novel_scores = self.novel_fc_cls(x)

        # normalize the input x along the `input_size` dimension
        
        # x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        # x_normalized = x.div(x_norm + 1e-5)
        # temp_norm = torch.norm(self.base_fc_cls.weight.data, p=2,
        #                        dim=1).unsqueeze(1).expand_as(
        #                            self.base_fc_cls.weight.data)
        # self.base_fc_cls.weight.data = self.base_fc_cls.weight.data.div(
        #     temp_norm + 1e-5)
        # cos_dist = self.base_fc_cls(x_normalized)
        # base_scores = self.scale * cos_dist
        

        
        # x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        # x_normalized = x.div(x_norm + 1e-5)
        # temp_norm = torch.norm(self.novel_fc_cls.weight.data, p=2,
        #                        dim=1).unsqueeze(1).expand_as(
        #                            self.novel_fc_cls.weight.data)
        # self.novel_fc_cls.weight.data = self.novel_fc_cls.weight.data.div(
        #     temp_norm + 1e-5)
        # cos_dist = self.novel_fc_cls(x_normalized)
        # novel_scores = self.scale * cos_dist
        # novel_scores[:] = -100
        scores = torch.cat([base_scores[:, :60], novel_scores, base_scores[:, 60:61]], -1)
        # scores = torch.cat([base_scores, novel_scores], -1)
        # return scores, bbox_preds
        return scores, bbox_preds

    def init_weights(self):
        torch.manual_seed(0)
        # conv layers are already initialized by ConvModule
        for module_list in [self.base_shared_fcs, self.novel_shared_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.base_fc_cls.weight, 0, 0.01)
        nn.init.normal_(self.novel_fc_cls.weight, 0, 0.01)
        nn.init.normal_(self.fc_reg.weight, 0, 0.001)
        nn.init.constant_(self.fc_reg.bias, 0)

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             gt_bboxes=None,
             gt_labels=None,
             img_metas=None):
        losses = super(TMBBoxHead,
                       self).loss(cls_score,
                                  bbox_pred,
                                  rois,
                                  labels,
                                  label_weights,
                                  bbox_targets,
                                  bbox_weights,
                                  reduction_override=reduction_override)

        # set-specialized margin loss
        if self.loss_margin is not None:
            losses.update(self.loss_margin(cls_score, labels, label_weights))

        return losses
