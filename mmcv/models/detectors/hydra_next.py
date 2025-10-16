import copy

import torch

from mmcv.models import DETECTORS
from mmcv.models.dense_heads.planning_head_plugin.metric_stp3 import PlanningMetric
from mmcv.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.models.utils.grid_mask import GridMask
from mmcv.utils import force_fp32, auto_fp16


@DETECTORS.register_module()
class HydraNext(MVXTwoStageDetector):
    """VAD model.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 prev_frame_num=0,
                 fut_ts=6,
                 fut_mode=6,
                 debug=False,
                 **kwargs
                 ):

        super(HydraNext,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head['valid_fut_ts']
        self.prev_frame_num = prev_frame_num
        self.prev_frame_infos = []
        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.planning_metric = None

    def obtain_hist_feats(self, imgs_queue):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        with torch.no_grad():
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            feats = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            return feats

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          map_gt_bboxes_ignore=None,
                          prev_bev=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None,
                          ego_status_feature=None,
                          gt_brake=None,
                          gt_throttle=None,
                          gt_steer=None,
                          ctrl_seq=None,
                          ctrl_mask=None,
                          ctrl_seq_dp=None,
                          ctrl_mask_dp=None,
                          local_command_xy=None
                          ):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat,
                                  ego_status_feature=ego_status_feature,
                                  local_command_xy=local_command_xy)
        env_tokens = outs['env_tokens']
        loss_inputs = [
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd,
            gt_brake, gt_throttle, gt_steer, ctrl_seq, ctrl_mask,
            ctrl_seq_dp, ctrl_mask_dp, env_tokens,
            img_metas
        ]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, inputs, return_loss=True, rescale=False):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            losses = self.forward_train(**inputs)
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(inputs['img_metas']))
            return outputs
        else:
            outputs = self.forward_test(**inputs, rescale=rescale)
            return outputs

    @force_fp32(apply_to=('img', 'points', 'prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None,
                      ego_status_feature=None,
                      gt_brake=None,
                      gt_throttle=None,
                      gt_steer=None,
                      ctrl_seq=None,
                      ctrl_mask=None,
                      ctrl_seq_dp=None,
                      ctrl_mask_dp=None,
                      local_command_xy=None,
                      **kwargs
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        # B T N C H W
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        img_metas = [each[len_queue - 1] for each in img_metas]
        # [B N C H W]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if len_queue > 1:
            hist_feats = self.obtain_hist_feats(prev_img)
            input = [hist_feats, img_feats]
        else:
            input = img_feats
        losses = dict()
        losses_pts = self.forward_pts_train(input, gt_bboxes_3d, gt_labels_3d,
                                            map_gt_bboxes_3d, map_gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, map_gt_bboxes_ignore, prev_bev=None,
                                            ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                            ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                            ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels,
                                            ego_status_feature=ego_status_feature,
                                            gt_brake=gt_brake,
                                            gt_throttle=gt_throttle,
                                            gt_steer=gt_steer,
                                            ctrl_seq=ctrl_seq,
                                            ctrl_mask=ctrl_mask,
                                            ctrl_seq_dp=ctrl_seq_dp,
                                            ctrl_mask_dp=ctrl_mask_dp,
                                            local_command_xy=local_command_xy)

        losses.update(losses_pts)
        return losses

    def forward_test(
            self,
            img_metas,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            img=None,
            ego_his_trajs=None,
            ego_fut_trajs=None,
            ego_fut_cmd=None,
            ego_lcf_feat=None,
            gt_attr_labels=None,
            ego_status_feature=None,
            local_command_xy=None,
            **kwargs
    ):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if self.prev_frame_num > 0:
            if len(self.prev_frame_infos) < self.prev_frame_num:
                self.prev_frame_info = {
                    "prev_bev": None,
                    "scene_token": None,
                    "prev_pos": 0,
                    "prev_angle": 0,
                }
            else:
                self.prev_frame_info = self.prev_frame_infos.pop(0)

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        if ego_his_trajs is not None:
            ego_his_trajs = ego_his_trajs[0]
        if ego_fut_trajs is not None:
            ego_fut_trajs = ego_fut_trajs[0]
        if ego_fut_cmd is not None:
            ego_fut_cmd = ego_fut_cmd[0]
        if ego_lcf_feat is not None:
            ego_lcf_feat = ego_lcf_feat[0]

        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0],
            img=img[0],
            prev_bev=self.prev_frame_info['prev_bev'],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
            ego_status_feature=ego_status_feature,
            local_command_xy=local_command_xy,
            **kwargs
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        if self.prev_frame_num > 0:
            self.prev_frame_infos.append(self.prev_frame_info)

        return bbox_results

    def simple_test(
            self,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            img=None,
            prev_bev=None,
            points=None,
            fut_valid_flag=None,
            rescale=False,
            ego_his_trajs=None,
            ego_fut_trajs=None,
            ego_fut_cmd=None,
            ego_lcf_feat=None,
            gt_attr_labels=None,
            ego_status_feature=None,
            local_command_xy=None,
            **kwargs
    ):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict, subscores = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
            ego_status_feature=ego_status_feature,
            local_command_xy=local_command_xy
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict
        for bbox, subscore, img_meta in zip(bbox_list, subscores, img_metas):
            bbox['subscores'] = subscore
            bbox['token'] = f'{img_meta["scene_token"]}+{img_meta["frame_idx"]}'
            # cl eval
            if 'fut_valid_flag' in bbox['metric_results']:
                bbox['valid'] = bbox['metric_results']['fut_valid_flag'][0].item()
            else:
                bbox['valid'] = True
        return new_prev_bev, bbox_list

    def simple_test_pts(
            self,
            x,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev=None,
            fut_valid_flag=None,
            rescale=False,
            start=None,
            ego_his_trajs=None,
            ego_fut_trajs=None,
            ego_fut_cmd=None,
            ego_lcf_feat=None,
            gt_attr_labels=None,
            ego_status_feature=None,
            local_command_xy=None
    ):
        if isinstance(ego_status_feature, list):
            ego_status_feature = ego_status_feature[0]
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat,
                                  ego_status_feature=ego_status_feature,
                                  local_command_xy=local_command_xy)
        bbox_list, subscores = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale,
                                                             local_command_xy=local_command_xy)

        bbox_results = []
        for i, result_dict in enumerate(bbox_list):
            bbox_result = {}
            bbox_result['ego_fut_pred'] = result_dict['trajectory'].cpu()
            bbox_result['brake'] = result_dict['brake'].cpu()
            bbox_result['throttle'] = result_dict['throttle'].cpu()
            bbox_result['steer'] = result_dict['steer'].cpu()
            bbox_result['brake_next'] = result_dict['brake_next'].cpu()
            bbox_result['throttle_next'] = result_dict['throttle_next'].cpu()
            bbox_result['steer_next'] = result_dict['steer_next'].cpu()
            bbox_result['env_tokens'] = result_dict['env_tokens'].cpu()
            bbox_result['dp_seq'] = result_dict['dp_seq'].cpu()
            bbox_result['ctrl_seq'] = result_dict['ctrl_seq'].cpu()
            bbox_result['dp_throttle'] = result_dict['dp_throttle'].cpu()
            bbox_result['dp_steer'] = result_dict['dp_steer'].cpu()
            bbox_result['dp_proposals'] = result_dict['dp_proposals'].cpu()
            bbox_result['dp_proposals_next'] = result_dict['dp_proposals_next'].cpu()
            bbox_result['trajectory_soft_ensemble'] = result_dict['trajectory_soft_ensemble'].cpu()
            bbox_result['ctrl_soft_ensemble'] = result_dict['ctrl_soft_ensemble'].cpu()
            bbox_result['ctrl_waypoints'] = result_dict['ctrl_waypoints'].cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        metric_dict = {}

        if gt_attr_labels is not None:
            assert len(bbox_results) == 1, 'only support batch_size=1 now'
            with torch.no_grad():
                c_bbox_results = copy.deepcopy(bbox_results)

                bbox_result = c_bbox_results[0]
                gt_bbox = gt_bboxes_3d[0][0]
                gt_attr_label = gt_attr_labels[0][0].to('cpu')
                assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
                ego_fut_pred = bbox_result['ego_fut_pred']
                ego_fut_trajs = ego_fut_trajs[0, 0]
                ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

                metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                    pred_ego_fut_trajs=ego_fut_pred[None],
                    gt_ego_fut_trajs=ego_fut_trajs[None],
                    gt_agent_boxes=gt_bbox,
                    gt_agent_feats=gt_attr_label.unsqueeze(0),
                    fut_valid_flag=fut_valid_flag
                )
                metric_dict.update(metric_dict_planner_stp3)

        return None, bbox_results, metric_dict, subscores

    def compute_planner_metric_stp3(
            self,
            pred_ego_fut_trajs,
            gt_ego_fut_trajs,
            gt_agent_boxes,
            gt_agent_feats,
            fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s': 0,
            'plan_L2_2s': 0,
            'plan_L2_3s': 0,
            'plan_obj_col_1s': 0,
            'plan_obj_col_2s': 0,
            'plan_obj_col_3s': 0,
            'plan_obj_box_col_1s': 0,
            'plan_obj_box_col_2s': 0,
            'plan_obj_box_col_3s': 0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i + 1) * 2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                metric_dict['plan_L2_{}s'.format(i + 1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i + 1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = obj_box_coll.mean().item()
            else:
                metric_dict['plan_L2_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = 0.0

        return metric_dict

    def set_epoch(self, epoch):
        self.pts_bbox_head.epoch = epoch
