import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision.ops import sigmoid_focal_loss

from mmcv.models import BaseModule
from mmcv.models.builder import HEADS
from mmcv.models.utils.attn import MemoryEffTransformer
from mmcv.utils import force_fp32


@torch.no_grad()
def forecast_ego_vehicle(steer, throttle, brake, speed, location, heading, time_step=0.1):
    throttle = torch.clip(throttle, 0., 1.0)

    steering_gain = 0.36848336
    rear_wheel_base = 1.4178275
    throttle_values = torch.tensor([
        9.63873001e-01, 4.37535692e-04, -3.80192912e-01, 1.74950069e+00, 9.16787414e-02, -7.05461530e-02,
        -1.05996152e-03, 6.71079346e-04
    ], device=steer.device, dtype=steer.dtype)
    brake_values = torch.tensor([
        9.31711370e-03, 8.20967431e-02, -2.83832427e-03, 5.06587474e-05, -4.90357228e-07, 2.44419284e-09,
        -4.91381935e-12
    ], device=steer.device, dtype=steer.dtype)
    front_wheel_base = -0.090769015
    # 20hz, 40 frames, 2 secs
    throttle_threshold_during_forecasting = 0.3

    wheel_angle = steering_gain * steer
    slip_angle = torch.arctan(rear_wheel_base / (front_wheel_base + rear_wheel_base) *
                              torch.tan(wheel_angle))

    next_x = location[..., 0] + speed * torch.sin(heading + slip_angle) * time_step
    next_y = location[..., 1] + speed * torch.cos(heading + slip_angle) * time_step
    next_heading = heading + speed / rear_wheel_base * torch.sin(slip_angle) * time_step
    next_location = torch.cat([next_x.unsqueeze(-1), next_y.unsqueeze(-1)], -1)

    # We use different polynomial models for estimating the speed if whether the ego vehicle brakes or not.
    next_speed = torch.zeros_like(speed)

    brake_mask = brake == 1.0
    no_brake_mask = torch.logical_not(brake_mask)
    throttle_smaller_than_thresh_mask = throttle < throttle_threshold_during_forecasting
    throttle_bigger_than_thresh_mask = torch.logical_not(throttle_smaller_than_thresh_mask)
    speed_kph = speed * 3.6

    # 3 cases:
    # case 1 : brake=0, throttle < 0.3
    next_speed += torch.logical_and(no_brake_mask, throttle_smaller_than_thresh_mask) * speed

    # case 2 : brake=1
    brake_speed = speed_kph.unsqueeze(-1) ** torch.arange(1, 8, device=steer.device, dtype=steer.dtype)
    brake_speed = brake_speed @ brake_values / 3.6
    next_speed += brake_mask * brake_speed

    # case 3 : brake=0, throttle >= 0.3
    speed_feats = [
        speed_kph,
        speed_kph ** 2,
        throttle,
        throttle ** 2,
        speed_kph * throttle,
        speed_kph * throttle ** 2,
        speed_kph ** 2 * throttle,
        speed_kph ** 2 * throttle ** 2
    ]
    speed_feats = [f.unsqueeze(-1) for f in speed_feats]
    throttle_speed = torch.cat(speed_feats, -1)
    throttle_speed = throttle_speed @ throttle_values / 3.6
    next_speed += torch.logical_and(no_brake_mask, throttle_bigger_than_thresh_mask) * throttle_speed

    # end of cases
    next_speed = torch.maximum(torch.zeros_like(next_speed), next_speed)

    return next_location, next_heading, next_speed


def focal_loss_multiclass(outputs, targets, alpha, gamma):
    alpha = alpha.to(outputs.device)
    ce_loss = F.cross_entropy(outputs, targets, reduction='none', weight=alpha)
    pt = torch.exp(-ce_loss)
    focal_loss = (1 - pt) ** gamma * ce_loss
    return focal_loss


def val2onehot(cls, tensor):
    """
    Args:
        cls: [N_CLASSES]
        tensor: [B, L] / [B]
    Returns:
        [B, L, N_CLASSES] / [B, N_CLASSES]
    """
    ori_len = len(tensor.shape)
    if ori_len == 1:
        tensor = tensor.unsqueeze(1)
    indices = torch.argmin(torch.abs(tensor.unsqueeze(-1) - cls), dim=-1)
    one_hot = torch.zeros((tensor.size(0), tensor.size(1), len(cls)), device=cls.device)
    one_hot.scatter_(-1, indices.unsqueeze(-1), 1)
    if ori_len == 1:
        one_hot = one_hot.squeeze(1)
    return one_hot


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SimpleDiffusionTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dp_nlayers, input_dim, obs_len,
                 if_dp_es=False,
                 if_traj_cond=False,
                 traj_cond_k=1):
        super().__init__()
        self.if_dp_es = if_dp_es
        self.if_traj_cond = if_traj_cond
        self.dp_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), dp_nlayers
        )
        self.input_emb = nn.Linear(input_dim, d_model)
        self.time_emb = SinusoidalPosEmb(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_emb = nn.Linear(d_model, input_dim)
        token_len = obs_len + 1
        if self.if_dp_es:
            token_len += 1
        if self.if_traj_cond:
            token_len += traj_cond_k

        self.cond_pos_emb = nn.Parameter(torch.zeros(1, token_len, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, SimpleDiffusionTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(self,
                sample,
                timestep,
                cond):
        B, HORIZON, DIM = sample.shape
        sample = sample.view(B, -1).float()
        input_emb = self.input_emb(sample)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,To,n_emb)
        cond_embeddings = torch.cat([time_emb, cond], dim=1)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
                              :, :tc, :
                              ]  # each position maps to a (learnable) vector
        x = cond_embeddings + position_embeddings
        memory = x
        # (B,T_cond,n_emb)

        # decoder
        token_embeddings = input_emb.unsqueeze(1)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
                              :, :t, :
                              ]  # each position maps to a (learnable) vector
        x = token_embeddings + position_embeddings
        # (B,T,n_emb)
        x = self.dp_transformer(
            tgt=x,
            memory=memory,
        )
        # (B,T,n_emb)
        x = self.ln_f(x)
        x = self.output_emb(x)
        return x.squeeze(1).view(B, HORIZON, DIM)


def vocab2ctrl_cls_idx(vocab2ctrl, ctrl_cls, device):
    steer_vocab = torch.from_numpy(vocab2ctrl).to(device)
    # K
    steer_closest = (steer_vocab[:, None] - ctrl_cls.to(device)).abs().argmin(-1)
    return steer_closest


def ctrlseq2waypoints(ctrlseq, velo, timestep):
    ctrlseq = ctrlseq.cpu().float()
    velo = velo.cpu().float()
    NUM_PROPOSALS = ctrlseq.shape[0]
    DP_HORIZON = ctrlseq.shape[1]
    next_location = torch.zeros((NUM_PROPOSALS, 1, 2)).float()
    next_heading = torch.zeros((NUM_PROPOSALS, 1)).float()
    next_speed = velo.repeat(NUM_PROPOSALS, 1)

    dp_locations = []
    for t in range(DP_HORIZON):
        control = ctrlseq[:, t]
        next_location, next_heading, next_speed = forecast_ego_vehicle(
            control[..., 2:3],
            control[..., 1:2],
            control[..., 0:1],
            next_speed,
            next_location,
            next_heading,
            time_step=timestep
        )
        dp_locations.append(next_location)
    # [B, 5, 2]
    dp_locations = torch.cat(dp_locations, 1)
    return dp_locations


@HEADS.register_module()
class HydraNext_head(BaseModule):
    def __init__(self,
                 queue_length,
                 ctrl_steps_dp,
                 # input
                 img_vert_anchors, img_horz_anchors, num_views, ego_status_feature_num,
                 # downscale & transformers
                 img_feat_c, d_model, nhead, d_ffn, nlayers,
                 # vocab
                 vocab_path, num_poses,
                 # vad
                 pc_range, fut_ts=6, valid_fut_ts=6,
                 # inference
                 heads=[],
                 inference_weights=dict(),
                 # loss
                 loss_weights=dict(), loss_weights_control=dict(),
                 col_path='', ctrl_drop=None, ctrl_steps=4,
                 ctrl_nlayers=3,
                 dp_nlayers=3,
                 if_focal_steerthrottle=False,
                 if_seed=False, if_decay=False,
                 ctrl_es=False, if_dp_es=False,
                 if_traj_cond=False,
                 traj_cond_k=1, topk=100, num_proposals=10,
                 **kwargs):
        super().__init__()
        self.num_proposals = num_proposals
        self.topk = topk
        self.pid_window_turn = None
        self.pid_window_speed = None
        self.traj_cond_k = traj_cond_k
        self.if_dp_es = if_dp_es
        self.if_traj_cond = if_traj_cond
        self.ctrl_steps_dp = ctrl_steps_dp
        self.if_seed = if_seed
        self.if_decay = if_decay
        self.d_model = d_model
        if self.if_seed:
            self.rand_enc = nn.Sequential(*[
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model)
            ])

        self.if_focal_steerthrottle = if_focal_steerthrottle
        self.queue_length = queue_length
        self.ctrl_steps = ctrl_steps
        self.inference_weights = inference_weights
        self.loss_weights = loss_weights
        self.loss_weights_control = loss_weights_control

        self.per_img_len = img_vert_anchors * img_horz_anchors
        self.num_views = num_views
        self.pc_range = pc_range

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (img_vert_anchors, img_horz_anchors)
        )
        self._keyval_embedding = nn.Embedding(
            img_vert_anchors * img_horz_anchors * num_views, d_model
        )
        self.downscale_layer = nn.Conv2d(img_feat_c, d_model, kernel_size=1)
        self._status_encoding = nn.Linear(ego_status_feature_num, d_model)

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), nlayers
        )
        self.transformer_ctrl = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), ctrl_nlayers
        )
        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(vocab_path)).float(),
            requires_grad=False
        )
        # ctrl branch
        self.ctrl_query = nn.Embedding(
            ctrl_steps, d_model
        )
        self.throttle_cls = torch.tensor(
            [0., 0.3, 0.5, 0.7, 1.],
            dtype=torch.float32
        )
        self.steer_cls = torch.tensor(
            [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 0.7],
            dtype=torch.float32
        )
        self.throttle_cls_weight = torch.tensor([
            0.5766136074183559, 0.9402093121323514, 0.9952421316272592, 0.7032529848489366,
            0.784681963973097
        ], dtype=torch.float32)
        self.steer_cls_weight = torch.tensor([
            0.96399013498375, 0.9753161618441958, 0.9568852787207851, 0.848646967870545, 0.4081969936405573,
            0.9127099404095089, 0.9466070903739421, 0.9985475084018759, 0.9890999237548399
        ], dtype=torch.float32)

        self.ctrl_heads = nn.ModuleDict({
            'brake': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'throttle': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, len(self.throttle_cls)),
            ),
            'steer': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, len(self.steer_cls)),
            )
        })
        if ctrl_drop is not None:
            for _, head in self.ctrl_heads.items():
                head.insert(2, nn.Dropout(ctrl_drop))

        self.hydra_heads = nn.ModuleDict({
            'imi': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )
        })
        for head in heads:
            self.hydra_heads[head] = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )

        self.head_keys = list(self.hydra_heads.keys())
        self.traj_encoder = MemoryEffTransformer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0
        )
        self.traj_embed = nn.Sequential(
            nn.Linear(num_poses * 2, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        if len(heads) > 0:
            self.gt_subscore = pickle.load(open(col_path, 'rb'))
            # preprocess col train pkl,
            # 1: collision, 0: noc -> 1: noc, 0: collision
            for k, v in self.gt_subscore.items():
                self.gt_subscore[k]['col'] = 1 - v['col']

        self.fut_ts = fut_ts
        self.valid_fut_ts = valid_fut_ts
        self.kv_prev = None

        if self.queue_length > 1:
            self.temporal_embedding = nn.Embedding(queue_length, d_model)
            self.temporal_fusion = nn.MultiheadAttention(
                d_model, nhead, batch_first=True
            )
        # dp branch
        self.dp_transformer = SimpleDiffusionTransformer(
            d_model, nhead, d_ffn, dp_nlayers,
            input_dim=2 * self.ctrl_steps_dp,
            obs_len=img_vert_anchors * img_horz_anchors * num_views,
            if_dp_es=if_dp_es,
            if_traj_cond=if_traj_cond,
            traj_cond_k=traj_cond_k
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',
            clip_sample=True,
            prediction_type='epsilon'
        )
        if if_traj_cond:
            self.traj_cond_embed = nn.Sequential(
                nn.Linear(num_poses * 2, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps

    def build_kv_1f(self, feat, num_cam):
        kv = []
        # multi-view cams -> blc
        for cam_idx in range(num_cam):
            img_feat_curr = feat[:, cam_idx]
            img_feat_curr = self.avgpool_img(img_feat_curr)
            img_feat_curr = self.downscale_layer(img_feat_curr)
            # bchw->bcl->blc
            img_feat_curr = img_feat_curr.flatten(-2, -1).permute(0, 2, 1)
            kv.append(img_feat_curr)
        kv = torch.cat(kv, 1).contiguous()
        kv += self._keyval_embedding.weight[None, ...]
        return kv

    def fuse(self, kv_prev, kv_curr):
        if kv_prev is None:
            kv_prev = kv_curr

        # first frame in inference
        kv_prev += self.temporal_embedding.weight[0].unsqueeze(0).unsqueeze(0)
        kv_curr += self.temporal_embedding.weight[1].unsqueeze(0).unsqueeze(0)
        kv_all = torch.cat([kv_prev, kv_curr], 1).contiguous()
        kv_all = self.temporal_fusion(
            kv_all, kv_all, kv_all, need_weights=False,
        )
        return kv_all[0]

    def save_hist(self, kv_prev):
        # called by team agent
        self.kv_prev = kv_prev

    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                ego_status_feature=None,
                local_command_xy=None
                ):
        if len(mlvl_feats) == 2:
            # training, we have hist_feats
            hist_feats, img_feats = mlvl_feats
            hist_feats = hist_feats[0]
            img_feats = img_feats[0]
            B, _, num_cam, _, _, _ = hist_feats.shape
            # currently queue_len = 1
            self.kv_prev = self.build_kv_1f(hist_feats[:, 0], num_cam)
        else:
            # inference, hist_feats is not available
            img_feats = mlvl_feats[0]
            B, num_cam, _, _, _ = img_feats.shape

        kv_curr = self.build_kv_1f(img_feats, num_cam)
        # temporal fusion
        if self.queue_length > 1:
            kv = self.fuse(self.kv_prev, kv_curr)
        else:
            kv = kv_curr
        dtype = img_feats.dtype

        # embed vocab
        vocab = self.vocab.data
        L, HORIZON, _ = vocab.shape
        embedded_vocab = self.traj_embed(vocab.view(L, -1))[None]
        embedded_vocab = self.traj_encoder(embedded_vocab).repeat(B, 1, 1)

        es_feat = self._status_encoding(ego_status_feature.to(dtype)).unsqueeze(1)

        # transformer decoder traj
        traj_tr_out = self.transformer(embedded_vocab, kv)

        ctrl_queries = self.ctrl_query.weight.unsqueeze(0).repeat(B, 1, 1)
        ctrl_tr_out = self.transformer_ctrl(
            ctrl_queries,
            kv
        )
        result = {}
        # ctrl heads
        for k, head in self.ctrl_heads.items():
            if k == 'brake':
                # focal loss no sigmoid for brake
                result[k] = head(ctrl_tr_out)
            else:
                result[k] = head(ctrl_tr_out)

        # hydra heads
        dist_status = traj_tr_out + es_feat
        for k, head in self.hydra_heads.items():
            if k == 'imi':
                result[k] = head(dist_status).squeeze(-1)
            else:
                result[k] = head(dist_status).squeeze(-1).sigmoid()
        scores = self.inference_weights['imi'] * result['imi'].softmax(-1).log()
        for k in self.head_keys:
            if k == 'imi':
                continue
            scores += self.inference_weights[k] * result[k].log()

        selected_indices = scores.argmax(1)
        result["trajectory"] = self.vocab.data[selected_indices]
        result["trajectory_vocab"] = self.vocab.data
        result["selected_indices"] = selected_indices
        result["ego_status_feature"] = ego_status_feature
        result["traj_scores"] = scores
        result["env_tokens"] = kv_curr
        if self.if_dp_es:
            result["env_tokens"] = torch.cat([
                result["env_tokens"],
                es_feat
            ], 1)
        if self.if_traj_cond:
            # B, K, 6, 2
            topk_trajs = self.vocab.data[scores.topk(k=self.traj_cond_k, sorted=True, dim=1).indices]
            # topk
            result["env_tokens"] = torch.cat([
                result["env_tokens"],
                self.traj_cond_embed(topk_trajs.view(B, self.traj_cond_k, -1))
            ], 1)
        # dp part
        if not self.training:
            NUM_PROPOSALS = self.num_proposals
            condition = result["env_tokens"].repeat(NUM_PROPOSALS, 1, 1)
            noise = torch.randn(
                size=(NUM_PROPOSALS, self.ctrl_steps_dp, 2),
                dtype=condition.dtype,
                device=condition.device,
            )

            # set step values
            self.noise_scheduler.set_timesteps(self.num_inference_steps)

            for t in self.noise_scheduler.timesteps:
                # 2. predict model output
                model_output = self.dp_transformer(
                    noise,
                    t,
                    condition
                )

                # 3. compute previous image: x_t -> x_t-1
                noise = self.noise_scheduler.step(
                    model_output, t, noise,
                ).prev_sample
            result['dp_output'] = noise
        return result

    def loss_planning(self,
                      vocab,
                      subscores,
                      ego_fut_gt,
                      ego_fut_masks,
                      ego_fut_cmd,
                      tokens):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, ego_fut_mode, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, ego_fut_mode]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, fut_mode]
        Returns:
            loss_plan_reg (Tensor): planning reg loss.
            loss_plan_bound (Tensor): planning map boundary constraint loss.
            loss_plan_col (Tensor): planning col constraint loss.
            loss_plan_dir (Tensor): planning directional constraint loss.
        """
        B = ego_fut_cmd.shape[0]
        ego_fut_gt = ego_fut_gt.cumsum(-2)
        valid_mask = ego_fut_masks.all(1)

        l2_distance = -(
                (vocab[:, ][None].repeat(B, 1, 1, 1) - ego_fut_gt[:, None]) ** 2
        ) / 0.5
        imi_loss = F.cross_entropy(
            subscores['imi'],
            l2_distance.sum((-2, -1)).softmax(1),
            reduction='none'
        )
        imi_loss = (imi_loss * valid_mask).sum() / valid_mask.shape[0]

        loss_plan_dict = {
            'loss_plan_imi': self.loss_weights['imi'] * imi_loss,
        }
        # get the scores
        for k in self.head_keys:
            if k == 'imi':
                continue
            gt_col_raw_scores = [self.gt_subscore[token][k][None] for token in tokens]
            gt_col_scores = (torch.from_numpy(
                np.concatenate(gt_col_raw_scores, axis=0)
            )).to(subscores[k].dtype).to(subscores[k].device)

            col_loss = F.binary_cross_entropy(
                subscores[k],
                gt_col_scores
            )
            col_loss = (col_loss * valid_mask).sum() / valid_mask.shape[0]
            loss_plan_dict[f'loss_plan_{k}'] = self.loss_weights[k] * col_loss

        return loss_plan_dict

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_brake=None,
             gt_throttle=None,
             gt_steer=None,
             ctrl_seq=None,  # brake, throttle, steer
             ctrl_mask=None,
             ctrl_seq_dp=None,
             ctrl_mask_dp=None,
             env_tokens=None,
             img_metas=None
             ):
        device = gt_brake.device

        ctrl_seq = ctrl_seq.float()
        gt_brake_seq = ctrl_seq[..., 0]
        gt_throttle_seq = val2onehot(self.throttle_cls.to(device), ctrl_seq[..., 1])
        gt_steer_seq = val2onehot(self.steer_cls.to(device), ctrl_seq[..., 2])

        B = ego_fut_gt.shape[0]
        loss_dict = dict()
        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        vocab = preds_dicts['trajectory_vocab']
        subscores = {
            k: preds_dicts[k] for k in self.head_keys
        }
        tokens = [
            f'{meta["scene_token"]}+{meta["frame_idx"]}' for meta in img_metas
        ]
        loss_plan_input = [
            vocab,
            subscores,
            ego_fut_gt,
            ego_fut_masks,
            ego_fut_cmd,
            tokens,
        ]

        loss_planning_dict = self.loss_planning(*loss_plan_input)
        loss_dict.update(loss_planning_dict)

        if self.if_decay:
            decay_factor = 0.99
            decay_weights = torch.tensor([decay_factor ** i for i in range(20)],
                                         device=device)
            decay_weights = decay_weights.unsqueeze(0).repeat(B, 1)
        else:
            decay_weights = torch.ones_like(ctrl_mask)

        # ctrl_seq: [B, 4, 3]: brake, throttle, steer
        loss_dict['loss_brake'] = (sigmoid_focal_loss(
            preds_dicts['brake'].squeeze(-1), gt_brake_seq,
            alpha=0.728,  # for brake=1
            gamma=2,
            reduction='none'
        ) * ctrl_mask * decay_weights).sum() * self.loss_weights_control['brake'] / B
        if self.if_focal_steerthrottle:
            # focal loss
            loss_dict['loss_throttle'] = (focal_loss_multiclass(
                preds_dicts['throttle'].permute(0, 2, 1).contiguous(),
                gt_throttle_seq.permute(0, 2, 1).contiguous(),
                gamma=2, alpha=self.throttle_cls_weight
            ) * ctrl_mask * decay_weights).sum() * self.loss_weights_control['throttle'] / B
            loss_dict['loss_steer'] = (focal_loss_multiclass(
                preds_dicts['steer'].permute(0, 2, 1).contiguous(),
                gt_steer_seq.permute(0, 2, 1).contiguous(),
                gamma=2, alpha=self.steer_cls_weight
            ) * ctrl_mask * decay_weights).sum() * self.loss_weights_control['steer'] / B
        else:
            # ce loss
            loss_dict['loss_throttle'] = (F.cross_entropy(
                preds_dicts['throttle'].permute(0, 2, 1).contiguous(),
                gt_throttle_seq.permute(0, 2, 1).contiguous(),
                reduction='none'
            ) * ctrl_mask * decay_weights).sum() * self.loss_weights_control['throttle'] / B
            loss_dict['loss_steer'] = (F.cross_entropy(
                preds_dicts['steer'].permute(0, 2, 1).contiguous(),
                gt_steer_seq.permute(0, 2, 1).contiguous(),
                reduction='none'
            ) * ctrl_mask * decay_weights).sum() * self.loss_weights_control['steer'] / B

        # Sample noise that we'll add to the images
        # preprocess ctrl_seq_dp into B, 5, 2
        dp_brake_seq = ctrl_seq_dp[..., 0]
        dp_throttle_seq = ctrl_seq_dp[..., 1]
        dp_steer_seq = ctrl_seq_dp[..., 2]
        dp_throttle_seq = torch.where(
            dp_brake_seq.bool(),
            -1 * torch.ones_like(dp_throttle_seq),
            dp_throttle_seq
        )
        ctrl_seq_dp_input = torch.cat([
            dp_throttle_seq.unsqueeze(-1),
            dp_steer_seq.unsqueeze(-1)
        ], -1)
        noise = torch.randn(ctrl_seq_dp_input.shape, device=device)
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_dp_input = self.noise_scheduler.add_noise(
            ctrl_seq_dp_input, noise, timesteps
        )

        # Predict the noise residual
        pred = self.dp_transformer(
            noisy_dp_input,
            timesteps,
            env_tokens
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss_dp = F.mse_loss(pred, target, reduction='none')
        loss_dp = loss_dp * ctrl_mask_dp.unsqueeze(-1)
        loss_dp = loss_dp.sum() * self.loss_weights_control['dp'] / B
        loss_dict['loss_dp'] = loss_dp
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False, local_command_xy=None):
        ret_list = []
        subscores = []
        num_samples = preds_dicts['trajectory'].shape[0]
        device = preds_dicts['trajectory'].device
        if isinstance(local_command_xy, list):
            local_command_xy = local_command_xy[0]

        for i in range(num_samples):
            brake_dp = (preds_dicts['dp_output'][:, 0, 0] < -0.5).float().unsqueeze(-1)
            throttle_dp = preds_dicts['dp_output'][:, 0, 0].clamp(0.0, 1.0).unsqueeze(-1)
            steer_dp = preds_dicts['dp_output'][:, 0, 1].clamp(-1.0, 1.0).unsqueeze(-1)
            dp_proposals = torch.cat([
                brake_dp, throttle_dp, steer_dp
            ], -1)
            brake_dp2 = (preds_dicts['dp_output'][:, 1, 0] < -0.5).float().unsqueeze(-1)
            throttle_dp2 = preds_dicts['dp_output'][:, 1, 0].clamp(0.0, 1.0).unsqueeze(-1)
            steer_dp2 = preds_dicts['dp_output'][:, 1, 1].clamp(-1.0, 1.0).unsqueeze(-1)
            dp_proposals2 = torch.cat([
                brake_dp2, throttle_dp2, steer_dp2
            ], -1)

            brake_dp_seq = (preds_dicts['dp_output'][:, :, 0] < -0.5).float().unsqueeze(-1)
            throttle_dp_seq = preds_dicts['dp_output'][:, :, 0].clamp(0.0, 1.0).unsqueeze(-1)
            steer_dp_seq = preds_dicts['dp_output'][:, :, 1].clamp(-1.0, 1.0).unsqueeze(-1)
            dp_seq = torch.cat([
                brake_dp_seq, throttle_dp_seq, steer_dp_seq
            ], -1)

            brake_ctrl_seq = (preds_dicts['brake'][i].sigmoid() > 0.6).float()
            throttle_ctrl_seq = self.throttle_cls.cuda()[preds_dicts['throttle'][i].argmax(-1)].unsqueeze(-1)
            throttle_ctrl_seq = torch.where(
                brake_ctrl_seq.bool(),
                torch.zeros_like(throttle_ctrl_seq),
                throttle_ctrl_seq
            )
            steer_ctrl_seq_1st = self.steer_cls.cuda()[preds_dicts['steer'][i].argmax(-1)].unsqueeze(-1)
            ctrl_seq = torch.cat([
                brake_ctrl_seq, throttle_ctrl_seq, steer_ctrl_seq_1st
            ], -1)

            ctrl_proposals = torch.cat([
                ctrl_seq[None],
            ], 0)
            if ctrl_seq.shape[0] == 4:
                interpolated_ctrl_proposals = F.interpolate(
                    ctrl_proposals.permute(0, 2, 1),
                    size=15,
                    mode='linear',
                    align_corners=True
                ).permute(0, 2, 1)
            else:
                interpolated_ctrl_proposals = ctrl_proposals
            speed = preds_dicts['ego_status_feature'][..., 0:1]
            # B, 20, 2
            ctrl_waypoints = ctrlseq2waypoints(interpolated_ctrl_proposals, speed, 0.1).to(device)

            if dp_seq.shape[1] == 3:
                dp_seq = F.interpolate(
                    dp_seq.permute(0, 2, 1),
                    size=10,
                    mode='linear',
                    align_corners=True
                ).permute(0, 2, 1)

            ctrl_waypoints2traj_dist = (
                    ((ctrl_waypoints[:, 9] - preds_dicts['trajectory'][:, 1]) ** 2).sum(-1) +
                    ((ctrl_waypoints[:, 4] - preds_dicts['trajectory'][:, 0]) ** 2).sum(-1)
            )
            ctrl_soft_ensemble = ctrl_proposals[ctrl_waypoints2traj_dist.argmin(0)]
            ret_list.append({
                'ctrl_waypoints': ctrl_waypoints,
                'ctrl_soft_ensemble': ctrl_soft_ensemble,
                'trajectory_soft_ensemble': torch.ones_like(ctrl_soft_ensemble),
                'trajectory': preds_dicts['trajectory'][i],
                'brake': preds_dicts['brake'][i, 0].sigmoid(),
                'throttle': preds_dicts['throttle'][i, 0],
                'steer': preds_dicts['steer'][i, 0],
                'brake_next': preds_dicts['brake'][i, 1].sigmoid(),
                'throttle_next': preds_dicts['throttle'][i, 1],
                'steer_next': preds_dicts['steer'][i, 1],
                'dp_throttle': preds_dicts['dp_output'][i, 0, 0],
                'dp_steer': preds_dicts['dp_output'][i, 0, 1],
                'dp_proposals': dp_proposals,
                'dp_proposals_next': dp_proposals2,
                'dp_seq': dp_seq,
                'ctrl_seq': ctrl_seq,
                'env_tokens': preds_dicts['env_tokens'][i]
            })
        for i in range(num_samples):
            subscores.append({
                k: preds_dicts[k][i] for k in self.head_keys
            })
        return ret_list, subscores
