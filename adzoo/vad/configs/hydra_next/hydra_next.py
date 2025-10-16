_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection

NameMapping = {
    # =================vehicle=================
    # bicycle
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    # car
    "vehicle.chevrolet.impala": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.dodge.charger_police": 'car',
    "vehicle.dodge.charger_police_2020": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.lincoln.mkz_2020": 'car',
    "vehicle.mini.cooper_s_2021": 'car',
    "vehicle.mercedes.coupe_2020": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.nissan.patrol_2021": 'car',
    "vehicle.audi.tt": 'car',
    "vehicle.audi.etron": 'car',
    "vehicle.ford.crown": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.tesla.model3": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": 'car',
    # bus
    # van
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
    "vehicle.ford.ambulance": "van",
    # truck
    "vehicle.carlamotors.firetruck": 'truck',
    # =========================================

    # =================traffic sign============
    # traffic.speed_limit
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.speed_limit.40": 'traffic_sign',
    "traffic.speed_limit.50": 'traffic_sign',
    "traffic.speed_limit.60": 'traffic_sign',
    "traffic.speed_limit.90": 'traffic_sign',
    "traffic.speed_limit.120": 'traffic_sign',

    "traffic.stop": 'traffic_sign',
    "traffic.yield": 'traffic_sign',
    "traffic.traffic_light": 'traffic_light',
    # =========================================

    # ===================Construction===========
    "static.prop.warningconstruction": 'traffic_cone',
    "static.prop.warningaccident": 'traffic_cone',
    "static.prop.trafficwarning": "traffic_cone",

    # ===================Construction===========
    "static.prop.constructioncone": 'traffic_cone',

    # =================pedestrian==============
    "walker.pedestrian.0001": 'pedestrian',
    "walker.pedestrian.0003": 'pedestrian',
    "walker.pedestrian.0004": 'pedestrian',
    "walker.pedestrian.0005": 'pedestrian',
    "walker.pedestrian.0007": 'pedestrian',
    "walker.pedestrian.0010": 'pedestrian',
    "walker.pedestrian.0013": 'pedestrian',
    "walker.pedestrian.0014": 'pedestrian',
    "walker.pedestrian.0015": 'pedestrian',
    "walker.pedestrian.0016": 'pedestrian',
    "walker.pedestrian.0017": 'pedestrian',
    "walker.pedestrian.0018": 'pedestrian',
    "walker.pedestrian.0019": 'pedestrian',
    "walker.pedestrian.0020": 'pedestrian',
    "walker.pedestrian.0021": 'pedestrian',
    "walker.pedestrian.0022": 'pedestrian',
    "walker.pedestrian.0025": 'pedestrian',
    "walker.pedestrian.0027": 'pedestrian',
    "walker.pedestrian.0030": 'pedestrian',
    "walker.pedestrian.0031": 'pedestrian',
    "walker.pedestrian.0032": 'pedestrian',
    "walker.pedestrian.0034": 'pedestrian',
    "walker.pedestrian.0035": 'pedestrian',
    "walker.pedestrian.0041": 'pedestrian',
    "walker.pedestrian.0042": 'pedestrian',
    "walker.pedestrian.0046": 'pedestrian',
    "walker.pedestrian.0047": 'pedestrian',

    # ==========================================
    "static.prop.dirtdebris01": 'others',
    "static.prop.dirtdebris02": 'others',
}

eval_cfg = {
    "eval_planning_only": True,
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "mean_ap_weight": 5,
    "class_names": ['car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian'],
    "tp_metrics": ['trans_err', 'scale_err', 'orient_err', 'vel_err'],
    "err_name_maping": {'trans_err': 'mATE', 'scale_err': 'mASE', 'orient_err': 'mAOE', 'vel_err': 'mAVE',
                        'attr_err': 'mAAE'},
    "class_range": {'car': (50, 50), 'van': (50, 50), 'truck': (50, 50), 'bicycle': (40, 40), 'traffic_sign': (30, 30),
                    'traffic_cone': (30, 30), 'traffic_light': (30, 30), 'pedestrian': (40, 40)}
}

class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian', 'others'
]
num_classes = len(class_names)

# map has classes: divider, ped_crossing, boundary
map_classes = ['Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20  # now only support fixed_pts > 0
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)
past_frames = 2
future_frames = 6
ego_status_feature_num = 7
cam_names = ['CAM_FRONT', 'CAM_BACK']

vocab_path = './traj_final/t2d/4096_kmeans_t2d.npy'
col_path = './data/infos/train_4096_t2d_latestv2.pkl'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 1  # each sequence contains `queue_length` frames.

ctrl_sample_interval_dp = 1
ctrl_steps_dp = 10

ctrl_steps = 4
total_epochs = 20
samples_per_gpu = 32
lr = 2e-4

model = dict(
    type='HydraNext',
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch'
    ),
    pts_bbox_head=dict(
        type='HydraNext_head',
        ctrl_steps_dp=ctrl_steps_dp,
        queue_length=queue_length,
        img_vert_anchors=23,
        img_horz_anchors=40,
        # vel (1), acc (3), steer (1), cmd (6)
        ego_status_feature_num=ego_status_feature_num,
        num_views=len(cam_names),
        img_feat_c=2048,
        d_model=256,
        nhead=8,
        d_ffn=1024,
        nlayers=3,
        vocab_path=vocab_path,
        col_path=col_path,
        pc_range=point_cloud_range,
        num_poses=6,
        valid_fut_ts=6,
        heads=['col', 'dir', 'ep'],
        inference_weights=dict(
            imi=0.5,
            col=0.1,
            dir=0.1,
            ep=0.1
        ),
        loss_weights=dict(
            imi=1,
            col=2,
            dir=1,
            ep=1
        ),
        loss_weights_control=dict(
            brake=5,
            throttle=1,
            steer=1,
            dp=1
        ),
        ctrl_drop=0.3,
        ctrl_steps=ctrl_steps,
        ctrl_nlayers=3,
        dp_nlayers=5
    ))

dataset_type = "B2D_VAD_Dataset"
data_root = "data/bench2drive"
info_root = "data/infos"
map_root = "data/bench2drive/maps"
map_file = "data/infos/b2d_map_infos.pkl"
file_client_args = dict(backend="disk")
ann_file_train = info_root + f"/b2d_infos_train.pkl"
ann_file_val = info_root + f"/b2d_infos_val.pkl"
ann_file_test = info_root + f"/b2d_infos_val.pkl"

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='VADFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs', 'gt_attr_labels', 'ego_fut_trajs',
               'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'ego_status_feature',
               'gt_brake', 'gt_throttle', 'gt_steer', 'ctrl_seq', 'ctrl_mask',
               'ctrl_seq_dp', 'ctrl_mask_dp', 'local_command_xy'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='VADFormatBundle3D', class_names=class_names, with_label=False, with_ego=True),
            dict(type='CustomCollect3D',
                 keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                       'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
                       'ego_lcf_feat', 'gt_attr_labels', 'ego_status_feature',
                       'local_command_xy'])])
]

inference_only_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='VADFormatBundle3D', class_names=class_names, with_label=False, with_ego=True),
            dict(type='CustomCollect3D', keys=['img', 'ego_fut_cmd',
                                               'ego_status_feature',
                                               'local_command_xy'])])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=6,
    train=dict(

        type=dataset_type,
        ctrl_steps=ctrl_steps,
        ego_status_feature_num=ego_status_feature_num,
        data_root=data_root,
        ctrl_sample_interval_dp=ctrl_sample_interval_dp,
        ctrl_steps_dp=ctrl_steps_dp,
        cam_names=cam_names,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        name_mapping=NameMapping,
        map_root=map_root,
        map_file=map_file,
        modality=input_modality,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        past_frames=past_frames,
        future_frames=future_frames,
        point_cloud_range=point_cloud_range,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        # custom_eval_version='vad_nusc_detection_cvpr_2019'
    ),
    val=dict(type=dataset_type,
             ctrl_steps=ctrl_steps,
             data_root=data_root,
             ego_status_feature_num=ego_status_feature_num,

             ctrl_sample_interval_dp=ctrl_sample_interval_dp,
             ctrl_steps_dp=ctrl_steps_dp,
             cam_names=cam_names,
             ann_file=ann_file_val,
             pipeline=test_pipeline,
             classes=class_names,
             name_mapping=NameMapping,
             map_root=map_root,
             map_file=map_file,
             modality=input_modality,
             bev_size=(bev_h_, bev_w_),
             queue_length=queue_length,
             past_frames=past_frames,
             future_frames=future_frames,
             point_cloud_range=point_cloud_range,
             polyline_points_num=map_fixed_ptsnum_per_gt_line,
             eval_cfg=eval_cfg
             ),
    test=dict(type=dataset_type,
              ctrl_steps=ctrl_steps,
              ctrl_sample_interval_dp=ctrl_sample_interval_dp,
              ctrl_steps_dp=ctrl_steps_dp,
              data_root=data_root,
              ego_status_feature_num=ego_status_feature_num,

              cam_names=cam_names,
              ann_file=ann_file_val,
              pipeline=test_pipeline,
              classes=class_names,
              name_mapping=NameMapping,
              map_root=map_root,
              map_file=map_file,
              modality=input_modality,
              bev_size=(bev_h_, bev_w_),
              queue_length=queue_length,
              past_frames=past_frames,
              future_frames=future_frames,
              point_cloud_range=point_cloud_range,
              polyline_points_num=map_fixed_ptsnum_per_gt_line,
              eval_cfg=eval_cfg
              ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=lr,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    by_epoch=False,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

evaluation = dict(interval=total_epochs + 100, pipeline=test_pipeline, metric='bbox', map_metric='chamfer')

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)
custom_hooks = [dict(type='CustomSetEpochInfoHook')]
