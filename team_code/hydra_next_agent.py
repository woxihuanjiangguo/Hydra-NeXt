import datetime
import json
import math
import os
import pathlib
import time
from collections import deque

import carla
import cv2
import numpy as np
import torch
from PIL import Image
from leaderboard.autoagents import autonomous_agent
from pyquaternion import Quaternion
from scipy.optimize import fsolve
from torchvision import transforms as T

from mmcv import Config
from mmcv.core.bbox import get_box_type
from mmcv.datasets.pipelines import Compose
from mmcv.models import build_model
from mmcv.parallel.collate import collate as mm_collate_to_batch_form
from mmcv.utils import (load_checkpoint)
from pid_controller import PIDController
from planner import RoutePlanner

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

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
    throttle_threshold_during_forecasting = 0.3

    wheel_angle = steering_gain * steer
    slip_angle = torch.arctan(rear_wheel_base / (front_wheel_base + rear_wheel_base) *
                              torch.tan(wheel_angle))

    next_x = location[..., 0] + speed * torch.sin(heading + slip_angle) * time_step
    next_y = location[..., 1] + speed * torch.cos(heading + slip_angle) * time_step
    next_heading = heading + speed / rear_wheel_base * torch.sin(slip_angle) * time_step
    next_location = torch.cat([next_x.unsqueeze(-1), next_y.unsqueeze(-1)], -1)

    next_speed = torch.zeros_like(speed)

    brake_mask = brake == 1.0
    no_brake_mask = torch.logical_not(brake_mask)
    throttle_smaller_than_thresh_mask = throttle < throttle_threshold_during_forecasting
    throttle_bigger_than_thresh_mask = torch.logical_not(throttle_smaller_than_thresh_mask)
    speed_kph = speed * 3.6

    next_speed += torch.logical_and(no_brake_mask, throttle_smaller_than_thresh_mask) * speed
    brake_speed = speed_kph.unsqueeze(-1) ** torch.arange(1, 8, device=steer.device, dtype=steer.dtype)
    brake_speed = brake_speed @ brake_values / 3.6
    next_speed += brake_mask * brake_speed
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

    next_speed = torch.maximum(torch.zeros_like(next_speed), next_speed)
    return next_location, next_heading, next_speed



def control_diff(control1, control2):
    return np.sum(np.abs(control2 - control1), -1)


def get_entry_point():
    return 'HydraNextAgent'


def find_closest_dp_traj(traj, dp_seq, velo, timestep=0.1):
    # traj: [B, 6, 2]
    traj = traj.unsqueeze(0)

    NUM_DP = dp_seq.shape[0]
    DP_HORIZON = dp_seq.shape[1]
    next_location = torch.zeros((NUM_DP, 1, 2)).float()
    next_heading = torch.zeros((NUM_DP, 1)).float()
    next_speed = velo.repeat(NUM_DP, 1)

    dp_locations = []
    for t in range(DP_HORIZON):
        control = dp_seq[:, t]
        next_location, next_heading, next_speed = forecast_ego_vehicle(
            control[..., 2].unsqueeze(-1),
            control[..., 1].unsqueeze(-1),
            control[..., 0].unsqueeze(-1),
            next_speed,
            next_location,
            next_heading,
            time_step=timestep
        )
        dp_locations.append(next_location)
    # [B, 10, 2]
    dp_locations = torch.cat(dp_locations, 1)

    traj_closest = (((dp_locations[:, -1] - traj[:, 1]) ** 2).sum(-1) +
                    ((dp_locations[:, 4] - traj[:, 0]) ** 2).sum(-1)).argmin(0)
    dp_action = dp_seq[traj_closest, 0]
    return dp_action



def find_closest_dp_ctrlwaypoints(ctrl_waypoints, dp_seq, velo, timestep=0.1):
    # traj: [1, 15, 2]
    traj = ctrl_waypoints

    NUM_DP = dp_seq.shape[0]
    DP_HORIZON = dp_seq.shape[1]
    next_location = torch.zeros((NUM_DP, 1, 2)).float()
    next_heading = torch.zeros((NUM_DP, 1)).float()
    next_speed = velo.repeat(NUM_DP, 1)

    dp_locations = []
    for t in range(DP_HORIZON):
        control = dp_seq[:, t]
        next_location, next_heading, next_speed = forecast_ego_vehicle(
            control[..., 2].unsqueeze(-1),
            control[..., 1].unsqueeze(-1),
            control[..., 0].unsqueeze(-1),
            next_speed,
            next_location,
            next_heading,
            time_step=timestep
        )
        dp_locations.append(next_location)
    # [B, 10, 2]
    dp_locations = torch.cat(dp_locations, 1)

    traj_closest = (((dp_locations[:, 9] - traj[:, 9]) ** 2).sum(-1) +
                    ((dp_locations[:, 4] - traj[:, 4]) ** 2).sum(-1)).argmin(0)
    dp_action = dp_seq[traj_closest, 0]
    return dp_action


class HistFeatsQueue:
    def __init__(self):
        self.queue = deque()
        self.current_time = -1

    def push(self, item):
        self.queue.append(item)
        self.current_time += 1
        # Remove items older than current_time - 10
        if self.current_time >= 10:
            self.queue.popleft()

    def is_empty(self):
        return len(self.queue) == 0

    def fetch(self):
        return self.queue[0]


class HydraNextAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steer = 0
        # ori
        self.pidcontroller = PIDController()

        self.config_path = path_to_conf_file.split('+')[0]
        self.ckpt_path = path_to_conf_file.split('+')[1]
        self.save_name = path_to_conf_file.split('+')[-1]
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        cfg = Config.fromfile(self.config_path)
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    plugin_dir = os.path.join("Bench2DriveZoo", plugin_dir)
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)

        self.cfg = cfg
        self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu', strict=True)
        self.model.cuda()
        self.model.eval()
        self.inference_only_pipeline = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
            if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph',
                                                       'LoadMultiViewImageFromFiles']:
                self.inference_only_pipeline.append(inference_only_pipeline)
        self.inference_only_pipeline = Compose(self.inference_only_pipeline)

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        self._im_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.lat_ref, self.lon_ref = 42.0, 2.0

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        self.prev_control = control
        self.prev_control_cache = []
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'rgb_front_right').mkdir()
            (self.save_path / 'rgb_front_left').mkdir()
            (self.save_path / 'rgb_back').mkdir()
            (self.save_path / 'rgb_back_right').mkdir()
            (self.save_path / 'rgb_back_left').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()

        self.lidar2img = {
            'CAM_FRONT': np.array([[1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
                                   [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                                   [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            'CAM_FRONT_LEFT': np.array([[6.03961325e-14, 1.39475744e+03, 0.00000000e+00, -9.20539908e+02],
                                        [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                        [-8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            'CAM_FRONT_RIGHT': np.array([[1.31064327e+03, -4.77035138e+02, 0.00000000e+00, -4.06010608e+02],
                                         [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                         [8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            'CAM_BACK': np.array([[-1.00000000e+00, -1.22464680e-16, 0.00000000e+00, -1.97168135e-16],
                                  [0.00000000e+00, 0.00000000e+00, -1.00000000e+00, -2.40000000e-01],
                                  [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, -1.61000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            'CAM_BACK_LEFT': np.array([[-1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -6.84385123e+02],
                                       [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                       [-9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

            'CAM_BACK_RIGHT': np.array([[3.60989788e+02, -1.34723223e+03, 0.00000000e+00, -1.04238127e+02],
                                        [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                        [9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        }
        self.lidar2cam = {
            'CAM_FRONT': np.array([[1., 0., 0., 0.],
                                   [0., 0., -1., -0.24],
                                   [0., 1., 0., -1.19],
                                   [0., 0., 0., 1.]]),
            'CAM_FRONT_LEFT': np.array([[0.57357644, 0.81915204, 0., -0.22517331],
                                        [0., 0., -1., -0.24],
                                        [-0.81915204, 0.57357644, 0., -0.82909407],
                                        [0., 0., 0., 1.]]),
            'CAM_FRONT_RIGHT': np.array([[0.57357644, -0.81915204, 0., 0.22517331],
                                         [0., 0., -1., -0.24],
                                         [0.81915204, 0.57357644, 0., -0.82909407],
                                         [0., 0., 0., 1.]]),
            'CAM_BACK': np.array([[-1., 0., 0., 0.],
                                  [0., 0., -1., -0.24],
                                  [0., -1., 0., -1.61],
                                  [0., 0., 0., 1.]]),

            'CAM_BACK_LEFT': np.array([[-0.34202014, 0.93969262, 0., -0.25388956],
                                       [0., 0., -1., -0.24],
                                       [-0.93969262, -0.34202014, 0., -0.49288953],
                                       [0., 0., 0., 1.]]),

            'CAM_BACK_RIGHT': np.array([[-0.34202014, -0.93969262, 0., 0.25388956],
                                        [0., 0., -1., -0.24],
                                        [0.93969262, -0.34202014, 0., -0.49288953],
                                        [0., 0., 0., 1.]])
        }
        self.lidar2ego = np.array([[0., 1., 0., -0.39],
                                   [-1., 0., 0., 0.],
                                   [0., 0., 1., 1.84],
                                   [0., 0., 0., 1.]])

        topdown_extrinsics = np.array(
            [[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array(
            [[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown
        self.hist_feats_queue = HistFeatsQueue()

    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0

            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(
                    x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(
                    x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(
                    math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]

            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}


    def sensors(self):
        sensors = [
            # camera rgb
            {
                'type': 'sensor.camera.rgb',
                'x': 0.80, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_LEFT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_RIGHT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -2.0, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                'width': 1600, 'height': 900, 'fov': 110,
                'id': 'CAM_BACK'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_LEFT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_RIGHT'
            },
            # imu
            {
                'type': 'sensor.other.imu',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'IMU'
            },
            # gps
            {
                'type': 'sensor.other.gnss',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'GPS'
            },
            # speed
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'SPEED'
            },
        ]
        if IS_BENCH2DRIVE:
            sensors += [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 50.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 512, 'height': 512, 'fov': 5 * 10.0,
                    'id': 'bev'
                }]
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)

        if (math.isnan(compass) == True):  # It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
            'imgs': imgs,
            'gps': gps,
            'pos': pos,
            'speed': speed,
            'compass': compass,
            'bev': bev,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
            'command_near': near_command,
            'command_near_xy': near_node
        }

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '
        results['frame_idx'] = 0
        results['timestamp'] = self.step / 20
        results['box_type_3d'], _ = get_box_type('LiDAR')
        cam_names = self.cfg.cam_names
        for cam in cam_names:
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'], axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'], axis=0)
        raw_theta = tick_data['compass'] if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi / 2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180
        results['can_bus'] = can_bus
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0:2] = can_bus[0:2].copy()
        ego_lcf_feat[2:4] = can_bus[10:12].copy()
        ego_lcf_feat[4] = rotation[-1]
        ego_lcf_feat[5] = 4.89238167
        ego_lcf_feat[6] = 1.83671331
        ego_lcf_feat[7] = np.sqrt(can_bus[0] ** 2 + can_bus[1] ** 2)

        if len(self.prev_control_cache) < 10:
            ego_lcf_feat[8] = 0
        else:
            ego_lcf_feat[8] = self.prev_control_cache[0].steer

        command = tick_data['command_near']
        if command < 0:
            command = 4
        command -= 1
        results['command'] = command
        command_onehot = np.zeros(6)
        command_onehot[command] = 1
        results['ego_fut_cmd'] = command_onehot
        theta_to_lidar = raw_theta
        command_near_xy = np.array(
            [tick_data['command_near_xy'][0] - can_bus[0], -tick_data['command_near_xy'][1] - can_bus[1]])
        rotation_matrix = np.array(
            [[np.cos(theta_to_lidar), -np.sin(theta_to_lidar)], [np.sin(theta_to_lidar), np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy

        ego2world = np.eye(4)
        ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2, 3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3, 0:3]
        results['l2g_t'] = lidar2global[0:3, 3]
        stacked_imgs = np.stack(results['img'], axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        if self.cfg.model.pts_bbox_head.ego_status_feature_num == 7:
            results['ego_status_feature'] = np.concatenate([
                can_bus[7:8],
                results['ego_fut_cmd']
            ])
        else:
            results['ego_status_feature'] = np.concatenate([
                can_bus[7:8],
                can_bus[10:13],
                ego_lcf_feat[8:9],
                results['ego_fut_cmd']
            ])

        if self.cfg.model.pts_bbox_head.ego_status_feature_num == 13:
            results['ego_status_feature'] = np.concatenate([
                results['ego_status_feature'],
                local_command_xy
            ])
        results['local_command_xy'] = local_command_xy
        print(local_command_xy[0], local_command_xy[1])
        assert self.cfg.model.pts_bbox_head.ego_status_feature_num == len(results['ego_status_feature'])

        results = self.inference_only_pipeline(results)
        self.device = "cuda"
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to(self.device)

        if not self.hist_feats_queue.is_empty():
            hist_feats = self.hist_feats_queue.fetch()
            self.model.pts_bbox_head.save_hist(torch.from_numpy(hist_feats).to(self.device)[None])

        output_data_batch = self.model(input_data_batch, return_loss=False, rescale=True)
        self.hist_feats_queue.push(output_data_batch[0]['pts_bbox']['env_tokens'].cpu().numpy())
        all_out_truck = output_data_batch[0]['pts_bbox']['ego_fut_pred'].cpu().numpy()
        brake = output_data_batch[0]['pts_bbox']['brake'].cpu().numpy()
        throttle = output_data_batch[0]['pts_bbox']['throttle'].cpu().numpy()
        steer = output_data_batch[0]['pts_bbox']['steer'].cpu().numpy()

        throttle_cls = self.model.pts_bbox_head.throttle_cls.cpu().numpy()
        steer_cls = self.model.pts_bbox_head.steer_cls.cpu().numpy()

        brake_thresh = 0.6
        brake_ctrl = float(brake > brake_thresh)
        steer_ctrl = steer_cls[steer.argmax()]
        throttle_ctrl = throttle_cls[throttle.argmax()] if not brake_ctrl else 0.0

        out_truck = all_out_truck
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out_truck,
                                                                                              tick_data['speed'],
                                                                                              local_command_xy)
        if brake_traj < 0.05:
            brake_traj = 0.0

        if throttle_traj > brake_traj:
            brake_traj = 0.0

        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        velo = torch.from_numpy(can_bus[7:8])[None]
        best_proposal = find_closest_dp_traj(output_data_batch[0]['pts_bbox']['ego_fut_pred'].cpu(),
                                             output_data_batch[0]['pts_bbox']['dp_seq'].cpu(),
                                             velo.cpu().float()).cpu().numpy()
        brake_dp = float(best_proposal[0])
        throttle_dp = float(best_proposal[1])
        steer_dp = float(best_proposal[2])

        best_proposal2 = find_closest_dp_ctrlwaypoints(
            output_data_batch[0]['pts_bbox']['ctrl_waypoints'].cpu(),
            output_data_batch[0]['pts_bbox']['dp_seq'].cpu(),
            velo.cpu().float()
        ).cpu().numpy()
        brake_dp2 = best_proposal2[0]
        throttle_dp2 = best_proposal2[1]
        steer_dp2 = best_proposal2[2]

        if_brake = (float(brake_traj) + float(brake_ctrl) + float(brake_dp) + float(brake_dp2)) >= 2

        final_throttle = (float(throttle_traj) + float(throttle_ctrl) + float(throttle_dp) + float(throttle_dp2)) / 4 \
            if not if_brake else 0.0
        final_steer = (float(steer_traj) + float(steer_ctrl) + float(steer_dp) + float(steer_dp2)) / 4

        final_brake = float(np.clip(if_brake, 0, 1))
        final_steer = np.clip(final_steer, -1, 1)
        final_throttle = np.clip(final_throttle, 0, 0.75)

        # tcp slow down during turns
        if abs(final_steer) > 0.1:  ## In turning
            speed_threshold = 1.0  ## Avoid stuck during turning
        else:
            speed_threshold = 1000  ## Avoid pass stop/red light/collision
        if float(tick_data['speed']) > speed_threshold:
            max_throttle = 0.5
        else:
            max_throttle = 0.75
        final_throttle = np.clip(final_throttle, a_min=0.0, a_max=max_throttle)

        # end
        control.steer = final_steer
        control.throttle = final_throttle
        control.brake = final_brake

        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake

        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)

        self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
        self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
        self.pid_metadata['brake_ctrl'] = float(brake_ctrl)

        self.pid_metadata['plan'] = out_truck.tolist()
        self.pid_metadata['command'] = command
        self.pid_metadata['all_plan'] = all_out_truck.tolist()

        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()
        if SAVE_PATH is not None and self.step % 10 == 0:
            pass
            # self.save(tick_data)

        self.prev_control = control

        if len(self.prev_control_cache) == 10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control

    def save(self, tick_data):
        frame = self.step // 10

        Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']).save(
            self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']).save(
            self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']).save(
            self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']).save(
            self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
