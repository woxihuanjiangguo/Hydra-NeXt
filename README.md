# Hydra-NeXt
The official repository for [Hydra-NeXt: Robust Closed-Loop Driving with Open-Loop Training](https://arxiv.org/abs/2503.12030).

# Installation
For training, please refer to [DATA_PREP.md](docs/DATA_PREP.md) and [INSTALL.md](docs/INSTALL.md). 

For closed-loop evaluation, please refer to [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) for the setup of CARLA.

# Training
Use this script for training:
```shell
bash adzoo/vad/dist_train.sh adzoo/vad/configs/hydra_next/hydra_next.py --gpus 8 --work-dir your/exp/path
```


# Evaluation
After [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) is prepared, rewrite the three variables in the [script](https://github.com/Thinklab-SJTU/Bench2Drive/blob/main/leaderboard/scripts/run_evaluation_multi_vad.sh) based on the script below:
```shell
cd your/path/to/Bench2Drive
train_dir=your/path/to/exp
save=your/path/to/save

TEAM_AGENT=hydra_next_agent.py \
TEAM_CONFIG=Bench2DriveZoo/adzoo/vad/configs/hydra_next/hydra_next_eval.py+${train_dir}/latest.pth \
ALGO=hydranext \
bash leaderboard/scripts/run_evaluation_multi_vad.sh
```

# Checkpoints
| Method     | DS   | SR   | CKPT         | RESULT                                                                                                                                                       |
|------------|------|------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hydra-NeXt | 73.9 | 50.0 | [CKPT](https://huggingface.co/Zzxxxxxxxx/HydraNeXt/blob/main/hydra_next.pth)     | [B2D](https://huggingface.co/Zzxxxxxxxx/HydraNeXt/blob/main/merged_new.json) / [CARLA v2](https://huggingface.co/Zzxxxxxxxx/HydraNeXt/blob/main/merged.json) |

# Citation
```
@article{li2025hydra,
  title={Hydra-next: Robust closed-loop driving with open-loop training},
  author={Li, Zhenxin and Wang, Shihao and Lan, Shiyi and Yu, Zhiding and Wu, Zuxuan and Alvarez, Jose M},
  journal={arXiv preprint arXiv:2503.12030},
  year={2025}
}
```