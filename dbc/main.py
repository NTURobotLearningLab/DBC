import sys, os
sys.path.insert(0, "./")

from functools import partial

import d4rl
import torch
import torch.nn as nn
import numpy as np
from rlf import run_policy, evaluate_policy
from rlf.algos import BaseAlgo, BehavioralCloning, DBC
from rlf.algos.il.base_il import BaseILAlgo
from rlf.algos.nested_algo import NestedAlgo
from rlf.policies import BasicPolicy
from rlf.args import str2bool
from rlf.rl.loggers.base_logger import BaseLogger
from rlf.rl.loggers.wb_logger import WbLogger
# from rlf.rl.loggers.wb_logger import WbLogger, get_wb_ray_config, get_wb_ray_kwargs
from rlf.rl.model import MLPBase, MLPBasic
from rlf.run_settings import RunSettings

import dbc.envs.ball_in_cup
import dbc.envs.d4rl
import dbc.envs.fetch
import dbc.envs.goal_check
import dbc.envs.gridworld
import dbc.envs.hand
import dbc.gym_minigrid
from dbc.envs.goal_traj_saver import GoalTrajSaver
from dbc.utils import trim_episodes_trans
from dbc.models import GwImgEncoder
from dbc.policies.grid_world_expert import GridWorldExpert
from typing import Dict, Optional, Tuple
from torch.utils.data import Dataset
import time


def get_basic_policy(env_name, args, is_stoch):
    if env_name.startswith("MiniGrid") and args.gw_img:
        return BasicPolicy(
            is_stoch=is_stoch, get_base_net_fn=lambda i_shape: GwImgEncoder(i_shape)
        )
    else:
        return BasicPolicy(
            is_stoch=is_stoch,
            get_base_net_fn=lambda i_shape: MLPBasic(
                i_shape[0],
                hidden_size=args.hidden_dim,
                num_layers=args.depth
            ),
        )
    return BasicPolicy()


def get_setup_dict():
    return {
        "bc": (BehavioralCloning(), partial(get_basic_policy, is_stoch=False)),
        "dbc": (DBC(), partial(get_basic_policy, is_stoch=False)),
    }


class DBCSettings(RunSettings):
    def get_policy(self):
        return get_setup_dict()[self.base_args.alg][1](
            self.base_args.env_name, self.base_args
        )

    def create_traj_saver(self, save_path):
        return GoalTrajSaver(save_path, False)

    def get_algo(self):
        algo = get_setup_dict()[self.base_args.alg][0]
        if isinstance(algo, NestedAlgo) and isinstance(algo.modules[0], BaseILAlgo):
            algo.modules[0].set_transform_dem_dataset_fn(trim_episodes_trans)
        return algo

    def get_logger(self):
        if self.base_args.no_wb:
            return BaseLogger()
        else:
            return WbLogger()

    def get_add_args(self, parser):
        parser.add_argument("--alg")
        parser.add_argument("--env-name")
        # Should always be true!
        parser.add_argument("--gw-img", type=str2bool, default=True)
        parser.add_argument("--no-wb", action="store_true", default=False)
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--hidden-dim", type=int, default=512)
        parser.add_argument("--depth", type=int, default=3)
        parser.add_argument("--weight-decay", type=float, default=0.0)
        parser.add_argument("--rnd-id", type=str, default='default')
        parser.add_argument("--collect-all", type=str2bool, default=False)
        parser.add_argument("--bc-state-norm", type=str2bool, default=True)
        parser.add_argument("--agent_expert_normalization", type=str2bool, default=True)

    def import_add(self):
        import dbc.envs.fetch
        import dbc.envs.goal_check


if __name__ == "__main__":
    start = time.time()
    run_policy(DBCSettings())
    end = time.time()
    print("The time used to execute this is:", end - start)
