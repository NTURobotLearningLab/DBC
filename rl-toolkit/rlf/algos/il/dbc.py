import os
import sys

import copy
import gym
import numpy as np
import rlf.algos.utils as autils
import rlf.rl.utils as rutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlf.algos.il.base_il import BaseILAlgo
from rlf.args import str2bool
from rlf.storage.base_storage import BaseStorage
from tqdm import tqdm
import wandb
from dbc import ddpm
import math


# def linear_beta_schedule(timesteps):
#     """
#     https://blog.csdn.net/g11d111/article/details/131326934
#     linear schedule, proposed in original ddpm paper
#     """
#     scale = 1000 / timesteps
#     beta_start = scale * 0.0001
#     beta_end = scale * 0.02
#     return torch.linspace(beta_start, beta_end, timesteps)

# def cosine_beta_schedule(timesteps, s=0.008):
#     """
#     cosine schedule as proposed in https://arxiv.org/abs/2102.09672
#     """
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps)
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0.0001, 0.9999)

# def quadratic_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.03 #0.02
#     return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class DBC(BaseILAlgo):
    """
    When used as a standalone updater, BC will perform a single update per call
    to update. The total number of udpates is # epochs * # batches in expert
    dataset. num-steps must be 0 and num-procs 1 as no experience should be collected in the
    environment. To see the performance, you must evaluate. To just evaluate at
    the end of training, set eval-interval to a large number that is greater
    than the number of updates. There will always be a final evaluation.
    """

    def __init__(self, set_arg_defs=True):
        super().__init__()
        self.set_arg_defs = set_arg_defs

    def init(self, policy, args):
        super().init(policy, args)
        self.agent_expert_normalization = args.agent_expert_normalization
        self.num_epochs = 0
        self.action_dim = rutils.get_ac_dim(self.policy.action_space)
        if self.args.bc_state_norm:
            self.norm_mean = self.expert_stats["state"][0]
            self.norm_std = self.expert_stats["state"][1]
        else:
            self.norm_mean = None
            self.norm_var = None
        self.num_bc_updates = 0
        self.L1 = nn.L1Loss().cuda()
        self.coeff = args.coeff
        self.coeff_bc = args.coeff_bc

        num_steps = 1000
        betas = sigmoid_beta_schedule(num_steps)

        if args.env_name[:4] == 'maze':
            dim = 8
            self.diff_model = ddpm.MLPDiffusion(num_steps,
                                                input_dim=dim,
                                                num_units=128,
                                                depth=self.args.ddpm_depth).to(self.args.device)
        elif args.env_name[:9] == 'FetchPick':
            dim = 20
            self.diff_model = ddpm.MLPDiffusion(num_steps,
                                                     input_dim=dim,
                                                     num_units=1024,
                                                     depth=self.args.ddpm_depth).to(self.args.device)

        elif args.env_name[:10] == 'CustomHand':
            dim = 88
            self.diff_model = ddpm.MLPDiffusion(num_steps,
                                                input_dim=dim,
                                                num_units=2048,
                                                depth=self.args.ddpm_depth).to(self.args.device)
        elif args.env_name[:6] == 'Walker':
            dim = 23
            self.diff_model = ddpm.MLPDiffusion(num_steps, 
                                                input_dim=dim,
                                                num_units=1024).to(self.args.device)
        elif args.env_name[:11] == 'HalfCheetah':
            dim = 23
            self.diff_model = ddpm.MLPDiffusion(num_steps, 
                                                input_dim=dim,
                                                num_units=1024).to(self.args.device)
        elif args.env_name[:3] == 'Ant':
            dim = 50
            self.diff_model = ddpm.MLPDiffusion(num_steps,
                                                input_dim=dim,
                                                num_units=1024).to(self.args.device)
        weight_path = self.args.ddpm_path
        self.diff_model.load_state_dict(torch.load(weight_path))

    # sample at any given time t, and calculate sampling loss
    def diffusion_loss_fn(self, model, x_0_pred, x_0_expert, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
        batch_size = x_0_pred.shape[0]

        t = torch.randint(0, n_steps, size=(batch_size//2,)).to(self.args.device)
        t = torch.cat([t, n_steps-1-t], dim=0) #[batch_size, 1]
        t = t.unsqueeze(-1)

        # coefficient of x0
        a = alphas_bar_sqrt[t].to(self.args.device)
        
        # coefficient of eps
        aml = one_minus_alphas_bar_sqrt[t].to(self.args.device)
        
        # generate random noise eps
        e = torch.randn_like(x_0_pred).to(self.args.device)
        
        # model input
        x = x_0_pred*a + e*aml
        x2 = x_0_expert*a + e*aml

        # get predicted randome noise at time t
        output = model(x, t.squeeze(-1).to(self.args.device))
        output2 = model(x2, t.squeeze(-1).to(self.args.device))
        
        # calculate the loss between actual noise and predicted noise
        loss = (e - output).square().mean()
        loss2 = (e - output2).square().mean()
        return loss, loss2


    def get_density(self, states, pred_action, expert_action):
        num_steps = 1000
        betas = sigmoid_beta_schedule(num_steps)
        betas = betas.to(self.args.device)
        
        alphas = 1-betas
        alphas_prod = torch.cumprod(alphas,0).to(self.args.device)
        alphas_prod_p = torch.cat([torch.tensor([1]).float().to(self.args.device),alphas_prod[:-1]],0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        
        pred = torch.cat((states, pred_action), 1)
        expert = torch.cat((states, expert_action), 1)
        pred_loss, expert_loss = self.diffusion_loss_fn(self.diff_model, pred, expert, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps) 
        return pred_loss, expert_loss
             

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if args.bc_state_norm:
            print("Setting environment state normalization")
            settings.state_fn = self._norm_state
        return settings

    def _norm_state(self, x):
        obs_x = torch.clamp(
            (rutils.get_def_obs(x) - self.norm_mean)
            / (self.norm_std + 1e-8),
            -10.0,
            10.0,
        )
        obs_x = obs_x*(self.norm_std != 0)
        if isinstance(x, dict):
            x["observation"] = obs_x
            return x
        else:
            return obs_x

    def get_num_updates(self):
        if self.exp_generator is None:
            return len(self.expert_train_loader) * self.args.bc_num_epochs
        else:
            return self.args.exp_gen_num_trans * self.args.bc_num_epochs

    def get_completed_update_steps(self, num_updates):
        return num_updates * self.args.traj_batch_size

    def _reset_data_fetcher(self):
        super()._reset_data_fetcher()
        self.num_epochs += 1

    def full_train(self, update_iter=0):
        action_loss = []
        diff_loss = []
        prev_num = 0

        # First BC
        with tqdm(total=self.args.bc_num_epochs) as pbar:
            while self.num_epochs < self.args.bc_num_epochs:
                super().pre_update(self.num_bc_updates)
                log_vals = self._bc_step(False)
                action_loss.append(log_vals["_pr_action_loss"])
                diff_loss.append(log_vals["_pr_diff_loss"])

                pbar.update(self.num_epochs - prev_num)
                prev_num = self.num_epochs

        rutils.plot_line(
            action_loss,
            f"action_loss_{update_iter}.png",
            self.args.vid_dir,
            not self.args.no_wb,
            self.get_completed_update_steps(self.update_i),
        )
        self.num_epochs = 0

    def pre_update(self, cur_update):
        # Override the learning rate decay
        pass
    
    def _bc_step(self, decay_lr):
        if decay_lr:
            super().pre_update(self.num_bc_updates)
        expert_batch = self._get_next_data()
        if expert_batch is None:
            self._reset_data_fetcher()
            expert_batch = self._get_next_data()

        states, true_actions = self._get_data(expert_batch)
        # states = self._norm_state(states)
        
        log_dict = {}
        pred_actions, _, _ = self.policy(states, None, None)
        
        if rutils.is_discrete(self.policy.action_space):
            pred_label = rutils.get_ac_compact(self.policy.action_space, pred_actions)
            acc = (pred_label == true_actions.long()).sum().float() / pred_label.shape[0]
            log_dict["_pr_acc"] = acc.item()
        loss_ = autils.compute_ac_loss(
            pred_actions,
            true_actions.view(-1, self.action_dim),
            self.policy.action_space,
        )
        loss = self.coeff_bc * loss_

        pred_loss, expert_loss = self.get_density(states, pred_actions, true_actions)
        
        
        if self.agent_expert_normalization:
            diff_loss_ = torch.clip((pred_loss - expert_loss), min=0)
            # diff_loss_ = pred_loss - expert_loss
        else:
            diff_loss_ = torch.clip(pred_loss, min=0) # TODO: check if this is correct
            # diff_loss_ = pred_loss
        diff_loss = self.coeff * diff_loss_
        total_loss = loss + diff_loss

        self._standard_step(total_loss) #backward
        self.num_bc_updates += 1

        val_loss = self._compute_val_loss()
        if val_loss is not None:
            log_dict["_pr_val_loss"] = val_loss.item()

        log_dict["_pr_action_loss"] = loss_.item()
        log_dict["_pr_diff_loss"] = diff_loss_.item()
        log_dict["pred_loss"] = pred_loss.item()
        log_dict["expert_loss"] = expert_loss.item()

        return log_dict

    def _get_data(self, batch):
        states = batch["state"].to(self.args.device)
        if self.args.bc_state_norm:
            states = self._norm_state(states)

        if self.args.bc_noise is not None:
            add_noise = torch.randn(states.shape) * self.args.bc_noise
            states += add_noise.to(self.args.device)
            states = states.detach()

        true_actions = batch["actions"].to(self.args.device)
        true_actions = self._adjust_action(true_actions)
        return states, true_actions

    def _compute_val_loss(self):
        if self.update_i % self.args.eval_interval != 0:
            return None
        if self.val_train_loader is None:
            return None
        with torch.no_grad():
            action_losses = []
            diff_losses = []
            for batch in self.val_train_loader:
                states, true_actions = self._get_data(batch)
                pred_actions, _, _ = self.policy(states, None, None)
                action_loss = autils.compute_ac_loss(
                    pred_actions,
                    true_actions.view(-1, self.action_dim),
                    self.policy.action_space,
                )
                action_losses.append(action_loss.item())
                
                pred_loss, expert_loss = self.get_density(states, pred_actions, true_actions)
                diff_loss = (pred_loss - expert_loss)
                diff_losses.append(diff_loss.item())
            return np.mean(action_losses + diff_losses)

    def update(self, storage):
        top_log_vals = super().update(storage) #actor_opt_lr
        log_vals = self._bc_step(True) #_pr_action_loss
        log_vals.update(top_log_vals) #_pr_action_loss
        return log_vals

    def get_storage_buffer(self, policy, envs, args):
        return BaseStorage()

    def get_add_args(self, parser):
        if not self.set_arg_defs:
            # This is set when BC is used at the same time as another optimizer
            # that also has a learning rate.
            self.set_arg_prefix("dbc")

        super().get_add_args(parser)
        #########################################
        # Overrides
        if self.set_arg_defs:
            parser.add_argument("--num-processes", type=int, default=1)
            parser.add_argument("--num-steps", type=int, default=0)
        parser.add_argument("--no-wb", default=False, action="store_true")

        #########################################
        # New args
        parser.add_argument("--bc-num-epochs", type=int, default=1)
        parser.add_argument("--bc-noise", type=float, default=None)
        parser.add_argument('--num-units', type=int, default=128) #hidden dim of ddpm
        parser.add_argument('--ddpm-depth', type=int, default=4)
