import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os, sys
import argparse
import numpy as np
import gym
import d4rl  # Import required to register environments
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def str2bool(v):
    return v.lower() == "true"


class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, input_dim=23, num_units=1024, depth=4, device="cuda"):
        super(MLPDiffusion, self).__init__()
        linears_list = []
        linears_list.append(nn.Linear(input_dim, num_units))
        linears_list.append(nn.ReLU())
        if depth > 1:
            for i in range(depth - 1):
                linears_list.append(nn.Linear(num_units, num_units))
                linears_list.append(nn.ReLU())
        linears_list.append(nn.Linear(num_units, input_dim))
        self.linears = nn.ModuleList(linears_list).to(device)

        embed_list = []
        for i in range(depth - 1):
            embed_list.append(nn.Embedding(n_steps, num_units))
        self.step_embeddings = nn.ModuleList(embed_list).to(device)

    def forward(self, x, t):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
        return x


def norm_vec(x, mean, std):
    obs_x = torch.clamp(
        (x - mean) / (std + 1e-8),
        -10.0,
        10.0,
    )
    return obs_x


########### training loss funciton,  sample at any given time t, and calculate sampling loss
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]
    # generate eandom t for a batch data
    t = torch.randint(0, n_steps, size=(batch_size // 2,)).to(device)
    t = torch.cat([t, n_steps - 1 - t], dim=0)  # [batch_size, 1]
    t = t.unsqueeze(-1)
    # coefficient of x0
    a = alphas_bar_sqrt[t]
    # coefficient of eps
    aml = one_minus_alphas_bar_sqrt[t]
    # generate random noise eps
    e = torch.randn_like(x_0)
    # model input
    x = x_0 * a + e * aml
    # get predicted random noise at time t
    if x.dtype == torch.float64:
        x = x.to(torch.float32)
    output = model(x, t.squeeze(-1))
    # calculate the loss between actual noise and predicted noise
    return (e - output).square().mean()


########### reverse diffusion sample function（inference）
def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    # generate[T-1]、x[T-2]|...x[0] from x[T]
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    # sample reconstruction data at time t drom x[T]
    t = torch.tensor([t]).to(device)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample


def reconstruct(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    # generate random t for a batch data
    t = torch.ones_like(x_0, dtype=torch.long).to(device) * n_steps
    # coefficient of x0
    a = alphas_bar_sqrt[t]
    # coefficient of eps
    aml = one_minus_alphas_bar_sqrt[t]
    # generate random noise eps
    e = torch.randn_like(x_0).to(device)
    # model input
    x_T = x_0 * a + e * aml
    if x_T.dtype == torch.float64:
        x_T = x_T.to(torch.float32)
    # generate[T-1]、x[T-2]|...x[0] from x[T]
    for i in reversed(range(n_steps)):
        # for i in reversed(range(1, n_steps+1)):
        x_T = p_sample(model, x_T, i, betas, one_minus_alphas_bar_sqrt)
    x_construct = x_T
    return x_construct


########### start training, print loss and print the medium reconstrction result
if __name__ == "__main__":
    # Create the environment
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-load-path', type=str, default='expert_datasets/maze.pt')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--scheduler-type', type=str, default='linear')
    # parser.add_argument('--achieve-range', type=float, default=0.1)
    # parser.add_argument('--data', type=int, default=100)
    parser.add_argument('--num-epoch', type=int, default=8000)
    parser.add_argument('--prefix', type=str, default='')
    args = parser.parse_args()
    print(f"Hidden dimension = {args.hidden_dim}")
    print(f"Depth = {args.depth}")

    ########### hyper parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    num_epoch = args.num_epoch
    num_steps = 1000
    betas = sigmoid_beta_schedule(num_steps)

    model_save_path = 'data/dm/trained_models'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    image_save_path = 'data/dm/trained_imgs'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, exist_ok=True)

    betas = torch.clip(betas, 0.0001, 0.9999).to(device)

    # calculate alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat(
        [torch.tensor([1]).float().to(device), alphas_prod[:-1]], 0
    )
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    assert (
        alphas.shape
        == alphas_prod.shape
        == alphas_prod_p.shape
        == alphas_bar_sqrt.shape
        == one_minus_alphas_bar_log.shape
        == one_minus_alphas_bar_sqrt.shape
    )
    print("all the same shape", betas.shape)

    env = args.traj_load_path.split('/')[-1][:-3]
    data = torch.load(args.traj_load_path)
    # Demonstration normalization
    obs = data["obs"]
    if args.norm:
        obs_mean = obs.mean(0)
        obs_std = obs.std(0)
        print(f"obs std: {obs_std}")
        obs = norm_vec(obs, obs_mean, obs_std)

    actions = data["actions"]
    if args.norm:
        actions_mean = actions.mean(0)
        actions_std = actions.std(0)
        print(f"actions std: {actions_std}")
        actions = norm_vec(actions, actions_mean, actions_std)

    dataset = torch.cat((obs, actions), 1)
    sample_num = dataset.size()[0]
    if sample_num % 2 == 1:
        dataset = dataset[1:sample_num, :]
    print("after", dataset.size())
    print("actions.dtype:", actions.dtype)

    print("Training model...")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # output dimension is state_dim + action_dim，inputs are x and step
    model = MLPDiffusion(
        num_steps,
        input_dim=dataset.shape[1],
        num_units=args.hidden_dim,
        depth=args.depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loss_list = []
    for t in tqdm(range(0, num_epoch)):
        total_loss = 0
        for idx, batch_x in enumerate(dataloader):
            batch_x = batch_x.squeeze().to(device)
            loss = diffusion_loss_fn(
                model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            loss = loss.cpu().detach()
            total_loss += loss
        ave_loss = total_loss / len(dataloader)
        train_loss_list.append(ave_loss)
        if t % 500 == 0:
            print(t, ":", ave_loss)
            with torch.no_grad():
                out = reconstruct(
                    model,
                    dataset.squeeze().to(device),
                    alphas_bar_sqrt,
                    one_minus_alphas_bar_sqrt,
                    num_steps - 1,
                )
                out = out.cpu().detach()
                fig, axs = plt.subplots(1, 2, figsize=(14, 6))
                axs[0].scatter(dataset[:5000, 0], dataset[:5000, 1], color='blue', edgecolor='white')
                axs[1].scatter(out[:5000, 0], out[:5000, 1], color='red', edgecolor='white')
                plt.savefig(f'data/dm/trained_imgs/{env}-pos.png')
                plt.close()
        if t % 20 == 0:
            train_iteration_list = list(range(len(train_loss_list)))
            plt.plot(train_iteration_list, train_loss_list, color='r')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(env + '_ddpm_loss')
            plt.savefig(f'data/dm/trained_imgs/{env}_ddpm_loss.png')
            plt.close()    
        if t % 1000 == 0:
            torch.save(model.state_dict(), f'data/dm/trained_models/{env}_ddpm.pt')
    torch.save(model.state_dict(), f'data/dm/trained_models/{env}_ddpm.pt')

