#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch
import cv2
import os
import numpy as np
import scipy.stats
from geomloss import SamplesLoss
import gym
import d4rl # Import required to register environments
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler, BatchSampler
from tqdm import tqdm

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

########### hyper parameter
num_steps = 100

# decide beta
betas = torch.linspace(-6,6,num_steps)
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

# calculate alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt
alphas = 1-betas
alphas_prod = torch.cumprod(alphas,0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
==one_minus_alphas_bar_sqrt.shape
print("all the same shape",betas.shape)

########### decide the sample during definite diffusion process
# calculate x on given time based on x_0 and re-parameterization
def q_x(x_0,t):
    """based on x[0], get x[t] on any given time t"""
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise) # adding noise based on x[0]在x[0]

########### gaussian distribution in reverse diffusion process
import torch
import torch.nn as nn

class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128, device='cuda'):
        super(MLPDiffusion,self).__init__()
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(8,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,8),
            ]
        ).to(device)
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
            ]
        ).to(device)
    def forward(self,x,t):
        #print("t:", t)
        #print(t.size())
        #x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx+1](x)
            
        x = self.linears[-1](x)
        
        return x   
    
########### training loss funciton
# sample at any given time t, and calculate sampling loss
def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
    batch_size = x_0.shape[0]
    
    # generate eandom t for a batch data
    t = torch.randint(0,n_steps,size=(batch_size//2,))
    t = torch.cat([t,n_steps-1-t],dim=0) #[batch_size, 1]
    t = t.unsqueeze(-1)
    
    # coefficient of x0
    a = alphas_bar_sqrt[t]
    
    # coefficient of eps
    aml = one_minus_alphas_bar_sqrt[t]
    
    # generate random noise eps
    e = torch.randn_like(x_0)
    
    # model input
    x = x_0*a+e*aml
    
    # get predicted randome noise at time t
    output = model(x,t.squeeze(-1))
    
    # calculate the loss between actual noise and predicted noise
    return (e - output).square().mean()

########### reverse diffusion sample function（inference）
def p_sample_loop(model,shape,n_steps,betas,one_minus_alphas_bar_sqrt):
    # generate[T-1]、x[T-2]|...x[0] from x[T]
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):
    # sample reconstruction data at time t drom x[T]
    t = torch.tensor([t])

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x,t)
 
    mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
    
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
   
    sample = mean + sigma_t * z
   
    return (sample)

########### start training, print loss and print the medium reconstrction result
seed = 1234

class EMA(): # Exponential Moving Average
    #EMA
    def __init__(self,mu=0.01):
        self.mu = mu
        self.shadow = {}
        
    def register(self,name,val):
        self.shadow[name] = val.clone()
        
    def __call__(self,name,x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0-self.mu)*self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

if __name__ == '__main__':
    # Create the environment
    env = gym.make('maze2d-umaze-v1')

    data = torch.load('expert_traj.pt')
    obs = data['obs']
    actions = data['actions']
    print("type(obs)", obs.size())
    print("type(actions):", actions.size())
    dataset = torch.cat((obs, actions), 1)
 
    '''
    print("done", data['done'].shape)
    print("obs", data['obs'][0])
    print("next_obs", data['next_obs'].shape)
    print("actions", data['actions'][0])

    '''
    print("actions.dtype:", actions.dtype)
    '''
    fig,ax = plt.subplots()
    ax.scatter(data[:,0], data[:,1], color='blue', edgecolor='white');
    ax.axis('off')
    plt.savefig("./origin.jpg")
    '''

    print('Training model...')
    batch_size = 128 #128

    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

    num_epoch = 4000
    plt.rc('text',color='blue')

    model = MLPDiffusion(num_steps) # output dimension is 8，inputs are x and step
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    flag = [0 for k in range(0,12)]
    for t in tqdm(range(1,num_epoch+1)):
        for idx, batch_x in enumerate(dataloader):
            batch_x = batch_x.squeeze()
            loss = diffusion_loss_fn(model,batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        
        if(t%100==0):
             print(t, ":", loss)

    torch.save(model.state_dict(), 'ddpm.pt')

