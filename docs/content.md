
---

## Abstract

Imitation learning addresses the challenge of learning by observing an expert’s demonstrations without access to reward signals from the environment. Most existing imitation learning methods that do not require interacting with environments either model the expert distribution as the conditional probability p(a|s) (e.g., behavioral cloning, BC) or the joint probability p(s, a) (e.g., implicit behavioral cloning). Despite its simplicity, modeling the conditional probability with BC usually struggles with generalization. While modeling the joint probability can lead to improved generalization performance, the inference procedure can be time-consuming and it often suffers from manifold overfitting. This work proposes an imitation learning framework that benefits from modeling both the conditional and joint probability of the expert distribution. Our proposed diffusion model-augmented behavioral cloning (DBC) employs a diffusion model trained to model expert behaviors and learns a policy to optimize both the BC loss (conditional) and our proposed diffusion model loss (joint). DBC outperforms baselines in various continuous control tasks in navigation, robot arm manipulation, dexterous manipulation, and locomotion. We design additional experiments to verify the limitations of modeling either the conditional probability or the joint probability of the expert distribution as well as compare different generative models.

----

## Framework Overview 

![](./img/method.jpg "Illustration of our model")

Our proposed method DBC augments behavioral cloning (BC) by employing a diffusion model.
**(a) Learning a Diffusion Model:** The diffusion model φ learns to model the distribution of concatenated state-action pairs sampled from the demonstration dataset D. It learns to reverse the diffusion process (i.e. denoise) by optimizing L<sub>diff</sub>.
**(b) Learning a Policy with the Learned Diffusion Model:** We propose a diffusion model objective L<sub>DM</sub> for policy learning and jointly optimize it with the BC objective L<sub>BC</sub>. Specifically, L<sub>DM</sub> is computed based on processing a sampled state-action pair (s, a) and a state-action pair (s, a&#770;) with the action a&#770; predicted by the policy π with L<sub>diff</sub>.

----

## Environments & Tasks

![](./img/env.png "Environments and Tasks")


**(a) Maze:** A point-mass agent (<span style="color:green">green</span>) in a 2D maze learns to navigate from its start location to a goal location (<span style="color:red">red</span>).
**(b)-(c) FetchPick and FetchPush:** The robot arm manipulation tasks employ a 7-DoF Fetch robotics arm. FetchPick requires picking up an object (<span style="color:#c2c20c">yellow</span> cube) from the table and moving it to a target location (<span style="color:red">red</span>); FetchPush requires the arm to push an object (black cube) to a target location (<span style="color:red">red</span>).
**(d) HandRotate:** This dexterous manipulation task requires a Shadow Dexterous Hand to in-hand rotate a block to a target orientation.
**(e) Walker:** This locomotion task requires learning a bipedal walker policy to walk as fast as possible while maintaining its balance.

----

## Quantitative Results

We report the mean and the standard deviation of success rate (Maze, FetchPick, FetchPush, HandRotate) and return (Walker), evaluated over three random seeds. Our proposed method (DBC) outperforms the baselines on Maze, FetchPick, FetchPush, and HandRotate, and performs competitively against the best-performing baseline on Walker.

![](./img/quantitative_results.jpg "Comparisons to other baselines")


----


## Qualitative Results

Rendered videos of the policies learned by our proposed framework (DBC) and the baselines. (<span style="color:green">green</span>: succeed; <span style="color:red">red</span>: fail)

![](./img/qualitative_results.gif "render gif")

----

## Generalization Experiments

We report the performance of our proposed framework DBC and the baselines regarding the mean and the standard deviation of the success rate with different levels of noise injected into the initial state and goal locations in FetchPick and FecthPush, evaluated over three random seeds.

![FetchPick generalization experimental result](./img/pick.png "FetchPick generalization experimental result")

![FetchPush generalization experimental result](./img/push.png "FetchPush generalization experimental result")

----

## Citation
```
@article{wang2023diffusion,
  title={Diffusion Model-Augmented Behavioral Cloning},
  author={Hsiang-Chun Wang and Shang-Fu Chen and Ming-Hao Hsu and Chun-Mao Lai and Shao-Hua Sun},
  journal={arXiv preprint arXiv:2302.13335},
  year={2023}
}
```
<br>
