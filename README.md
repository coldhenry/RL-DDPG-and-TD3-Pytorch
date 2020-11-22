# Deep Determinisitic Policy Gradient and TD3 using Pytorch

<img src="https://github.com/coldhenry/RL-DDPG-and-TD3-Pytorch/blob/main/pic/openai-pytorch.jpg" width="600" height="200" />

## DDPG

Deep Deterministic Policy Gradient (DDPG) is an off-policy algorithm that learns a Q-function and a policy. It is suitable for continuous action spaces and can be thought of a deep Q-learning algorithm for continuous action spaces.

In this implementation, there are three tricks applied for performance stability and covergence:
1. **Experience Replay Buffer**: A buﬀer that stores previous state-transitions and rewards functions, which is used to train the Q(s, a) and π(a)
2. **Target Actor and Critic Networks**: Classic Actor-Critic method, which combines with target networks to delay the update, and make MSBE loss more stable.
3. **Adding noise to actions**: since the policy is deterministic, we add noise to actions to explore better. The author of the original paper used time-correlated OU noise while we use uncorrelated, mean-zero Gaussian noise, which seems to work fine and much simpler.

## TD3

Twin Delayed DDPG (TD3) is a modified version of DDPG, which aims to solve a major problem of DDPG: the policy breaking because of the overestimated Q-values from the learned Q-function, which exploits the errors.

As a result, TD3 introduces three crucial modifications:
1. **Clipped Double-Q learning**: In TD3, it uses 2 Q-functions (critic networks) and pick the minimal one to construct the targets in the Bellman error loss functions.
2. **Delayed Policy Update**: if updates the target networks less frequently than the Q-function.
3. **Target Policy Smoothing Regularization**: the noise is added in the "target" action. By smoothing out Q, it is harder for the policy to exploit Q-function errors.

## Experiment Environment and Results

We use [modified-gym-env](https://github.com/ucsdarclab/pybullet-gym-env.git) for this implementation.

env: modified_gym_env:ReacherPyBulletEnv-v1

### DDPG training reward 
Use 2 seeds for testing

<img src="https://github.com/coldhenry/RL-DDPG-and-TD3-Pytorch/blob/main/pic/Train-Rewards.jpg" width="300" height="300" /><img src="https://github.com/coldhenry/RL-DDPG-and-TD3-Pytorch/blob/main/pic/Train-Rewards-2000.jpg" width="300" height="300" />

### TD3 training reward
<img src="https://github.com/coldhenry/RL-DDPG-and-TD3-Pytorch/blob/main/pic/Train-Rewards- TD3.jpg" width="300" height="300" />

## Reference 
[CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1509.02971.pdf)
