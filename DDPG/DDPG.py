"""

@author: coldhenry
####################################################
#                                                  #
#     Deep Determinisitc Policy Gradient (DDPG)    #
#                                                  #
####################################################

"""



""" Learn a policy using DDPG for the reach task"""
import time
import copy

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pytorch_utils as tools

import gym
import pybullet
import pybulletgym.envs

from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

np.random.seed(3000)

state_space = 8
action_space = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighSync(target_model, source_model, tau=0.001):
    param_t = target_model.parameters()
    param_s = source_model.parameters()
    for p_target, p_source in zip(param_t, param_s):
        updated = tau * p_source.data + (1 - tau) * p_target.data
        p_target.data.copy_(updated)


class Replay:
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """
        self.buffer = deque(maxlen=buffer_size)
        self.size_limit = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        # initilization
        state = env.reset()
        for i in range(init_length):
            action = env.action_space.sample()
            next_state, r, done, _ = env.step(action)
            self.buffer.append([state, action, r, next_state, int(done)])
            state = next_state
            if done:
                state = env.reset()

    def buffer_add(self, exp):
        """
        A function to add a list to the buffer
        param: exp : A list consisting of state, action, reward , next state and done flag
        """
        self.buffer.append(exp)

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        if len(self.buffer) < N:
            raise ValueError("Not enough elements")
        else:
            samples = random.sample(self.buffer, N)
            s_cur, a_cur, r_set, s_next, d_set = ([], [], [], [], [])
            for s in samples:
                s_cur.append(np.array(s[0]))
                a_cur.append(np.array(s[1]))
                r_set.append(np.array(s[2]))
                s_next.append(np.array(s[3]))
                d_set.append(np.array(s[4]))
            return (
                samples,
                np.array(s_cur),
                np.array(a_cur),
                np.array(r_set).reshape(-1, 1),
                np.array(s_next),
                np.array(d_set).reshape(-1, 1),
            )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        super(Actor, self).__init__()

        num_hidden = 400
        num_hidden2 = 300

        self.l1 = nn.Linear(state_space, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_hidden2)
        self.l3 = nn.Linear(num_hidden2, action_space)

    def forward(self, state):
        """
        Define the forward pass
        param: state: The state of the environment
        """
        # fully connected model
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        super(Critic, self).__init__()

        num_hidden = 400
        num_hidden2 = 300

        self.l1 = nn.Linear(state_space + action_space, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_hidden2)
        self.l3 = nn.Linear(num_hidden2, 1)

    def forward(self, state, action):
        """
        Define the forward pass of the critic
        """
        # fully connected model
        x = F.relu(self.l1(torch.cat([state, action], dim=-1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG:
    def __init__(
        self,
        env,
        action_dim,
        state_dim,
        critic_lr=1e-3,
        actor_lr=1e-3,
        gamma=0.99,
        batch_size=100,
    ):
        """
        param: env: An gym environment
        param: action_dim: Size of action space
        param: state_dim: Size of state space
        param: critic_lr: Learning rate of the critic
        param: actor_lr: Learning rate of the actor
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        """
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.env_eval = copy.deepcopy(env)
        self.lr_a = actor_lr
        self.lr_c = critic_lr

        ## initialize actor network and actor_target network
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)

        ## initialize critic network and critic_target network
        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.ReplayBuffer = Replay(
            buffer_size=10000,
            init_length=1000,
            state_dim=state_dim,
            action_dim=action_dim,
            env=env,
        )

    def update_target_networks(self):
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor)
        weighSync(self.critic_target, self.critic)

    def loss(self, s_set, a_set, r_set, ns_set, d_set):
        """
        param: s_set: tensor of states
        param: a_set: tensor of actions
        param: r_set: tensor of rewards
        param: ns_set: tensor of next states
        param: d_set: tensor of mask of (not done) 
        """

        q_behavior = self.critic(s_set, a_set)
        q_target = self.critic_target(ns_set, self.actor_target(ns_set))  # dim: (100,1)

        y = r_set + (self.gamma * d_set * q_target).detach()

        loss = F.mse_loss(q_behavior, y)

        return loss

    def policyEval(self):
        state = self.env_eval.reset()
        done = False
        step = 0
        reward = 0
        while not done:
            step += 1
            action = (
                self.actor(torch.FloatTensor(state).to(device))
                .cpu()
                .detach()
                .squeeze()
                .numpy()
            )
            next_state, r, done, _ = self.env_eval.step(action)
            reward += r
            state = next_state
        return step, reward

    def train(self, num_steps):
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """
        rewards = []
        actor_losses = []
        count = 0
        print("Start Training...")
        state = env.reset()
        for it in tqdm(range(int(num_steps))):

            count += 1

            # select an action
            # self.actor.eval()
            action = self.actor(
                torch.FloatTensor(state.reshape(1, -1)).to(device)
            ).cpu()
            action_noised = np.clip(
                action.data.numpy().flatten() + np.random.normal(0, 0.1, size=2), -1, 1
            )
            # print(action_noised)
            # self.actor.train()

            # take action and observe
            next_state, r, done, _ = env.step(action_noised)

            # add new data into buffer
            self.ReplayBuffer.buffer_add(
                [state, action_noised, r, next_state, int(done)]
            )

            # sample a random minibatch of transitions from buffer
            (
                samples,
                s_cur,
                a_cur,
                r_set,
                s_next,
                d_set,
            ) = self.ReplayBuffer.buffer_sample(self.batch_size)

            # convert to tensor
            s_cur_tensor = torch.FloatTensor(s_cur).to(device)
            a_cut_tensor = torch.FloatTensor(a_cur).to(device)
            r_set_tensor = torch.FloatTensor(r_set).to(device)
            s_next_tensor = torch.FloatTensor(s_next).to(device)
            d_set_tensor = torch.FloatTensor(1 - d_set).to(device)

            # update Critic network
            self.optimizer_critic.zero_grad()
            loss_critic = self.loss(
                s_cur_tensor, a_cut_tensor, r_set_tensor, s_next_tensor, d_set_tensor
            )
            loss_critic.backward()
            self.optimizer_critic.step()

            # update Actor network
            self.optimizer_actor.zero_grad()
            loss_actor = -(self.critic(s_cur_tensor, self.actor(s_cur_tensor))).mean()
            # print("loss actor: ", loss_actor.data)
            actor_losses.append(loss_actor.data)
            loss_actor.backward()
            self.optimizer_actor.step()

            self.update_target_networks()

            if done:
                state = env.reset()
            else:
                state = next_state

            if count % 500 == 0:
                tools.TD3_ckp(
                    it,
                    self.actor,
                    self.actor_target,
                    self.optimizer_actor,
                    self.critic,
                    self.critic_target,
                    self.optimizer_critic,
                )

            if count % 100 == 0:
                _, reward = self.policyEval()
                rewards.append(reward)
            if count % 5000 == 0:
                print("Eval steps: {}".format(reward))

        tools.saveNetwork(self.critic, "", "DDPG_critic_3000")
        tools.saveNetwork(self.actor, "", "DDPG_actor_3000")

        t = np.arange(0, int(num_steps), 100)
        plt.figure(figsize=(9, 9))
        plt.plot(t, rewards)
        plt.xlabel("Iterations")
        plt.ylabel("Rewards")
        plt.title("Training Rewards - seed 3000")
        plt.savefig("Train-Rewards-3000.jpg")
        plt.show()

        return rewards, actor_losses


if __name__ == "__main__":
    # Define the environment
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)

    ddpg_object = DDPG(
        env,
        8,
        2,
        critic_lr=1e-3,
        actor_lr=1e-3,
        gamma=0.99,
        batch_size=100,
    )

    # Train the policy
    rewards, actor_losses = ddpg_object.train(2e5)

    # Evaluate the final policy
    state = env.reset()
    done = False
    while not done:
        action = ddpg_object.actor(state).detach().squeeze().numpy()
        next_state, r, done, _ = env.step(action)
        env.render()
        time.sleep(0.1)
        state = next_state
