import gym
import criss_cross
import tianshou as ts
import torch
import numpy as np
from torch import nn


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 20), nn.ReLU(inplace=True),
            nn.Linear(20, 20), nn.ReLU(inplace=True),
            nn.Linear(20, 20), nn.ReLU(inplace=True),
            nn.Linear(20, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state





#env = gym.make('criss_cross-v0')
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
#env = make_vec_env('criss_cross-v0', n_envs=8)
env = gym.make('criss_cross-v0')

# model = PPO(MlpPolicy, env, verbose=1, n_steps=128*16*8)
# model.learn(total_timesteps=1000000)

n_steps = 100
av_reward = np.zeros(n_steps)
av_r = 0.
env.reset()
for i in range(n_steps):
    # Random action
    k = 0
    done = False
    env.reset()
    av_r = 0.
    while not done:
        action = 1#env.action_space.sample()
        obs, reward, done, info = env.step(action)
        av_r = reward +  av_r
        k = k + 1

    av_reward[i]=av_r


print(av_reward)


# state_shape = 3
# action_shape = env.action_space.shape or env.action_space.n
# net = Net(state_shape, action_shape)
# optim = torch.optim.Adam(net.parameters(), lr=1e-3)
#
# train_envs = ts.env.DummyVectorEnv([lambda: gym.make('criss_cross-v0') for _ in range(8)])
# test_envs = ts.env.DummyVectorEnv([lambda: gym.make('criss_cross-v0') for _ in range(100)])
#
# policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.995, estimation_step=20, target_update_freq=320)
# train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=20000))
# test_collector = ts.data.Collector(policy, test_envs)
#
#
# result = ts.trainer.offpolicy_trainer(
#     policy, train_collector, test_collector,
#     max_epoch=20, step_per_epoch=10000, collect_per_step=100,
#     episode_per_test=100000, batch_size=64,
#     train_fn=lambda epoch, env_step: policy.set_eps(0.1),
#     test_fn=lambda epoch, env_step: policy.set_eps(0.05),
#     #stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
#     writer=None)
# print(f'Finished training! Use {result["duration"]}')