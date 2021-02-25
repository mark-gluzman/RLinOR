import gym
import criss_cross
env = gym.make('criss_cross-v0')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample()) # take a random action

env.render()
env.close()