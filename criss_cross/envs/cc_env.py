import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from io import StringIO
import sys

class CrissCross(gym.Env):

    def __init__(self, load=0.8):
        self.alpha = np.asarray([load, 0, load])  # arrival rate
        self.mu = np.asarray([2., 1., 2.])  # service rates

        self.uniform_rate = np.sum(self.alpha) + np.sum(self.mu)  # uniform rate for uniformization
        self.p_arriving = np.divide(self.alpha, self.uniform_rate)  # arrival rates in the uniformized queueing network
        self.p_compl = np.divide(self.mu, self.uniform_rate)  # service complition rates in the uniformized queueing network
        self.cumsum_rates = np.unique(np.cumsum(np.asarray([self.p_arriving, self.p_compl])))

        self.state = np.array([0, 0, 0])  # Start at beginning of the chain
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(1000),
            spaces.Discrete(1000),
            spaces.Discrete(1000)))
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        reward = - np.sum(self.state)
        xi = self.np_random.rand()

        activity = 0
        while xi > self.cumsum_rates[activity]:
            activity += 1  # activity that will be processed

        if activity == 0:
            self.state = self.state + [1, 0, 0]
        elif activity == 1:
            self.state = self.state + [0, 0, 1]
        elif activity == 2 and action == 1 and self.state[0] > 0:
            self.state = self.state + [-1, 0, 0]
        elif activity == 3 and self.state[1] > 0:
            self.state = self.state + [0, -1, 0]
        elif activity == 4 and action == 2 and self.state[2] > 0:
            self.state = self.state + [0, 0, -1]

        done = False
        return tuple(self.state), reward, done, {}

    def render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write(np.array2string(self.state))

        # No need to return anything for human
        if mode != 'human':
            return outfile

    def reset(self):
        self.state = np.array([0, 0, 0])
        return self.state
