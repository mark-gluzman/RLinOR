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

        self.state = np.array([2, 2, 2])  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiDiscrete([ 1000, 1000, 1000])
        # spaces.Tuple((
        #     spaces.Discrete(1000),
        #     spaces.Discrete(1000),
        #     spaces.Discrete(1000)))
        self._seed()
        metadata = {'render.modes': ['ansi']}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        reward = - 1.
        xi = self.np_random.rand()

        activity = 0
        while xi > self.cumsum_rates[activity]:
            activity += 1  # activity that will be processed

        if activity == 0:
            self.state = self.state + [1, 0, 0]
        elif activity == 1:
            self.state = self.state + [0, 0, 1]
        elif activity == 2 and (action == 0 or self.state[2]==0) and self.state[0] > 0:
            self.state = self.state + [-1, 1, 0]
        elif activity == 3 and self.state[1] > 0:
            self.state = self.state + [0, -1, 0]
        elif activity == 4 and (action == 1 or self.state[0]==0) and self.state[2] > 0:
            self.state = self.state + [0, 0, -1]

        done = (np.sum(self.state) == 0)
        return self.state, reward, done, {}

    def render(self, mode='ansi', close=False):
        outfile = StringIO() if mode == 'ansi' else super(CrissCross, self).render(mode=mode)
        outfile.write(np.array2string(self.state))



    def reset(self):
        self.state = np.array([2, 2, 2])
        return self.state
