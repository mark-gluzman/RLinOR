import gym
import numpy as np
import sys

class CrissCross(gym.Env):

    def __init__(self, load=0.5):

        self.alpha = np.asarray([load, 0, load])  # arrival rates for each job class
        self.nu = np.asarray([2., 1., 2.])  # service completion rates for each job class

        self.uniform_rate = np.sum(self.alpha) + np.sum(self.nu)  # uniform rate
        self.p_arriving = np.divide(self.alpha, self.uniform_rate)  # arrival rates in the  queueing network after the uniformization
        self.p_compl = np.divide(self.nu, self.uniform_rate)  # service completion rates in the  queueing network after the uniformization
        self.cumsum_rates = np.unique(np.cumsum(np.asarray([self.p_arriving, self.p_compl]))) # the cumulative sum of probabilities of all possible events
        self.state = np.array([1, 0, 0])  # initial state
        self.action_space = gym.spaces.Discrete(2) #definition of the action space
        self.observation_space = gym.spaces.MultiDiscrete([ 1000, 1000, 1000]) #definition of the state space
        self.seed() # seeding method
        metadata = {'render.modes': ['human']} # available modes for render

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)  # ensure that the action is valid

        ### make sure that the policy is work-conserving ###
        if self.state[0] == 0:
            action = 1
        if self.state[2] == 0:
            action = 0
        ###

        reward = - np.sum(self.state)  # definition of the one-step reward function

        xi = self.np_random.rand()  # generate a random variable

        ###   determine potential event (activity) based on the r.v. realization ####
        activity = 0
        while xi > self.cumsum_rates[activity]:
            activity += 1  # activity that will be processed
        ###

        if activity == 0:  # class 1 job arriving
            self.state = self.state + [1, 0, 0]
        elif activity == 1:  # class 3 job arriving
            self.state = self.state + [0, 0, 1]
        # class 1 job service completion
        elif activity == 2 and action == 0 and self.state[0] > 0:  # class 1 job service completion
            self.state = self.state + [-1, 1, 0]
        # class 2 job service completion
        elif activity == 3 and self.state[1] > 0:
            self.state = self.state + [0, -1, 0]
        # class 3 job service completion
        elif activity == 4 and action == 1 and self.state[2] > 0:
            self.state = self.state + [0, 0, -1]
        # else `fake' event; state does not change

        done = (np.sum(self.state) == 0)  # flag that indicates the end of the episode

        return self.state, reward, done, {}

    def render(self, mode='human'):
        outfile = sys.stdout if mode == 'human' else super(CrissCross, self).render(mode=mode)
        outfile.write(np.array2string(self.state) + '\n')

    def reset(self):
        self.state = np.array([1, 0, 0])
        return self.state
