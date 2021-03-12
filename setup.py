from gym.envs.registration import register

register(
    id='criss_cross-v0', # id or name of the environment
    entry_point='criss_cross.envs:CrissCross', # entry point of the environment definition
    max_episode_steps=100000, # max number of time-steps in one episode
    kwargs={'load' : 0.5} # the criss-cross network load
)