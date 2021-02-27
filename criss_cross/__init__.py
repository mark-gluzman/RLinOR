from gym.envs.registration import register

register(
    id='criss_cross-v0',
    entry_point='criss_cross.envs:CrissCross',
    max_episode_steps=100000
)