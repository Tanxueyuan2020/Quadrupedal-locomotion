from gym.envs.registration import register

register(
         id='Real-v0',
         entry_point='gym_real.envs:RealEnv',
         max_episode_steps=1000,
         reward_threshold=6000.0,
         )