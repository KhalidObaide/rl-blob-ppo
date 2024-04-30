from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback 
from gym_env import setup_sample_env

checkpoint_callback = CheckpointCallback(save_freq=25_000, save_path="./checkpoints/")

env = setup_sample_env()
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=600_000, callback=checkpoint_callback)
model.save("final_model")

