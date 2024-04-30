from stable_baselines3 import PPO
from gym_env import setup_sample_env

env = setup_sample_env()
model = PPO.load("final_model.zip")

for episode in range(30):
    obs, _ = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        action, _ = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render(True)

