from stable_baselines3 import PPO
from engine import Sprite, GameObject, Game
from gym_env import GameGymEnv

game = Game((20, 20))
player = GameObject(Sprite('🟩'), (10, 10))
food = GameObject(Sprite('🟥'), (18, 18))
game.register_game_object(player)
game.register_game_object(food)
env = GameGymEnv(game, player, food)
model = PPO.load("blob_final")

EPISODES = 30

for episode in range(EPISODES):
    obs, _ = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        action, _ = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render(True)

