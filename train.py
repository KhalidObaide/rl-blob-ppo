from stable_baselines3 import PPO
from engine import Sprite, GameObject, Game
from gym_env import GameGymEnv

game = Game((20, 20))
player = GameObject(Sprite('🟩'), (10, 10))
food = GameObject(Sprite('🟥'), (18, 18))
game.register_game_object(player)
game.register_game_object(food)
env = GameGymEnv(game, player, food)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=250_000)
model.save("blob_final")

