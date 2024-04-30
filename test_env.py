from engine import Sprite, GameObject, Game
from gym_env import GameGymEnv

game = Game((20, 20))
player = GameObject(Sprite('🟩'), (10, 10))
food = GameObject(Sprite('🟥'), (18, 18))
game.register_game_object(player)
game.register_game_object(food)

EPISODES = 10

env = GameGymEnv(game, player, food)
for episode in range(EPISODES):
    observation, info = env.reset()
    total_reward = 0
    terminated, truncated = False, False
    while not terminated and not truncated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render(True)
    print(f"{episode}: {total_reward}")

