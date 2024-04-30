from os import truncate
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time

from engine import Sprite, GameObject, Game

class GameGymEnv(gym.Env):
    def __init__(
        self, 
        game: Game, 
        agent: GameObject, 
        target: GameObject
    ):
        assert game is not None
        self.game = game

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.game.size-1, shape=(2,), dtype=np.int8),
                "target": spaces.Box(0, self.game.size-1, shape=(2,), dtype=np.int8),
            }
        )

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self._agent = agent
        self._target = target

        self.frame_limit = 300

    def _get_obs(self):
        return {
            "agent": self._agent.position, 
            "target": self._target.position
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent.position - self._target.position, ord=1
            )
        }

    def reset(self):
        game_width, game_height = game.size
        new_position = (
            random.randint(0, game_width-1), 
            random.randint(0, game_height-1)
        )
        self._agent.update_position(new_position)
        self._target.update_position(new_position)
        while np.array_equal(self._target.position, self._agent.position):
            self._target.update_position((
                random.randint(0, game_width-1), 
                random.randint(0, game_height-1)
            ))

        observation = self._get_obs()
        info = self._get_info()
        self.current_frame = 0

        return observation, info


    def step(self, action):
        direction = self._action_to_direction[action]
        new_x, new_y = np.clip(
            self._agent.position + direction, 0, self.game.size - 1
        )
        self._agent.update_position((new_x, new_y))
        terminated = np.array_equal(self._agent.position, self._target.position)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()
        truncated = self.current_frame > self.frame_limit
        self.current_frame += 1

        return observation, reward, terminated, truncated, info

    def render(self, clear_terminal: bool = False):
        self.game.draw(clear_terminal)
        time.sleep(0.2)

    def close(self):
        pass


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

