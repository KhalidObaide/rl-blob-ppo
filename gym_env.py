import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time

from engine import GameObject, Game, Sprite

class GameGymEnv(gym.Env):
    def __init__(
        self, 
        game: Game, 
        agent: GameObject, 
        target: GameObject,
        avoid: GameObject
    ):
        assert game is not None
        self.game = game

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.game.size-1, shape=(2,), dtype=np.int8),
                "target": spaces.Box(0, self.game.size-1, shape=(2,), dtype=np.int8),
                "avoid": spaces.Box(0, self.game.size-1, shape=(2,), dtype=np.int8),
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
        self._avoid = avoid

        self.frame_limit = 300

    def _get_obs(self):
        return {
            "agent": self._agent.position, 
            "target": self._target.position,
            "avoid": self._avoid.position
        }

    def _get_info(self):
        return {
            "target_distance": np.linalg.norm(
                self._agent.position - self._target.position, ord=1
            ),
            "avoid_distance": np.linalg.norm(
                self._agent.position - self._avoid.position, ord=1
            ),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        game_width, game_height = self.game.size
        new_position = (
            random.randint(0, game_width-1), 
            random.randint(0, game_height-1)
        )
        self._agent.update_position(new_position)
        self._target.update_position(new_position)
        self._avoid.update_position(new_position)
        self._avoid_last_move = np.array([1, 0]) # to keep the enemey moving
        while np.array_equal(self._target.position, self._agent.position):
            self._target.update_position((
                random.randint(0, game_width-1), 
                random.randint(0, game_height-1)
            ))
        while np.array_equal(self._avoid.position, self._agent.position):
            self._avoid.update_position((
                random.randint(0, game_width-1), 
                random.randint(0, game_height-1)
            ))

        observation = self._get_obs()
        info = self._get_info()
        self.current_frame = 0

        return observation, info


    def step(self, action):
        direction = self._action_to_direction[int(action)]
        new_x, new_y = np.clip(
            self._agent.position + direction, 0, self.game.size - 1
        )
        self._agent.update_position((new_x, new_y))

        # enemy movement ( side to side )
        avoid_movement = self._avoid_last_move
        if self._avoid.position[0] >= self.game.size[0] - 1:
            avoid_movement = [-1, 0]
        elif self._avoid.position[0] <= 0:
            avoid_movement = [1, 0]
        avoid_x, avoid_y = self._avoid.position + avoid_movement
        self._avoid.update_position((avoid_x, avoid_y))
        self._avoid_last_move = avoid_movement

        terminated_good = np.array_equal(self._agent.position, self._target.position)
        terminated_bad = np.array_equal(self._agent.position, self._avoid.position)
        if terminated_good:
            reward = 1
        elif terminated_bad:
            reward = -1
        else:
            reward = 0
        observation = self._get_obs()
        info = self._get_info()
        truncated = self.current_frame > self.frame_limit
        self.current_frame += 1

        return observation, reward, terminated_good or terminated_bad, truncated, info

    def render(self, clear_terminal: bool = False):
        self.game.draw(clear_terminal)
        time.sleep(0.1)

    def close(self):
        pass


def setup_sample_env() -> GameGymEnv:
    game = Game((20, 20))
    player = GameObject(Sprite('🟦'), (10, 10))
    food = GameObject(Sprite('🟩'), (18, 18))
    enemy = GameObject(Sprite('🟥'), (8, 8))
    game.register_game_object(player)
    game.register_game_object(food)
    game.register_game_object(enemy)
    env = GameGymEnv(game, player, food, enemy)
    return env

