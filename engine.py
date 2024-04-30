from typing import List, Tuple, Optional 
import numpy as np
import os

class Sprite:
    def __init__(self, display_char: str):
        self.display_char = display_char

class GameObject:
    def __init__(self, sprite: Sprite, position: Tuple[int, int]):
        self.sprite = sprite
        self.position = np.array(position)

    def update_position(self, new_position: Tuple[int, int]):
        self.position = np.array(new_position)

class Game:
    def __init__(self, size: Tuple[int, int]):
        self.size = np.array(size)
        self.game_objects: List[GameObject] = []

    def register_game_object(self, game_object: GameObject):
        self.game_objects.append(game_object)

    def draw(self, clear_terminal: bool = False):
        if clear_terminal:
            os.system('cls' if os.name == 'nt' else 'clear')

        game_width, game_height = self.size
        text = [['⬜️' for _ in range(game_width)] for _ in range(game_height)]
        for game_object in self.game_objects:
            x, y = game_object.position
            text[y][x] = game_object.sprite.display_char
        text = '\n'.join([' '.join(row) for row in text])
        print(f'\n{text}\n')

