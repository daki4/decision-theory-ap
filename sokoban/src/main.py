from abc import ABC, abstractmethod
from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

class SokobanEnv(gym.Env):
    playing_area: list[list[int]]
    actions = ['up', 'down', 'left', 'right']

    def __init__(self, playing_area: list[list[int]]=None, dp: list[tuple[int, int]]=None) -> None:
        if playing_area is None:
            
            # -1  - empty
            #  0  - wall
            #  1  - player
            #  2  - box
            playing_area = [
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, 0, 0],
                ]
            dp = [(1, 1)]
        observation_space = Box(low=0, high=1, shape=(5, len(playing_area), len(playing_area[0])), dtype=np.int32)
        self.action_space = Discrete(4)

        self.playing_area = playing_area
        self.dp = dp
        self.initial_playing_area: list[int] = [i.copy() for i in playing_area]

        
        super().__init__()
    
    def reset(self, seed=None):
        _ = seed

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)
    
    def get_moves(self):
        pass

    def render(self):
        pass

    def clone(self):
        pass

    def filter_observations(self):
        pass
    
    def can_move_in_direction(self,  action: str):
        if box.is_immovable():
            return False
        
        direction = [0, 0]
        if action == 'up':
            direction = [0, -1]
        if action == 'down':
            direction = [0, 1]
        if action == 'left':
            direction = [-1, 0]
        if action == 'right':
            direction = [0, 1]

        if type(self.playing_area[box.x + direction[0]][box.y + direction[1]]) == 0 or DesiredPosition:
            return True
        return False
