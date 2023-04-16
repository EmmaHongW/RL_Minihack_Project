import numpy as np
from minihack import MiniHackNavigation
from minihack import LevelGenerator
from minihack import RewardManager
import random as rd
import gym
import matplotlib.pyplot as plt


# Reward Function
# Intuition: make the reward function denser, so that the agent can learn how to reach the target quicker.
def reward_check(state,target_pos,size):
    x,y = state
    x1,y1 = target_pos
    # For size*size map
    distance = np.sqrt((x-x1)**2+(y-y1)**2)
    max_distance = np.sqrt(size**2+size**2)
    score = - distance/max_distance
    # The closer the agent is with to the target, the larger the score
    # Add extra bonus when agent reach certain area closer enough to the target, so that the agent won't go back easily
    if distance < 0.85 * max_distance:
        return score+ 0.05
    elif distance < 0.7 * max_distance:
        return score+ 0.1
    elif distance < 0.6 * max_distance :
        return score+ 0.2
    elif distance < 0.5 * max_distance :
        return score+ 0.3
    elif distance < 0.33 * max_distance:
        return score+ 0.5
    elif distance < 0.25 * max_distance:
        return score+ 0.7
    elif distance < 0.15 * max_distance:
        return score+ 0.99
    else:
        return score
    

# Define Environment
class MiniHackRoom(MiniHackNavigation):
    """Customized environment"""
    def __init__(
        self,
        *args,
        size=5,
        n_monster=0,
        n_trap=0,
        n_teleport=0,
        lit=True,
        level=1,
        fixend=None,
        **kwargs
    ):
        kwargs["max_episode_steps"] = kwargs.pop(
            "max_episode_steps", size * 200
        )

        lvl_gen = LevelGenerator(w=size, h=size, lit=lit)
        self.x_start = 0
        self.y_start = 0

        self.x_target = size-1 if fixend is not None else rd.randint(0, size-1)
        self.y_target = size-1 if fixend is not None else rd.randint(0, size-1)
        lvl_gen.add_goal_pos((self.x_target, self.y_target))
        lvl_gen.set_start_pos((self.x_start, self.y_start))

        # Add some trees
        if level == 1:
            lvl_gen.fill_terrain("rect", "T", 5, 5, 9, 9)
        elif level == 2:
            lvl_gen.fill_terrain("rect", "T", 0, 5, 3, 5)
            lvl_gen.fill_terrain("rect", "T", 9, 10, 14, 10) 
        elif level == 3:
            lvl_gen.fill_terrain("rect", "T", 12, 2, 13, 6)
            lvl_gen.fill_terrain("rect", "T", 1, 15, 6, 16)

        # Add monster
        for _ in range(n_monster):
            lvl_gen.add_monster('hobbit',args=["peaceful"])

        # Sink
        for _ in range(n_trap):
            lvl_gen.add_sink()

        # Add Teleport
        for _ in range(n_teleport):
            lvl_gen.add_trap('teleport')
        
        # Add extra harmful objects
        if level == 3:
            lvl_gen.add_object("axe", ")",(12,10))
            lvl_gen.add_object("apple", "%",(10,17))
            lvl_gen.fill_terrain("rect",'P', 0,5,4,8) 
            lvl_gen.fill_terrain("rect",'P', 14,9,19,11)

        # Define a reward manager
        reward_manager = RewardManager()
        if level == 3:
            adj_x, adj_y = 30,1
        else:
            adj_x, adj_y = 32,3

        # -2 reward for standing on a sink
        if level == 2:
            reward_manager.add_location_event("sink", reward=-2, terminal_required=False)
        elif level == 3:
            reward_manager.add_location_event("sink", reward=-2, terminal_required=False)
            reward_manager.add_location_event('axe', reward=-3, terminal_required=False,repeatable=True)
            reward_manager.add_location_event("apple",reward=-2, terminal_required=False,repeatable=True)
        reward_manager.add_coordinate_event((self.x_target+adj_x, self.y_target+adj_y), reward=20, terminal_required=True,terminal_sufficient=True)
        
        for i in range(size):
            for j in range(size):
                rewardcustom = reward_check((i,j),(self.x_target, self.y_target),size)
                reward_manager.add_coordinate_event((i+adj_x,j+adj_y), reward = rewardcustom, repeatable=True, terminal_required=False)

        super().__init__(*args, des_file=lvl_gen.get_des(), reward_manager=reward_manager, **kwargs)

class MiniHackRoom15x15L1(MiniHackRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size = 15, fixend = True, n_monster=0, n_trap=0, n_teleport=0, level = 1, **kwargs)

class MiniHackRoom15x15L2(MiniHackRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size = 15, fixend = True, n_monster=1, n_trap=1, n_teleport=1, level = 2, **kwargs)

class MiniHackRoom20x20L3(MiniHackRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size = 20, n_monster=2, n_trap=2, n_teleport=2, level = 3, **kwargs)
