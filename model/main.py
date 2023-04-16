
import gym
import numpy as np
from minihack import MiniHackNavigation
from minihack import LevelGenerator
from minihack import RewardManager
import random as rd
import matplotlib.pyplot as plt
from IPython import display

from init import get_image,get_des_file_rendering
from environment import reward_check, MiniHackRoom, MiniHackRoom15x15L1, MiniHackRoom15x15L2, MiniHackRoom20x20L3
from qlearning import Agent, QLearning, test_minihack_qlearning
from dqn import DQN, ExperienceReplay, QNetwork,test_dqn_minihack
from expectedSarsa import ESarsa, test_minihack_expectedsarsa
from plot_func import plot_performance, live_play_viz, live_play_viz_es, extract_values, plot_SVfunction



# Create the environments and visualize them
env1 = MiniHackRoom15x15L1(observation_keys=('pixel', 'tty_cursor'),\
                           penalty_step = -0.1, penalty_time = -0.1, reward_lose=-10) 
env2 = MiniHackRoom15x15L2(observation_keys=('pixel', 'tty_cursor'),\
                           penalty_step = -0.1, penalty_time = -0.1, reward_lose=-10) 
env3 = MiniHackRoom20x20L3(observation_keys=('pixel', 'tty_cursor'),\
                           penalty_step = -0.1, penalty_time = -0.1, reward_lose=-10) 

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

# Visualize the initial environment for env1
state1 = env1.reset()
axes[0].imshow(get_image(state1['pixel']))
axes[0].set_title('Env 1: target={}'.format((env1.x_target, env1.y_target)))
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].axis('off')

# Visualize the initial environment for env2
state2 = env2.reset()
axes[1].imshow(get_image(state2['pixel']))
axes[1].set_title('Env 2: target={}'.format((env2.x_target, env2.y_target)))
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].axis('off')

# Visualize the initial environment for env3
state3 = env3.reset()
axes[2].imshow(get_image(state3['pixel']))
axes[2].set_title('Env 3: target={}'.format((env3.x_target, env3.y_target)))
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].axis('off')

plt.show()


# Try to walk by hand, for example 
env = env1
# 0: up, 1: right, 2: down, 3: left, 4: up right, 5: down right, 6: down left, 7: up left
state, reward, done, info = env.step(7) 
print("The current coordinates are:",state['tty_cursor'])
print("The current reward is:",reward)

next_state_image = get_image(state['pixel'])
plt.imshow(next_state_image)


env1 = MiniHackRoom15x15L1(observation_keys=('pixel', 'tty_cursor'),\
                           penalty_step = -0.1, penalty_time = -0.1, reward_lose=-10) 
env2 = MiniHackRoom15x15L2(observation_keys=('pixel', 'tty_cursor'),\
                           penalty_step = -0.1, penalty_time = -0.1, reward_lose=-10) 
env3 = MiniHackRoom20x20L3(observation_keys=('pixel', 'tty_cursor'),\
                           penalty_step = -0.1, penalty_time = -0.1, reward_lose=-10) 

qlearning_agent_L1, all_rewards_L1 = test_minihack_qlearning(env1,"_L1")
qlearning_agent_L2, all_rewards_L2 = test_minihack_qlearning(env2,"L2")
qlearning_agent_L3, all_rewards_L3 = test_minihack_qlearning(env3,"L3")

plot_performance(all_rewards_L1,'1')
plot_performance(all_rewards_L2,'2')
plot_performance(all_rewards_L3,'3')
live_play_viz(env1, qlearning_agent_L1)
live_play_viz(env2, qlearning_agent_L2)
live_play_viz(env3, qlearning_agent_L3)
plot_SVfunction(qlearning_agent_L1,"Q-Learning Level 1")
plot_SVfunction(qlearning_agent_L2,"Q-Learning Level 2")
plot_SVfunction(qlearning_agent_L3,"Q-Learning Level 3")

esarsa_agent_L1, all_rewards_L1 = test_minihack_expectedsarsa(env1,"_L1")
esarsa_agent_L2, all_rewards_L2 = test_minihack_expectedsarsa(env2,"_L2")
esarsa_agent_L3, all_rewards_L3 = test_minihack_expectedsarsa(env3,"_L3")
plot_performance(all_rewards_L1,'1')
plot_performance(all_rewards_L2,'2')
plot_performance(all_rewards_L3,'3')
live_play_viz_es(env1, esarsa_agent_L1)
live_play_viz_es(env2, esarsa_agent_L2)
live_play_viz_es(env3, esarsa_agent_L3)
plot_SVfunction(esarsa_agent_L1,"Expected_Sarsa Level 1")
plot_SVfunction(esarsa_agent_L2,"Expected_Sarsa Level 2")
plot_SVfunction(esarsa_agent_L3,"Expected_Sarsa Level 3")


