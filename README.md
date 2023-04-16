# RL_Minihack_Project
Reinforcement learning project training agents based on customized minihack environment

There will be **three levels**:
1. The environment in the primary level only has an agent, trees and walls, and the agent should somehow find the fixed exit point through learning while avoid running into an obstacle; 

2. The second level has an agent, obstacles, a sink, a teleport portal and a moving monster. If the agent comes across the monster, it will lose points; 

3. The third level includes an agent, obstacles, 2 moving monster and 2 portals which transfers the agent to a closer location to the exit, 2 sinks, an axe and an apple. In addition, the exit is randomly assigned for each play;

**Game maps of the 3 levels:**
![image](https://user-images.githubusercontent.com/81871673/232309321-cb20fa47-f2fa-4697-9e04-acee4171d3d8.png)

We tested two different agents: Q-Learning and Expected-Sarsa, the performance of the agents is successful. With q-learning explored more aggressively, while e-sarsa more conservatively.

**Example play of Level 3 Q-Learning:**

![image](https://github.com/EmmaHongW/RL_Minihack_Project/blob/main/Live_play/Level_3_qlearning.gif)
