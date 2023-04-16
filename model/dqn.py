import collections
from copy import deepcopy
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List


class DQN(Agent):

    def __init__(self, network: nn.Module, actions: List, alpha: float, gamma: float, eps: float, c: int = 128, t: int = 1024, capacity: int = 1024, bs: int = 32, device='cpu'):
        super().__init__()

        self.actions = {i: action for i, action in enumerate(actions)}
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.bs = bs
        self.c = c
        self.t = t

        self.device = device

        self.buffer = ExperienceReplay(capacity, device)
        self.Q = network.to(device)
        self.Q_prime = deepcopy(self.Q).to(device).eval()

        self.loss = nn.MSELoss()
        self.opt = torch.optim.AdamW(self.Q.parameters(), lr=self.alpha)
        self.i = 0  # counter used to trigger the update of Q_prime with Q

        self.prev_state = None
        self.prev_action = None

    def _action_value(self, state, action=None, clone: bool = False):
        """ If clone is False, the `self.Q` network is used, otherwise, `self.Q_prime` is used. """
        Q = self.Q if not clone else self.Q_prime
        n = state.shape[0]
        state = state.to(self.device)
        if action is not None:
            value = Q(state)[list(range(n)), action]
        else:
            value = Q(state)
        return value

    def _get_action(self, state, eps):
        """ Return an eps-greedy action to be taken from this state.    """
        with torch.no_grad():
            if np.random.rand() < eps:  
                return torch.from_numpy(np.random.choice(list(self.actions.keys()), size=(state.shape[0],)))
            actions = self._action_value(state=state, clone=True).argmax(dim=1)
            return actions

    def update(self, state:torch.Tensor, reward:float):
        """ Update state-action value of previous (state, action).
        Args:
            state (Any): The new state representation.
            reward (float): Reward received upon the transaction to `state`.
        Note:
            - The parameter ``state`` should be a tensor with the leading batch dimension.
        """
        state = self.decode_state(state).cpu()

        # register history
        self.buffer.append((self.prev_state, self.prev_action, torch.tensor(reward).unsqueeze(0).float(), state))

        # sample batch_size
        states, actions, rewards, next_states = self.buffer.sample(self.bs)
        gt = rewards + self.gamma * self._action_value(next_states, clone=True).max(dim=1)[0]
        pred = self._action_value(states, actions, clone=False)
        loss = self.loss(pred, gt)

        # update Q
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.i == self.c:
            # update Q_prim
            self.i = 0
            self.Q_prime = deepcopy(self.Q).eval()
        self.i += 1

        try:
            return loss.item()
        except:
            return None

    def take_action(self, state):
        """ Choose an eps-greedy action to be taken from this state. 
        Args:
            state (Any): The current state representation. After fed to ``decode_state``, the output should be eligible to be a network input.
        """
        state = self.decode_state(state)
        assert state.shape[0] == 1
        
        action = self._get_action(state, self.eps).cpu()
        
        self.prev_action = action
        self.prev_state = state
        return self.actions[action.item()]

    def save(self, path: str):
        """ Save state-action value table in `path`.npy
        Args:
            path (str): The location of where to store the state-action value table.
        """
        super().save(path)
        torch.save(self.Q.state_dict(), path + '.pth')

    def load(self, path):
        """ Load state-action value table.
        If it doesn't exist, a randomly-initialized table is used.
        Args:
            path (str): The location of where the state-action value table resides.
        """

        try:
            self.Q.load_state_dict(torch.load( path + '.pth'))
            self.Q = self.Q.to(self.device)
            self.Q_prime = deepcopy(self.Q).to(self.device).eval()
        except:
            print("No file is found in:", path)


class ExperienceReplay:

    def __init__(self, capacity, device):
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        try:
            indices = np.random.choice(
                len(self.buffer), batch_size, replace=False)
        except:
            indices = np.random.choice(
                len(self.buffer), batch_size, replace=True)

        states, actions, rewards, next_states = map(lambda x: torch.cat(x, dim=0).to(self.device), zip(*(self.buffer[idx] for idx in indices)))
        return states, actions, rewards, next_states


# Define the network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# Define the function to test dqn agent
def test_dqn_minihack(env,size):
    ALPHA = 0.1 # Learning rate
    GAMMA = 1 # discount
    EPS = 0.05 # exploration rate
    ITERS = 10
    class MyAgent(DQN):

        def decode_state(self, state):
            s = torch.from_numpy(np.array(
                tuple(map(tuple, state['chars_crop'])))).flatten().unsqueeze(0).float()
            return s

    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dqnlearner = MyAgent(nn.Linear(size*size,env.action_space.n), actions=list(range(env.action_space.n)),
                       alpha=ALPHA, gamma=GAMMA, eps=EPS,device = device)
    #dqnlearner = MyAgent(QNetwork(size*size,env.action_space.n), actions=list(range(env.action_space.n)),
    #                   alpha=ALPHA, gamma=GAMMA, eps=EPS,device = device)
    
    dqnlearner.load('dqnlearner_minihack')

    all_rewards = []
    for i in range(ITERS):
        state = env.reset()
        # Get the target postision(Here we fixed it at the bottom right)
        n = 0
        done = False
        rewards = 0
        while not done:
            n += 1
            action = dqnlearner.take_action(state)
            state, reward, done, info = env.step(action)
            rewards+=reward
            dqnlearner.update(state, reward)

        dqnlearner.save('dqnlearner_minihack')
        all_rewards.append(rewards)
        print('>'*40, f'Episode {i+1} is finished in {n} steps, the reward is {rewards}')

    return dqnlearner,all_rewards

