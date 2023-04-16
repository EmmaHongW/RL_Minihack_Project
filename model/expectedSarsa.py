from abc import ABC, abstractmethod
from typing import Any

from os.path import split, exists
from os import makedirs

from typing import List
import numpy as np

import sys
import gym


# Define a default class for later use
class Agent(ABC):

    def __init__(self):
        try:
            # The decode_state function is to make the state into an immutable format(not necessary if we preprocessed.)
            self.decode_state(1)
        except NotImplementedError:
            print("Please implement the static method 'decode_state' before procedding.")
            exit(-1)
        except:
            pass

    def save(self, path):
        dir = split(path)[0]
        if dir == '':
            dir = '.'
        if not exists(dir):
            makedirs(dir)


class ESarsa(Agent):

    def __init__(self, actions: List, alpha: float, gamma: float, eps: float):
        super().__init__()

        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.q = {}
        self.sar = []

    def _action_value(self, state, action):
        """ Compute state-action value of this pair."""
        return self.q.get((state, action), 1e-3 * np.random.randn())

    def _get_action(self, state, eps):
        """ Return an eps-greedy action to be taken from this state.    """
        if np.random.rand() < eps:
            return np.random.choice(self.actions)
        action = max(self.actions, key=lambda action: self._action_value(state=state, action=action))
        return action

    def update(self, reward):
        """ Update state-action value of previous (state, action).
        Args:
            reward (float): Reward received upon the transaction to `state`.
        """
        self.sar[-1][-1] = reward
        if len(self.sar) < 2:
            return
        state = self.sar[1][0]
        action = self.sar[1][1]
        reward = self.sar[0][-1]
        prev_state = self.sar[0][0]
        prev_action = self.sar[0][1]

        # Compute the expected Q-value of the next state
        next_state = self.sar[-1][0]
        next_actions = [a for a in self.actions if (next_state, a) in self.q]
        if not next_actions:
            next_q = 0
        else:
            next_q = np.mean([self._action_value(next_state, a) for a in next_actions])

        # Compute the TD error and update the Q-value of the current state-action pair
        q = self._action_value(state=prev_state, action=prev_action)
        tmp = reward + self.gamma * next_q - q
        self.q[(prev_state, prev_action)] = q + self.alpha * tmp

        del self.sar[0]

    def take_action(self, state):
        """ Choose an eps-greedy action to be taken from this state. 
        Args:
            state (Any): The current state representation. It should be an immutable type since it's used as a key.
        """
        state = self.decode_state(state)
        action = self._get_action(state, self.eps)
        self.sar.append([state, action, 0])
        return action

    def end_episode(self):
        """ Update state-action value of the last (state, action) pair. 
        """
        prev_state = self.sar[0][0]
        prev_action = self.sar[0][1]
        
        q = self._action_value(state=prev_state, action=prev_action)

        self.q[(self.sar[0][0], self.sar[0][1])] = q + self.alpha * (self.sar[0][2] - q)
        
        self.sar = []

    def save(self, path: str):
        """ Save state-action value table in `path`.npy
        Args:
            path (str): The location of where to store the state-action value table.
        """
        super().save(path)
        np.save(path + '.npy', self.q)

    def load(self, path):
        """ Load state-action value table.
        If it doesn't exist, a randomly-initialized table is used.
        Args:
            path (str): The location of where the state-action value table resides.
        """

        try:
            self.q = np.load(path + '.npy', allow_pickle='TRUE').item()
        except:
            self.q = {}
            print("No file is found in:", path)

# Define the function to test q learning agent
def test_minihack_expectedsarsa(env,level):
    ALPHA = 0.1 # Learning rate
    GAMMA = 1 # discount
    EPS = 0.05 # exploration rate, the empirical value from the individual assignment
    ITERS = 200

    class MyAgent(ESarsa):
        def decode_state(self, state):
            return tuple(state['tty_cursor'])  

    esarsa = MyAgent(actions=list(range(env.action_space.n)),
                       alpha=ALPHA, gamma=GAMMA, eps=EPS)
    esarsa.load('esarsa_minihack'+level)

    all_rewards = []
    for i in range(ITERS):
        state = env.reset()
        # Get the target postision(Here we fixed it at the bottom right)
        n = 0
        done = False
        rewards = 0
        while not done:
            n += 1
            action = esarsa.take_action(state)
            state, reward, done, info = env.step(action)
            rewards+=reward
            esarsa.update(reward)

        esarsa.save('esarsa_minihack'+level)
        all_rewards.append(rewards)
        print('>'*40, f'Episode {i+1} is finished in {n} steps, the reward is {rewards}')

    return esarsa,all_rewards
