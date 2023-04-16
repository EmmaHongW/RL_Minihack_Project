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


# Define the q learning agent class
class QLearning(Agent):
    def __init__(self, actions: List, alpha: float, gamma: float, eps: float):
        super().__init__()

        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.rand_generator = np.random.RandomState(42)
        # Initialize a q value dictionary
        self.q = {}
        self.prev_state = None
        self.prev_action = None

    def _argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)
        # If we use the argmax function defined by numpy, the index will always be the same
        return self.rand_generator.choice(ties)
    
    def _action_value(self, state, action):
        """ Compute state-action value of this pair."""
        return self.q.get((state, action), 1e-2*np.random.randn())

    def _get_action(self, state, eps):
        """ Return an eps-greedy action to be taken from this state."""
        if np.random.rand() < eps:
            return np.random.choice(self.actions)
        q_values = [self._action_value(state=state, action=action) for action in self.actions]
        action = self._argmax(q_values)
        return action

    def update(self, state, reward):
        """ Update state-action value of previous (state, action).
        Args:
            state (Any): The new state representation. Here we use the coordinate of the agent on the map.
            reward (float): Reward received upon the transaction to `state`.
        Note:
            - The parameter ``state`` should be an immutable type since it's used as a key.
        """
        state = self.decode_state(state)
        q = self._action_value(state=self.prev_state, action=self.prev_action)
        self.q[(self.prev_state, self.prev_action)] = q + self.alpha * \
            (reward - q + self.gamma * self._action_value(state, self._get_action(state, 0)))

    def take_action(self, state):
        """ Choose an eps-greedy action to be taken from this state. 
        Args:
            state (Any): The current state representation. It should be an immutable type since it's used as a key.
        """
        state = self.decode_state(state)
        action = self._get_action(state, self.eps)
        
        self.prev_action = action
        self.prev_state = state
        return action

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
def test_minihack_qlearning(env,level):
    ALPHA = 0.1 # Learning rate
    GAMMA = 1 # discount
    EPS = 0.05 # exploration rate, the empirical value from the individual assignment
    ITERS = 200

    class MyAgent(QLearning):
        def decode_state(self, state):
            return tuple(state['tty_cursor'])  

    qlearner = MyAgent(actions=list(range(env.action_space.n)),
                       alpha=ALPHA, gamma=GAMMA, eps=EPS)
    qlearner.load('qlearner_minihack'+level)

    all_rewards = []
    for i in range(ITERS):
        state = env.reset()
        # Get the target postision(Here we fixed it at the bottom right)
        n = 0
        done = False
        rewards = 0
        while not done:
            n += 1
            action = qlearner.take_action(state)
            state, reward, done, info = env.step(action)
            rewards+=reward
            qlearner.update(state, reward)

        qlearner.save('qlearner_minihack'+level)
        all_rewards.append(rewards)
        print('>'*40, f'Episode {i+1} is finished in {n} steps, the reward is {rewards}')

    return qlearner,all_rewards