import os
import torch
import numpy as np
from tqdm import tqdm
import pdb

CONST_EPS = 1e-10


class OnlineReplayBuffer:
    _device: torch.device
    _state: np.ndarray
    _action: np.ndarray
    _reward: np.ndarray
    _next_state: np.ndarray
    _next_action: np.ndarray
    _not_done: np.ndarray
    _return: np.ndarray
    _size: int


    def __init__(
        self, 
        device: torch.device, 
        state_dim: int, action_dim: int, max_size: int
    ) -> None:

        self._device = device

        self._state = np.zeros((max_size, state_dim))
        self._action = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        self._next_state = np.zeros((max_size, state_dim))
        self._next_action = np.zeros((max_size, action_dim))
        self._not_done = np.zeros((max_size, 1))
        self._return = np.zeros((max_size, 1))
        self._advantage = np.zeros((max_size, 1))

        self._size = 0


    def store(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        s_p: np.ndarray,
        a_p: np.ndarray,
        not_done: bool
    ) -> None:

        self._state[self._size] = s
        self._action[self._size] = a
        self._reward[self._size] = r
        self._next_state[self._size] = s_p
        self._next_action[self._size] = a_p
        self._not_done[self._size] = not_done
        self._size += 1


    def compute_return(
        self, gamma: float
    ) -> None:

        pre_return = 0
        for i in tqdm(reversed(range(self._size)), desc='Computing the returns'):
            self._return[i] = self._reward[i] + gamma * pre_return * self._not_done[i]
            pre_return = self._return[i]


    def compute_advantage(
        self, gamma:float, lamda: float, value
    ) -> None:
        delta = np.zeros_like(self._reward)

        pre_value = 0
        pre_advantage = 0

        for i in tqdm(reversed(range(self._size)), 'Computing the advantage'):
            current_state = torch.FloatTensor(self._state[i]).to(self._device)
            current_value = value(current_state).cpu().data.numpy().flatten()

            delta[i] = self._reward[i] + gamma * pre_value * self._not_done[i] - current_value
            self._advantage[i] = delta[i] + gamma * lamda * pre_advantage * self._not_done[i]

            pre_value = current_value
            pre_advantage = self._advantage[i]

        self._advantage = (self._advantage - self._advantage.mean()) / (self._advantage.std() + CONST_EPS)

    def sample(
        self, batch_size: int
    ) -> tuple:

        ind = np.random.randint(0, self._size, size=batch_size)

        return (
            torch.FloatTensor(self._state[ind]).to(self._device),
            torch.FloatTensor(self._action[ind]).to(self._device),
            torch.FloatTensor(self._reward[ind]).to(self._device),
            torch.FloatTensor(self._next_state[ind]).to(self._device),
            torch.FloatTensor(self._next_action[ind]).to(self._device),
            torch.FloatTensor(self._not_done[ind]).to(self._device),
            torch.FloatTensor(self._return[ind]).to(self._device),
            torch.FloatTensor(self._advantage[ind]).to(self._device)
        )


class EpisodicReplayBuffer(OnlineReplayBuffer):

    def __init__(
        self, device: torch.device, 
        state_dim: int, action_dim: int, max_size: int
    ) -> None:
        super().__init__(device, state_dim, action_dim, max_size)
    

    def sample(self, batch_size: int) -> tuple:
        ind = np.random.randint(0, self._num_episodes, size=batch_size)
        states = self._episode_states[ind]
        actions = self._episode_actions[ind]
        masks = self._episode_masks[ind]
        returns = self._return[ind]
        return (
            torch.FloatTensor(states).to(self._device), 
            torch.FloatTensor(actions).to(self._device), 
            torch.FloatTensor(masks).to(self._device),
            torch.FloatTensor(returns).to(self._device)
            )

    def filter_dataset(self, cutoff_return: float):
        idxs = np.argwhere(self._return[:, 0] >= cutoff_return)[:, 0]
        self._episode_states = self._episode_states[idxs]
        self._episode_actions = self._episode_actions[idxs]
        self._episode_masks = self._episode_masks[idxs]
        self._return = self._return[idxs]
        self._num_episodes = len(idxs)


    def load_dataset(
        self, dataset: dict
    ) -> None:
        
        self._state = dataset['observations'][:-1, :]
        self._action = dataset['actions'][:-1, :]
        self._reward = dataset['rewards'].reshape(-1, 1)[:-1, :]
        self._next_state = dataset['observations'][1:, :]
        self._next_action = dataset['actions'][1:, :]
        self._not_done = 1. - (dataset['terminals'].reshape(-1, 1)[:-1, :] | dataset['timeouts'].reshape(-1, 1)[:-1, :])
        self._size = len(dataset['actions']) - 1
        # pdb.set_trace()

        e_idx = np.argwhere(dataset['terminals'][:-1] | dataset['timeouts'][:-1] > 0)
        self._e_idx = np.concatenate([[-1,], e_idx[:, 0], [dataset['observations'].shape[0] - 1,]]) + 1
        self._episode_lens = self._e_idx[1:] - self._e_idx[:-1] 
        self._max_ep_len = max(self._episode_lens)
        self._num_episodes = len(self._episode_lens) 

        if len(dataset['observations'].shape) > 1:
            self._episode_states = np.zeros((self._num_episodes, self._max_ep_len, dataset['observations'].shape[-1]))
        else:
            self._episode_states = np.zeros((self._num_episodes, self._max_ep_len))
        if len(dataset['actions'].shape) > 1:
            self._episode_actions = np.zeros((self._num_episodes, self._max_ep_len, dataset['actions'].shape[-1]))
        else:
            self._episode_actions = np.zeros((self._num_episodes, self._max_ep_len))
        self._episode_masks = np.zeros((self._num_episodes, self._max_ep_len))
        
        for i in tqdm(range(self._num_episodes), desc='Computing the episodes'):
            length = self._e_idx[i+1] - self._e_idx[i]
            self._episode_states[i, :length] = dataset['observations'][self._e_idx[i]: self._e_idx[i+1]]
            self._episode_actions[i, :length] = dataset['actions'][self._e_idx[i]: self._e_idx[i+1]]
            self._episode_masks[i, :length] = np.ones((length,))

    def compute_return(self, gamma: float) -> None:
        super().compute_return(gamma)
        # pdb.set_trace()
        self._return = self._return[self._e_idx[:-1]]

    def normalize_state(
        self
    ) -> tuple:

        mean = self._state.mean(0, keepdims=True)
        std = self._state.std(0, keepdims=True) + CONST_EPS
        self._state = (self._state - mean) / std
        self._next_state = (self._next_state - mean) / std
        return (mean, std)