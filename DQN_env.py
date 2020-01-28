import numpy as np
import gym
from collections import deque


class AtariDQNEnv(gym.Wrapper):

    def __init__(self, env, skip=4):
        super(AtariDQNEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    # Preprocess a (210, 160, 3) image into a (80, 80, 1) image in grey scale
    def preprocessed(self, state):
        state = state.dot([0.299, 0.587, 0.114])
        state = state[35:195]
        state = state[::2, ::2]
        state = state[:, :, np.newaxis]
        return state.astype(np.uint8)

    def step(self, action):
        total_reward = 0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done: break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return self.preprocessed(max_frame), total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return self.preprocessed(obs)
