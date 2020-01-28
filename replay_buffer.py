import numpy as np


class ReplayBuffer:

    def __init__(self, buffer_size, frame_shape, state_history,
                 batch_size, state_high):
        self.buffer_size = buffer_size
        self.state_history = state_history
        self.batch_size = batch_size
        self.state_high = state_high
        self.frames = np.zeros((buffer_size, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros((buffer_size), dtype=np.uint8)
        self.rewards = np.zeros((buffer_size), dtype=np.float32)
        self.dones = np.zeros((buffer_size), dtype=bool)
        self.num_stored = 0


    def store_step(self, state, action, reward, done):
        self.frames[self.num_stored % self.buffer_size] = state
        self.actions[self.num_stored % self.buffer_size] = action
        self.rewards[self.num_stored % self.buffer_size] = reward
        self.dones[self.num_stored % self.buffer_size] = done
        self.num_stored += 1

    def sample_state(self, frame, index=None):
        if index is None: index = self.num_stored
        if frame is None: frame = self.frames[index % self.buffer_size]
        history_range = range(index - self.state_history + 1, index)
        history = self.frames.take(history_range, axis=0, mode='wrap')
        state = np.vstack((history, [frame]))
        state_dones = self.dones.take(history_range, axis=0, mode='wrap')
        if state_dones.any(): state[:np.argwhere(state_dones).max()+1] = 0
        state = np.concatenate(state, axis=2).astype(np.float32)
        return state / self.state_high

    def sample_batches(self):
        low = max(self.state_history-1,
                  self.num_stored-self.buffer_size+self.state_history-1)
        high = self.num_stored-1
        indices = np.random.randint(low=low, high=high, size=self.batch_size)
        states = [self.sample_state(None, i) for i in indices]
        new_states = [self.sample_state(None, i) for i in indices+1]
        indices = indices % self.buffer_size
        return (np.stack(states), self.actions[indices], self.rewards[indices],
                np.stack(new_states), self.dones[indices])
