import tensorflow as tf
import numpy as np
import gym
from DQN_env import AtariDQNEnv
from replay_buffer import ReplayBuffer

np.set_printoptions(precision=3, suppress=True)


# Hyperparameters
nsteps_train       = 5000000
batch_size         = 32
buffer_size        = 2000
target_update_freq = 10000
gamma              = 0.99
learning_freq      = 4
state_history      = 4
skip_frame         = 4
lr_begin           = 0.00025
lr_end             = 0.00005
lr_nsteps          = nsteps_train/2
eps_begin          = 1
eps_end            = 0.1
eps_nsteps         = 1000000
learning_start     = 1000
num_episodes_test  = 50
eval_freq          = 250000
save_freq          = 250000
soft_epsilon       = 0.05
state_high         = 255.


# Training variables
epsilon = lambda t: (eps_end if t > eps_nsteps else
                     eps_begin + ((eps_end - eps_begin) / eps_nsteps) * t)
preprocessed_frame_shape = (80, 80, 1)
state_shape = preprocessed_frame_shape * np.array([1, 1, state_history])
env = AtariDQNEnv(gym.make('Pong-v4'), skip=skip_frame)
num_actions = env.action_space.n
replay_buffer = ReplayBuffer(buffer_size, preprocessed_frame_shape,
                             state_history, batch_size, state_high)


# Model
q = tf.keras.models.Sequential()
q.add(tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu',
                             input_shape=state_shape))
q.add(tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu'))
q.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu'))
q.add(tf.keras.layers.Flatten())
q.add(tf.keras.layers.Dense(512, activation='relu'))
q.add(tf.keras.layers.Dense(num_actions))

q_target = tf.keras.models.clone_model(q)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_begin)
loss_object = tf.keras.losses.MeanSquaredError()

def update_target():
    for i, layer in enumerate(q.layers):
        q_target.layers[i].set_weights(layer.get_weights())


# Policy
def policy(state, epsilon):
    return (env.action_space.sample() if np.random.random() < epsilon
            else tf.argmax(tf.squeeze(q(np.array([state])))))


# Training step
@tf.function
def train_step(states, actions, rewards, new_states, dones):
    g = gamma * (1. - tf.cast(dones, tf.float32))
    targets = rewards + g * tf.reduce_max(q_target(new_states), axis=1)
    action_mask = tf.one_hot(actions, num_actions)
    with tf.GradientTape() as tape:
        predictions = tf.reduce_sum(q(states) * action_mask, axis=1)
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, q.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q.trainable_variables))


# Training loop
def train():
    t = 0
    while t < nsteps_train:
        frame = env.reset()
        while True:
            t += 1
            if t % 1000 == 0: print('t:', t)
            action = policy(replay_buffer.sample_state(frame), epsilon(t))
            new_frame, reward, done, _ = env.step(action)
            replay_buffer.store_step(frame, action, reward, done)
            frame = new_frame

            if t > learning_start and t % learning_freq == 0:
                train_step(*replay_buffer.sample_batches())

            if t > learning_start and t % target_update_freq == 0:
                update_target()

            if t > learning_start and t % eval_freq == 0:
                evaluate()

            if t % save_freq == 0:
                q.save_weights('t' + str(t) + '.h5')

            if t >= nsteps_train or done:
                break


# Evaluation procedure
def evaluate():
    returns = []
    for _ in range(num_episodes_test):
        episode_return = 0
        frame = env.reset()
        while True:
            action = policy(replay_buffer.sample_state(frame), soft_epsilon)
            new_frame, reward, done, _ = env.step(action)
            replay_buffer.store_step(frame, action, reward, done)
            frame = new_frame

            episode_return += reward
            if done: break
        returns.append(episode_return)
    print('Average return: {:04.2f}'.format(np.mean(returns)))


# Showcase results
def showcase(random_play=False):
    q.load_weights('checkpoint.h5')
    showcase_buffer = ReplayBuffer(buffer_size, preprocessed_frame_shape,
                                   state_history, batch_size, state_high)
    for _ in range(10):
        frame = env.reset()
        while True:
            env.render()
            action = (env.action_space.sample() if random_play else
                      policy(showcase_buffer.sample_state(frame), soft_epsilon))
            new_frame, reward, done, _ = env.step(action)
            showcase_buffer.store_step(frame, action, reward, done)
            frame = new_frame
            if done: break
        print('aaa')


# train()
showcase()
