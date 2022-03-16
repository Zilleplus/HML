import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from collections import deque

env = gym.make("CartPole-v0")
input_shape = [4]  # env.observation_space.shape
n_outputs = 2  # env.action_space.shape


model = keras.models.Sequential([
    keras.layers.Dense(units=32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(units=32, activation="elu"),
    keras.layers.Dense(n_outputs)
])


def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])


replay_buffer = deque(maxlen=2000)


def sample_experiences(batch_size):
    # Generate indices to get "batch_size" random samples from the replay.
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    # get all the states, actions, rewards, next_states, dones into
    # seperate arrays. Instead of a one array with tuples.
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error


def training_step(batch_size):
    # get a bunch of random training examples
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    # predict the next Q-values
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values)
    target_Q_values = (rewards +
                       (1 - dones) * discount_factor * max_next_Q_values)
    # go from (batch_size,) to (batch_size, 1)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))  # TD-error
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


for episode in range(600):
    if((episode % 10) == 0):
        print("executin episode " + str(episode))
    obs = env.reset()
    for step in range(200):
        # decrease the value of epsilone as the episodes progress
        epsilon = max(1 - (episode / 500), 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            print("episode done at step " + str(step))
            break
    if episode > 50:
        training_step(batch_size)

obs = env.reset()
for i in range(100):
    Q_values = model.predict(obs[np.newaxis])
    action = np.argmax(Q_values[0])
    next_state, reward, done, info = env.step(action)
    if done:
        print("done at step" + str(i))
        break
