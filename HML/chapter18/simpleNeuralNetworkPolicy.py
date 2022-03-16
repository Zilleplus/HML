import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

env = gym.make("CartPole-v1")

obs = env.reset()

env.render()

n_inputs = 4  # 4 states in the obs [x, \dot{x}, \theta, \dot{\theta}]
model = keras.models.Sequential([
    keras.layers.Dense(units=5, activation='elu', input_shape=[n_inputs]),
    keras.layers.Dense(units=1, activation='sigmoid'),
])


def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        # obs.shape = (4,), obs[np.newaxis] makes it (1, 4)
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform(shape=[1, 1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        # y_target =
        #     -1 if action==False : first action in left_proba
        #     1 if action==True : second action in left_proba
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads


def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episodes in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        # play out a game
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        # save all the gradients of the current game
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads


def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    # range(begin, end, step) -> so loop from the back,
    # as the current reward depends on future rewards.
    for step in range(len(rewards)-2, -1, -1):
        discounted[step] += discounted[step+1]*discount_factor
    return discounted


def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std
            for discounted_rewards in all_discounted_rewards]


n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200  # max number of steps taken per time you play
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=0.01)
loss_fn = keras.losses.binary_crossentropy
for iteration in range(n_iterations):
    all_rewards, all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model, loss_fn)
    all_final_rewards = discount_and_normalize_rewards(
        all_rewards, discount_factor)
    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        mean_grads = tf.reduce_mean(
          [final_reward*all_grads[episode_index][step][var_index]
           for episode_index, final_rewards in enumerate(all_final_rewards)
              for step, final_reward in enumerate(final_rewards)], axis=0)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))


for i in range(10):
    obs = env.reset()
    done = False
    for i in range(1000):
        p = model.predict(obs[np.newaxis])
        print(p)
        _ = env.render()
        if done:
            print(str(i))
            break
        if p <= 0.5:
            obs, reward, done, info = env.step(0)
        else:
            obs, reward, done, info = env.step(1)
