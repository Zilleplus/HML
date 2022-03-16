import gym
import time

env = gym.make("CartPole-v1")

obs = env.reset()

env.render()

action = 1
for i in range(13):
    (obs, reward, done, info) = env.step(action)
    if not done:
        env.render()
        time.sleep(1)
    else:
        print("is done")
        break


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1


totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)
print(sum(totals)/len(totals))
