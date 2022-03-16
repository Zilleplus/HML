from tf_agents.environments import suite_gym
import matplotlib.pyplot as plt

env = suite_gym.load("Breakout-v4")

# Differences with open-ai gym environments
# Env returns a TimeStep object that wraps the observation, as well
# as some extra information.
env.reset()

# returns TimeStep object as well
env.step(1)

# env returns a RBG image
img = env.render()
plt.figure(1)
plt.imshow(img)
plt.show()

env.observation_spec()

env.action_spec()

from tf_agents.environments.wrappers import ActionRepeat

repeating_env = ActionRepeat(env, times=4)

from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

max_episode_steps = 27000
environment_name = "BreakoutNoFrameskip-v4"

env_preprocessed = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])

from tf_agents.environments.tf_py_environment import TFPyEnvironment

# This makes the environment usuable from withing a TensorFlow Graph.
tf_env = TFPyEnvironment(env_preprocessed)
