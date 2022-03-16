import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tf_agents.networks.q_network import QNetwork
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.eval.metric_utils import log_metrics
import logging
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.utils.common import function
from tf_agents.metrics import tf_metrics
from tf_agents.utils.common import Checkpointer


max_episode_steps = 27000
environment_name = "BreakoutNoFrameskip-v4"

env_preprocessed = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])
tf_env = TFPyEnvironment(env_preprocessed)

# cast down, to save memory
preprocessing_layers = keras.layers.Lambda(
    lambda obs: tf.cast(obs, np.float32)/255.)
# 3 filters:
#  1. 32 times 8*8 filter, stride of 4
#  2. 64 times 4*4 filter, stride of 2
#  3. 64 times 3*3 filter, stride of 1
conv_layer_params = [
    (32, (8, 8), 4),
    (64, (4, 4), 2),
    (64, (3, 3), 1)]
# use output layer of 512 units
fc_layer_params = [512]

q_net = QNetwork(
    input_tensor_spec=tf_env.observation_spec(),
    action_spec=tf_env.action_spec(),
    preprocessing_layers=preprocessing_layers,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)


train_step = tf.Variable(0)
update_period = 4  # train the model every 4 steps
optimizer = keras.optimizers.RMSprop(
    lr=2.5e-4,
    rho=0.95,
    momentum=0.0,
    epsilon=0.00001,
    centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0,
    decay_steps=250000 // update_period,  # <=> 1,000,000 ALE frames
    end_learning_rate=0.01)  # final /epsilon
agent = DqnAgent(time_step_spec=tf_env.time_step_spec(),
                 action_spec=tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000,  # <=> 32,000 ALE frames
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99,  # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    # I only do 1/5th of the replay buffer used in the book
    # as by pc can't handle any bigger.
    max_length=20000)

replay_buffer_observer = replay_buffer.add_batch


# Alternatively we could define out own observer
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric()
    ]

# Old logger is used in the book:
# logging.get_logger().set_level(logging.INFO)
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

collect_driver = DynamicStepDriver(
    env=tf_env,
    policy=agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period)  # collectoin 4 steps for each training iteration


# warm up with 20,000 random steps
initial_collect_policy = RandomTFPolicy(
    time_step_spec=tf_env.time_step_spec(),
    action_spec=tf_env.action_spec())
init_driver = DynamicStepDriver(
    env=tf_env,
    policy=initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000)

final_step_step, final_policy_state = init_driver.run()

# sample some of the generated trajectories
trajectories, buffer_info = replay_buffer.get_next(
    sample_batch_size=2, num_steps=3)

trajectories._fields

# 2 tajectories
# 3 steps per trajectory
# 84*84 images (*4 -> not sure why this is 4, I would have expected 3)
trajectories.observation.shape

time_steps, action_steps, next_time_steps = to_transition(trajectories)
time_steps.observation.shape

# convert the replay buffer in a dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=2).prefetch(3)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

# optional stuff to use checkpoint
global_step = tf.compat.v1.train.get_global_step()
check_pointer = Checkpointer(
    ckpt_dir='./checkpoints',
    max_to_keep=3,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer
)
check_pointer.initialize_or_restore()


def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
            check_pointer.save(global_step)  # extra -> use checkpoint


train_agent(300000)
