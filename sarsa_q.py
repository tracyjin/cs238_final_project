"""
Contains the agents that can be used.
"""

import json
import pickle
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from bottleneck import move_mean
from gym import logger
from gym.wrappers import Monitor
from utils import get_action
from utils import get_observation


class RandomAgent:
    """
    Base/random class for RL agents.
    """

    def __init__(self, config):
        """
        Agent initialization.
        :param config:
        """

        # Running configuration
        self.run = 0
        self.step = 0
        self.episode = 0
        self.episode_count = config['EPISODES']

        # Env
        self.env_id = config['ENV_ID']
        self.env_seed = config['ENV_SEED']
        self.env = gym.make(self.env_id)
        self.env.seed(self.env_seed)

        # Get random number generator
        self.prng = np.random.RandomState(self.env_seed)

        # Score/rewards over time
        # Deque allows quick appends and pops and has a max length
        self.score = deque(maxlen=self.episode_count)
        # self.actions = deque(maxlen=self.episode_count)
        self.actions = []
        self.rand_actions = []
        self.rewards = []
        self.score_100 = deque(maxlen=100)  # for keeping track of mean of last 100

    def act(self, *args):
        """
        Perform a random action!
        :param args:
        :return:
        """
        return self.prng.randint(self.env.action_space.n)

    def do_episode(self, config):
        """

        :param config:
        :return:
        """

        # Initial values
        done = False
        score_e = 0
        step_e = 0

        # Reset environment
        self.env.reset()

        # Continue while not crashed
        while not done:
            # Act
            action = self.act()
            _, reward, done, _ = self.env.step(action)

            # Increment score and steps
            score_e += reward
            step_e += 1
            self.step += 1

        # Append score
        self.score.append(score_e)
        self.score_100.append(score_e)
        mean_score = np.mean(self.score_100)

        # Increment episode
        self.episode += 1

    def get_best_score(self):
        """

        :return:
        """

        # Best score is defined as highest 100-episode score reached (+ episode) when score < 200,
        # or the episode when score >= 200
        score_100 = move_mean(self.score, window=(100 if len(self.score) > 99 else len(self.score)), min_count=1)

        # Get max
        ep_max = np.argmax(score_100)
        score_max = score_100[ep_max]

        if score_max >= 200.0:
            ep_max = np.argmax(score_100 >= 200.0)
            score_max = 200.0  # to ensure equivalence

        return int(ep_max), float(score_max)


class SarsaAgent(RandomAgent):
    """
    Agent that makes use of Sarsa (on-policy TD control).
    """

    def __init__(self, config):
        """

        :param config:
        """

        # Initialize base class
        super().__init__(config)

        # State
        self.state_bounds = config['STATE_BOUNDS']
        self.state_bins = tuple(config['STATE_BINS'])

        # Float conversion
        for i, lr in enumerate(config['LEARNING_RATE']):
            if type(lr) is str:
                config['LEARNING_RATE'][i] = float(lr)
        for i, eps in enumerate(config['E_GREEDY']):
            if type(eps) is str:
                config['E_GREEDY'][i] = float(eps)

        # Learning parameters
        # First linear decay, then exponential decay
        self.alpha_start, self.alpha_end, self.alpha_steps, self.alpha_decay = config['LEARNING_RATE']
        self.epsilon_start, self.epsilon_end, self.epsilon_steps, self.epsilon_decay = config['E_GREEDY']
        self.alpha, self.epsilon = self.alpha_start, self.epsilon_start
        self.gamma = float(config['DISCOUNT_RATE'])

        # Q-table
        self.q_table = self.prng.uniform(low=-1.0, high=1.0, size=self.state_bins + (self.env.action_space.n,))

    def act(self, state):
        """

        :param state:
        :return:
        """

        if self.prng.random_sample() < self.epsilon:
            return self.prng.randint(self.env.action_space.n)
        else:
            return np.argmax(self.q_table[state])

    def discretize_state(self, state):
        """

        :param state:
        :return:
        """

        # First calculate the ratios, then convert to bin indices
        ratios = [(state[i] + abs(self.state_bounds[i][0])) / (self.state_bounds[i][1] - self.state_bounds[i][0]) for i
                  in range(len(state))]
        state_d = [int(round((self.state_bins[i] - 1) * ratios[i])) for i in range(len(state))]
        state_d = [min(self.state_bins[i] - 1, max(0, state_d[i])) for i in range(len(state))]

        return tuple(state_d)

    def learn(self, done, state, action, reward, state_, action_):
        """

        :param done:
        :param state:
        :param action:
        :param reward:
        :param state_:
        :param action_:
        :return:
        """

        # Get current Q(s, a)
        q_value = self.q_table[state][action]

        # Check if next state is terminal, get next Q(s', a')
        if not done:
            q_value_ = reward + self.gamma * self.q_table[state_][action_]
        else:
            q_value_ = reward

        # Update current Q(s, a)
        self.q_table[state][action] += self.alpha * (q_value_ - q_value)

    def do_episode(self, config):
        """

        :param config:
        :return:
        """

        # Initial values
        done = False
        score_e = 0
        step_e = 0

        # Get epsilon for initial state
        self.update_epsilon_step()

        # Episodic decay (only after linear decay)
        self.update_alpha_episode()
        self.update_epsilon_episode()

        # Get current state s, act based on s
        state = self.discretize_state(self.env.reset())
        action = self.act(state)


        # Continue while not crashed
        all_acts = []
        rand_acts = []
        all_rewards = []
        while not done:

            # Update for other steps
            self.update_alpha_step()
            self.update_epsilon_step()

            # Get next state s' and reward, act based on s'
            state_, reward, done, _ = self.env.step(action)
            if config['rand_obs'] == 1:
                state_ = get_observation(state_, option=1)
            state_ = self.discretize_state(state_)
            action_ = self.act(state_)
            if config['rand_act'] == 1:
                action, is_rand = get_action(action)
            else:
                action, is_rand = action, 0
            all_acts.append(action)
            if is_rand:
                rand_acts.append(1)
            else:
                rand_acts.append(0)
            # Learn
            self.learn(done, state, action, reward, state_, action_)
            all_rewards.append(reward)

            # Set next state and action to current
            state = state_
            action = action_

            # Increment score and steps
            score_e += reward
            step_e += 1
            self.step += 1

        # Append score
        self.score.append(score_e)
        self.score_100.append(score_e)
        self.actions.append(np.array(all_acts))
        self.rand_actions.append(np.array(rand_acts))
        self.rewards.append(np.array(all_rewards))
        mean_score = np.mean(self.score_100)

        # Increment episode
        self.episode += 1

    def update_alpha_step(self):
        """

        :return:
        """

        # Linear decay
        if self.step <= self.alpha_steps and self.alpha_steps > 0:
            self.alpha = self.alpha_start - self.step * (self.alpha_start - self.alpha_end) / self.alpha_steps

    def update_epsilon_step(self):
        """

        :return:
        """

        # Linear decay
        if self.step <= self.epsilon_steps and self.epsilon_steps > 0:
            self.epsilon = self.epsilon_start - self.step * (self.epsilon_start - self.epsilon_end) / self.epsilon_steps

    def update_alpha_episode(self):
        """

        :return:
        """

        # Exponential decay
        if self.step > self.alpha_steps:
            self.alpha *= self.alpha_decay

    def update_epsilon_episode(self):
        """

        :return:
        """

        # Exponential decay
        if self.step > self.epsilon_steps:
            self.epsilon *= self.epsilon_decay


class QAgent(SarsaAgent):
    """
    Agent that makes use of Q-learning (off-policy TD control).
    """

    def __init__(self, config):
        """

        :param config:
        """
        super().__init__(config)

    def learn(self, done, state, action, reward, state_, action_=None):
        """

        :param done:
        :param state:
        :param action:
        :param reward:
        :param state_:
        :param action_:
        :return:
        """

        # Get current Q(s, a)
        q_value = self.q_table[state][action]

        # Check if next state is terminal, get next maximum Q-value
        if not done:
            q_value_ = reward + self.gamma * max(self.q_table[state_])
        else:
            q_value_ = reward

        # Update current Q(s, a)
        self.q_table[state][action] += self.alpha * (q_value_ - q_value)

    def do_episode(self, config):
        """

        :param config:
        :return:
        """

        # Initial values
        done = False
        score_e = 0
        step_e = 0

        # Episodic decay (only after linear decay)
        self.update_alpha_episode()
        self.update_epsilon_episode()

        # Get current state s
        state = self.discretize_state(self.env.reset())

        # Continue while not crashed
        while not done:

            # Show on screen
            if config['VERBOSE'] > 1:
                self.env.render()

            # Get learning parameters
            self.update_alpha_step()
            self.update_epsilon_step()

            # Act based on current state s
            action = self.act(state)
            state_, reward, done, _ = self.env.step(action)
            state_ = self.discretize_state(state_)

            # Learn
            self.learn(done, state, action, reward, state_)

            # Set next state to current
            state = state_

            # Increment score and steps
            score_e += reward
            step_e += 1
            self.step += 1

        # Append score
        self.score.append(score_e)
        self.score_100.append(score_e)
        mean_score = np.mean(self.score_100)

        # Increment episode
        self.episode += 1

        if config['VERBOSE'] > 0:
            logger.info(f'[Episode {self.episode}] - score: {score_e:.2f}, steps: {step_e}, e: {self.epsilon:.4f}, '
                        f'a: {self.alpha:.4f}, 100-score: {mean_score:.2f}.')

