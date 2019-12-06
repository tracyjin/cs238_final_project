import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear
import torch
import torch.optim as optim
from a3c_test import test

import gym

import numpy as np

from dqn import DQN
from a3c import ActorCritic

import numpy as np


def get_observation(obs, option=0, noise_obs_level=0.01):
	# add gaussian noise
	if option == 0:
		noise = np.random.normal(size=obs.shape) * noise_obs_level
		noise = np.clip(noise, -0.5, 0.5)
		obs += noise
	# add uniform noise
	elif option == 1:
		noise = np.random.uniform(size=obs.shape) * noise_obs_level
		obs += noise
	# remove last one element
	elif option == 2:
		obs = obs[:-1]
	# remove the sixth element
	elif option == 3:
		obs = np.concatenate((obs[:5], obs[6:]))
	return obs

def get_action(act, option=0, noise_act_level=0.1):
	# add random action
	if option == 0:
		poss = np.random.uniform(size=1)
		if poss >= 1 - noise_act_level:
			rand_action = np.arange(4)
			np.random.shuffle(rand_action)
			return rand_action[0], 1
	return act, 0