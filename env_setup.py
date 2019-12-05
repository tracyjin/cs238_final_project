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
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

# action space: 0 do nothing
#				1 fire left engine
#				2 fire down engine
#				3 fire right engine
# state space: (x coord, y coord, x velocity, y velocity, angle, angular velocity,
#				left leg contact on land, right leg contact on land)
print(env.action_space)
print(env.observation_space)


def get_observation(obs, option=0):
	# add gaussian noise
	if option == 0:
		noise = np.random.normal(size=obs.shape) * 0.005
		obs += noise
	# add uniform noise
	elif option == 1:
		noise = np.random.uniform(size=obs.shape) * 0.01
		obs += noise
	# remove last one element
	elif option == 2:
		obs = obs[:-1]
	# remove the sixth element
	elif option == 3:
		obs = np.concatenate((obs[:5], obs[6:]))
	return obs

def get_action(act, option=0):
	# add random action
	if option == 0:
		poss = np.random.uniform(size=1)
		if poss >= 0.8:
			rand_action = np.arange(4)
			np.random.shuffle(rand_action)
			return rand_action[0]
	return act


def train_dqn(episode):
    loss = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 3000
        for i in range(max_steps):
            state = get_observation(state, option=1)
            action = agent.act(state)
            action = get_action(action)
            # env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        # if is_solved > 50:
        #     print('\n Task Completed! \n')
        #     break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return loss

def train_a3c(episode):
    # Defaults parameters:
    #    gamma = 0.99
    #    lr = 0.02
    #    betas = (0.9, 0.999)
    #    random_seed = 543

    render = False
    gamma = 0.99
    lr = 0.02
    betas = (0.9, 0.999)
    random_seed = 543

    torch.manual_seed(random_seed)

    policy = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    print(lr, betas)

    running_reward = 0
    loss_ls = []
    for i_episode in range(0, episode):
        state = env.reset()
        score = 0
        for t in range(10000):
            state = get_observation(state, option=1)
            action = policy(state)
            action = get_action(action)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            running_reward += reward
            score += reward
            if render and i_episode > 1000:
                env.render()
            if done:
                break
        loss_ls.append(score)
        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()
        policy.clearMemory()

        # # saving the model if episodes > 999 OR avg reward > 200
        # if i_episode > 999:
        #     torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))

        if running_reward > 4000:
            torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            print("########## Solved! ##########")
            test(name='LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            break

        if i_episode % 20 == 0:
            running_reward = running_reward / 20
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0

    return loss_ls

def train_ddpg(episode):
    pass

if __name__ == '__main__':

    print(env.observation_space)
    print(env.action_space)
    episodes = 3000
    dqn_loss = train_dqn(episodes)
    #a3c_loss = train_a3c(episodes)
    #plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
    plt.plot([i + 1 for i in range(0, len(dqn_loss), 2)], dqn_loss[::2])
    plt.savefig("result.png")


