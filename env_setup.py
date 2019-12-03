import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

import numpy as np

from dqn import DQN

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


def observation(obs, option=0):
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

def action(act, option=0):
	# add random action
	if option == 0:
		poss = np.random.uniform(size=1)
		if poss >= 0.9:
			rand_action = np.arange(4) + 1
			np.random.shuffle(rand_action)
			return rand_action[0]
	else:
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
            state = observation(state, option=1)
            action = agent.act(state)
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

if __name__ == '__main__':

    print(env.observation_space)
    print(env.action_space)
    episodes = 400
    loss = train_dqn(episodes)
    plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
    plt.savefig("result.png")


