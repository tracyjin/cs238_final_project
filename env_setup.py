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

from sarsa_q import SarsaAgent
from a3c_test import test
from utils import get_observation
from utils import get_action

import gym

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



def train_dqn(episode, rand_obs=0, rand_act=0, noise_obs_level=0.01, noise_act_level=0.1):
    loss = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    all_actions = []
    all_rand_acts = []
    all_rewards = []
    for e in range(episode):
        curr_acts = []
        curr_rand_acts = []
        curr_rewards = []
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 5000
        for i in range(max_steps):
            if rand_obs == 1:
                state = get_observation(state, option=0, noise_obs_level=noise_obs_level)
            action = agent.act(state)
            if rand_act == 1:
                action, is_rand = get_action(action)
            else:
                action, is_rand = action, 0
            curr_acts.append(action)
            curr_rand_acts.append(is_rand)
            # env.render()
            next_state, reward, done, _ = env.step(action)
            curr_rewards.append(reward)
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)
        all_actions.append(np.array(curr_acts))
        all_rand_acts.append(np.array(curr_rand_acts))
        all_rewards.append(np.array(curr_rewards))
        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        # if is_solved > 50:
        #     print('\n Task Completed! \n')
        #     break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    # np.savez("./saved/dqn_rand_act_" + str(rand_act) + "_rand_obs_" + str(rand_obs) + ".npz",
    #                       acts=np.array(all_actions),
    #                       rand_actions=np.array(all_rand_acts),
    #                       rewards=np.array(all_rewards),
    #                       scores=np.array(loss))
    # np.savez("./saved_dqn/dqn_rand_act_" + str(rand_act) + "_rand_obs_" + str(rand_obs) + "_noise_obs_lvl_" + str(noise_obs_level) + ".npz",
    #                       acts=np.array(all_actions),
    #                       rand_actions=np.array(all_rand_acts),
    #                       rewards=np.array(all_rewards),
    #                       scores=np.array(loss))
    np.savez("./saved_dqn/dqn_rand_act_" + str(rand_act) + "_rand_obs_" + str(rand_obs) + "_noise_act_lvl_" + str(noise_act_level) + ".npz",
                          acts=np.array(all_actions),
                          rand_actions=np.array(all_rand_acts),
                          rewards=np.array(all_rewards),
                          scores=np.array(loss))
    return loss

def train_a3c(episode, rand_obs=0, rand_act=0):
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
    all_actions = []
    all_rand_acts = []
    all_rewards = []
    for i_episode in range(0, episode):
        curr_acts = []
        curr_rand_acts = []
        curr_rewards = []
        state = env.reset()
        score = 0
        for t in range(10000):
            if rand_obs == 1:
                state = get_observation(state, option=1)
            # action = agent.act(state)
            # state = get_observation(state, option=1)
            action = policy(state)
            if rand_act == 1:
                action, is_rand = get_action(action)
            else:
                action, is_rand = action, 0
            curr_acts.append(action)
            curr_rand_acts.append(is_rand)
            # action = get_action(action)
            state, reward, done, _ = env.step(action)
            curr_rewards.append(reward)

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
        all_actions.append(np.array(curr_acts))
        all_rand_acts.append(np.array(curr_rand_acts))
        all_rewards.append(np.array(curr_rewards))


        # # saving the model if episodes > 999 OR avg reward > 200
        # if i_episode > 999:
        #     torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))

        # if running_reward > 4000:
        #     torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
        #     print("########## Solved! ##########")
        #     test(name='LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
        #     break

        if i_episode % 20 == 0:
            running_reward = running_reward / 20
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0
    np.savez("./saved/a3c_rand_act_" + str(rand_act) + "_rand_obs_" + str(rand_obs) + ".npz",
                          acts=np.array(all_actions),
                          rand_actions=np.array(all_rand_acts),
                          rewards=np.array(all_rewards),
                          scores=np.array(loss_ls))
    return loss_ls

def train_ddpg(episode):
    pass


def train_sarsa(episode, rand_obs=0, rand_act=0):
    config = {"ENV_ID": 'LunarLander-v2',
    "ENV_SEED": 0,
    "AGENT": 'sarsa',
    "EPISODES": 5000,
    "SAVE_EVERY": 100,
    "STATE_BINS": [5, 5, 5, 5, 5, 5, 2, 2],
    "STATE_BOUNDS": [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],
                   [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
    "CONTINUE": False,
    "E_GREEDY": [1.0, 0.05, 1e5, 0.97],
    "LEARNING_RATE": [0.2, 0.2, 0, 1],
    "DISCOUNT_RATE": 0.99,
    "rand_act": rand_act,
    "rand_obs": rand_obs}
    agent = SarsaAgent(config)
    while agent.episode < agent.episode_count:

        # Do episode
        agent.do_episode(config)

        print("episode: {}/{}, score: {}".format(agent.episode, agent.episode_count, agent.score[-1]))
        # Save every nth episode
        # if agent.episode % config['SAVE_EVERY'] == 0 and config['VERBOSE'] > 0:
        #     agent.save_checkpoint(config)
        #     figure.savefig(config['RECORD_DIR'] + 'score.pdf')

        # Break when goal of 100-score > 200 is reached
        if np.mean(agent.score_100) >= 200.0:
            print('\n Task Completed! \n')
            break
    # Close
    np.savez("./saved/sarsa_rand_act_" + str(rand_act) + "_rand_obs_" + str(rand_obs) + ".npz",
                          acts=np.array(agent.actions),
                          rand_actions=np.array(agent.rand_actions),
                          rewards=np.array(agent.rewards),
                          scores=np.array(agent.score))
    agent.env.close()

if __name__ == '__main__':

    print(env.observation_space)
    print(env.action_space)
    episodes = 1500
    # dqn_loss = train_dqn(episodes)
    # dqn = train_dqn(episodes, rand_obs=1, rand_act=1)
    # dqn = train_dqn(episodes, rand_obs=1, rand_act=0)
    # dqn = train_dqn(episodes, rand_obs=0, rand_act=1)
    # dqn = train_dqn(episodes, rand_obs=0, rand_act=0)

    # dqn = train_dqn(episodes, rand_obs=1, rand_act=0, noise_obs_level=0.0)
    # dqn = train_dqn(episodes, rand_obs=1, rand_act=0, noise_obs_level=0.001)
    # dqn = train_dqn(episodes, rand_obs=1, rand_act=0, noise_obs_level=0.005)
    # dqn = train_dqn(episodes, rand_obs=1, rand_act=0, noise_obs_level=0.01)
    # dqn = train_dqn(episodes, rand_obs=1, rand_act=0, noise_obs_level=0.05)
    # dqn = train_dqn(episodes, rand_obs=1, rand_act=0, noise_obs_level=0.1)
    # dqn = train_dqn(episodes, rand_obs=1, rand_act=0, noise_obs_level=0.2)
    # dqn = train_dqn(episodes, rand_obs=1, rand_act=0, noise_obs_level=0.4)

    # dqn = train_dqn(episodes, rand_obs=0, rand_act=1, noise_act_level=0.0)
    # dqn = train_dqn(episodes, rand_obs=0, rand_act=1, noise_act_level=0.05)
    # dqn = train_dqn(episodes, rand_obs=0, rand_act=1, noise_act_level=0.1)
    # dqn = train_dqn(episodes, rand_obs=0, rand_act=1, noise_act_level=0.15)
    dqn = train_dqn(episodes, rand_obs=0, rand_act=1, noise_act_level=0.2)


    # a3c = train_a3c(episodes, rand_obs=1, rand_act=1)
    # a3c = train_a3c(episodes, rand_obs=1, rand_act=0)
    # a3c = train_a3c(episodes, rand_obs=0, rand_act=1)
    # a3c = train_a3c(episodes, rand_obs=0, rand_act=0)

    # sarsa = train_sarsa(episodes, rand_obs=1, rand_act=1)
    # sarsa = train_sarsa(episodes, rand_obs=1, rand_act=0)
    # sarsa = train_sarsa(episodes, rand_obs=0, rand_act=1)
    # sarsa = train_sarsa(episodes, rand_obs=0, rand_act=0)



    #a3c_loss = train_a3c(episodes)
    #plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
    # plt.plot([i + 1 for i in range(0, len(dqn_loss), 2)], dqn_loss[::2])
    # plt.savefig("result.png")


