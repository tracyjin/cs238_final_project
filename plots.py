import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import interp1d

def plot_scores(scores, name):
	all_s = []
	for ii in range(len(scores) // 10 - 1):
		all_s.append(np.mean(scores[ii * 10 : (ii + 1) * 10]))
	all_s = np.array(all_s)
	all_s = np.clip(all_s, -500, 500)
	plt.plot(np.arange(len(all_s)) * 10, all_s, 'k')
	# window = 30
	# w = np.array([1/window for _ in range(window)])
	# weights = lfilter(w, 1, all_s)
	# x = np.arange(window//2, len(all_s) - window//2)
	# plt.plot(x, weights[window:len(all_s)], 'r--')
	plt.xlabel('Num episodes')
	plt.ylabel('Scores')
	plt.savefig('./figs/' + name + ".png")
	plt.close()

def plot_diff_algo_scores(scores_a3c, scores_dqn, scores_sarsa, name):
	all_s_a3c = []
	for ii in range(len(scores_a3c) // 10 - 1):
		all_s_a3c.append(np.mean(scores_a3c[ii * 10 : (ii + 1) * 10]))
	all_s_a3c = np.array(all_s_a3c)
	all_s_a3c = np.clip(all_s_a3c, -500, 500)
	# f = interp1d(np.arange(len(all_s_a3c)) * 10, all_s_a3c, kind='cubic')
	plt.plot(np.arange(len(all_s_a3c)) * 10, all_s_a3c, 'k', color='r', label="a3c")

	all_s_dqn = []
	for ii in range(len(scores_dqn) // 10 - 1):
		all_s_dqn.append(np.mean(scores_dqn[ii * 10 : (ii + 1) * 10]))
	all_s_dqn = np.array(all_s_dqn)
	all_s_dqn = np.clip(all_s_dqn, -500, 500)
	plt.plot(np.arange(len(all_s_dqn)) * 10, all_s_dqn, 'k', color='b', label="dqn")

	all_s_sarsa = []
	for ii in range(len(scores_sarsa) // 10 - 1):
		all_s_sarsa.append(np.mean(scores_sarsa[ii * 10 : (ii + 1) * 10]))
	all_s_sarsa = np.array(all_s_sarsa)
	all_s_sarsa = np.clip(all_s_sarsa, -500, 500)
	plt.plot(np.arange(len(all_s_sarsa)) * 10, all_s_sarsa, 'k', color='g', label="sarsa")
	# window = 30
	# w = np.array([1/window for _ in range(window)])
	# weights = lfilter(w, 1, all_s)
	# x = np.arange(window//2, len(all_s) - window//2)
	# plt.plot(x, weights[window:len(all_s)], 'r--')
	plt.xlabel('Num episodes')
	plt.ylabel('Scores')
	plt.legend()
	plt.savefig('./figs/' + name + ".png")
	plt.close()


def plot_diff_noise_scores(scores_a_0_o_0, scores_a_0_o_1, scores_a_1_o_0, scores_a_1_o_1, name):
	all_s_a_0_o_0 = []
	for ii in range(len(scores_a_0_o_0) // 20 - 1):
		all_s_a_0_o_0.append(np.mean(scores_a_0_o_0[ii * 20 : (ii + 1) * 20]))
	all_s_a_0_o_0 = np.array(all_s_a_0_o_0)
	all_s_a_0_o_0 = np.clip(all_s_a_0_o_0, -500, 500)
	# f = interp1d(np.arange(len(all_s_a_0_o_0)) * 20, all_s_a_0_o_0, kind='cubic')
	plt.plot(np.arange(len(all_s_a_0_o_0)) * 20, all_s_a_0_o_0, 'k', color='r', label="no noise")

	all_s_a_1_o_0 = []
	for ii in range(len(scores_a_1_o_0) // 20 - 1):
		all_s_a_1_o_0.append(np.mean(scores_a_1_o_0[ii * 20 : (ii + 1) * 20]))
	all_s_a_1_o_0 = np.array(all_s_a_1_o_0)
	all_s_a_1_o_0 = np.clip(all_s_a_1_o_0, -500, 500)
	# f = interp1d(np.arange(len(all_s_a_1_o_0)) * 20, all_s_a_1_o_0, kind='cubic')
	plt.plot(np.arange(len(all_s_a_1_o_0)) * 20, all_s_a_1_o_0, 'k', color='b', label="action noise")

	all_s_a_0_o_1 = []
	for ii in range(len(scores_a_0_o_1) // 20 - 1):
		all_s_a_0_o_1.append(np.mean(scores_a_0_o_1[ii * 20 : (ii + 1) * 20]))
	all_s_a_0_o_1 = np.array(all_s_a_0_o_1)
	all_s_a_0_o_1 = np.clip(all_s_a_0_o_1, -500, 500)
	# f = interp1d(np.arange(len(all_s_a_0_o_1)) * 20, all_s_a_0_o_1, kind='cubic')
	plt.plot(np.arange(len(all_s_a_0_o_1)) * 20, all_s_a_0_o_1, 'k', color='g', label="state noise")

	all_s_a_1_o_1 = []
	for ii in range(len(scores_a_1_o_1) // 20 - 1):
		all_s_a_1_o_1.append(np.mean(scores_a_1_o_1[ii * 20 : (ii + 1) * 20]))
	all_s_a_1_o_1 = np.array(all_s_a_1_o_1)
	all_s_a_1_o_1 = np.clip(all_s_a_1_o_1, -500, 500)
	# f = interp1d(np.arange(len(all_s_a_1_o_1)) * 20, all_s_a_1_o_1, kind='cubic')
	plt.plot(np.arange(len(all_s_a_1_o_1)) * 20, all_s_a_1_o_1, 'k', color='orange', label="action noise, state noise")
	# window = 30
	# w = np.array([1/window for _ in range(window)])
	# weights = lfilter(w, 1, all_s)
	# x = np.arange(window//2, len(all_s) - window//2)
	# plt.plot(x, weights[window:len(all_s)], 'r--')
	plt.xlabel('Num episodes')
	plt.ylabel('Scores')
	plt.legend()
	plt.savefig('./figs/' + name + ".png")
	plt.close()


def plot_dqn_diff_noise(scores, name):
	colors = ['r', 'g', 'b', 'orange', 'pink', 'brown']
	for key, value in scores.items():
		curr_noise = key
		curr_scores = value
		all_s_scores = []
		for ii in range(len(curr_scores) // 20 - 1):
			all_s_scores.append(np.mean(curr_scores[ii * 20 : (ii + 1) * 20]))
		all_s_scores = np.array(all_s_scores)
		plt.plot(np.arange(len(all_s_scores)) * 20, all_s_scores, 'k', color=colors[0], label=key)
		colors = colors[1:]
	plt.xlabel('Num episodes')
	plt.ylabel('Scores')
	plt.legend()
	plt.savefig('./figs/' + name + ".png")
	plt.close()




def load_plot_data():
	a3c_rand_act_0_rand_obs_0 = np.load("./saved/a3c_rand_act_0_rand_obs_0.npz")
	a3c_rand_act_0_rand_obs_1 = np.load("./saved/a3c_rand_act_0_rand_obs_1.npz")
	a3c_rand_act_1_rand_obs_0 = np.load("./saved/a3c_rand_act_1_rand_obs_0.npz")
	a3c_rand_act_1_rand_obs_1 = np.load("./saved/a3c_rand_act_1_rand_obs_1.npz")

	plot_scores(a3c_rand_act_0_rand_obs_0['scores'][:2000], "scores_a3c_rand_act_0_rand_obs_0")
	plot_scores(a3c_rand_act_0_rand_obs_1['scores'][:2000], "scores_a3c_rand_act_0_rand_obs_1")
	plot_scores(a3c_rand_act_1_rand_obs_0['scores'][:2000], "scores_a3c_rand_act_1_rand_obs_0")
	plot_scores(a3c_rand_act_1_rand_obs_1['scores'][:2000], "scores_a3c_rand_act_1_rand_obs_1")

	dqn_rand_act_0_rand_obs_0 = np.load("./saved/dqn_rand_act_0_rand_obs_0.npz")
	dqn_rand_act_0_rand_obs_1 = np.load("./saved/dqn_rand_act_0_rand_obs_1.npz")
	dqn_rand_act_1_rand_obs_0 = np.load("./saved/dqn_rand_act_1_rand_obs_0.npz")
	dqn_rand_act_1_rand_obs_1 = np.load("./saved/dqn_rand_act_1_rand_obs_1.npz")

	plot_scores(dqn_rand_act_0_rand_obs_0['scores'][:2000], "scores_dqn_rand_act_0_rand_obs_0")
	plot_scores(dqn_rand_act_0_rand_obs_1['scores'][:2000], "scores_dqn_rand_act_0_rand_obs_1")
	plot_scores(dqn_rand_act_1_rand_obs_0['scores'][:2000], "scores_dqn_rand_act_1_rand_obs_0")
	plot_scores(dqn_rand_act_1_rand_obs_1['scores'][:2000], "scores_dqn_rand_act_1_rand_obs_1")

	sarsa_rand_act_0_rand_obs_0 = np.load("./saved/sarsa_rand_act_0_rand_obs_0.npz")
	sarsa_rand_act_0_rand_obs_1 = np.load("./saved/sarsa_rand_act_0_rand_obs_1.npz")
	sarsa_rand_act_1_rand_obs_0 = np.load("./saved/sarsa_rand_act_1_rand_obs_0.npz")
	sarsa_rand_act_1_rand_obs_1 = np.load("./saved/sarsa_rand_act_1_rand_obs_1.npz")

	plot_scores(sarsa_rand_act_0_rand_obs_0['scores'][:2000], "scores_sarsa_rand_act_0_rand_obs_0")
	plot_scores(sarsa_rand_act_0_rand_obs_1['scores'][:2000], "scores_sarsa_rand_act_0_rand_obs_1")
	plot_scores(sarsa_rand_act_1_rand_obs_0['scores'][:2000], "scores_sarsa_rand_act_1_rand_obs_0")
	plot_scores(sarsa_rand_act_1_rand_obs_1['scores'][:2000], "scores_sarsa_rand_act_1_rand_obs_1")


	plot_diff_algo_scores(a3c_rand_act_0_rand_obs_0['scores'][:2000],
						  dqn_rand_act_0_rand_obs_0['scores'][:2000],
						  sarsa_rand_act_0_rand_obs_0['scores'][:2000],
						  "scores_diff_algo_rand_act_0_rand_obs_0")

	plot_diff_algo_scores(a3c_rand_act_0_rand_obs_1['scores'][:1500],
						  dqn_rand_act_0_rand_obs_1['scores'][:1500],
						  sarsa_rand_act_0_rand_obs_1['scores'][:1500],
						  "scores_diff_algo_rand_act_0_rand_obs_1")

	plot_diff_algo_scores(a3c_rand_act_1_rand_obs_0['scores'][:1200],
						  dqn_rand_act_1_rand_obs_0['scores'][:1200],
						  sarsa_rand_act_1_rand_obs_0['scores'][:1200],
						  "scores_diff_algo_rand_act_1_rand_obs_0")

	plot_diff_algo_scores(a3c_rand_act_1_rand_obs_1['scores'][:1500],
						  dqn_rand_act_1_rand_obs_1['scores'][:1500],
						  sarsa_rand_act_1_rand_obs_1['scores'][:1500],
						  "scores_diff_algo_rand_act_1_rand_obs_1")

	plot_diff_noise_scores(a3c_rand_act_0_rand_obs_0['scores'][:2000],
						   a3c_rand_act_0_rand_obs_1['scores'][:2000],
						   a3c_rand_act_1_rand_obs_0['scores'][:2000],
						   a3c_rand_act_1_rand_obs_1['scores'][:2000],
						   "scores_diff_noise_a3c")

	plot_diff_noise_scores(dqn_rand_act_0_rand_obs_0['scores'][:2000],
						   dqn_rand_act_0_rand_obs_1['scores'][:2000],
						   dqn_rand_act_1_rand_obs_0['scores'][:2000],
						   dqn_rand_act_1_rand_obs_1['scores'][:2000],
						   "scores_diff_noise_dqn")

	plot_diff_noise_scores(sarsa_rand_act_0_rand_obs_0['scores'][:2000],
						   sarsa_rand_act_0_rand_obs_1['scores'][:2000],
						   sarsa_rand_act_1_rand_obs_0['scores'][:2000],
						   sarsa_rand_act_1_rand_obs_1['scores'][:2000],
						   "scores_diff_noise_sarsa")


def load_dqn_obs_plot_data():
	dqn_nl_0 = np.load("./saved_dqn/dqn_rand_act_0_rand_obs_1_noise_obs_lvl_0.0.npz")
	dqn_nl_0_001 = np.load("./saved_dqn/dqn_rand_act_0_rand_obs_1_noise_obs_lvl_0.001.npz")
	dqn_nl_0_005 = np.load("./saved_dqn/dqn_rand_act_0_rand_obs_1_noise_obs_lvl_0.005.npz")
	dqn_nl_0_01 = np.load("./saved_dqn/dqn_rand_act_0_rand_obs_1_noise_obs_lvl_0.01.npz")
	dqn_nl_0_05 = np.load("./saved_dqn/dqn_rand_act_0_rand_obs_1_noise_obs_lvl_0.05.npz")
	dqn_nl_0_1 = np.load("./saved_dqn/dqn_rand_act_0_rand_obs_1_noise_obs_lvl_0.1.npz")

	scores = {"0": dqn_nl_0['scores'][:1500],
			  "0.001": dqn_nl_0_001['scores'][:1500],
			  "0.005": dqn_nl_0_005['scores'][:1500],
			  "0.01": dqn_nl_0_01['scores'][:1500],
			  "0.05": dqn_nl_0_05['scores'][:1500],
			  "0.1": dqn_nl_0_1['scores'][:1500]}
	plot_dqn_diff_noise(scores, "scores_diff_obs_lvl_dqn")

def load_dqn_act_plot_data():
	dqn_nl_0 = np.load("./saved_dqn/dqn_rand_act_1_rand_obs_0_noise_act_lvl_0.0.npz")
	dqn_nl_0_05 = np.load("./saved_dqn/dqn_rand_act_1_rand_obs_0_noise_act_lvl_0.05.npz")
	dqn_nl_0_1 = np.load("./saved_dqn/dqn_rand_act_1_rand_obs_0_noise_act_lvl_0.1.npz")
	dqn_nl_0_15 = np.load("./saved_dqn/dqn_rand_act_1_rand_obs_0_noise_act_lvl_0.15.npz")
	dqn_nl_0_2 = np.load("./saved_dqn/dqn_rand_act_1_rand_obs_0_noise_act_lvl_0.2.npz")

	scores = {"0": dqn_nl_0['scores'][:1500],
			  "0.05": dqn_nl_0_05['scores'][:1500],
			  "0.1": dqn_nl_0_1['scores'][:1500],
			  "0.15": dqn_nl_0_15['scores'][:1500],
			  "0.2": dqn_nl_0_2['scores'][:1500]}
	plot_dqn_diff_noise(scores, "scores_diff_act_lvl_dqn")








# load_plot_data()
# load_dqn_obs_plot_data()
load_dqn_act_plot_data()



