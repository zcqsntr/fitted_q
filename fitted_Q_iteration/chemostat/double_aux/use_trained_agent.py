import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# file path for fitted_Q_agents
FQ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FQ_DIR)

# file path for chemostat_env
C_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
C_DIR = os.path.join(C_DIR, 'chemostat_env')
sys.path.append(C_DIR)


from chemostat_envs import *
from fitted_Q_agents import *



def fig_6_reward_function(state, action, next_state):
    # make everything negative here
    N1_targ = 250
    N2_targ = 550
    targ = np.array([N1_targ, N2_targ])

    SSE = sum((state-targ)**2)

    reward = (1 - SSE/(sum(targ**2)))/10
    done = False

    '''
    if any(state < 20): # try and prevent extinction
        reward = - 1
    '''

    if any(state < 10):
        reward = - 1
        done = True

    return reward, done

def fig_6_reward_function_two_step(state, action, next_state):

    N1_targ = 250
    N2_targ = 550
    targ = np.array([N1_targ, N2_targ])
    current_state = state[2:4]
    SSE = sum((current_state-targ)**2)

    reward = (1 - SSE/(sum(targ**2)))/10

    done = False

    if any(current_state < 10):
        reward = - 1
        done = True

    return reward, done


param_path = os.path.join(C_DIR, 'parameter_files/smaller_target_good_ICs.yaml')

save_path = 'use_trained_agent'

update_timesteps = 1
sampling_time = 4
delta_mode = False
tmax = 1000
explore_rate = 0

env = ChemostatEnv(param_path, sampling_time, update_timesteps, delta_mode)

agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_Cin_states**env.num_controlled_species], cost_function = fig_6_reward_function)
agent.predict(np.array([0,0]))

agent.load_network('/Users/ntreloar/Desktop/Projects/summer/fitted_Q_iteration/chemostat/double_aux/decaying_sample_time/repeat1/saved_network.h5')
#agent.save_network_tensorflow(os.path.dirname(os.path.abspath(__file__)) + '/100eps/training_on_random/')
#agent.load_network_tensorflow('/Users/Neythen/Desktop/summer/fitted_Q_iteration/chemostat/100eps/training_on_random')

trajectory = agent.run_online_episode(env, explore_rate, tmax, train = False)
test_r = np.array([t[2] for t in trajectory])
test_a = np.array([t[1] for t in trajectory])
values = np.array(agent.values)

env.plot_trajectory([0,1])
os.makedirs(save_path, exist_ok = True)
plt.savefig(save_path + '/populations.png')
np.save(save_path + '/trajectory.npy', env.sSol)


plt.figure()
plt.plot(test_r)
np.save(save_path + '/rewards.npy', test_r)
plt.savefig(save_path + '/rewards.png')


plt.figure()
plt.plot(test_a)
np.save(save_path + '/actions.npy', test_a)
plt.savefig(save_path + '/actions.png')

plt.figure()
for i in range(4):
    plt.plot(values[:, i], label = 'action ' + str(i))
plt.legend()

plt.savefig(save_path + '/values.png')
