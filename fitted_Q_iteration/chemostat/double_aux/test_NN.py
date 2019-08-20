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
from argparse import ArgumentParser

np.set_printoptions(precision = 16)
def no_LV_reward_function_new_target(state, action, next_state):

    N1_targ = 10000
    N2_targ = 20000
    targ = np.array([N1_targ, N2_targ])
    SE = sum(np.abs(state-targ))

    reward = (1 - sum(SE/targ)/2)/10
    done = False


    if any(state < 1000):
        reward = - 1
        done = True

    return reward, done

def no_LV_reward_function_new_target_two_step(state, action, next_state):

    N1_targ = 10000
    N2_targ = 20000
    targ = np.array([N1_targ, N2_targ])
    state = state[2:4]
    SE = sum(np.abs(state-targ))

    reward = (1 - sum(SE/targ)/2)/10
    done = False


    if any(state < 0):
        reward = - 1
        done = True

    return reward, done


def entry():
    '''
    Entry point for command line application handle the parsing of arguments and runs the relevant agent
    '''
    # define arguments
    parser = ArgumentParser(description = 'Bacterial control app')
    parser.add_argument('-s', '--save_path')
    parser.add_argument('-r', '--repeat')
    arguments = parser.parse_args()

    # get number of repeats, if not supplied set to 1
    repeat = int(arguments.repeat)

    save_path = os.path.join(arguments.save_path, 'repeat' + str(repeat))
    print(save_path)
    run_test(save_path)

def run_test(save_path):
    param_path = os.path.join(C_DIR, 'parameter_files/smaller_target_good_ICs_no_LV.yaml')
    update_timesteps = 1
    one_min = 0.016666666667
    n_mins = 1
    sampling_time = n_mins*one_min
    delta_mode = False
    tmax = 100
    pop_scaling = 100000

    env = ChemostatEnv(param_path, no_LV_reward_function_new_target, sampling_time, update_timesteps, pop_scaling, delta_mode)
    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_Cin_states**env.num_controlled_species])
    state = env.reset()


    trajectory = []
    actions = []
    rewards = []


    for n_mins in [5]:
        sampling_time = n_mins*one_min
        delta_mode = False
        tmax = int((24*60)/n_mins) # set this to 24 hours
        tmax = 100
        print('tmax: ', tmax)
        train_times = []
        train_rewards = []
        test_times = []
        test_rewards = []

        env = ChemostatEnv(param_path, no_LV_reward_function_new_target, sampling_time, update_timesteps, pop_scaling, delta_mode)

        explore_rate = 1
        all_pred_rewards = []
        all_actual_rewards = []
        n_repeats = 100
        os.makedirs(save_path, exist_ok = True)
        for repeat in range(n_repeats):

            #agent.load_network('/Users/ntreloar/Desktop/Projects/summer/fitted_Q_iteration/chemostat/double_aux/new_target/repeat9/saved_network.h5')
            #agent.load_network('/Users/ntreloar/Desktop/Projects/summer/fitted_Q_iteration/chemostat/double_aux/results/100eps/training_on_random/saved_network.h5')
            # generate training data
            env.reset()
            agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_Cin_states**env.num_controlled_species])
            #env.state = (np.random.uniform(-0.5, 0.5), 0, np.random.uniform(-0.5, 0.5), 0)
            train_trajectory, train_r = agent.run_episode(env, explore_rate, tmax)

            for i in range(20):
                agent.fitted_Q_update()

            values = []

            #predict values after training
            agent.values = []

            for state in train_trajectory[:,0:2]:
                agent.get_action(state/100000, 0) # appends to agent.values


            print(len(agent.values))
            print(len(agent.actions))

            values = np.array(agent.values)

            pred_rewards = []

            for i in range(len(agent.actions)):
                action_values = values[i]
                action_taken = agent.actions[i]
                pred_rewards.append(action_values[action_taken])

            all_pred_rewards.append(pred_rewards)
            all_actual_rewards.append(agent.single_ep_reward)



            print('pred_rewards:', pred_rewards)
            print()
            print('single_ep_reward:', agent.single_ep_reward)

            plt.figure()
            plt.plot(agent.single_ep_reward, label = 'actual')
            plt.plot(pred_rewards, label = 'pred')
            plt.legend()
            plt.show()
            '''
            plt.savefig(save_path + '/' + str(n_mins))
            '''
        np.save(save_path + '/' + str(n_mins) +'all_actual_r', all_actual_rewards)
        np.save(save_path + '/' + str(n_mins) + 'all_pred_r', all_pred_rewards)
        np.save(save_path + '/' + str(n_mins) + 'av_SSE', sum((np.array(all_actual_rewards) - np.array(all_pred_rewards))**2))
        print('all_actual: ',all_actual_rewards)
        print('all_pred:' ,all_pred_rewards)
        print(sum(sum((np.array(all_actual_rewards) - np.array(all_pred_rewards))**2)))




    '''
    train_trajectory, train_r = agent.run_episode(env, 0, tmax)
    print(train_r)
    # plot the last train trajectory
    plt.figure()
    xSol = np.array(train_trajectory)
    for i in [0,1]:
        plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = env.labels[i])
    plt.legend()
    plt.show()
    '''




if __name__ == '__main__':
    entry()
