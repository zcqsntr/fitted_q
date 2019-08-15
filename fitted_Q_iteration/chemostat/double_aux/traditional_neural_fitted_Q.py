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


# Fig 6 in preprint
def fig_6_reward_function(state, action, next_state):

    N1_targ = 250
    N2_targ = 550
    targ = np.array([N1_targ, N2_targ])

    SSE = sum((state-targ)**2)

    reward = (1 - SSE/(sum(targ**2)))/10

    done = False


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

def smaller_target_reward(X, action, next_state):
    if 100 < X[0] < 400 and 400 < X[1] < 700:
        reward = 1
    else:
        reward = -1

    done = False
    if any(X < 10):
        reward = - 10
        done = True
    return reward, done

def fig_6_reward_function_new_target(state, action, next_state):

    N1_targ = 250
    N2_targ = 700
    targ = np.array([N1_targ, N2_targ])

    SE = sum(np.abs(state-targ))

    reward = (1 - sum(SE/targ)/2)/10
    done = False


    if any(state < 1):
        reward = - 1
        done = True

    return reward, done

def fig_6_reward_function_new_target_two_step(state, action, next_state):

    N1_targ = 250
    N2_targ = 700
    targ = np.array([N1_targ, N2_targ])
    state = state[2:4]
    SE = sum(np.abs(state-targ))

    reward = (1 - sum(SE/targ)/2)/10
    done = False


    if any(state < 1):
        reward = - 1
        done = True

    return reward, done

def no_LV_reward_function_new_target(state, action, next_state):

    N1_targ = 20000
    N2_targ = 30000
    targ = np.array([N1_targ, N2_targ])
    SE = sum(np.abs(state-targ))

    reward = (1 - sum(SE/targ)/2)/10
    done = False


    if any(state < 100):
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
    delta_mode = False
    tmax = 1000
    n_episodes = 1
    one_min = 0.016666666667
    n_mins = 10

    sampling_time = n_mins*one_min
    tmax = int((24*60)/n_mins)
    times = []
    rewards = []
    env = ChemostatEnv(param_path, sampling_time, update_timesteps, delta_mode)

    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_controlled_species*env.num_Cin_states], cost_function = no_LV_reward_function_new_target)


    # generate data, need to turn update Q off for this
    for i in range(n_episodes):
        print('EPISODE: ', i)

        # training EPISODE
        #explore_rate = 0
        #explore_rate = agent.get_rate(i, 0, 1, 2.5)
        explore_rate = 1
        print(explore_rate)

        env.reset()
        #env.state = (np.random.uniform(-0.5, 0.5), 0, np.random.uniform(-0.5, 0.5), 0)
        trajectory, train_r = agent.run_episode(env, explore_rate, tmax, train= True)

        env.plot_trajectory([0,1])
        plt.show()
    print('number of training points: ', len(trajectory))


    # train iteratively on data
    train_rs = []
    losses = []
    for i in range(100):
        print('EPISODE: ', i)
        history = agent.fitted_Q_update()
        print()


        if i%10 == 0:
            explore_rate = 0
            env.reset()
            trajectory, train_r = agent.run_episode(env, explore_rate, tmax, train= False)
            if train_r > 9:
                env.plot_trajectory([0,1])
                plt.show()
            losses.append(history.history['loss'])
            train_rs.append(train_r)

    plt.figure()
    plt.plot(losses)
    plt.figure()
    plt.plot(train_rs)
    print(train_r)

    explore_rate = 0
    env.reset()
    trajectory, train_r = agent.run_episode(env, explore_rate, tmax, train= False)


    rewards = np.array(rewards)
    os.makedirs(save_path, exist_ok = True)
    agent.save_results(save_path)
    agent.save_network(save_path)
    plt.figure()
    plt.plot(times)

    plt.xlabel('Timestep')
    plt.ylabel('Timesteps until terminal state')
    plt.savefig(save_path + '/times.png')

    env.plot_trajectory([0,1])
    plt.savefig(save_path + '/populations.png')
    np.save(save_path + '/trajectory.npy', env.sSol)

    plt.figure()
    plt.plot(rewards)
    plt.savefig(save_path + '/episode_rewards.png')

if __name__ == '__main__':
    entry()
