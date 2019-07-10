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


def fig_6_reward_function_new(state, action, next_state):

    N1_targ = 250
    N2_targ = 550
    targ = np.array([N1_targ, N2_targ])

    SE = sum(np.abs(state-targ))

    reward = (1 - sum(SE/targ)/2)/10
    done = False


    if any(state < 10):
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
    param_path = os.path.join(C_DIR, 'parameter_files/smaller_target_good_ICs.yaml')
    update_timesteps = 1
    sampling_time = 4
    delta_mode = False
    tmax = 1000
    n_episodes = 50
    train_times = []
    train_rewards = []
    test_times = []
    test_rewards = []
    env = ChemostatEnv(param_path, sampling_time, update_timesteps, delta_mode)

    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_Cin_states**env.num_controlled_species], cost_function = fig_6_reward_function_new)

    for i in range(n_episodes):

        env = ChemostatEnv(param_path, sampling_time, update_timesteps, delta_mode)
        print('EPISODE: ', i)
        # training EPISODE
        #explore_rate = 0
        explore_rate = agent.get_rate(i, 0, 1, n_episodes/10)
        #explore_rate = 1
        print(explore_rate)
        env.reset()
        #env.state = (np.random.uniform(-0.5, 0.5), 0, np.random.uniform(-0.5, 0.5), 0)
        trajectory, train_r = agent.run_episode(env, explore_rate, tmax)
        train_times.append(len(trajectory))
        train_rewards.append(train_r)

        '''
        env.reset()
        explore_rate = 0.1
        #env.state = (np.random.uniform(-0.5, 0.5), 0, np.random.uniform(-0.5, 0.5), 0)
        trajectory, train_r = agent.run_episode(env, explore_rate, tmax, train = True)

        #hint_to_goal(agent.network, agent.optimiser)
        print('Train Time: ', len(trajectory))
        '''
        # testing EPISODE
        explore_rate = 0
        env.reset()
        #env.state = (np.random.uniform(-1, 1), 0, np.random.uniform(-0.5, 0.5), 0)
        trajectory, test_r = agent.run_episode(env, explore_rate, tmax, train = False)
        print('Test Time: ', len(trajectory))

        test_times.append(len(trajectory))
        test_rewards.append(test_r)
        '''
        if test_r > 30:
            env.plot_trajectory([0,1])
            plt.show()
        '''
        print()



    test_rewards = np.array(test_rewards)
    train_rewards = np.array(train_rewards)
    test_times = np.array(test_times)
    train_times = np.array(train_times)

    os.makedirs(save_path, exist_ok = True)
    np.save(save_path + '/test_rewards.npy', test_rewards)
    np.save(save_path + '/train_rewards.npy', train_rewards)
    np.save(save_path + '/test_times.npy', test_times)
    np.save(save_path + '/train_times.npy', train_times)

    agent.save_network(save_path)

    plt.figure()
    plt.plot(train_times)
    plt.xlabel('Timestep')
    plt.ylabel('Timesteps until terminal state')
    plt.savefig(save_path + '/train_times.png')

    plt.figure()
    plt.plot(test_times)
    plt.xlabel('Timestep')
    plt.ylabel('Timesteps until terminal state')
    plt.savefig(save_path + '/test_times.png')

    env.plot_trajectory([0,1])
    plt.savefig(save_path + '/populations.png')
    np.save(save_path + '/trajectory.npy', env.sSol)

    plt.figure()
    plt.plot(test_rewards)
    plt.savefig(save_path + '/test_rewards.png')
    plt.figure()
    plt.plot(train_rewards)
    plt.savefig(save_path + '/train_rewards.png')


    # test trained policy with smaller time step


if __name__ == '__main__':
    entry()