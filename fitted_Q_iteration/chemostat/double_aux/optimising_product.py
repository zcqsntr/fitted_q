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

from double_aux_rewards import *

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
    param_path = os.path.join(C_DIR, 'parameter_files/product.yaml')
    update_timesteps = 1
    one_min = 0.016666666667
    n_mins = 10

    sampling_time = n_mins*one_min
    delta_mode = False
    tmax = int((24*60)/n_mins) # set this to 24 hours
    #tmax = 10
    print('tmax: ', tmax)
    n_episodes = 30
    train_times = []
    train_rewards = []
    test_times = []
    test_rewards = []
    pop_scaling = 100000
    env = ProductEnv(param_path, no_LV_reward_function_new_target, sampling_time, update_timesteps, pop_scaling, delta_mode)

    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_Cin_states**env.num_controlled_species])
    #agent.load_network('/Users/ntreloar/Desktop/Projects/summer/fitted_Q_iteration/chemostat/double_aux/new_target/repeat9/saved_network.h5')
    #agent.load_network('/Users/ntreloar/Desktop/Projects/summer/fitted_Q_iteration/chemostat/double_aux/results/100eps/training_on_random/saved_network.h5')
    train_trajectorys = []
    for i in range(n_episodes):
        print('EPISODE: ', i)
        print('train: ')
        # training EPISODE
        #explore_rate = 0
        explore_rate = agent.get_rate(i, 0, 1, n_episodes/10)
        #explore_rate = 1
        print(explore_rate)
        env.reset()
        #env.state = (np.random.uniform(-0.5, 0.5), 0, np.random.uniform(-0.5, 0.5), 0)
        train_trajectory, train_r = agent.run_episode(env, explore_rate, tmax)
        train_trajectorys.append(train_trajectory)
        train_times.append(len(train_trajectory))
        train_rewards.append(train_r)
        train_actions = np.array(agent.actions)

        values = np.array(agent.values)
        env.plot_trajectory([0,1])


        '''
        plt.figure()
        for i in range(4):
            plt.plot(values[:, i], label = 'action ' + str(i))
        plt.legend()

        plt.figure()

        plt.plot(agent.single_ep_reward)

        env.plot_trajectory([0,1])

        plt.show()
        '''

        # testing EPISODE
        explore_rate = 0
        print('test: ')
        env.reset()
        #env.state = (np.random.uniform(-1, 1), 0, np.random.uniform(-0.5, 0.5), 0)
        test_trajectory, test_r = agent.run_episode(env, explore_rate, tmax, train = False)
        test_actions = np.array(agent.actions)
        print('Test Time: ', len(test_trajectory))
        env.plot_trajectory([0,1])


        test_times.append(len(test_trajectory))
        test_rewards.append(test_r)
        print(test_rewards)

        '''
        if test_r > 10:
            env.plot_trajectory([0,1])
            plt.show()
        '''
        print()

    os.makedirs(save_path, exist_ok = True)

    # use trained policy on env with smaller smaplingn time
    #sampling_time = 0.1

    exploit_env = ProductEnv(param_path, no_LV_reward_function_new_target, sampling_time, update_timesteps, pop_scaling, delta_mode)
    # testing EPISODE
    explore_rate = 0
    print('test: ')
    exploit_env.reset()
    tmax = 100
    #env.state = (np.random.uniform(-1, 1), 0, np.random.uniform(-0.5, 0.5), 0)
    exploit_trajectory, exploit_r = agent.run_episode(exploit_env, explore_rate, tmax, train = False)
    exploit_env.plot_trajectory([0,1]) # the last test_trajectory
    plt.savefig(save_path + '/exploit_populations.png')
    np.save(save_path + '/exploit_trajectory.npy', exploit_trajectory)
    np.save(save_path + '/exploit_actions.npy', np.array(agent.actions))

    np.save(save_path + '/train_trajectories.npy', np.array(train_trajectorys))
    test_rewards = np.array(test_rewards)
    train_rewards = np.array(train_rewards)
    test_times = np.array(test_times)
    train_times = np.array(train_times)


    np.save(save_path + '/test_rewards.npy', test_rewards)
    np.save(save_path + '/train_rewards.npy', train_rewards)
    np.save(save_path + '/test_times.npy', test_times)
    np.save(save_path + '/train_times.npy', train_times)
    np.save(save_path + '/test_actions.npy', test_actions)
    np.save(save_path + '/train_actions.npy', train_actions)

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

    env.plot_trajectory([0,1]) # the last test_trajectory
    plt.savefig(save_path + '/test_populations.png')
    np.save(save_path + '/test_trajectory.npy', env.sSol)


    # plot the last train trajectory
    plt.figure()
    xSol = np.array(train_trajectory)
    for i in [0,1]:
        plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = env.labels[i])
    plt.legend()
    plt.savefig(save_path + '/train_populations.png')
    np.save(save_path + '/train_trajectory.npy', train_trajectory)


    plt.figure()
    plt.plot(test_rewards)
    plt.savefig(save_path + '/test_rewards.png')
    plt.figure()
    plt.plot(train_rewards)
    plt.savefig(save_path + '/train_rewards.png')

    values = np.array(agent.values)
    plt.figure()
    for i in range(4):
        plt.plot(values[:, i], label = 'action ' + str(i))
    plt.legend()

    plt.savefig(save_path + '/values.png')

    print(env.sSol)
    print()
    values = np.array(agent.values)



    # test trained policy with smaller time step


if __name__ == '__main__':
    entry()
