import sys
import os
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
    param_path = os.path.join(C_DIR, 'parameter_files/smaller_target_good_ICs_no_LV.yaml')
    update_timesteps = 1
    n_mins = 5
    one_min = 0.016666666667

    sampling_time = n_mins*one_min
    delta_mode = False
    #tmax = int((24*60)/n_mins) # set this to 24 hours
    tmax = 10
    print('tmax: ', tmax)
    n_episodes = 29
    train_times = []
    train_rewards = []

    pop_scaling = 100000
    os.makedirs(save_path, exist_ok = True)
    os.makedirs(save_path + '/after_heuristic', exist_ok = True)
    env = ChemostatEnv(param_path, no_LV_reward_function_new_target, sampling_time, update_timesteps, pop_scaling, delta_mode)

    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_Cin_states**env.num_controlled_species])

    agent.learn_heuristic()

    agent.save_network(save_path + '/after_heuristic')
    agent.saved = save_path + '/after_heuristic'
    print(agent.saved)

    train_trajectory, train_r = agent.run_episode(env, 0, 100, train = False)
    env.plot_trajectory([0,1])
    plt.savefig(save_path + 'after_heuristic.png')
    env.reset()
    agent.memory = []


    overall_traj = []
    print(env.S)
    for i in range(n_episodes):

        print()
        print('EPISODE: ', i)
        print('train: ')
        # training EPISODE
        #
        #agent.memory = []

        explore_rate = agent.get_rate(i, 0.05, 1., n_episodes/10)

        #explore_rate = 1
        print(explore_rate)


        train_trajectory, train_r = agent.run_episode(env, explore_rate, tmax)
        print(env.S)
        os.makedirs(save_path + '/episode' + str(i), exist_ok = True)
        agent.save_network(save_path + '/episode' + str(i))

        train_times.append(len(train_trajectory))
        train_rewards.append(train_r)

        values = np.array(agent.values)

    for i in range(5):
        train_trajectory, train_r = agent.run_episode(env, 0, tmax)
        train_rewards.append(train_r)
        agent.save_network(save_path + '/episode' + str(i+n_episodes))

        values = np.array(agent.values)




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

        '''

        # testing EPISODE
        explore_rate = 0
        print('test: ')
        env.reset()
        test_trajectory, test_r = agent.run_episode(env, explore_rate, tmax, train = False)
        print('Test Time: ', len(test_trajectory))
        env.plot_trajectory([0,1])
        plt.show()

        test_times.append(len(test_trajectory))
        test_rewards.append(test_r)
        print(test_rewards)
        '''
        '''
        if test_r > 10:
            env.plot_trajectory([0,1])
            plt.show()
        '''
        print()
    env.plot_trajectory([0,1])
    plt.hlines([20000, 30000], 0, 400, color = 'g', label = 'target')
    env.plot_trajectory([2,3,4])

    plt.show()


    # use trained policy on env with smaller smaplingn time
    #sampling_time = 0.1

    exploit_env = ChemostatEnv(param_path, no_LV_reward_function_new_target, sampling_time, update_timesteps, pop_scaling, delta_mode)
    # testing EPISODE
    explore_rate = 0
    print('test: ')
    exploit_env.reset()
    tmax = 100
    #env.state = (np.random.uniform(-1, 1), 0, np.random.uniform(-0.5, 0.5), 0)
    exploit_trajectory, exploit_r = agent.run_episode(exploit_env, explore_rate, tmax, train = False)
    exploit_env.plot_trajectory([0,1]) # the last test_trajectory
    plt.hlines([20000, 30000], 0, 400, color = 'g', label = 'target')
    plt.savefig(save_path + '/exploit_populations.png')
    np.save(save_path + '/exploit_trajectory.npy', exploit_trajectory)

    train_rewards = np.array(train_rewards)


    np.save(save_path + '/train_rewards.npy', train_rewards)
    np.save(save_path + '/train_times.npy', train_times)

    agent.save_network(save_path)

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
    plt.plot(train_rewards)
    plt.savefig(save_path + '/train_rewards.png')

    values = np.array(agent.values)
    plt.figure()
    for i in range(4):
        plt.plot(values[:, i], label = 'action ' + str(i))
    plt.legend()

    plt.savefig(save_path + '/values.png')


    print()
    values = np.array(agent.values)



    # test trained policy with smaller time step


if __name__ == '__main__':
    entry()
