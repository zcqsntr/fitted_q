from chemostat_envs import *

import yaml
import matplotlib.pyplot as plt

def fig_6_reward_function(state, action, next_state):

    N1_targ = 250
    N2_targ = 550
    targ = np.array([N1_targ, N2_targ])

    SSE = sum((state-targ)**2)

    reward = (1 - SSE/(sum(targ**2)))/10
    done = False
    if reward < 0:
        print('Reward smaller than 0: ', reward)

    if any(state < 10):
        reward = - 1
        done = True

    return reward, done

def test_trajectory():
    param_file = '/Users/Neythen/Desktop/summer/chemostat_env/parameter_files/smaller_target_good_ICs.yaml'


    update_timesteps = 1
    sampling_time = 4
    env = ChemostatEnv(param_file, sampling_time, update_timesteps, False)
    rew = 0

    actions = []
    for i in range(1000):

        #a = np.random.choice(range(4))

        #a = 3
        #print(a)
        '''
        a = 3
        if i == 400:
            a = 2

        if i == 500:
            a = 1
        '''
        a = 2

        state = env.get_state()
        r, done = fig_6_reward_function(state, None, None)
        print(r)
        rew += r
        env.step(a)
        if done:
            break

        actions.append(a)
    print(actions)

    env.plot_trajectory([0,1,2,3,4])
    plt.show()

    print(rew)
if __name__ == '__main__':
    test_trajectory()
