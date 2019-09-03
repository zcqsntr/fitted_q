import numpy as np

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


    if any(state < 1):
        reward = - 1
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


    if any(state < 1000):
        reward = - 1
        done = True

    return reward, done

def no_LV_reward_function_new_target_two_step(state, action, next_state):

    N1_targ = 20000
    N2_targ = 30000
    targ = np.array([N1_targ, N2_targ])
    state = state[2:4]
    SE = sum(np.abs(state-targ))

    reward = (1 - sum(SE/targ)/2)/10
    done = False


    if any(state < 1000):
        reward = - 1
        done = True

    return reward, done
