import sys
import os


import numpy as np
import tensorflow as tf
import math
import random

from tensorflow import keras


'''
import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
'''
import matplotlib.pyplot as plt

class FittedQAgent():

    '''
    abstract class for the Torch and Keras implimentations, dont use directly

    '''
    def get_action(self, state, explore_rate):


        if np.random.random() < explore_rate:
            action = np.random.choice(range(self.layer_sizes[-1]))
        else:
            values = self.predict(state)
            self.values.append(values)
            action = np.argmax(values)
        assert action < self.n_actions, 'Invalid action'
        return action

    def get_inputs_targets_online(self): #YES

        # DO THIS WITH NUMPY TO MAKE IT FASTER
        inputs = []
        targets = []

        for transition in self.memory:

            # CHEKC TARGET IS BUILT CORRECTLY

            state, action, cost, next_state, done = transition
            inputs.append(state)
            # construct target
            values = self.predict(state)
            next_values = self.predict(next_state)

            assert len(values) == self.n_actions, 'neural network returning wrong number of values'
            assert len(next_values) == self.n_actions, 'neural network returning wrong number of values'

            #update the value for the taken action using cost function and current Q


            if not done:
                values[action] = cost + self.gamma*np.max(next_values) # could introduce step size here, maybe not needed for neural agent
            else:
                values[action] = cost

            targets.append(values)

        inputs, targets = np.array(inputs), np.array(targets)

        # output = transition_cost(state_i, action_i, next_state_i) + y*min_action Q(state, action)
        # predicted_Qs is the minimum Q_value for the next state in every transition in memory
        # targets is the transition cost of every transition in memory + y*predicted_Qs
        return inputs, targets

    def get_inputs_targets(self): #YES
        inputs = []
        targets = []

        # DO THIS WITH NUMPY TO MAKE IT FASTER
        for trajectory in self.memory:

            for transition in trajectory:
                # CHEKC TARGET IS BUILT CORRECTLY

                state, action, cost, next_state, done = transition
                inputs.append(state)
                # construct target
                values = self.predict(state)
                next_values = self.predict(next_state)

                assert len(values) == self.n_actions, 'neural network returning wrong number of values'
                assert len(next_values) == self.n_actions, 'neural network returning wrong number of values'

                #update the value for the taken action using cost function and current Q
                if not done:
                    values[action] = cost + self.gamma*np.max(next_values) # could introduce step size here, maybe not needed for neural agent
                else:
                    values[action] = cost

                targets.append(values)

        # shuffle inputs and target for IID
        inputs, targets  = np.array(inputs), np.array(targets)

        # shuffle works
        randomize = np.arange(len(inputs))
        np.random.shuffle(randomize)
        inputs = inputs[randomize]
        targets = targets[randomize]


        # output = transition_cost(state_i, action_i, next_state_i) + y*min_action Q(state, action)
        # predicted_Qs is the minimum Q_value for the next state in every transition in memory
        # targets is the transition cost of every transition in memory + y*predicted_Qs


        assert inputs.shape[1] == self.state_size, 'inputs to network wrong size'
        assert targets.shape[1] == self.n_actions, 'targets for network wrong size'
        return inputs, targets

    def fitted_Q_update(self):
        inputs, targets = self.get_inputs_targets()
        #
        #tf.initialize_all_variables() # resinitialise netowrk without adding to tensorflow graph
        # try RMSprop and adam and maybe some from here https://arxiv.org/abs/1609.04747
        self.reset_weights()

        history = self.fit(inputs, targets)
        print('---------', history.history['loss'][-1])
        return history
        #print('loss:', history.history)

    def online_Q_learning_update(self, transition):
        state, action, cost, next_state, done = transition

        # construct target
        values = self.predict(state)
        next_values = self.predict(next_state)


        #update the value for the taken action using cost function and current Q
        if not done:
            values[action] = cost + self.gamma*np.max(next_values) # could introduce step size here, maybe not needed for neural agent
        else:
            values[action] = cost

        loss = self.fit(state.reshape(1, -1), values.reshape(1, -1))

    def online_fitted_Q_update(self, transition):
        inputs, targets = self.get_inputs_targets_online()
        #
        #tf.initialize_all_variables() # resinitialise netowrk without adding to tensorflow graph
        # try RMSprop and adam and maybe some from here https://arxiv.org/abs/1609.04747
        #self.reset_weights()

        loss = self.fit(inputs, targets)
        self.total_loss += loss

    def run_episode(self, env, explore_rate, tmax, train = True, render = False):
        # run trajectory with current policy and add to memory
        trajectory = []
        actions = []
        state = np.array(env.get_state())
        print('initial state: ', state)
        episode_reward = 0
        print('n_vars: ', len(tf.all_variables()))

        for i in range(tmax):
            if render: env.render()
            action = self.get_action(state, explore_rate)
            actions.append(action)

            next_state, reward, done, info = env.step(action)

            #cost = -cost # as cartpole default returns a reward
            assert len(next_state) == self.state_size, 'env return state of wrong size'

            reward, done = self.transition_cost(state, action, next_state) # use this for custom transition cost
            if done:
                print(reward)
            transition = (state, action, reward, next_state, done)
            state = next_state
            trajectory.append(transition)
            episode_reward += reward
            if done: break

        self.episode_lengths.append(i)
        self.episode_rewards.append(episode_reward)
        print(actions)
        print('reward:', episode_reward)

        if train:
            self.memory.append(trajectory)

            '''
            if len(self.memory) > 10:
                self.memory = self.memory[1:]
            '''
            #self.memory = trajectory
            self.fitted_Q_update()

        #env.plot_trajectory()
        #plt.show()
        return trajectory, episode_reward

    def run_online_episode(self, env, explore_rate, tmax, train = True, render = False):
        # run trajectory with current policy and add to memory
        trajectory = []
        actions = []
        state = np.array(env.get_state())
        print('initial state: ', state)

        print('n_vars: ', len(tf.all_variables()))

        for i in range(tmax):
            print(i)
            if render: env.render()
            action = self.get_action(state, explore_rate)
            actions.append(action)

            next_state, reward, done, info = env.step(action)

            #cost = -cost # as cartpole default returns a reward
            assert len(next_state) == self.state_size, 'env return state of wrong size'

            reward, done = self.transition_cost(state, action, next_state) # use this for custom transition cost

            transition = (state, action, reward, next_state, done)

            # update Q function online
            if train:
                #self.memory.append(transition)
                self.online_Q_learning_update(transition)


            state = next_state
            trajectory.append(transition)

            if done: break

        self.episode_lengths.append(i)
        print(actions)

        #env.plot_trajectory()
        #plt.show()
        return trajectory

    def run_mutation_episode(self, env, explore_rate, tmax, train = True, render = False):
        # run trajectory with current policy and add to memory
        trajectory = []
        actions = []
        state = np.array(env.get_state())
        print('initial state: ', state)

        print('n_vars: ', len(tf.all_variables()))

        for i in range(tmax):
            print(i)
            if render: env.render()
            action = self.get_action(state, explore_rate)
            actions.append(action)

            next_state, reward, done, info = env.step_mutation_experiment(action)

            #cost = -cost # as cartpole default returns a reward
            assert len(next_state) == self.state_size, 'env return state of wrong size'

            reward, done = self.transition_cost(state, action, next_state) # use this for custom transition cost

            transition = (state, action, reward, next_state, done)

            # update Q function online
            if train:
                print(len(self.memory))
                if len(self.memory) > 60:
                    self.memory = self.memory[1:]
                self.memory.append(transition)
                self.online_Q_learning_update(transition)


            state = next_state
            trajectory.append(transition)

            if done: break

        self.episode_lengths.append(i)
        print(actions)

        #env.plot_trajectory()
        #plt.show()
        return trajectory

    def neural_fitted_Q(self, env, n_episodes, tmax):
        # initialise agent

        times = []
        for i in range(n_episodes):
            print()
            print('EPISODE', i)


            # CONSTANT EXPLORE RATE OF 0.1 worked well
            explore_rate = self.get_rate(i, 0, 1, 2.5)
            #explore_rate = 0.1
            #explore_rate = 0
            print('explore_rate:', explore_rate)
            env.reset()
            trajectory, reward = self.run_episode(env, explore_rate, tmax)

            time = len(trajectory)
            print('Time: ', time)
            times.append(time)

        print(times)

    def plot_rewards(self):
        plt.figure(figsize = (16.0,12.0))

        plt.plot(self.episode_rewards)

    def save_results(self, save_path):
        np.save(save_path + '/survival_times', self.episode_lengths)
        np.save(save_path + '/episode_rewards', self.episode_rewards)

    def get_rate(self, episode, MIN_LEARNING_RATE,  MAX_LEARNING_RATE, denominator):
        '''
        Calculates the logarithmically decreasing explore or learning rate

        Parameters:
            episode: the current episode
            MIN_LEARNING_RATE: the minimum possible step size
            MAX_LEARNING_RATE: maximum step size
            denominator: controls the rate of decay of the step size
        Returns:
            step_size: the Q-learning step size
        '''

        # input validation
        if not 0 <= MIN_LEARNING_RATE <= 1:
            raise ValueError("MIN_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 <= MAX_LEARNING_RATE <= 1:
            raise ValueError("MAX_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 < denominator:
            raise ValueError("denominator needs to be above 0")

        rate = max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, 1.0 - math.log10((episode+1)/denominator)))

        return rate

class KerasFittedQAgent(FittedQAgent):
    def __init__(self, layer_sizes = [2,20,20,4], cost_function = False):
        self.memory = []
        self.layer_sizes = layer_sizes
        self.network = self.initialise_network(layer_sizes)
        self.gamma = 0.95
        self.state_size = layer_sizes[0]
        self.n_actions = layer_sizes[-1]
        self.episode_lengths = []
        self.episode_rewards = []
        self.total_loss = 0
        self.values = []

        print(cost_function)
        if cost_function:
            self.transition_cost = cost_function

    def initialise_network(self, layer_sizes): #YES

        initialiser = keras.initializers.RandomUniform(minval = -0.5, maxval = 0.5, seed = None)
        network = keras.Sequential([
            keras.layers.InputLayer([layer_sizes[0]]),
            keras.layers.Dense(layer_sizes[1], activation = tf.nn.sigmoid, kernel_initializer = initialiser),
            keras.layers.Dense(layer_sizes[2], activation = tf.nn.sigmoid, kernel_initializer = initialiser),
            keras.layers.Dense(layer_sizes[3], kernel_initializer = initialiser) # linear output layer
        ])

        network.compile(optimizer = 'adam', loss = 'mean_squared_error') # TRY DIFFERENT OPTIMISERS
        return network

    def predict(self, state): #YES

        return self.network.predict(state.reshape(1,-1))[0]

    def fit(self, inputs, targets):
        history = self.network.fit(inputs, targets,  epochs = 300, verbose = 0) # TRY DIFFERENT EPOCHS
        return history

    def reset_weights(model): # verified working
        sess = tf.keras.backend.get_session()
        #print(sess.run(tf.all_variables()[0]))
        sess.run(tf.global_variables_initializer())
        #print(sess.run(tf.all_variables()[0]))
    def save_network(self, save_path): # tested
        #print(self.network.layers[1].get_weights())
        self.network.save(save_path + '/saved_network.h5')

    def save_network_tensorflow(self, save_path):
        saver = tf.train.Saver()
        print('-------------------------', save_path)
        sess = tf.keras.backend.get_session()
        path = saver.save(sess, save_path + "/saved/model.cpkt")
        print('---------', path)

    def load_network_tensorflow(self, save_path):
        saver = tf.train.Saver()
        #print(self.network.layers[1].get_weights())
        sess = tf.keras.backend.get_session()
        saver.restore(sess, save_path +"/saved/model.cpkt")
        #print(self.network.layers[1].get_weights())


    def load_network(self, load_path): #tested

        try:
            self.network = keras.models.load_model(load_path) # sometimes this crashes, apparently a bug in keras
        except:
            self.network.load_weights(load_path) # this requires model to be initialised exactly the same

'''
class TorchFittedQAgent(FittedQAgent):

    def __init__(self, layer_sizes = [2,20,20,4], cost_function = False):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('DEVICE: ', self.dev)
        self.memory = []
        self.layer_sizes = layer_sizes
        self.network = self.initialise_network(layer_sizes)
        self.gamma = 0.95
        self.state_size = layer_sizes[0]
        self.n_actions = layer_sizes[-1]
        self.episode_lengths = []
        self.episode_costs = []
        self.total_loss = 0

        if cost_function:
            self.transition_cost = cost_function


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.uniform_(m.weight, a = -0.5, b = 0.5)

    def initialise_network(self, layer_sizes): #YES

        network = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.Sigmoid(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.Sigmoid(),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.Sigmoid()

        ).to(self.dev)


        layers = []
        for i in range(len(layer_sizes)-2):
            #hidden layers
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.Sigmoid())
        i += 1
        # output layer
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        layers.append(nn.Tanh())
        network = nn.Sequential(*layers).to(self.dev)

        #network.apply(self.init_weights)
        self.optimiser = optim.Rprop(network.parameters(), lr=1e-2)
        return network.double()

    def predict(self, state): #YES
        state = torch.from_numpy(state).to(self.dev)
        return self.network(state).detach().numpy()

    def fit(self, inputs, targets): #YES
        inputs = torch.from_numpy(inputs).to(self.dev)
        targets = torch.from_numpy(targets).to(self.dev)
        for _ in range(300):

            predicted_Qs = self.network(inputs).to(self.dev)
            loss = F.mse_loss(predicted_Qs, targets).to(self.dev)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        return loss

    def reset_weights(self):
        self.network = self.initialise_network()

'''

'''
import gym
env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)

for episode in range(20):
    observation = env.reset()
    for t in range(10):
        env.render()
        action = 0
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("episode failed after {} timesteps".format(t+1))
            break

env.close()
'''
