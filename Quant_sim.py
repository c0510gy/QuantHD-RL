import gym
import copy
import numpy as np
import torch
from time import time
from tqdm import tqdm
from QHD_QuantModel import QHD_Model
import json
from copy import deepcopy


def save_as_json(filename, data):

    with open(filename, 'w', encoding='utf8') as f:
        json.dump(data, f)


def gen_prob_table(prob, bits):

    table = [[prob * 100. / 2, prob * 100. / 2] for _ in range(2**bits)]
    table[0] = [prob * 100., 0.]
    table[-1] = [0., prob * 100.]

    return table


def run(env, bits, dimension, prob, noise_on_inference_only, device):

    history = []

    prob_table = gen_prob_table(prob, bits)

    env = env.unwrapped
    ts = 5 
    tau = 1 #update every 1 episode
    tau_step = 100 #update every 10 steps

    if task == 'cartpole':
        epsilon = 0.2
        epsilon_decay = 0.99
        reward_decay = 0.9
        EPISODES = 201

    minimum_epsilon = 0.01
    n_actions = env.action_space.n
    n_obs = env.observation_space.shape[0]
    model = QHD_Model(dimension, n_actions, n_obs, epsilon, epsilon_decay,
                    minimum_epsilon, reward_decay, train_sample_size=ts, lr=0.05, device=device, bits=bits) #lr=0.05
    copied_model = deepcopy(model)
    #with open(filename,'a') as f:
    #    f.write("Episode,Steps,Reward,Runtime\n")

    total_runtime = 0
    total_step = 0
    for episode in tqdm(range(EPISODES)):
        start = time()
        rewards_sum = 0
        obs = env.reset()
        model.n_steps = 0

        while True:
            #env.render()
            if noise_on_inference_only:
                copied_model = deepcopy(model)
                copied_env = deepcopy(env)
            
            action = int(model.act(obs))
            new_obs, reward, done, info = env.step(action)
            if task == 'cartpole':
                if done:
                    reward = -5
            model.store_transition(obs, action, reward, new_obs, done)
            
            model.feedback()

            if noise_on_inference_only:
                copied_model.random_bit_flip_by_prob(prob_table)

                action = int(copied_model.act(obs))
                new_obs, reward, done, info = copied_env.step(action)
                rewards_sum += reward
            else:
                rewards_sum += reward
                model.random_bit_flip_by_prob(prob_table)

            total_step += 1
            if task == 'cartpole':
                if total_step % tau_step == 0:
                    model.delay_model = copy.deepcopy(model.model)

            if rewards_sum > 1000:
                done = True

            # if done:
            #     if task == 'cartpole':
            #         if episode % tau == 0: # 5
            #             model.delay_model = copy.deepcopy(model.model)

            if done:
                end = time()
                total_runtime += end - start
                #print('Episode: ', episode)
                #print('Episode Rewards: ', rewards_sum)

                history.append(rewards_sum)
                #print('Total Runtime: ', total_runtime)
                #with open(filename,'a') as f:
                #    f.write(str(episode)+','+str(total_step)+','+str(rewards_sum)+','+str(total_runtime)+'\n')
                break

            model.n_steps += 1
            obs = new_obs

        model.epsilon = max(model.epsilon * model.epsilon_decay, 
                                model.minimum_epsilon)
    
    return history

def run_all(dir='./'):
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    tasks = ['cartpole']
    for task in tasks:
        if task == 'cartpole':
            env = gym.make('CartPole-v0')
        
        dims = [10000]
        probs = [0., 0.20, 0.50, 0.90]

        for dimension in dims:
            for prob in probs:

                filename = f'{dir}01/qhd_{task}_{dimension}_{prob}_results.json'
                print(filename)

                history = run(env, 3, dimension, prob, True, device)
                
                save_as_json(filename, history)
