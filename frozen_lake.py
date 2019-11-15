import numpy as np
import gym
import random



env = gym.make("FrozenLake-v0")

action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))

#hyperparameters

total_episodes = 15000
learning_rate = 0.8
max_steps  = 99
gamma = 0.95

#Exploration parameters

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.001
decay_rate = 0.005

# List of rewards
rewards = []



for episode in range(total_episodes):
    #reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0,1)

        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])

        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)

        #Update qtable
        
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards += reward

        state = new_state

        if done == True:
            break

        episode += 1

# reduce epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)

print("Score over time:", str(sum(rewards)/total_episodes))
print(qtable)

env.reset()


for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("--------------------------")
    print ("Episode ", episode)
    for step in range(max_steps):
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)

        if done:
            env.render()
            print("number of steps", step)
            break
        state = new_state

env.close()






