import numpy as np


# Define the states
location_to_state = {
    'L1' : 0,
    'L2' : 1,
    'L3' : 2,
    'L4' : 3,
    'L5' : 4
    'L6' : 5
    'L7' : 6
    'L8' : 7

}

#Define the rewards

rewards = np.array([[0,1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0]])


#maps indices to locations

state_to_location = dict((state, location) for location, state in location_to_state.items())

#initialize parameters
gamma = 0.75 # discount factor
alpha = 0.9 # learning rate

#init q-values
Q = np.array(np.zeros([8,8]))

#copy the rewards matrix to new matrix
rewards_copy = np.copy(rewards)