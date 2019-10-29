import numpy as np


# Define the states
location_to_state = {
    'L1' : 0,
    'L2' : 1,
    'L3' : 2,
    'L4' : 3,
    'L5' : 4,
    'L6' : 5,
    'L7' : 6,
    'L8' : 7

}

#Define the actions
actions = [0,1,2,3,4,5,6,7]

#Define the rewards
rewards = np.array([[0,1,0,0,0,0,0,0],
    [1,0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,1,1,0,1,0,0,0],
    [0,0,0,1,0,1,0,0],
    [0,0,0,0,1,0,1,0],
    [0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,1,0]])


#maps indices to locations
state_to_location = dict((state, location) for location, state in location_to_state.items())


#initialize parameters
gamma = 0.75 # discount factor
alpha = 0.9 # learning rate

#init q-values
Q = np.array(np.zeros([7,7]))

#--------------------------------
#--------------------------------
#--------------------------------

class QAgent():

    # Init alpha, gamma, states, actions, rewards, and Q-values
    def __init__(self, alpha, gamma, location_to_state, actions, rewards, state_location, Q):

            self.gamma = gamma
            self.alpha = alpha

            self.location_to_state = location_to_state
            self.actions = actions
            self.rewards = rewards
            self.state_to_location = state_to_location

            self.Q = Q

    def training(self, start_location, end_location, iterations):

        #def get_optimal_route(start_location, end_location):
        #copy the rewards matrix to new matrix
        rewards_new = np.copy(rewards)

        # get the ending state corresponding to the ending location as given
        ending_state = location_to_state[end_location]

        #with the above information automatically set the priority of the 
        #given ending state to the highest one
        rewards_new[ending_state,ending_state] = 999

        # ----------------Q-Learning algorithm-----------------

        for i in range(iterations):
            #pick up a state randomly
            current_state = np.random.randint(0,8)
            playable_actions = []

        #iterate through the new rewards martix and get the actions > 0
            for j in range(8):
                if rewards_new[current_state,j] > 0 :
                    playable_actions.append(j)

        # pick an action randomly from the list of playable actions leading us 
        # to the next state
            next_state = np.random.choice(playable_actions)

        # compute the temporal difference
        # the action here exactly refers to going to the next state

            TD = rewards_new[current_state, next_state] + self.gamma * self.Q[next_state, np.argmax(self.Q[next_state,])] - self.Q[current_state, next_state]

        # Update the Q-Value using the Bellman equation
            self.Q[current_state, next_state] +=  self.alpha * TD


        # Initialize the optimal route with the starting location
        route = [start_location]

        next_location = start_location

        self.get_optimal_route(start_location, end_location, next_location, route, self.Q)

        # We don´t know about the exact number of iterations needed to reach to the final
        # location hence while loop will be a good choice for iterating

    def get_optimal_route(self, start_location, end_location, next_location, route, Q):
        while(next_location != end_location):
            #fetch the starting state
            starting_state = self.location_to_state[start_location]
            # Fetch the highest Q-value pertaining to starting state
            next_state = np.argmax(Q[starting_state,])
            # We got the index of the next state. But we need the corresponding letter
            next_location = self.state_to_location[next_state]
            route.append(next_location)
            # Update the starting location for the next iteration
            start_location = next_location

        print(route)



qagent = QAgent(alpha, gamma, location_to_state, actions, rewards, state_to_location, Q)
qagent.training('L8', 'L1', 1000)