"""
This is Jack's Car Rental Environment

"""
from typing import Tuple
import numpy as np
import scipy.stats

class jack_car_rental():
    
    max_req_ret = 20
    
    max_cars = 20
    max_poisson = 11 # covers 99.7% of the probability
    max_transfer = 5

    reward_rental = 10
    reward_move = -2
    
    lamb_car_req_lot_1 = 3
    lamb_car_req_lot_2 = 4
    lamb_car_ret_lot_1 = 3
    lamb_car_ret_lot_2 = 2
    
    requests_lambda = np.array([lamb_car_req_lot_1, lamb_car_req_lot_2])
    returns_lambda = np.array([lamb_car_ret_lot_1, lamb_car_ret_lot_2])
    
    action_space = np.arange(-5,6)
    # self.state_space = np.asarray(list(itertools.product(range(21),range(21))))
    
    def __init__(self):
        
        #caching probabilities of requests and returns

        # n_requests_returns is a matrix with all combinations of possible values for requests and returns
        # requests are the 2 first columns; returns are the 2 last columns
        n_requests_returns = np.indices([self.max_poisson] * 4).reshape([4, -1]).T
        self.n_requests = n_requests_returns[:, :2]
        self.n_returns = n_requests_returns[:, 2:]
        
        # Put lambdas in array format
        requests_lambda = np.array([self.lamb_car_req_lot_1, self.lamb_car_req_lot_2])
        returns_lambda = np.array([self.lamb_car_ret_lot_1, self.lamb_car_ret_lot_2])
        
        dist_requests = scipy.stats.poisson(requests_lambda)
        dist_returns = scipy.stats.poisson(returns_lambda)
        
        prob_requests = dist_requests.pmf(self.n_requests)
        prob_returns = dist_returns.pmf(self.n_requests)
        
        # calculating probability of each 2 reqs and 2 rets happening, by multiplying all probabilities
        self.p_event = np.prod(prob_requests * prob_returns, axis=1)
        
        # Normalizing to 100%
        self.p_event /= np.sum(self.p_event)
        
        self.reset()
        
    def reset(self, state=None):
        
        if state is None:
            # Random Start
            state = np.random.randint(0, self.max_cars, size=2)
            self.state = (state[0], state[1])
        else:
            self.state = (state[0],state[1])
            
        return self.state
    
    def step(self, action):
        reward = 0
        transfer = action - self.max_transfer
        
        cars_in_lot = np.zeros(2, dtype=int)
        
        (cars_in_lot[0], cars_in_lot[1]) = self.state
            
        # This works for making an environment, but not to calculate V(s)
        cars_req_lot_1 = min(np.random.poisson(self.lamb_car_req_lot_1), self.max_req_ret)
        cars_req_lot_2 = min(np.random.poisson(self.lamb_car_req_lot_2), self.max_req_ret)
        cars_req = ([cars_req_lot_1, cars_req_lot_2])
        
        cars_ret_lot_1 = min(np.random.poisson(self.lamb_car_ret_lot_1), self.max_req_ret)
        cars_ret_lot_2 = min(np.random.poisson(self.lamb_car_ret_lot_2), self.max_req_ret)
        cars_ret = ([cars_ret_lot_1, cars_ret_lot_2])
        
        # Overnight transfer & cost
        if transfer > 0:
            # transfer from lot 1 to lot 2
            transferred_lot_1 = -min(abs(transfer), cars_in_lot[0])
            transferred_lot_2 = +min(abs(transferred_lot_1), self.max_cars - cars_in_lot[1])
            transferred = -transferred_lot_1
        else:
            # transfer from lot 2 to lot 2
            transferred_lot_2 = -min(abs(transfer), cars_in_lot[1])
            transferred_lot_1 = +min(abs(transferred_lot_2), self.max_cars - cars_in_lot[0])
            transferred = -transferred_lot_2
                              
        cars_in_lot = cars_in_lot + np.array([transferred_lot_1, transferred_lot_2])
        
        transfer_reward = abs(transferred) * self.reward_move
        
        # Rentals
        
        rented_cars = np.minimum(cars_req, cars_in_lot)
        cars_in_lot = cars_in_lot - rented_cars
        
        # Returns
        returned_cars = np.minimum(cars_ret, (self.max_cars - cars_in_lot))
        cars_in_lot = cars_in_lot + returned_cars
            
        # Next State
        self.state = cars_in_lot
        
        # Reward Calculation
        rental_revenue = self.reward_rental * np.sum(rented_cars)
        rewards = rental_revenue + transfer_reward
        
        return self.state, reward, rented_cars, returned_cars

    def expected_return(self, state_values, action, state, gamma):
        
        transfer = action - self.max_transfer
        
        cars_in_lot = np.zeros(2, dtype=int)
        
        (cars_in_lot[0], cars_in_lot[1]) = state
        
        #print(f"Cars in lot 1: {cars_in_lot[0]}, Cars in lot 2: {cars_in_lot[1]}, Action: {action}, Transfer Action: {transfer}")
                
        # Overnight transfer & cost
        if transfer > 0:
            # transfer from lot 1 to lot 2
            transferred_lot_1 = -min(abs(transfer), cars_in_lot[0])
            transferred_lot_2 = +min(abs(transferred_lot_1), self.max_cars - cars_in_lot[1])
            transferred = -transferred_lot_1
        else:
            # transfer from lot 2 to lot 2
            transferred_lot_2 = -min(abs(transfer), cars_in_lot[1])
            transferred_lot_1 = +min(abs(transferred_lot_2), self.max_cars - cars_in_lot[0])
            transferred = -transferred_lot_2
              
        #print(f"Cars Transferred: {transferred}")
                              
        cars_in_lot = cars_in_lot + np.array([transferred_lot_1, transferred_lot_2])
        
        #print(f"Cars in lot 1: {cars_in_lot[0]}, Cars in lot 2: {cars_in_lot[1]}")
                              
        transfer_reward = abs(transferred) * self.reward_move
              
        #print(f"Transfer Cost: {transfer_reward}")
        
        # Rentals
        rented_cars = np.minimum(self.n_requests, cars_in_lot)
        cars_in_lot = cars_in_lot - rented_cars
              
        #print(f"Cars Rented: {rented_cars}, Cars in lot 1: {cars_in_lot[0]}, Cars in lot 2: {cars_in_lot[1]}")
        
        #print(f"{cars_in_lot}")
        
        # Returns  
        returned_cars = np.minimum(self.n_returns, (self.max_cars - cars_in_lot))
        cars_in_lot = cars_in_lot + returned_cars
              
        #print(f"Cars Returned: {returned_cars}, Cars in lot 1: {cars_in_lot[0]}, Cars in lot 2: {cars_in_lot[1]}")
        
        # Reward Calculation
        rental_revenue = self.reward_rental * np.sum(rented_cars, axis=1)
        rewards = rental_revenue + transfer_reward
              
        #print(f"Rental Revenue: {rental_revenue}, Rewards: {rewards}")
        
        returns = rewards + gamma * state_values[tuple(cars_in_lot.T)]
              
        #print(f"Returns: {returns}")
        
        expected_return = np.sum(self.p_event * returns)
              
        #print(f"Expected Return: {expected_return}")
        
        return expected_return