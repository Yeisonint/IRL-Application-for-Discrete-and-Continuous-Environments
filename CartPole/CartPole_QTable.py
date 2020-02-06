# Import the necessary libraries
import numpy as np
import gym 
import math  
from collections import deque
import matplotlib.pyplot as plt
import pandas

# The dir (class) function returns the attributes of a class
class Qlearning_CartPole:

    # Definition of the number of states
    buckets = (5, 5, 12, 9)
    scores = deque(maxlen=150)
    total_score = []
    states_count = {}
    route = []

    # Initialization Function
    def __init__(self,environment = 'CartPole-v0', num_episodes = 1000, gamma = 1, min_alpha = 0.1, min_epsilon = 0.1):
        # Define the local variables
        self.env = gym.make(environment) # Environment
        self.num_episodes = num_episodes # Number of episodes until finished
        self.gamma = gamma               # Discount factor
        self.min_alpha = min_alpha       # Learning rate
        self.min_epsilon = min_epsilon   # Exploration rate
        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))
        self.states_count = np.zeros(list(self.buckets))

    # Function decreasing over time
    def dec(self,value,time):
        return max(value, min(1.0, 1.0 - math.log10((time + 1)/25)))
    
    def saveRoute(self,name):
        np.save(name,np.asarray(self.route))
        
    def gaussian_function(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    # Disctretization function
    def discretize(self, Observation):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(Observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(Observation))]
        dis_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(Observation))]
        dis_obs = [min(self.buckets[i] - 1, max(0, dis_obs[i])) for i in range(len(Observation))]
        return tuple(dis_obs)
    
    # Take a sample or find in QTable
    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q_table[state])
    
    # Learning begins
    def start(self):
        self.scores = deque(maxlen=150)
        self.states_count = {}
        for e in range(self.num_episodes):
            current_state_cont = self.env.reset()
            current_state = self.discretize(current_state_cont)
            alpha = self.dec(self.min_alpha,e)
            epsilon = self.dec(self.min_epsilon,e)
            done = False
            i = 0
            while not done:
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                reward = self.gaussian_function(0,obs[0],2.4) + self.gaussian_function(0,obs[1],1) + self.gaussian_function(0,obs[2],12.) + self.gaussian_function(0,obs[3],1)
                new_state = self.discretize(obs)
                self.Q_table[current_state][action] += alpha * (reward + self.gamma * np.max(self.Q_table[new_state]) - self.Q_table[current_state][action])
                self.states_count[str((current_state[0],current_state[2],current_state[3]))] = self.states_count.get(str((current_state[0],current_state[2],current_state[3])),0) +1
                current_state = new_state
                i += 1
            self.scores.append(i)
            self.total_score.append(i)
            mean=np.mean(self.scores)
            print((mean,len(self.scores),e))
            if mean >= 195:
                return "Ok"
            self.env.close()
        return "Not complete"

    def getSocores(self):
        return self.scores
    
    # Function that stores the routes taken by the agent with learning (for IRL)
    def makeRoute(self,num):
        for n in range(num):
            cont = self.env.reset()
            current_state = self.discretize(cont)
            done = False
            i=0
            reward_total=0
            current_route = []
            while i<5000 and done == False:
                self.env.render(mode  =  'rgb_array')
                action = np.argmax(self.Q_table[current_state])
                cont, reward, done, _ = self.env.step(action)
                current_route.append((cont,action,reward))
                new_state = self.discretize(cont)
                current_state = new_state
                reward_total+=reward
                i += 1
            print((reward_total,n))
            self.route.append(current_route)
            self.env.close()
    
    def printGraph(self):
        # Graph the lengths of the episodes
        plt.subplot(2, 1, 1)
        plt.plot(self.total_score, label='Scores')
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.title('Learning curve')
        # Graph the agent's animated route
        plt.subplot(2, 1, 2)
        plt.title('Histogram of visited states')
        data = []
        for key in self.states_count:
            data.append(self.states_count[key])
        plt.hist(data, len(data), facecolor='green',edgecolor='k', alpha=0.75)
        print(len(data))
        plt.show()

# Environment 'CartPole-v0'
Cart = Qlearning_CartPole('CartPole-v0',100000,1,0.001,0.1)
print(Cart.start())
Cart.printGraph()
