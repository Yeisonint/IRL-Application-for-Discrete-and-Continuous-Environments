import sys
import gym
import pylab
import numpy as np
import math
import cvxpy as cp
import random
import time
from scipy import interpolate
import matplotlib.pyplot as plt

#from app import *

n_states = 16
n_actions = 4
q_table = np.zeros((500,6))

gamma = 0.99
q_learning_rate = 0.1

w,scores = [],[]

class TaxiFeatureEstimate:
    count=1
    def __init__(self, env):
        self.env = env
        self.feature_num = 3
        self.feature = np.ones(self.feature_num)

    def gaussian_function(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def get_features(self, state):
        if(type(state)==type((0,0))):
            decState = state[1:6]
        else:
            decState = list(self.env.decode(state))
        if decState[2]!=4:
            passagerPosition = (0,0) if decState[2]==0 else (0,4) if decState[2]==1 else (4,0) if decState[2]==2 else (4,3)
            self.feature[0]=1-(self.gaussian_function(decState[0],passagerPosition[0],1)/2+self.gaussian_function(decState[1],passagerPosition[1],1)/2)
            self.feature[1]=1
        else:
            destinationPosition = (0,0) if decState[3]==0 else (0,4) if decState[3]==1 else (4,0) if decState[3]==2 else (4,3)
            self.feature[0]=0
            self.feature[1]=1-(self.gaussian_function(decState[0],destinationPosition[0],1)/2+self.gaussian_function(decState[1],destinationPosition[1],1)/2)
        self.feature[2]=1-self.gaussian_function(0,self.count,1)
        self.count=self.count+1
        return self.feature

def calc_feature_expectation(gamma, q_table, demonstrations, env):
    feature_estimate = TaxiFeatureEstimate(env)
    feature_expectations = np.zeros(2)
    demo_num = len(demonstrations)
    
    for _ in range(demo_num):
        state = env.reset()
        demo_length = 0
        done = False
        
        while not done:
            demo_length += 1
            action = np.argmax(q_table[state])
            next_state, reward, done, _ = env.step(action)
            features = feature_estimate.get_features(next_state)
            feature_expectations += (gamma**(demo_length)) * np.array(features)

            state = next_state
    
    feature_expectations = feature_expectations/ demo_num

    return feature_expectations

def expert_feature_expectation(gamma, demonstrations, env):
    feature_estimate = TaxiFeatureEstimate(env)
    feature_expectations = np.zeros(3)
    print(demonstrations.shape)
    for demo_num in range(len(demonstrations)):
        for demo_length in range(len(demonstrations[demo_num])):
            state = demonstrations[demo_num][demo_length][0]
            features = feature_estimate.get_features(state)
            feature_expectations += (gamma**(demo_length)) * np.array(features)
    
    feature_expectations = feature_expectations / len(demonstrations)
    
    return feature_expectations


def QP_optimizer(learner, expert):
    w = cp.Variable(3)
    
    obj_func = cp.Minimize(cp.norm(w))
    constraints = [(np.linalg.norm(expert)-np.linalg.norm(learner)) * w >= 2] 

    prob = cp.Problem(obj_func, constraints)
    prob.solve()

    if prob.status == "optimal":
        print("status:", prob.status)
        print("optimal value", prob.value)
    
        weights = np.squeeze(np.asarray(w.value))
        return weights, prob.status
    else:
        print("status:", prob.status)
        
        weights = np.zeros(feature_num)
        return weights, prob.status

# Actualiza tabla Q
def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action]
    q_2 = reward + gamma * max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)

def subtract_feature_expectation(learner):
    # if status is infeasible, subtract first feature expectation
    learner = learner[1:][:]
    return learner

def add_feature_expectation(learner, temp_learner):
    # save new feature expectation to list after RL step
    learner = np.vstack([learner, temp_learner])
    return learner

# Prueba el aprendizaje y retorna la recompensa total
def trylearning():
    env = gym.make('Taxi-v3')
    currentState = env.reset()
    done = False
    total_reward=0
    while True:
        env.render('human')
        action = np.argmax(q_table[currentState])
        newState, reward, done, info = env.step(action)
        currentState = newState
        total_reward += reward
        time.sleep(0.1)
        if done:
            env.render('human')
            break
    print("Recompensa total: "+str(total_reward))

def printGraph():
    plt.plot(scores)
    plt.ylabel('Iteraciones hasta terminar')
    plt.xlabel('Episodios')
    plt.show()

# Inicia IRL
def main():
    # Carga el entorno
    env = gym.make('Taxi-v3')
    # Obtiene las demostraciones del experto
    demonstrations = np.load("expert_demo.npy",allow_pickle=True)
    
    # Caracteriza las condiciones del entorno dentro de un objeto
    feature_estimate = TaxiFeatureEstimate(env)
    
    # Ni idea, pero creo que en base al gamma, da mayor peso a los estados cuando son visitados las primeras veces
    learner = calc_feature_expectation(gamma, q_table, demonstrations, env)
    learner = np.matrix([learner])
    
    # Lo mismo que el anterior pero con el experto
    expert = expert_feature_expectation(gamma, demonstrations, env)
    expert = np.matrix([expert])
    
    # Inicializa los pesos
    global w, scores
    w, status = QP_optimizer(learner, expert)
    episodes, scores = [], []
    
    for episode in range(60000):
        state = env.reset()
        score = 0

        while True:
            action = np.argmax(q_table[state]) if random.randint(0,100)>10 else random.choice(list(np.where([0,0,0,0] == np.amax([0,0,0,0]))[0]))
            next_state, reward, done, _ = env.step(action)
            
            features = feature_estimate.get_features(state)
            irl_reward = np.dot(w, features)
            
            update_q_table(state, action, irl_reward, next_state)

            score += reward
            state = next_state

            if done:
                scores.append(score)
                episodes.append(episode)
                break

        if episode % 1000 == 0:
            score_avg = np.mean(scores)
            print('{} episode score is {:.2f}'.format(episode, score_avg))
            np.save("app_q_table", arr=q_table)

        if episode % 5000 == 0:
            # optimize weight per 5000 episode
            status = "infeasible"
            temp_learner = calc_feature_expectation(gamma, q_table, demonstrations, env)
            learner = add_feature_expectation(learner, temp_learner)
            
            while status=="infeasible":
                w, status = QP_optimizer(learner, expert)
                if status=="infeasible":
                    learner = subtract_feature_expectation(learner)

if __name__ == '__main__':
    main()