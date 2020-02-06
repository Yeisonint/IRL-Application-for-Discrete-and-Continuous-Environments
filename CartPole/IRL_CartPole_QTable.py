import sys
import gym
import pylab
import numpy as np
import math
import cvxpy as cp
import random
import time
from scipy import interpolate

# Estado - accion
DISC_STATES = (5, 5, 12, 9)
q_table = np.zeros(DISC_STATES + (2,))
scores = []

gamma = 0.995
q_learning_rate = 0.1

class CartPoleFeatureEstimate:
    count=1
    def __init__(self, env):
        self.env = env
        self.feature_num = 4
        self.feature = np.ones(self.feature_num)
    
    # Funcion de disctretizacion
    def discretize(self, Observation):
        # Limites superiores e inferiores
        # ~[4.8,0.5,24,50] los dos ultimos son grados, pero se añaden a la tabla en radianes.
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        # Proporcion entre la observacion y los limites
        ratios = [(Observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(Observation))]
        dis_obs = [int(round((DISC_STATES[i] - 1) * ratios[i])) for i in range(len(Observation))]
        dis_obs = [min(DISC_STATES[i] - 1, max(0, dis_obs[i])) for i in range(len(Observation))]
        return tuple(dis_obs)

    def gaussian_function(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def get_features(self, state):
        self.feature[0]=self.gaussian_function(0,state[0],2.4)
        self.feature[1]=self.gaussian_function(0,state[0],1.)
        self.feature[2]=self.gaussian_function(0,state[2],12.)
        self.feature[3]=self.gaussian_function(0,state[0],1.)
        return self.feature

def save(self,name):
        np.save(name+"_QTABLE",np.asarray(q_table))
        np.save(name+"_SCORES",np.asarray(scores))

def calc_feature_expectation(gamma, q_table, demonstrations, env):
    feature_estimate = CartPoleFeatureEstimate(env)
    feature_expectations = np.zeros(4)
    demo_num = len(demonstrations)
    
    for _ in range(demo_num):
        #state = env.reset()
        cont_state = env.reset()
        state = feature_estimate.discretize(cont_state)
        demo_length = 0
        done = False
        
        while not done:
            demo_length += 1
            action = np.argmax(q_table[state])
            next_state, reward, done, _ = env.step(action)
            cont_state = next_state
            next_state = feature_estimate.discretize(cont_state)
            features = feature_estimate.get_features(cont_state)
            feature_expectations += (gamma**(demo_length)) * np.array(4)

            state = next_state
    
    feature_expectations = feature_expectations/ demo_num

    return feature_expectations

def expert_feature_expectation(gamma, demonstrations, env):
    feature_estimate = CartPoleFeatureEstimate(env)
    feature_expectations = np.zeros(4)
    print(demonstrations.shape)
    for demo_num in range(len(demonstrations)):
        for demo_length in range(len(demonstrations[demo_num])):
            cont_state = demonstrations[demo_num][demo_length][0]
            state = feature_estimate.discretize(cont_state)
            features = feature_estimate.get_features(cont_state)
            feature_expectations += (gamma**(demo_length)) * np.array(4)
    
    feature_expectations = feature_expectations / len(demonstrations)
    
    return feature_expectations


def QP_optimizer(learner, expert):
    w = cp.Variable(4)
    
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
    env = gym.make('CartPole-v0')
    feature_estimate = CartPoleFeatureEstimate(env)
    currentState = feature_estimate.discretize(env.reset())
    done = False
    total_reward=0
    while not done:
        env.render('human')
        action = np.argmax(q_table[currentState])
        newState, reward, done, info = env.step(action)
        currentState = feature_estimate.discretize(newState)
        total_reward += reward
    env.close()
    print("Recompensa total: "+str(total_reward))

# Inicia IRL
def main():
    # Carga el entorno
    env = gym.make('CartPole-v0')
    # Obtiene las demostraciones del experto
    demonstrations = np.load("Test.npy",allow_pickle=True)
    
    # Caracteriza las condiciones del entorno dentro de un objeto
    feature_estimate = CartPoleFeatureEstimate(env)
    
    # Ni idea, pero creo que en base al gamma, da mayor peso a los estados cuando son visitados las primeras veces
    learner = calc_feature_expectation(gamma, q_table, demonstrations, env)
    learner = np.matrix([learner])
    
    # Lo mismo que el anterior pero con el experto
    expert = expert_feature_expectation(gamma, demonstrations, env)
    expert = np.matrix([expert])
    
    # Inicializa los pesos
    w, status = QP_optimizer(learner, expert)
    
    episodes, scores = [], []
    
    for episode in range(1000000):
        cont_state = env.reset()
        state = feature_estimate.discretize(cont_state)
        score = 0

        while True:
            #env.render('human')
            action = np.argmax(q_table[state]) if random.randint(0,100)>80 else env.action_space.sample()
            #print((state,action))
            next_state, reward, done, _ = env.step(action)
            
            features = feature_estimate.get_features(cont_state)
            cont_state = next_state
            irl_reward = np.dot(w, features)
            
            next_state = feature_estimate.discretize(cont_state)
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