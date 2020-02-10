import sys
import gym
import pylab
import numpy as np
import math
import cvxpy as cp
import random
from scipy import interpolate
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import rmsprop, Adam
from collections import deque
import matplotlib.pyplot as plt
import pandas
import time
import os

# Hiperparametros necesarios
LEARNING_RATE = 0.001
BATCH_SIZE = 64
GAMMA = 1
EXPLORATION_RATE = 1
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.01
MAX_MEMORY = 50000

class CartPoleFeatureEstimate:
    count=1
    def __init__(self, env):
        self.env = env
        self.feature_num = 4
        self.feature = np.ones(self.feature_num)

    def gaussian_function(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def get_features(self, state):
        #state=state[0]
        try:
            self.feature[0]=self.gaussian_function(0,state[0],1.5)
            self.feature[1]=self.gaussian_function(0,state[1],0.5)
            self.feature[2]=self.gaussian_function(0,state[2],6.)
            self.feature[3]=self.gaussian_function(0,state[3],0.5)
        except:
            state=state[0]
            self.feature[0]=self.gaussian_function(0,state[0],1.5)
            self.feature[1]=self.gaussian_function(0,state[1],0.5)
            self.feature[2]=self.gaussian_function(0,state[2],6.)
            self.feature[3]=self.gaussian_function(0,state[3],0.5)
        return self.feature

# Clase Neuronal Network Agent
class NNAgent:
    
    # Inicializa los parametros 
    def __init__(self, env, action_space, observation_space, alpha = 0.001, batch_size = 10, gamma = 1, exploration_rate = 1, exploration_decay = 0.9, min_exploration = 0.1, max_memory = 1000):
        self.env = env
        self.action_space = action_space
        self.observation_space = observation_space
        self.alpha = alpha
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.memory = deque(maxlen=max_memory)
        self.model = None
    
    # Construye un modelo en blanco con los parametros de inicializacion y otros adicionales con respecto al numero de neuronas y su activacion
    # "layers" es un vector de tuplas que contiene el numero de neuronas y su activacion [(#neuronas,'activacion'),...]
    def makeNewModel(self, layers = [(24,'relu'),(24,'relu')]):
        self.model = Sequential()
        self.model.add(Dense(layers[0][0], input_shape=(self.observation_space,), activation=layers[0][1])) # Capa de entrada
        for layer in layers[1:]:
            self.model.add(Dense(layer[0], activation=layer[1])) # Demas capas de la red (capas ocultas)
        self.model.add(Dense(self.action_space,activation='linear')) # Capa de salida
        #self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha)) # Compilacion del modelo
        self.model.compile(loss='mse', optimizer=Adam(), metrics=['mae']) # Compilacion del modelo
        
    # Memoria de la red neuronal
    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    # Toma una accion aleatoria si el numero generado es menor a la tasa de exploracion, de lo contrario devuelve la accion que dice la red
    def take_action(self, state):
        return random.randrange(0, self.action_space) if np.random.rand() < self.exploration_rate else np.argmax(self.model.predict(state)[0])
    
    # Empieza a "recordar" su trayectoria para que la red neuronal no se sobreentrene con los ultimos datos que haya visto
    # Aqui es donde aplica QLearning, al interpretar la funcion como un problema de optimizacion
    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            #minibatch = random.sample(self.memory, len(self.memory))
            return
        else:
            minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, state_next, done in minibatch:
            Q = reward
            if not done:
                Q = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            Q_values = self.model.predict(np.array(state))[0]
            Q_values[action] = Q
            self.model.fit(state, Q_values.reshape(-1, self.action_space), verbose=0)
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.min_exploration, self.exploration_rate)
    
    # Guarda los pesos actuales del modelo
    def save(self, filename):
        self.model.save_weights(filename)
    
    # Carga los pesos al modelo actual
    def load(self, filename):
        self.model.load_weights(filename)

class IRL_EnvironmentSolver:
    
    # Constructor de la clase
    def __init__(self, solver):
        self.total_score = []
        self.solver = solver
        self.w = []
    
    # Funcion decreciente con el tiempo
    def dec(self,value,time):
        return max(value, min(1.0, 1.0 - math.log10((time + 1)/100)))
    
    # Ejecuta el entorno con el modelo aprendido hasta el momento
    def testTrain(self):
        print("Inicia prueba del aprendizaje")
        observation_space = self.solver.observation_space
        action_space = self.solver.action_space
        state = self.solver.env.reset()
        done = False
        total_reward=0
        while not done:
            state = np.array([state])
            self.solver.env.render('human')
            action = np.argmax(self.solver.model.predict(state)[0])
            state_next, reward, done, info = self.solver.env.step(action)
            state = state_next
            total_reward += reward
        env.close()
        print("Recompensa total: "+str(total_reward))
        print("Termino")
    
    # Imprime las recompensas
    def printGraph(self):
        plt.plot(self.total_score)
        plt.ylabel('Recompensas')
        plt.xlabel('Episodios')
        plt.show()
    
    # Guarda el modelo y el puntaje dentro de la carpeta especificada
    def saveTrain(self,name):
        try:
            os.mkdir(name)
        except:
            pass
        #pandas.DataFrame(np.array(self.total_score).transpose(),columns= ['Reward']).to_csv("./"+name+"/scores.csv",index_label="Episodes")
        self.solver.save("./"+name+"/w")
        np.save("./"+name+"/scores", np.array(self.total_score))
        np.save("./"+name+"/IRL_W", np.array(self.w))
        np.save("./"+name+"/memory", np.array(self.solver.memory))
    
    def loadTrain(self,name):
        self.solver.load("./"+name+"/w")
        self.total_score=list(np.load("./"+name+"/scores.npy",allow_pickle=True))
        self.w=list(np.load("./"+name+"/IRL_W.npy",allow_pickle=True))
        self.solver.memory=list(np.load("./"+name+"/memory.npy",allow_pickle=True))
    
    # Optimizador de la funcion de recompensa
    def QP_optimizer(self,learner, expert, numFeatures):
        w = cp.Variable(numFeatures)
        
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
            
            weights = np.zeros(numFeatures)
            return weights, prob.status
    
    def subtract_feature_expectation(self,learner):
        # if status is infeasible, subtract first feature expectation
        learner = learner[1:][:]
        return learner

    def add_feature_expectation(self,learner, temp_learner):
        # save new feature expectation to list after RL step
        learner = np.vstack([learner, temp_learner])
        return learner
    
    def calc_feature_expectation(self,gamma, demonstrations, numFeatures):
        feature_estimate = CartPoleFeatureEstimate(self.solver.env)
        feature_expectations = np.zeros(numFeatures)
        demo_num = len(demonstrations)
        for _ in range(demo_num):
            #state = env.reset()
            state = self.solver.env.reset()
            demo_length = 0
            done = False
            state = np.array([state])
            i=0
            while i<500 and not done:
                demo_length += 1
                action = np.argmax(self.solver.model.predict(state)[0])
                next_state, reward, done, _ = self.solver.env.step(action)
                state = np.array([next_state])
                features = feature_estimate.get_features(state)
                feature_expectations += (gamma**(demo_length)) * np.array(numFeatures)
                i+=1
        feature_expectations = feature_expectations/ demo_num
        return feature_expectations

    def expert_feature_expectation(self,gamma, demonstrations, numFeatures):
        feature_estimate = CartPoleFeatureEstimate(self.solver.env)
        feature_expectations = np.zeros(numFeatures)
        print(demonstrations.shape)
        for demo_num in range(len(demonstrations)):
            for demo_length in range(len(demonstrations[demo_num])):
                state = demonstrations[demo_num][demo_length][0]
                features = feature_estimate.get_features(state)
                feature_expectations += (gamma**(demo_length)) * np.array(numFeatures)
        feature_expectations = feature_expectations / len(demonstrations)
        return feature_expectations
    
    def train(self, episodes, dem_path):
        demonstrations = np.load(dem_path,allow_pickle=True)
        feature_estimate = CartPoleFeatureEstimate(self.solver.env)
        # Aprendiz
        learner = self.calc_feature_expectation(self.solver.gamma, demonstrations, 4)
        learner = np.matrix([learner])
        # Experto
        expert = self.expert_feature_expectation(self.solver.gamma, demonstrations, 4)
        expert = np.matrix([expert])
        # Inicializa los pesos
        self.w, status = self.QP_optimizer(learner, expert,4)
        
        self.episodes, self.scores = [], []
        
        for episode in range(len(self.total_score),episodes):
            state = self.solver.env.reset()
            state = np.array([state])
            score = 0
            episodeReward = [0,0]
            done = False
            i=0
            while i<500 and done == False:
                action = self.solver.take_action(state)
                state_next, reward, done, info = self.solver.env.step(action)
                episodeReward[0] += reward
                features = feature_estimate.get_features(state)
                irl_reward = np.dot(self.w, features)
                episodeReward[1]=irl_reward
                state_next = np.array([state_next])
                self.solver.add_to_memory(state, action, irl_reward, state_next, done)
                state = state_next
                self.solver.experience_replay()
                i+=1
            self.total_score.append(episodeReward)
            if episode % 1 == 0:
                score_avg = np.mean(self.total_score)
                print('{} episode score is {:.2f}'.format(episode, score_avg))
            if episode % 10 == 0:
                # optimize weight per 5000 episode
                status = "infeasible"
                temp_learner = self.calc_feature_expectation(self.solver.gamma, demonstrations, 4)
                learner = self.add_feature_expectation(learner, temp_learner)
                self.saveTrain('lastTrain_'+str(episode))
                while status=="infeasible":
                    self.w, status = self.QP_optimizer(learner, expert, 4)
                    if status=="infeasible":
                        learner = self.subtract_feature_expectation(learner)
        print(self.w)
        self.solver.env.close()

