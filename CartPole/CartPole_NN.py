# Proyecto Fina Bioinspirados
# Entorno Taxi en pygame con Redes Neuronales por Yeison Suarez
import gym
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import rmsprop, Adam
import numpy as np
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

class EnvironmentSolver:
    route = []
    # Incializacion de la clase
    def __init__(self, solver):
        self.total_score = []
        self.solver = solver
    
     # Funcion decreciente con el tiempo
    def dec(self,value,time):
        return max(value, min(1.0, 1.0 - math.log10((time + 1)/100)))
    
    # Comienza el entrenamiento del entorno
    def train(self, episodes, render):
        observation_space = self.solver.observation_space
        action_space = self.solver.action_space
        fe = CartPoleFeatureEstimate(self.solver.env)
        print("Inicia el entrenamiento")
        for episode in range(episodes):
            state = self.solver.env.reset()
            state = np.array([state])
            step = 0
            done = False
            episodeReward = 0
            while not done:
                if render:
                    self.solver.env.render('human')
                step += 1
                action = self.solver.take_action(state)
                state_next, reward, done, info = self.solver.env.step(action)
                reward = np.sum(fe.get_features(state))
                episodeReward += reward
                state_next = np.array([state_next])
                self.solver.add_to_memory(state, action, reward, state_next, done)
                state = state_next
                self.solver.experience_replay()
                
            print("Run: " + str(episode) + ", exploration: " + str(self.solver.exploration_rate) + ", Episode Reward: " + str(episodeReward)+ ", Steps: "+ str(step))
            self.total_score.append(episodeReward)
        self.solver.env.close()
    
    def saveRoute(self,name):
        try:
            os.mkdir(name)
        except:
            pass
        np.save("./"+name+"/route",np.asarray(self.route))
        np.save("./"+name+"/score",np.asarray(self.total_score))
    
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

    def makeRoute(self,num):
        for n in range(num):
            observation_space = self.solver.observation_space
            action_space = self.solver.action_space
            state = self.solver.env.reset()
            done = False
            # Hace un ultimo recorrido en el cual usa la tabla Q aprendida
            i=0
            reward_total=0
            current_route = []
            while i<500 and done == False:
                #self.env.render(mode  =  'rgb_array')
                state = np.array([state])
                action = np.argmax(self.solver.model.predict(state)[0])
                new_state, reward, done, _ = self.solver.env.step(action)
                current_route.append((state,action,reward))
                state = new_state
                reward_total+=reward
                i += 1
            print((reward_total,n))
            if(reward_total==500):
                self.route.append(current_route)
            self.solver.env.close()
    
    def printGraph(self):
        plt.plot(self.total_score)
        plt.ylabel('Iteraciones hasta terminar')
        plt.xlabel('Episodios')
        plt.show()
    
    def safeTrain(self,name):
        try:
            os.mkdir(name)
        except:
            pass
        pandas.DataFrame(np.array(self.total_score).transpose(),columns= ['Reward']).to_csv("./"+name+"/scores.csv",index_label="Episodes")
        self.solver.save("./"+name+"/w")
        np.save("./"+name+"/scores", np.array(self.total_score))
    
    def loadTrain(self,name):
        self.solver.load("./"+name+"/w")
        self.total_score=list(np.load("./"+name+"/scores.npy"))
        
env = gym.make('CartPole-v1').env
solver = NNAgent(env, env.action_space.n, env.observation_space.shape[0], LEARNING_RATE, BATCH_SIZE, GAMMA, EXPLORATION_RATE, EXPLORATION_DECAY, EXPLORATION_MIN, MAX_MEMORY)
solver.makeNewModel()
Agent = EnvironmentSolver(solver)