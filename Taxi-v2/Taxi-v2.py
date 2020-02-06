# Importa las librerias necesarias
import gym
import numpy as np
import math
import time
import matplotlib.pyplot as plt

# Crea el entorno
env = gym.make('Taxi-v3')

# Establece los hiperparametros
min_epsilon = 0.1
min_alpha = 0.1
#min_gamma = 0.9
total_episodes = 10000
total_score = []

# Creamos la tabla Q
QTable = np.zeros((env.observation_space.n, env.action_space.n))

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

# Funcion que retorna una accion de un estado, e-greedy policy
def choose_action(state,epsilon):
    return env.action_space.sample() if np.random.uniform(0, 1) < epsilon else np.argmax(QTable[state, :])

# Funcion decreciente con el tiempo
def decValue(value,time):
    return max(value, min(1.0, 1.0 - math.log10((time + 1)/25)))

# Actualiza la tabla Q
def updateQtable(currentState, newState, reward, action, alpha, gamma):
    QTable[currentState][action] = QTable[currentState][action] + alpha * (reward + gamma * np.max(QTable[newState,:]) - QTable[currentState][action])

# Prueba el aprendizaje y retorna la recompensa total
def trylearning():
    currentState = env.reset()
    done = False
    total_reward=0
    time.sleep(3)
    while not done:
        env.render('human')
        action = np.argmax(QTable[currentState])
        newState, reward, done, info = env.step(action)
        currentState = newState
        total_reward += reward
        time.sleep(0.2)
    time.sleep(3)
    print("Recompensa total: "+str(total_reward))

# Imprime los puntajes
def printGraph():
    plt.plot(total_score)
    plt.ylabel('Iteraciones hasta terminar')
    plt.xlabel('Episodios')
    plt.show()

# Inicia los episodios
def startLearning(render):
    tfe = TaxiFeatureEstimate(env)
    for episode in range(total_episodes):
        print("Episodio: "+str(episode))
        # Reinicia el entorno
        currentState = env.reset()
        alpha = decValue(min_alpha,episode)
        epsilon = decValue(min_epsilon,episode)
        gamma = 0.9
        i=0
        done=False
        # Inicia el aprendizaje
        while not done and i<200:
            if render:
                #print("")
                env.render() # Muestra el entorno
                #print("")
            action = choose_action(currentState,epsilon)  # Selecciona la accion usando e-greedy policy
            newState, reward, done, info = env.step(action) # Realiza la accion en el entorno y obtiene las caracteristicas del nuevo estado
            # [-3.95896383e-11, -3.95896383e-11, -4.05666286e-11]
            reward = np.dot(np.asarray([-0.00404298, -0.00404298, -0.00404298]), tfe.get_features(currentState))
            updateQtable(currentState, newState, reward, action,alpha,gamma) # Actualiza la tabla Q con la nueva informacion
            currentState = newState
            if done:
                break
            i=i+1
        print("Termino el aprendizaje: "+str(i))
        total_score.append(i)
