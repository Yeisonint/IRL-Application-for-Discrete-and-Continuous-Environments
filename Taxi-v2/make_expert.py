import gym
#import readchar
import numpy as np
import os
import time

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

env = gym.make('Taxi-v2')

QTable=np.load('QTable.npy',allow_pickle=True)

trajectories = []
episode_step = 0
tries = 0
goals=0

while episode_step < 20: # n_trajectories : 20
    trajectory = []
    step = 0
    state=env.reset()
    while True:
        os.system('clear')
        print("Episodio: ", episode_step)
        print("Paso: ", step)
        env.render()
        #print(state)
        action=np.argmax(QTable[state])
        state, reward, done, _ = env.step(action)
        trajectory.append((state, action))
        step += 1
        time.sleep(0.1)
        if done:
            env.render()
            tries+=1
            break
    if state >= 15 and done:
        trajectory_numpy = np.array(trajectory, float)
        print("trajectory_numpy.shape", trajectory_numpy.shape)
        trajectories.append(trajectory)
        episode_step += 1
        goals += 1

print("Intentos: "+str(tries)+" Aciertos: "+str(goals))
np.save("expert_demo",trajectories)