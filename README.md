# Inverse Reinforcement Learning application for discrete and continuous environments

Application of inverse reinforcement learning (IRL) in discrete and continuous environments based on the apprenticeship focus.

You can read the 
[document](./Inverse_reinforcement_learning_application_for_discrete_and_continuous_environments.pdf) within this repository for more information about the code and the concept.

## Installing dependences for python 3 (using ubuntu 18.04)

``` bash
sudo apt install python3-pip python3-tk python3-dev
sudo -H python3 -m pip install setuptools wheel numpy gym matplotlib pandas tensorflow keras cvxpy
```

If you want to install tensorflow with GPU support, read the official [guide](https://www.tensorflow.org/install/gpu).

## Running CartPole

### QLearning Cartpole with QTable

Run a terminal with python -i and the code:

``` bash
python3 -i ./CartPole/CartPole_QTable.py
```

Inside the Python RELP you can interact with *Cart* objetct, you can train and test with:

``` bash
Cart = Qlearning_CartPole('CartPole-v1',1000,1,0.01,0.1)
Cart.start()
Cart.printGraph()
```
### QLearning Cartpole with Neuronal Network

Run a terminal with python -i and the code:

``` bash
python3 -i ./CartPole/CartPole_NN.py
```

Inside the Python RELP you can create the environment, solver and agent, you can train and test with:

``` bash
env = gym.make('CartPole-v1').env
solver = NNAgent(env, env.action_space.n, env.observation_space.shape[0], LEARNING_RATE, BATCH_SIZE, GAMMA, EXPLORATION_RATE, EXPLORATION_DECAY, EXPLORATION_MIN, MAX_MEMORY)
solver.makeNewModel()
Agent = EnvironmentSolver(solver)
Agent.printGraph()
```

## Running Taxi