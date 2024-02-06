---
layout: default
title: Reinforcement Learning
nav_order: 10
has_children: false
parent: Lectures
permalink: /lectures/RL/RL
---

# Reinforcement Learning

Reinforcement learning is the process of learning by interacting with an environment. Reinforcement learning is also blessed with a lot of history and hence terminology, that does not always make it easier. So I have tried to put it into context so I can understand and hopefully others as well. It does not intend to give a complete history lesson but the key concepts are important to understand because they are often referred to in modern neural network reinforcement learning. 




```python
import gym # openAi gym
from gym import envs
import numpy as np 
import datetime
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import sleep

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
print("OK")
```

    OK
    

    Using TensorFlow backend.
    

# Building Environment with Gym
Gym is released by Open AI in 2016 (http://gym.openai.com/docs/). It is a toolkit for developing and comparing reinforcement learning algorithms. OpenAI’s mission is to ensure that artificial general intelligence benefits all of humanity.

Let's start learning using some build-in games in Gym that do not require additonal installs. Let's start with a very basic game called Taxi.
  



```python
env = gym.make('Taxi-v2')
env.reset()
env.render()
```

    +---------+
    |R: | : :[35mG[0m|
    | : : : : |
    | : : : : |
    | | : | : |
    |[34;1mY[0m|[43m [0m: |B: |
    +---------+
    
    

# The Taxi game
There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drive to the passenger's location, pick up the passenger, drive to the passenger's destination (another one of the four specified locations), and then drop off the passenger. Once the passenger is dropped off, the episode ends. The taxi cannot pass thru a wall.

Actions: 
There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
Rewards: 
There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
    
Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations


https://gym.openai.com/envs/Taxi-v2/




# Interacting with the Gym environment  

gym makes it relative straightforward to interact with the game.  

<img src="https://cdn-images-1.medium.com/max/800/1*7Ae4mf9gVvpuMgenwtf8wA.png">

Each timestep, the agent chooses an action, and the environment returns an observation and a reward.  

*observation, reward, done, info = env.step(action) *  
* observation (object): an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game line Taxi.
* reward (float): amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
* done (boolean): whether it’s time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
* info (dict): ignore, diagnostic information useful for debugging. Official evaluations of your agent are not allowed to use this for learning.  

Let's first do some random steps in the game so you see how the game looks like



```python
# Let's first do some random steps in the game so you see how the game looks like

rew_tot=0
obs= env.reset()
env.render()
for _ in range(6):
    action = env.action_space.sample() #take step using random action from possible actions (actio_space)
    obs, rew, done, info = env.step(action) 
    rew_tot = rew_tot + rew
    env.render()
#Print the reward of these random action
print("Reward: %r" % rew_tot)    
    
```

    +---------+
    |[35mR[0m: | : :[43mG[0m|
    | : : : : |
    | : : : : |
    | | : | : |
    |[34;1mY[0m| : |B: |
    +---------+
    
    +---------+
    |[35mR[0m: | : :[43mG[0m|
    | : : : : |
    | : : : : |
    | | : | : |
    |[34;1mY[0m| : |B: |
    +---------+
      (Pickup)
    +---------+
    |[35mR[0m: | : :[43mG[0m|
    | : : : : |
    | : : : : |
    | | : | : |
    |[34;1mY[0m| : |B: |
    +---------+
      (Dropoff)
    +---------+
    |[35mR[0m: | : :G|
    | : : : :[43m [0m|
    | : : : : |
    | | : | : |
    |[34;1mY[0m| : |B: |
    +---------+
      (South)
    +---------+
    |[35mR[0m: | : :G|
    | : : :[43m [0m: |
    | : : : : |
    | | : | : |
    |[34;1mY[0m| : |B: |
    +---------+
      (West)
    +---------+
    |[35mR[0m: | : :G|
    | : :[43m [0m: : |
    | : : : : |
    | | : | : |
    |[34;1mY[0m| : |B: |
    +---------+
      (West)
    +---------+
    |[35mR[0m: | : :G|
    | :[43m [0m: : : |
    | : : : : |
    | | : | : |
    |[34;1mY[0m| : |B: |
    +---------+
      (West)
    Reward: -24
    

# Action 
Action (a): the input the agent provides to the environment. So what are the action commands the agents can give to the enironment? The env.action_space will tell you

What is the meaning of the actions? For the deep learning algorithm it should not matter, it should sort it out independent of the meaning of the action. But for humans it is handy to have the description, so we can understand the actions.   

In case of the Taxi game [0..5]:  
* 0: move south
* 1: move north
* 2: move east 
* 3: move west 
* 4: pickup passenger
* 5: dropoff passenger
  



```python
# action space has 6 possible actions, the meaning of the actions is nice to know for us humans but the neural network will figure it out
print(env.action_space)
NUM_ACTIONS = env.action_space.n
print("Possible actions: [0..%a]" % (NUM_ACTIONS-1))


```

    Discrete(6)
    Possible actions: [0..5]
    

# State  
State (s): This represents the board state of the game and in gym returned it is returned as observation. State: a numeric representation of what the agent is observing at a particular moment of time in the environment.  
In case of Taxi the observation is an integer, 500 different states are possible that translate to a nice graphic visual format with the render function. Note that this is specific for the Taxi game, in case of e.g. an Atari game the observation is the game screen with many coloured pixels.



```python
print(env.observation_space)
print()
env.env.s=42 # some random number, you might recognize it
env.render()
env.env.s = 222 # and some other
env.render()
```

    Discrete(500)
    
    +---------+
    |[34;1mR[0m: |[43m [0m: :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |[35mY[0m| : |B: |
    +---------+
      (West)
    +---------+
    |[34;1mR[0m: | : :G|
    | : : : : |
    | :[43m [0m: : : |
    | | : | : |
    |[35mY[0m| : |B: |
    +---------+
      (West)
    

# Markov decision process(MDP)
The Taxi game is an example of an [Markov decision process ](https://en.wikipedia.org/wiki/Markov_decision_process). The game can be described in states, possible actions in a state (leading to a next state with a certain probability) with a reward.

The word Markov refers to [Markovian property](https://en.wikipedia.org/wiki/Markov_property) which means that the state is independent of any previous states history, not on the sequence of events that preceded it. The current state encapsulates all that is needed to decide the future actions, no memory needed.

[small video to explain Markov model](https://www.youtube.com/watch?v=EqUfuT3CC8s)  

In terms of Reinforcement Learning the Environment is modelled as a markov model and the agent needs to take actions in this environment to maximize the amount of reward. Since the agent sees only the outside of the environment (the effects of it actions) it is often referred to as the hidden markov model which needs to be learned.

# Policy
Policy (π): The strategy that the agent employs to determine next action 'a'  in state 's'. Note that it does not state if it is a good or bad policy, it is a policy.   The policy is normally noted with the greek letter π.
Optimal policy (π*), policy which maximizes the expected reward. Among all the policies taken, the optimal policy is the one that optimizes to maximize the amount of reward received or expected to receive over a lifetime.  

So how do we find the optimal policy (π*) that is maximize our reward (and win the game) given the Taxi environment with the Markov model .  


# Bellman equation
We will make use of the basic Bellman equation for deterministic environments to solve a problem that is described as a Markov model, see figure below:

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/06039a80d0b3ef5b1eee6f05b9ae77867cda3026">

Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Q-l%C3%A6ring_formel_1.png)

where
* R(s,a) = Reward of action a in state s
* P(s'|s,a)= Probability of going to state s' given action a in state s. The Taxi game actions are deterministic (no such a thing as if I want to go north there is an 80% chance to go north and 10% chance to go west and 10% chance to go east). so the probability that selected action will lead to expected state is 100%. So ignore it for this game, it is always 1.
* γ = Discount factor gamma, how much discount is applicable for the future rewards. It must be between 0 and 1. The higher gamma the higher the focus on long term rewards

The value iteration algorithm makes use of the equation in the form of:  
* Value V(s): The expected long-term return with discount, as opposed to the short-term reward R. Vπ(s) is defined as the expected long-term return of the current state sunder policy π.  

The Q learning algorithm makes use of the equation in the form of:   
* Q-matrix or action-value Q(s,a): Q-matrix is similar to Value, except that it takes an extra parameter, the action a. Qπ(s, a) refers to the long-term return of the current state s, taking action a under policy π.

# Value iteration algorithm
 So' let's start first in "memory lane", the early days of reinforcement learning. Value iteration is the "hello world" of reinforcement learning methods to find an optimal policy to maximize reward for e.g. a Markov decision process problem.   

The value iteration is centred around the game states. The core of the idea is to calculate the value (expected long-term maximum result) of each state. The algorithm loops over all states (s) and possible actions (a) to explore rewards of a given action and calculates the maximum possible action/reward and stores it in V[s]. The algorithm iterates/repeats until V[s] is not (significantly) improving anymore. The Optimal policy (π*) is then to take every time the action to go state with the highest value. This value iteration algorithm is an example of what is referred to as [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) (DP) in literature. There are other DP techniques to solve this like policy iteration, etc but it can also be solved by a recursive program (a function that calls itself, look at the Bellman equation, it is a recursive definition).  
Anyhow, lets see how this value iteration works.




```python
# Value iteration algorithem
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
V = np.zeros([NUM_STATES]) # The Value for each state
Pi = np.zeros([NUM_STATES], dtype=int)  # Our policy with we keep updating to get the optimal policy
gamma = 0.9 # discount factor
significant_improvement = 0.01

def best_action_value(s):
    # finds the highest value action (max_a) in state s
    best_a = None
    best_value = float('-inf')

    # loop through all possible actions to find the best current action
    for a in range (0, NUM_ACTIONS):
        env.env.s = s
        s_new, rew, done, info = env.step(a) #take the action
        v = rew + gamma * V[s_new]
        if v > best_value:
            best_value = v
            best_a = a
    return best_a

iteration = 0
while True:
    # biggest_change is referred to by the mathematical symbol delta in equations
    biggest_change = 0
    for s in range (0, NUM_STATES):
        old_v = V[s]
        action = best_action_value(s) #choosing an action with the highest future reward
        env.env.s = s # goto the state
        s_new, rew, done, info = env.step(action) #take the action
        V[s] = rew + gamma * V[s_new] #Update Value for the state using Bellman equation
        Pi[s] = action
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    iteration += 1
    if biggest_change < significant_improvement:
        print (iteration,' iterations done')
        break
```

    74  iterations done
    


```python
# Let's see how the algorithm solves the taxi game
rew_tot=0
obs= env.reset()
env.render()
done=False
while done != True: 
    action = Pi[obs]
    obs, rew, done, info = env.step(action) #take step using selected action
    rew_tot = rew_tot + rew
    env.render()
#Print the reward of these actions
print("Reward: %r" % rew_tot)  
```

    +---------+
    |[35mR[0m: | : :[34;1mG[0m|
    | : : :[43m [0m: |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
    
    +---------+
    |[35mR[0m: | :[43m [0m:[34;1mG[0m|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
      (North)
    +---------+
    |[35mR[0m: | : :[34;1m[43mG[0m[0m|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
      (East)
    +---------+
    |[35mR[0m: | : :[42mG[0m|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
      (Pickup)
    +---------+
    |[35mR[0m: | : :G|
    | : : : :[42m_[0m|
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
      (South)
    +---------+
    |[35mR[0m: | : :G|
    | : : :[42m_[0m: |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
      (West)
    +---------+
    |[35mR[0m: | : :G|
    | : :[42m_[0m: : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
      (West)
    +---------+
    |[35mR[0m: | : :G|
    | :[42m_[0m: : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
      (West)
    +---------+
    |[35mR[0m:[42m_[0m| : :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
      (North)
    +---------+
    |[35m[42mR[0m[0m: | : :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
      (West)
    +---------+
    |[35m[42mR[0m[0m: | : :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
      (Dropoff)
    Reward: 11
    

# Model vs Model-free based methods
It nicely solves it. Still it feels a bit like cheating in terms of reinforcement learning. We have to know all environment states/transitions upfront so the algorithm works. In RL literature they refer to it as model based methods.
What if not all states are known upfront and we need to find out while we are learning. Hence we enter the realm of model-free based methods.

 

# Basic Q-learning algorithm  
The[ Q-learning](https://en.wikipedia.org/wiki/Q-learning) algorithm is centred around the actor (in this case the Taxi) and starts exploring based on trial-and-error to update its knowledge about the model and hence path to the best reward.
The core of the idea is the Q-matrix Q(s, a). It contains the maximum discounted future reward when we perform action a in state s. Or in other words Q(s, a) gives estimates the best course of action a in state s. Q-learning learns by trail and error and updates its policy (Q-matrix) based on reward.  to state it simple: the best it can do given a state it is in.  
After every step we update Q(s,a) using the reward, and the max Q value for new state resulting from the action. This update is done using the action value formula (based upon the Bellman equation) and allows state-action pairs to be updated in a recursive fashion (based on future values).   
The Bellman equation is extended with a learning rate (if you put learning rate = 1 it comes back to the basic Bellman equation) :

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Q-l%C3%A6ring_formel_1.png/800px-Q-l%C3%A6ring_formel_1.png">

Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Q-l%C3%A6ring_formel_1.png)

Note: [Temporal difference learning ](https://en.wikipedia.org/wiki/Temporal_difference_learning) and [Sarsa](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action) algorithems explored simular value expressions. Q-learning was the basis for Deep Q-learning (Deep referring to Neural Network technology). so, let's see how the Q-learning algorithm works.



```python
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
Q = np.zeros([NUM_STATES, NUM_ACTIONS]) #You could also make this dynamic if you don't know all games states upfront
gamma = 0.9 # discount factor
alpha = 0.9 # learning rate
for episode in range(1,1001):
    done = False
    rew_tot = 0
    obs = env.reset()
    while done != True:
            action = np.argmax(Q[obs]) #choosing the action with the highest Q value 
            obs2, rew, done, info = env.step(action) #take the action
            Q[obs,action] += alpha * (rew + gamma * np.max(Q[obs2]) - Q[obs,action]) #Update Q-marix using Bellman equation
            #Q[obs,action] = rew + gamma * np.max(Q[obs2]) # same equation but with learning rate = 1 returns the basic Bellman equation
            rew_tot = rew_tot + rew
            obs = obs2   
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode,rew_tot))
```

    Episode 50 Total Reward: -61
    Episode 100 Total Reward: -93
    Episode 150 Total Reward: -12
    Episode 200 Total Reward: 6
    Episode 250 Total Reward: 5
    Episode 300 Total Reward: 9
    Episode 350 Total Reward: 5
    Episode 400 Total Reward: 7
    Episode 450 Total Reward: 10
    Episode 500 Total Reward: 4
    Episode 550 Total Reward: 7
    Episode 600 Total Reward: 13
    Episode 650 Total Reward: 6
    Episode 700 Total Reward: 6
    Episode 750 Total Reward: 12
    Episode 800 Total Reward: 7
    Episode 850 Total Reward: 9
    Episode 900 Total Reward: 7
    Episode 950 Total Reward: 10
    Episode 1000 Total Reward: 6
    

So, what is the magic, how does it solve it? 

The Q-matrix is initialized with zero's. So initially it starts moving randomly until it hits a state/action with rewards or state/actions with a penalty. For understanding, let's simplify the problem that it needs to go to a certain drop-off position to get a reward. So random moves get no rewards but by luck (brute force enough tries) the state/action is found where a reward is given. So next game the immediate actions preceding this state/action will direct toward it by use of the Q-Matrix. The next iteration the actions before that, etc, etc. In other words, it solves "the puzzle" backwards from end-result (drop-off passenger) towards steps to be taken to get there in a iterative fashion.  

Note that in case of the Taxi game there is a reward of -1 for each action. So if in a state the algorithm explored eg south which let to no value the Q-matrix is updated to -1 so next iteration (because values were initialized on 0) it will try an action that is not yet tried and still on 0. So also by design it encourages systematic exploration of states and actions 

If you put the learning rate on 1 the game also is solved. Reason is that there is only one reward (dropoff passenger), so the algorithm will find it whatever learning rate. In case a game has more reward places the learning rate determines if it should prioritize longer term or shorter term rewards



```python
# Let's see how the algorithm solves the taxi game by following the policy to take actions delivering max value

rew_tot=0
obs= env.reset()
env.render()
done=False
while done != True: 
    action = np.argmax(Q[obs])
    obs, rew, done, info = env.step(action) #take step using selected action
    rew_tot = rew_tot + rew
    env.render()
#Print the reward of these actions
print("Reward: %r" % rew_tot)  
```

    +---------+
    |R: | :[43m [0m:G|
    | : : : : |
    | : : : : |
    | | : | : |
    |[34;1mY[0m| : |[35mB[0m: |
    +---------+
    
    +---------+
    |R: | : :G|
    | : : :[43m [0m: |
    | : : : : |
    | | : | : |
    |[34;1mY[0m| : |[35mB[0m: |
    +---------+
      (South)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : :[43m [0m: |
    | | : | : |
    |[34;1mY[0m| : |[35mB[0m: |
    +---------+
      (South)
    +---------+
    |R: | : :G|
    | : : : : |
    | : :[43m [0m: : |
    | | : | : |
    |[34;1mY[0m| : |[35mB[0m: |
    +---------+
      (West)
    +---------+
    |R: | : :G|
    | : : : : |
    | :[43m [0m: : : |
    | | : | : |
    |[34;1mY[0m| : |[35mB[0m: |
    +---------+
      (West)
    +---------+
    |R: | : :G|
    | : : : : |
    |[43m [0m: : : : |
    | | : | : |
    |[34;1mY[0m| : |[35mB[0m: |
    +---------+
      (West)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    |[43m [0m| : | : |
    |[34;1mY[0m| : |[35mB[0m: |
    +---------+
      (South)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |[34;1m[43mY[0m[0m| : |[35mB[0m: |
    +---------+
      (South)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |[42mY[0m| : |[35mB[0m: |
    +---------+
      (Pickup)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    |[42m_[0m| : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (North)
    +---------+
    |R: | : :G|
    | : : : : |
    |[42m_[0m: : : : |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (North)
    +---------+
    |R: | : :G|
    | : : : : |
    | :[42m_[0m: : : |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (East)
    +---------+
    |R: | : :G|
    | : : : : |
    | : :[42m_[0m: : |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (East)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : :[42m_[0m: |
    | | : | : |
    |Y| : |[35mB[0m: |
    +---------+
      (East)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    | | : |[42m_[0m: |
    |Y| : |[35mB[0m: |
    +---------+
      (South)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |[35m[42mB[0m[0m: |
    +---------+
      (South)
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |[35m[42mB[0m[0m: |
    +---------+
      (Dropoff)
    Reward: 5
    

# exploration vs. exploitation

The taxi game has one place with the reward: dropoff passenger +20. What if it also had a reward: stop at coffee shop +2 points. If the trial-and-error found that value first and in subsequent iterations started to optimize the route to it, how do we know it would also find the bigger reward of dropoff passenger?  
What if the the Taxi game actions are not deterministic (eg due to slippery roads I want to go north there is an 80% chance to go north and 10% chance to go west and 10% chance to go east), how would we ensure we still find the optimal policy?

Our algorithm to exploit action = np.argmax(Q[obs]) time and time again will not cope with these more complex environments. In Reinforcement literature this is called the crucial tradeoff between "exploitation" and "exploration".
* Exploitation: Make the best decision given current information (Go to the restaurent you know you like)
* Exploration: Gather more information (go to a new restaurent to find out if you like it)

Some approaches:  

* Epsilon Greedy  
We exploit the current situation with probability 1 — epsilon and explore a new option with probability epsilon,  the rates of exploration and exploitation are fixed
* Epsilon-Decreasing  
We exploit the current situation with probability 1 — epsilon and explore a new option with probability epsilon, with epsilon decreasing over time. 
* Thompson sampling: the rates of exploration and exploitation are dynamically updated with respect to the entire probability distribution of each arm
* Epsilon-Decreasing with Softmax
We exploit the current situation with probability 1 — epsilon and explore a new option with probability epsilon, with epsilon decreasing over time.  In the case of exploring a new option, we don’t just pick an option at random, but instead we estimate the outcome of each option, and then pick based on that (this is the softmax part).  
* Etc  

The code below I tried with epsilon-greedy

A nice place to test this is in the game [Frozen lakes](https://gym.openai.com/envs/FrozenLake-v0/) of OpenAI/Gym.  

 Short description: "Winter is here. You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen, but there are a few holes where the ice has melted. If you step into one of those holes, you'll fall into the freezing water. At this time, there's an international frisbee shortage, so it's absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won't always move in the direction you intend."  

Notice that the game is not deterministic anymore: "won't always move in the direction you intend". Note it is really slippery, the chance you move into the direction you want is actually not that big

S- Start  
G - Goal  
F- Frozen (safe)  
H- Hole (dead)  


```python
# Let's show the game layout

env = gym.make('FrozenLake-v0')
rew_tot=0
obs= env.reset()
env.render()

```

    
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
    


```python
env = gym.make('FrozenLake-v0')
env.reset()
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
Q = np.zeros([NUM_STATES, NUM_ACTIONS]) #You could also make this dynamic if you don't know all games states upfront
gamma = 0.95 # discount factor
alpha = 0.01 # learning rate
epsilon = 0.1 #
for episode in range(1,500001):
    done = False
    obs = env.reset()
    while done != True:
        if np.random.rand(1) < epsilon:
            # exploration with a new option with probability epsilon, the epsilon greedy approach
            action = env.action_space.sample()
        else:
            # exploitation
            action = np.argmax(Q[obs])
        obs2, rew, done, info = env.step(action) #take the action
        Q[obs,action] += alpha * (rew + gamma * np.max(Q[obs2]) - Q[obs,action]) #Update Q-marix using Bellman equation
        obs = obs2   
        
    if episode % 5000 == 0:
        #report every 5000 steps, test 100 games to get avarage point score for statistics and verify if it is solved
        rew_average = 0.
        for i in range(100):
            obs= env.reset()
            done=False
            while done != True: 
                action = np.argmax(Q[obs])
                obs, rew, done, info = env.step(action) #take step using selected action
                rew_average += rew
        rew_average=rew_average/100
        print('Episode {} avarage reward: {}'.format(episode,rew_average))
        
        if rew_average > 0.8:
            # FrozenLake-v0 defines "solving" as getting average reward of 0.78 over 100 consecutive trials.
            # Test it on 0.8 so it is not a one-off lucky shot solving it
            print("Frozen lake solved")
            break
 
```

    Episode 5000 avarage reward: 0.0
    Episode 10000 avarage reward: 0.0
    Episode 15000 avarage reward: 0.24
    Episode 20000 avarage reward: 0.52
    Episode 25000 avarage reward: 0.78
    Episode 30000 avarage reward: 0.6
    Episode 35000 avarage reward: 0.67
    Episode 40000 avarage reward: 0.69
    Episode 45000 avarage reward: 0.71
    Episode 50000 avarage reward: 0.77
    Episode 55000 avarage reward: 0.64
    Episode 60000 avarage reward: 0.75
    Episode 65000 avarage reward: 0.73
    Episode 70000 avarage reward: 0.63
    Episode 75000 avarage reward: 0.72
    Episode 80000 avarage reward: 0.68
    Episode 85000 avarage reward: 0.7
    Episode 90000 avarage reward: 0.75
    Episode 95000 avarage reward: 0.76
    Episode 100000 avarage reward: 0.8
    Episode 105000 avarage reward: 0.8
    Episode 110000 avarage reward: 0.75
    Episode 115000 avarage reward: 0.69
    Episode 120000 avarage reward: 0.74
    Episode 125000 avarage reward: 0.7
    Episode 130000 avarage reward: 0.7
    Episode 135000 avarage reward: 0.7
    Episode 140000 avarage reward: 0.76
    Episode 145000 avarage reward: 0.76
    Episode 150000 avarage reward: 0.76
    Episode 155000 avarage reward: 0.71
    Episode 160000 avarage reward: 0.76
    Episode 165000 avarage reward: 0.73
    Episode 170000 avarage reward: 0.81
    Frozen lake solved
    


```python
# Let's see how the algorithm solves the frozen-lakes game

rew_tot=0.
obs= env.reset()
done=False
while done != True: 
    action = np.argmax(Q[obs])
    obs, rew, done, info = env.step(action) #take step using selected action
    rew_tot += rew
    env.render()

print("Reward:", rew_tot)  
```

      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    FHFH
    F[41mF[0mFH
    HFFG
      (Down)
    SFFF
    FHFH
    FFFH
    H[41mF[0mFG
      (Right)
    SFFF
    FHFH
    FFFH
    HF[41mF[0mG
      (Down)
    SFFF
    FHFH
    FFFH
    HFF[41mG[0m
    Reward: 1.0
    

Analyzing the moves. It looks like that if you want to move rigth there is a significant chance you move up or down, simular if you want to move up there is a significant chance you move left or right, etc. So the algorithm learned that if you are on the frozen tile left column second row and you want to move down it is risky to give the down command because you could move to the right into the hole. So it gives the left command because if will keep you on the tile or move you up or down, but not to thr right.  
Or in other words, the algorithm learned to take that actions with the least risk to (accidently slip) drown into a hole. Also interesting to se it learned as first move to go left, this to avoid you move right which is the more dangerous road.  

Note: there is no 100% score possible. By consitently moving away from a hole you can safely traverse all fields except 1 (second row, thrid column) on which you could glide into due to slippery ice.  

Also good to notice the algorithm uses tenthousands of iterations to find the optimal policy, while this is a 4 by 4 playing field...

# DQN
Nice, isn't it. In case of 500 observation states and 6 actions (Taxi game) or 16 observations and 4 moves (Frozen-lake game) the Q-matrix is a manageable matrix. Imagine you got a full Atari game screen of pixels as an observation and it becomes quickly visible the Q-matrix solution will not cope. Also the Q-learning agent does not have the ability to estimate value for unseen states, it has no clue which action to take and goes back to random action as best.  

To deal with these problems, Deep Q-Network (DQN) removes the two-dimensional Q-matrix by introducing a Neural Network. So it leverages a Neural Network to estimate the Q-value function. The input for the network is the current game state, while the output is the corresponding Q-value for each of the actions.  
![alt text](https://cdn-images-1.medium.com/max/800/1*3ZgGbUpEyAZb9POWijRq4Q.png)

In 2014 Google DeepMind published a [paper](https://arxiv.org/pdf/1312.5602.pdf) titled "Playing Atari with Deep Reinforcement Learning" that can play Atari 2600 games at expert human levels. This was the first breakthruogh in applying deep neural networks for reinforcement learning.

![alt text](https://cdn-images-1.medium.com/max/800/1*vUMIoHkl-PuIjbTqbtn8dA.png)



# Reinforcement learning developments
After the first publication of DQN many deeplearning Reinforcement Learning algorithms have been invented/tried, Some main ones in chronological order:  DQN, Double DQN, Duelling DQN, Deep Deterministic Policy Gradient, Continuous DQN (CDQN or NAF) ,  A2C/A3C,  Proximal Policy Optimization Algorithms, etc, etc. 



## Resources

- https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym


