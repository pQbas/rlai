import numpy as np
import gymnasium as gym

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')
env.reset()
env.render()

print('action_space:', env.action_space,'-> {0: left, 1: down, 2: right, 3: up}')
print('observation_space:', env.observation_space)

STATES = range(env.observation_space.n)
ACTIONS = range(env.action_space.n)

policy = np.random.rand(len(ACTIONS), len(STATES))
policy /= policy.sum(axis=0, keepdims=True)
print("Policy Matrix (Probability Distribution):\n",policy)



def maxActionAtState(currentState):

    actionReturn = []

    for action in ACTIONS:
        expectReturn = 0

        for transProb, nextState, reward, _ in env.P[currentState][action]:
            expectReturn += transProb*(reward + gamma*valueF[nextState])

        actionReturn.append({
            "action": action,
            "return": expectReturn
        })

    maxAction = max(actionReturn, key=lambda x: x['return'])

    return maxAction["action"] 


import random

# initialize value table with zeros
valueF = np.zeros(env.observation_space.n)
policy = [random.randint(0,env.action_space.n-1) for _ in range(env.observation_space.n)]

print('policy',policy)
threshold = 1e-50
gamma = 0.9


policyStable = False

while(not policyStable):
    # Policy Evaluation
    while True:
        delta = 0

        for state in STATES:
            # copy the old value of value function
            oldVal = np.copy(valueF[state])

            # compute the new value for current state
            newVal = 0
            action = policy[state]
            for transProb, nextState, reward, _ in env.P[state][action]:
                newVal += transProb*(reward + gamma*valueF[nextState])
            valueF[state] = newVal

            # compute the difference
            delta = np.max([delta, np.abs(valueF[state] - oldVal)])

        if(delta) < threshold: break


    # Policy Improvement
    policyStable = True

    for state in STATES:
        oldAction = policy[state]
        policy[state] = maxActionAtState(state) 
        if(oldAction != policy[state]): policyStable = False

    if policyStable:
        #print('Policy stable')
        break
    else:
        print('policy', policy)


## Applying the policy

state = env.env.s
for episode in range(50):
    action = policy[state]
    state, reward, terminated, truncated, info  = env.step(action=action)

    print(f'action:{action} -> nextstate: {state}, reward: {reward}')
    print(f'terminated:{terminated}, info: {info}')
   
    if terminated: break



