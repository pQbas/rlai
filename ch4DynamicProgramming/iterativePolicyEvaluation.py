import numpy as np
import gymnasium as gym

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')
env.reset()
env.render()

print('action_space:', env.action_space,'-> {0: left, 1: down, 2: right, 3: up}')
print('observation_space:', env.observation_space)

STATES = range(env.observation_space.n)
ACTIONS = range(env.action_space.n)


policy = np.random.randint(low=0, high=4, size=env.observation_space.n)
print(f'policy:{policy}')


policy = np.random.rand(len(ACTIONS), len(STATES))
policy /= policy.sum(axis=1, keepdims=True)
print("Policy Matrix (Probability Distribution):\n",policy)

# initialize value table with zeros
valueF = np.zeros(env.observation_space.n)

# set the threshold
threshold = 1e-10

# gamma
gamma = 0.9

while True:
    delta = 0

    for state in STATES:
        oldVal = np.copy(valueF[state])

        # compute the new value for current state
        newVal = 0
        for action in ACTIONS:
            expectReturn = 0
            for transProb, nextState, reward, _ in env.P[state][action]:
                expectReturn += transProb*(reward + gamma*valueF[nextState])
            newVal += policy[action][state]*expectReturn
        valueF[state] = newVal

        delta = np.max([delta, np.abs(valueF[state] - oldVal)])

    if(delta) < threshold: break

print('value function:\n', valueF)


