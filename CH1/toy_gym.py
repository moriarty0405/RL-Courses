import gym

# classic cart-pole problem

'''
# Without observation 

env = gym.make('CartPole-v0')
env.reset()

# 1000 timestep
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
'''

# With observation
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        # Each timestep, the agent chooses an action, and the environment returns an observation and a reward
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break