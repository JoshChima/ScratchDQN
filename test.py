import gym
# import forex

env = gym.make('Breakout-v0')

print(env.reset().shape)
print(env.step(env.action_space.sample())[0].shape)

# env = gym.make('Forex-v0')
# env = forex.envs.ForexRender(env)


# print('observation space:', env.observation_space)
# print('action space:', env.action_space)

# obs = env.reset()
# env.render('none')

# print('initial observation:', obs)

# action = env.action_space.sample()  # take a random action

# obs, r, done, info = env.step(action)
# print('next observation:', obs)
# print('reward:', r)
# print('done:', done)
# print('info:', info)
# env.reset()

# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         #print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

# print(env.FT.df.isnull().any())