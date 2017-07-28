
# coding: utf-8

# In[13]:


import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr


# In[14]:


def rargmax(vector):
    m = np.amax(vector)
    print(m)
    indices = np.nonzero(vector == m)[0]
    print(indices)
    return pr.choice(indices)


# In[15]:


rargmax([1,3,5])


# In[16]:


rargmax([1,3,2,3,1,5,0])


# In[17]:


register(id='FrozenLake-v3', 
         entry_point='gym.envs.toy_text:FrozenLakeEnv', 
         kwargs={'map_name': '4x4', 'is_slippery': False})


# In[18]:


env = gym.make('FrozenLake-v3')


# In[19]:


Q = np.zeros([env.observation_space.n, env.action_space.n])
dis = 0.99
num_episodes = 200


# In[20]:


rList = []


# In[21]:


for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    print(i)    
    while not done:
        action = rargmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))
        new_state, reward, done, _ = env.step(action)
        Q[state, action] = reward + dis * np.max(Q[new_state, :])
        rAll += reward
        state = new_state
    
    rList.append(rAll)


# In[22]:


print("Success rate:" + str(sum(rList)/num_episodes))
print("final Q-table values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()


# In[ ]:




