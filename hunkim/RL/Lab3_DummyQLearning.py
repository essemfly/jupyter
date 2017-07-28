
# coding: utf-8

# In[4]:


import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr


# In[8]:


def rargmax(vector):
    m = np.amax(vector)
    print(m)
    indices = np.nonzero(vector == m)[0]
    print(indices)
    return pr.choice(indices)


# In[9]:


rargmax([1,3,5])


# In[10]:


rargmax([1,3,2,3,1,5,0])


# In[11]:


register(id='FrozenLake-v3', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'map_name': '4x4', 'is_slippery': False})


# In[12]:


env = gym.make('FrozenLake-v3')


# In[13]:


Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000


# In[14]:


rList = []


# In[15]:


for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    
    while not done:
        action = rargmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        Q[state, action] = reward + np.max(Q[new_state, :])
        rAll += reward
        state = new_state
    
    rList.append(rAll)


# In[18]:


print("Success rate:" + str(sum(rList)/num_episodes))
print("final Q-table values")
print ("left down light up")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()


# In[ ]:




