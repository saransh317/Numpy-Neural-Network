#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    cache = x
    return s, cache


# In[4]:


def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape==Z.shape)
    cache = Z
    return A, cache


# In[5]:


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0]==0
    assert(dZ.shape==Z.shape)
    return dZ    


# In[6]:


def sigmoid_backward(dA, cache):
    Z = cache
    s = sigmoid(Z)[0]
    dZ = dA*s*(1-s)
    assert(dZ.shape==Z.shape)
    
    return dZ


# In[ ]:




