#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os


os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'




get_ipython().system('hf download cardiffnlp/twitter-roberta-base-irony')
get_ipython().system('hf download j-hartmann/emotion-english-distilroberta-base')
get_ipython().system('hf download unitary/toxic-bert')




# In[ ]:




