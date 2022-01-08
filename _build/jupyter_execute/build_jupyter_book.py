#!/usr/bin/env python
# coding: utf-8

# In[42]:


import os

os.system('jupyter-book build ./')


# In[23]:


pwd


# In[34]:


os.system('ghp-import -n -p -f _build/html')


# In[ ]:




