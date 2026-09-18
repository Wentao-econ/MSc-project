#!/usr/bin/env python
# coding: utf-8

# In[24]:


# ==========================================
# Peek at the Interactions Dataset
# ==========================================
import pandas as pd

# Read the first 5 rows of the compressed CSV
df_interactions = pd.read_csv('interactions.csv.gz', compression='gzip', nrows=5)
print("--- Interactions Data ---")
print(df_interactions.head())


# In[26]:


# ==========================================
# Peek at the User Posts Dataset
# ==========================================
import tarfile
import json

print("\n--- User Posts JSON Data ---")
with tarfile.open('user_posts.tar.gz', 'r:gz') as tar:
    # Find the first ordinary file in the compressed package
    for member in tar:
        if member.isfile():
            f = tar.extractfile(member)
            # Read the first line of the file
            first_line = f.readline().decode('utf-8')
            # Parse it into a JSON dictionary and format it for printing
            post_data = json.loads(first_line)
            print(json.dumps(post_data, indent=4))
            break  # Stop just by looking at the first point to prevent it from getting stuck


# In[22]:


# ==========================================
# Peek at the Followers Dataset
# ==========================================
# Read only the first 5 rows to quickly verify the column structure.
# We set header=None assuming it shares the same format as interactions.csv.
df_followers = pd.read_csv('followers.csv.gz', compression='gzip', nrows=5, header=None)

print("--- Followers Data ---")
print(df_followers.head())


# In[ ]:




