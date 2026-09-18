#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time
import gc


# In[3]:


# ==========================================
# 1. Configuration & Load Whitelist
# ==========================================
print("Loading target users whitelist...")
df_users = pd.read_csv('target_users_whitelist.csv')
df_users['user_id'] = df_users['user_id'].astype(str).str.replace(r'\.0$', '', regex=True)
target_users_set = set(df_users['user_id'])
print(f"Loaded {len(target_users_set)} clean target users.")


# In[5]:


# ==========================================
# 2. Extract Lightweight Interaction History
# ==========================================
INTERACTIONS_FILE = 'interactions.csv.gz'
OUTPUT_FILE = 'whitelist_interactions_history.csv'
chunk_size = 1000000
processed_rows = 0

print(f"\nScanning {INTERACTIONS_FILE} to extract interaction history for whitelist users...")
start_time = time.time()

# Create/overwrite the output file with headers
pd.DataFrame(columns=['actor', 'target', 'interaction_type', 'date']).to_csv(OUTPUT_FILE, index=False)

null_strings = {'nan', 'NaN', 'None', '', 'null'}

for chunk in pd.read_csv(INTERACTIONS_FILE, compression='gzip', header=None, chunksize=chunk_size, dtype=str):
    chunk.columns = ['actor', 'replied', 'thread', 'reposted', 'quoted', 'date']
    chunk['actor'] = chunk['actor'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # Fast filter: Keep only interactions where the actor is in our whitelist
    relevant = chunk[chunk['actor'].isin(target_users_set)]
    
    if not relevant.empty:
        records = []
        for _, row in relevant.iterrows():
            actor = row['actor']
            date_str = str(row['date']).strip()
            
            # Extract Comments (replied)
            replied = str(row['replied']).strip().replace('.0', '')
            if replied not in null_strings:
                records.append({'actor': actor, 'target': replied, 'interaction_type': 'comment', 'date': date_str})
                
            # Extract Reposts
            reposted = str(row['reposted']).strip().replace('.0', '')
            if reposted not in null_strings:
                records.append({'actor': actor, 'target': reposted, 'interaction_type': 'repost', 'date': date_str})
                
            # Extract Quotes
            quoted = str(row['quoted']).strip().replace('.0', '')
            if quoted not in null_strings:
                records.append({'actor': actor, 'target': quoted, 'interaction_type': 'quote', 'date': date_str})
        
        # Append to CSV dynamically to save RAM
        if records:
            pd.DataFrame(records).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            
    processed_rows += chunk_size
    if processed_rows % 5000000 == 0:
        print(f"Processed {processed_rows:,} raw interaction records...")
        gc.collect() # Free up memory

print(f"\nExtraction complete! History saved to '{OUTPUT_FILE}' in {time.time() - start_time:.2f} seconds.")


# In[ ]:




