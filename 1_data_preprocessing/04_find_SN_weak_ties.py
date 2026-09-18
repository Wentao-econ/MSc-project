#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random
from collections import defaultdict
import time


# In[13]:


# ==========================================
# 1. Configuration & Load Whitelist
# ==========================================
# Set random seed for reproducibility (crucial for academic research)
random.seed(42)  

print("Loading target users whitelist...")
df_users = pd.read_csv('target_users_whitelist.csv')

df_users['user_id'] = df_users['user_id'].astype(str).str.replace(r'\.0$', '', regex=True)
target_users_set = set(df_users['user_id'])

print(f"Loaded {len(target_users_set)} clean target users.")


# In[15]:


# ==========================================
# 2. Data Structures for Weak Ties
# ==========================================
# Using defaultdict(set) for O(1) lookup and automatic deduplication
following_dict = defaultdict(set)  # SN_Out: Users that the target user follows
followers_dict = defaultdict(set)  # SN_In: Users that follow the target user


# In[17]:


# ==========================================
# 3. Read Follower Graph in Chunks (Memory Safe)
# ==========================================
FOLLOWERS_FILE = 'followers.csv.gz'
chunk_size = 1000000  # Process 1 million rows at a time to prevent RAM overflow
processed_rows = 0

print(f"\nScanning {FOLLOWERS_FILE} to build weak tie networks...")
start_time = time.time()

# Read the CSV in chunks. Based on the data peek, there is no header.
# Column 0 is the follower (source), Column 1 is the followee (target).
for chunk in pd.read_csv(FOLLOWERS_FILE, compression='gzip', header=None, chunksize=chunk_size, dtype=str):
    
    # Assign clear column names based on dataset documentation
    chunk.columns = ['source', 'target'] 
    
    # Memory optimization: Only keep edges where at least one node is in our whitelist
    mask_source = chunk['source'].isin(target_users_set)
    mask_target = chunk['target'].isin(target_users_set)
    relevant_edges = chunk[mask_source | mask_target]
    
    # Populate the dictionaries
    for _, row in relevant_edges.iterrows():
        src = row['source']
        tgt = row['target']
        
        # If the source is a whitelist user, they follow the target
        if src in target_users_set:
            following_dict[src].add(tgt)
        # If the target is a whitelist user, they are followed by the source
        if tgt in target_users_set:
            followers_dict[tgt].add(src)
            
    processed_rows += chunk_size
    if processed_rows % 5000000 == 0:
        print(f"Processed {processed_rows:,} edges...")

print(f"Network build complete in {time.time() - start_time:.2f} seconds.")


# In[18]:


# ==========================================
# 4. Extract Top-K (K=5) Weak Tie Neighbors
# ==========================================
K = 5
sn_weak_records = []

print("\nSampling up to K=5 neighbors for each target user...")

for user in target_users_set:
    # Retrieve the full neighbor sets
    out_set = following_dict.get(user, set())
    in_set = followers_dict.get(user, set())
    
    # Calculate mutuals (intersection of following and followers)
    mut_set = out_set.intersection(in_set)
    
    # Uniform random sampling without replacement (up to K)
    sn_out = random.sample(list(out_set), min(K, len(out_set)))
    sn_in  = random.sample(list(in_set), min(K, len(in_set)))
    sn_mut = random.sample(list(mut_set), min(K, len(mut_set)))
    
    # Store the results as comma-separated strings for easy CSV storage
    sn_weak_records.append({
        'user_id': user,
        'SN_Out': ','.join(sn_out) if sn_out else None,
        'SN_In': ','.join(sn_in) if sn_in else None,
        'SN_Mut': ','.join(sn_mut) if sn_mut else None
    })


# In[21]:


# ==========================================
# 5. Save Results
# ==========================================
df_sn_weak = pd.DataFrame(sn_weak_records)
df_sn_weak.to_csv('sn_weak_ties_K5.csv', index=False)

print(f"\nSaved weak tie definitions for {len(df_sn_weak)} users to 'sn_weak_ties_K5.csv'.")
print("Preview of the data:")
print(df_sn_weak.head())


# In[23]:


# ==========================================
# 6. Summary Statistics for Thesis (Weak Ties)
# ==========================================
print("\n" + "="*50)
print("🎓 NETWORK STATISTICS: WEAK TIES")
print("="*50)

total_users = len(df_sn_weak)
print(f"Total target users analyzed: {total_users:,}")

def count_neighbors(x):
    """Helper function to count the number of neighbors from a comma-separated string."""
    if pd.isna(x) or str(x).strip() == '':
        return 0
    return len(str(x).split(','))

for col in ['SN_Out', 'SN_In', 'SN_Mut']:
    # Count how many neighbors each user has (0 to 5)
    neighbor_counts = df_sn_weak[col].apply(count_neighbors)
    
    # Calculate key statistics
    users_with_ties = (neighbor_counts > 0).sum()
    pct_with_ties = (users_with_ties / total_users) * 100
    avg_neighbors = neighbor_counts.mean()
    users_capped = (neighbor_counts == 5).sum()
    pct_capped = (users_capped / total_users) * 100

    print(f"\n[{col}]")
    print(f"- Users with >= 1 neighbor : {users_with_ties:,} ({pct_with_ties:.1f}%)")
    print(f"- Users hitting the cap (5): {users_capped:,} ({pct_capped:.1f}%)")
    print(f"- Average neighbors per user : {avg_neighbors:.2f} (Max: 5)")

print("="*50)


# In[ ]:




