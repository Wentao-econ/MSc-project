#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from collections import defaultdict
import time
from datetime import timedelta


# In[5]:


# ==========================================
# 1. Configuration & Load Required Data
# ==========================================
print("Loading target users, posts, and positive samples...")

# Load whitelist users and clean IDs
df_users = pd.read_csv('target_users_whitelist.csv')
df_users['user_id'] = df_users['user_id'].astype(str).str.replace(r'\.0$', '', regex=True)
target_users_set = set(df_users['user_id'])

# Load target posts to extract timestamps (T_M)
df_m = pd.read_csv('target_posts_M.csv')
df_m['author_id'] = df_m['author_id'].astype(str).str.replace(r'\.0$', '', regex=True)
df_m['post_id'] = df_m['post_id'].astype(str).str.replace(r'\.0$', '', regex=True)

# Load positive samples to link reposters to post timestamps
df_pos = pd.read_csv('positive_samples.csv')
df_pos['reposter_id'] = df_pos['reposter_id'].astype(str).str.replace(r'\.0$', '', regex=True)
df_pos['post_id'] = df_pos['post_id'].astype(str).str.replace(r'\.0$', '', regex=True)


# In[ ]:


# ==========================================
# 2. Build 30-Day Time Windows for Each User
# ==========================================
print("Calculating 30-day interaction windows for each user...")

# Dictionary to store the latest interaction timestamp for each user
user_latest_timestamp = {}

# Map post_id to its creation timestamp
post_time_map = dict(zip(df_m['post_id'], df_m['timestamp'].astype(str)))

# Assign T_M to authors (U_S)
for _, row in df_m.iterrows():
    author = row['author_id']
    ts = str(row['timestamp']).strip()
    if author in target_users_set:
        # Keep the latest timestamp if a user authored multiple posts
        user_latest_timestamp[author] = max(user_latest_timestamp.get(author, '000000000000'), ts)

# Assign T_M to reposters (U_R) based on the post they interacted with
for _, row in df_pos.iterrows():
    reposter = row['reposter_id']
    pid = row['post_id']
    ts = post_time_map.get(pid, '000000000000')
    if reposter in target_users_set:
        user_latest_timestamp[reposter] = max(user_latest_timestamp.get(reposter, '000000000000'), ts)

# Calculate exactly [T_M - 30 days, T_M] and format back to string for fast string comparison
user_time_windows = {}
for user, ts in user_latest_timestamp.items():
    try:
        # Convert YYYYMMDDHHMM to datetime
        dt_end = pd.to_datetime(ts, format='%Y%m%d%H%M')
        dt_start = dt_end - timedelta(days=30)
        
        # Store as comparable strings
        user_time_windows[user] = {
            'start': dt_start.strftime('%Y%m%d%H%M'),
            'end': dt_end.strftime('%Y%m%d%H%M')
        }
    except Exception:
        # Fallback for malformed timestamps: allow a very wide window
        user_time_windows[user] = {'start': '200001010000', 'end': '209912312359'}


# In[8]:


# ==========================================
# 3. Track Frequencies (Nested Dicts)
# ==========================================
# Format: dict[actor_id][target_id] = count
reply_counts = defaultdict(lambda: defaultdict(int))
repost_counts = defaultdict(lambda: defaultdict(int))
quote_counts = defaultdict(lambda: defaultdict(int))


# In[9]:


# ==========================================
# 4. Scan Interactions with Time Filtering
# ==========================================
INTERACTIONS_FILE = 'interactions.csv.gz'
chunk_size = 1000000
processed_rows = 0

print(f"\nScanning {INTERACTIONS_FILE} to extract Strong Ties...")
start_time = time.time()

# Ensure we read everything as strings to prevent Pandas from mangling IDs
for chunk in pd.read_csv(INTERACTIONS_FILE, compression='gzip', header=None, chunksize=chunk_size, dtype=str):
    
    # 0:actor, 1:replied, 2:thread_root, 3:reposted, 4:quoted, 5:date
    chunk.columns = ['actor', 'replied', 'thread', 'reposted', 'quoted', 'date']
    chunk['actor'] = chunk['actor'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # Filter to target users only
    valid_actors = chunk['actor'].isin(target_users_set)
    relevant_chunk = chunk[valid_actors]
    
    # Set to check for null string representations
    null_strings = {'nan', 'NaN', 'None', '', 'null'}
    
    for _, row in relevant_chunk.iterrows():
        actor = row['actor']
        date_str = str(row['date']).strip()
        
        window = user_time_windows.get(actor)
        if not window:
            continue
            
        # Strict 30-day temporal filter
        if window['start'] <= date_str <= window['end']:
            
            # 1. Check Reply (Comment)
            replied = str(row['replied']).strip().replace('.0', '')
            if replied not in null_strings:
                reply_counts[actor][replied] += 1
                
            # 2. Check Repost
            reposted = str(row['reposted']).strip().replace('.0', '')
            if reposted not in null_strings:
                repost_counts[actor][reposted] += 1
                
            # 3. Check Quote
            quoted = str(row['quoted']).strip().replace('.0', '')
            if quoted not in null_strings:
                quote_counts[actor][quoted] += 1

    processed_rows += chunk_size
    if processed_rows % 5000000 == 0:
        print(f"Processed {processed_rows:,} interaction records...")

print(f"Frequency counting complete in {time.time() - start_time:.2f} seconds.")


# In[11]:


# ==========================================
# 5. Extract Top-K (K=5) and Aggregate 'Any'
# ==========================================
print("\nExtracting Top 5 interacted neighbors per category...")
K = 5
sn_strong_records = []

def get_top_k(counts_dict, k=5):
    """Sorts dictionary by frequency (descending), returns top K keys."""
    if not counts_dict:
        return None
    # Sort by count desc. Secondary sort by ID string to resolve ties deterministically.
    sorted_items = sorted(counts_dict.items(), key=lambda x: (-x[1], x[0]))
    top_k = [target for target, count in sorted_items[:k]]
    return ','.join(top_k)

for user in target_users_set:
    user_reply = reply_counts.get(user, {})
    user_repost = repost_counts.get(user, {})
    user_quote = quote_counts.get(user, {})
    
    # Calculate SN_Any by summing all interaction types
    user_any = defaultdict(int)
    for tgt, count in user_reply.items(): user_any[tgt] += count
    for tgt, count in user_repost.items(): user_any[tgt] += count
    for tgt, count in user_quote.items(): user_any[tgt] += count
    
    sn_strong_records.append({
        'user_id': user,
        'SN_Comment': get_top_k(user_reply, K),
        'SN_Repost': get_top_k(user_repost, K),
        'SN_Quote': get_top_k(user_quote, K),
        'SN_Any': get_top_k(user_any, K)
    })


# In[13]:


# ==========================================
# 6. Save Results
# ==========================================
df_sn_strong = pd.DataFrame(sn_strong_records)
df_sn_strong.to_csv('sn_strong_ties_K5.csv', index=False)

print(f"\nSaved strong tie definitions for {len(df_sn_strong)} users to 'sn_strong_ties_K5.csv'.")
print("Preview of the data:")
print(df_sn_strong.head())


# In[15]:


# ==========================================
# 7. Summary Statistics for Thesis (Strong Ties)
# ==========================================
print("\n" + "="*50)
print("🎓 NETWORK STATISTICS: STRONG TIES (30-DAY WINDOW)")
print("="*50)

total_users = len(df_sn_strong)
print(f"Total target users analyzed: {total_users:,}")

def count_neighbors(x):
    """Helper function to count the number of neighbors from a comma-separated string."""
    if pd.isna(x) or str(x).strip() == '':
        return 0
    return len(str(x).split(','))

for col in ['SN_Comment', 'SN_Repost', 'SN_Quote', 'SN_Any']:
    # Count how many active neighbors each user has (0 to 5)
    neighbor_counts = df_sn_strong[col].apply(count_neighbors)
    
    # Calculate key statistics
    users_with_ties = (neighbor_counts > 0).sum()
    pct_with_ties = (users_with_ties / total_users) * 100
    avg_neighbors = neighbor_counts.mean()

    print(f"\n[{col}]")
    print(f"- Users with active ties (>=1): {users_with_ties:,} ({pct_with_ties:.1f}%)")
    print(f"- Average strong ties per user: {avg_neighbors:.2f} (Max: 5)")

print("="*50)


# In[12]:


import pandas as pd
import numpy as np

df_sn_strong = pd.read_csv('sn_strong_ties_K5.csv')

# ==========================================
# 7. Summary Statistics for Thesis (Strong Ties)
# ==========================================
print("\n" + "="*50)
print("🎓 NETWORK STATISTICS: STRONG TIES (30-DAY WINDOW)")
print("="*50)

total_users = len(df_sn_strong)
print(f"Total target users analyzed: {total_users:,}")

def count_neighbors(x):
    """Helper function to count the number of neighbors from a comma-separated string."""
    if pd.isna(x) or str(x).strip() == '':
        return 0
    return len(str(x).split(','))

for col in ['SN_Comment', 'SN_Repost', 'SN_Quote', 'SN_Any']:
    # Count how many active neighbors each user has (0 to 5)
    neighbor_counts = df_sn_strong[col].apply(count_neighbors)
    
    # Calculate key statistics
    users_with_ties = (neighbor_counts > 0).sum()
    pct_with_ties = (users_with_ties / total_users) * 100
    avg_neighbors = neighbor_counts.mean()
    
    users_capped = (neighbor_counts == 5).sum()
    pct_capped = (users_capped / total_users) * 100

    print(f"\n[{col}]")
    print(f"- Users with active ties (>=1): {users_with_ties:,} ({pct_with_ties:.1f}%)")
    print(f"- Users hitting the cap (5): {users_capped:,} ({pct_capped:.1f}%)")
    print(f"- Average strong ties per user: {avg_neighbors:.2f} (Max: 5)")

print("="*50)


# In[ ]:




