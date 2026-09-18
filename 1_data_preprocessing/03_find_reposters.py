#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tarfile
import json
import pandas as pd
import time


# In[29]:


# ==========================================
# 1. Configuration & Load Target Posts
# ==========================================
FILE_PATH = 'user_posts.tar.gz'

# Load the candidate original posts collected in the previous step
print("Loading candidate posts from 'target_posts_M.csv'...")
df_m = pd.read_csv('target_posts_M.csv')

# Create a fast-lookup set for post IDs and extract unique authors (U_S)
target_post_ids = set(df_m['post_id'].astype(str))
unique_authors = set(df_m['author_id'].astype(str))

print(f"Loaded {len(target_post_ids)} candidate posts authored by {len(unique_authors)} unique users.")


# In[31]:


# ==========================================
# 2. Tracking Dictionaries for Reposters
# ==========================================
# Dictionary to store the list of reposter IDs for each post (Limit: 3 per post)
reposters_dict = {post_id: [] for post_id in target_post_ids}

# To keep track of unique reposters (U_R) globally
unique_reposters = set()

# Since reposts could be scattered anywhere, we scan a large portion of users.
# You can adjust this based on how long you are willing to let it run.
MAX_USERS_TO_SCAN = 4000000  

print(f"\nStarting to search for reposts (Scanning up to {MAX_USERS_TO_SCAN} users)...")
start_time = time.time()


# In[33]:


# ==========================================
# 3. Scan for Repost Interactions
# ==========================================
processed_users = 0
reposts_found = 0

with tarfile.open(FILE_PATH, 'r:gz') as tar:
    for member in tar:
        if member.isfile():
            f = tar.extractfile(member)
            if f is not None:
                for line in f:
                    try:
                        post_data = json.loads(line.decode('utf-8'))
                        
                        repost_from = str(post_data.get('repost_from', ''))
                        
                        # If this post is a repost AND the original post is in our target set
                        if repost_from in target_post_ids:
                            reposter_id = str(post_data.get('user_id'))
                            
                            # CRITICAL CONSTRAINT: Sample up to 3 reposters per post
                            if len(reposters_dict[repost_from]) < 3:
                                reposters_dict[repost_from].append(reposter_id)
                                unique_reposters.add(reposter_id)
                                reposts_found += 1
                                
                    except Exception:
                        continue
            
            processed_users += 1
            if processed_users % 50000 == 0:
                print(f"Scanned {processed_users} users... Found {reposts_found} valid repost interactions so far.")
            
            if processed_users >= MAX_USERS_TO_SCAN:
                break

end_time = time.time()
print(f"\nSearch completed! Total time elapsed: {end_time - start_time:.2f} seconds.")


# In[35]:


# ==========================================
# 4. Generate Dataset Statistics for Thesis
# ==========================================
# Calculate statistics strictly mirroring Jonas's methodology
total_posts = len(target_post_ids)
posts_with_reposts = sum(1 for reposter_list in reposters_dict.values() if len(reposter_list) > 0)
percentage_reposted = (posts_with_reposts / total_posts) * 100

total_repost_edges_stored = sum(len(reposter_list) for reposter_list in reposters_dict.values())
avg_reposters_per_reposted_post = total_repost_edges_stored / posts_with_reposts if posts_with_reposts > 0 else 0

full_user_set = unique_authors.union(unique_reposters)

print("\n" + "="*50)
print("🎓 DATASET STATISTICS FOR YOUR THESIS (DATA SECTION)")
print("="*50)
print(f"- Total posts collected: {total_posts:,}")
print(f"- Unique authors (U_S): {len(unique_authors):,}")
print(f"- Posts with at least one repost: {posts_with_reposts:,} ({percentage_reposted:.2f}%)")
print(f"- Average stored reposters per post (for posts > 0 reposts): {avg_reposters_per_reposted_post:.2f}")
print(f"- Full user set (Authors + Reposters): {len(full_user_set):,}")
print("="*50)


# In[37]:


# ==========================================
# 5. Save Positive Samples and User Set
# ==========================================
# 5a. Save positive interaction edges (U_S -> U_R over M)
positive_edges = []
for post_id, reposter_list in reposters_dict.items():
    for reposter_id in reposter_list:
        positive_edges.append({
            'post_id': post_id,
            'reposter_id': reposter_id,
            'label': 1  # 1 indicates a positive sample (reposted)
        })

df_positive = pd.DataFrame(positive_edges)
df_positive.to_csv('positive_samples.csv', index=False)
print(f"\nSaved {len(df_positive)} positive interaction edges to 'positive_samples.csv'.")

# 5b. Save the ultimate global target user list
df_users = pd.DataFrame({'user_id': list(full_user_set)})
df_users.to_csv('target_users_whitelist.csv', index=False)
print(f"Saved {len(df_users)} unique users to 'target_users_whitelist.csv'.")


# In[43]:


import pandas as pd

# ==========================================
# Fast Patch: Merge Sender and Date into Positive Samples
# ==========================================
print("Loading existing positive_samples.csv...")
df_pos = pd.read_csv('positive_samples.csv')

print("Loading target_posts_M.csv to extract author and timestamp...")
df_m = pd.read_csv('target_posts_M.csv')

# Standardize post_id format for accurate merging
df_pos['post_id'] = df_pos['post_id'].astype(str).str.replace(r'\.0$', '', regex=True)
df_m['post_id'] = df_m['post_id'].astype(str).str.replace(r'\.0$', '', regex=True)

# Exact column names confirmed from your 02_extract_target_posts.py script!
author_col = 'author_id'
time_col = 'timestamp'

# Extract only the necessary columns from the M table
df_m_subset = df_m[['post_id', author_col, time_col]]

# Merge the author and timestamp into the positive samples based on post_id
print("Merging datasets...")
df_merged = pd.merge(df_pos, df_m_subset, on='post_id', how='left')

# Rename the columns to perfectly match the U-feature script requirements
df_merged.rename(columns={
    author_col: 'sender_id',
    'reposter_id': 'recipient_id',
    time_col: 'created_at'
}, inplace=True)

# Reorder the columns for a clean, professional look
df_final = df_merged[['sender_id', 'recipient_id', 'post_id', 'created_at', 'label']]

# Drop duplicates just in case
df_final = df_final.drop_duplicates()

# Overwrite the old positive_samples.csv with the upgraded one
df_final.to_csv('positive_samples.csv', index=False)

print(f"✅ Successfully updated! The file now contains {len(df_final)} rows with sender_id and created_at.")
print("You can now safely run the big U-feature extraction code!")


# In[39]:


df_m = pd.read_csv('target_posts_M.csv')
df_positive = pd.read_csv('positive_samples.csv')

df_m['post_id'] = df_m['post_id'].astype(str).str.replace(r'\.0$', '', regex=True)
df_positive['post_id'] = df_positive['post_id'].astype(str).str.replace(r'\.0$', '', regex=True)

total_counts = df_m.groupby('hashtag')['post_id'].nunique().reset_index()
total_counts.columns = ['Hashtag', 'Total Posts Gathered']

reposted_post_ids = df_positive['post_id'].unique()
df_m_positive = df_m[df_m['post_id'].isin(reposted_post_ids)]
positive_counts = df_m_positive.groupby('hashtag')['post_id'].nunique().reset_index()
positive_counts.columns = ['Hashtag', 'Positive Instances']

table_3_1 = pd.merge(total_counts, positive_counts, on='Hashtag', how='left').fillna(0)
table_3_1['Positive Instances'] = table_3_1['Positive Instances'].astype(int)

print("\n=== Table 3.1: Positive instances and total instances gathered per hashtag ===")
print(table_3_1.to_markdown(index=False))


# In[ ]:




