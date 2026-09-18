#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import random
from collections import defaultdict

random.seed(42)


df_pos = pd.read_csv('positive_samples.csv', dtype=str)
df_pos['post_id'] = df_pos['post_id'].str.replace(r'\.0$', '', regex=True)
df_pos['sender_id'] = df_pos['sender_id'].str.replace(r'\.0$', '', regex=True)
df_pos['recipient_id'] = df_pos['recipient_id'].str.replace(r'\.0$', '', regex=True)

df_pos['created_at'] = df_pos['created_at'].str.replace(r'\.0$', '', regex=True)
df_pos['created_at'] = pd.to_datetime(df_pos['created_at'], format='%Y%m%d%H%M', errors='coerce')


df_m = pd.read_csv('target_posts_M.csv', dtype=str)
df_m['post_id'] = df_m['post_id'].str.replace(r'\.0$', '', regex=True)
df_m['author_id'] = df_m['author_id'].str.replace(r'\.0$', '', regex=True)


df_m['timestamp'] = df_m['timestamp'].str.replace(r'\.0$', '', regex=True)

df_users = pd.read_csv('target_users_whitelist.csv', dtype=str)
df_users['user_id'] = df_users['user_id'].str.replace(r'\.0$', '', regex=True)
all_users_pool = list(set(df_users['user_id']))


true_reposters_dict = defaultdict(set)
for _, row in df_pos.iterrows():
    true_reposters_dict[row['post_id']].add(row['recipient_id'])

hashtag_to_posts = defaultdict(list)
for _, row in df_m.iterrows():
    hashtag_to_posts[row['hashtag']].append({
        'post_id': row['post_id'],
        'sender_id': row['author_id'],
        'created_at': row['timestamp']
    })


df_pos_sample = df_pos.sample(n=2000, random_state=42).copy()
df_pos_sample['label'] = 1
pos_list = df_pos_sample[['post_id', 'sender_id', 'recipient_id', 'created_at', 'label']].to_dict('records')


neg_list = []

post_to_hashtag = df_m.set_index('post_id')['hashtag'].to_dict()

for pos_row in pos_list:
    pos_post = pos_row['post_id']
    pos_hashtag = post_to_hashtag.get(pos_post)
    
    candidate_posts = hashtag_to_posts.get(pos_hashtag, [])
    
    neg_count = 0
    while neg_count < 5:
        sampled_post_info = random.choice(candidate_posts)
        neg_post_id = sampled_post_info['post_id']
        neg_sender_id = sampled_post_info['sender_id']
        
        neg_recipient_id = random.choice(all_users_pool)
        
        if neg_recipient_id != neg_sender_id and neg_recipient_id not in true_reposters_dict[neg_post_id]:
            neg_list.append({
                'post_id': neg_post_id,
                'sender_id': neg_sender_id,
                'recipient_id': neg_recipient_id,
                'created_at': pd.to_datetime(sampled_post_info['created_at'], format='mixed'),
                'label': 0
            })
            neg_count += 1

final_experiment_list = pos_list + neg_list
df_experiment = pd.DataFrame(final_experiment_list)

df_experiment = df_experiment.sample(frac=1, random_state=42).reset_index(drop=True)

df_experiment.to_csv('experiment_12000_rows_1_to_5.csv', index=False)

df_1_to_1 = pd.concat([df_experiment[df_experiment['label'] == 1], 
                       df_experiment[df_experiment['label'] == 0].head(2000)])
df_1_to_1 = df_1_to_1.sample(frac=1, random_state=42).reset_index(drop=True)
df_1_to_1.to_csv('experiment_4000_rows_1_to_1.csv', index=False)



# In[ ]:




