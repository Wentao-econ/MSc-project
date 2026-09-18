#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tarfile
import json
import pandas as pd
import time
import gc


df_pool = pd.read_csv('user_pool_12k.csv', dtype=str)
df_pool['user_id'] = df_pool['user_id'].str.replace(r'\.0$', '', regex=True)
user_pool = set(df_pool['user_id'])


# In[3]:



FILE_PATH = 'user_posts.tar.gz'
historical_posts = []
user_post_counts = {u: 0 for u in user_pool}

start_time = time.time()
processed_users = 0
chunk_index = 1
POST_LIMIT = 200000

with tarfile.open(FILE_PATH, 'r:gz') as tar:
    for member in tar:
        if member.isfile():
            f = tar.extractfile(member)
            if f is not None:
                for line in f:
                    try:
                        post_data = json.loads(line.decode('utf-8'))
                        author_id = str(post_data.get('user_id', '')).replace('.0', '')
                        
                        if author_id in user_pool:
                            user_post_counts[author_id] += 1
                            
                            if post_data.get('repost_from') is not None:
                                continue
                            
                            text = post_data.get('text', '')
                            if text:
                                post_id = str(post_data.get('post_id', '')).replace('.0', '')
                                timestamp = str(post_data.get('date', ''))
                                if not post_id or post_id == 'None':
                                    post_id = f"{author_id}_{timestamp}"
                                    
                                historical_posts.append({
                                    'post_id': post_id,
                                    'author_id': author_id,
                                    'created_at': timestamp,
                                    'text': text
                                })
                    except Exception:
                        continue
            
            processed_users += 1
            
  
            if len(historical_posts) >= POST_LIMIT:
                chunk_name = f'historical_posts_12k_part{chunk_index}.pkl'
                pd.DataFrame(historical_posts).to_pickle(chunk_name)
    
                historical_posts.clear()  
                gc.collect()    
                chunk_index += 1

            if processed_users % 200000 == 0:


# In[5]:


if historical_posts:
    chunk_name = f'historical_posts_12k_part{chunk_index}.pkl'
    pd.DataFrame(historical_posts).to_pickle(chunk_name)

df_post_counts = pd.DataFrame(list(user_post_counts.items()), columns=['user_id', 'total_post_num'])
df_post_counts.to_csv('user_post_counts_12k.csv', index=False)



# In[ ]:




