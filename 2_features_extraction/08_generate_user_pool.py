#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import timedelta
import time


# In[3]:



df_exp = pd.read_csv('experiment_12000_rows_1_to_5.csv', dtype=str)
df_exp['created_at'] = pd.to_datetime(df_exp['created_at'])


df_weak = pd.read_csv('sn_weak_ties_K5.csv', dtype={'user_id': str})
df_weak['user_id'] = df_weak['user_id'].str.replace(r'\.0$', '', regex=True)
weak_dict = df_weak.set_index('user_id').to_dict(orient='index')


df_interact = pd.read_csv('whitelist_interactions_history.csv', dtype=str)
df_interact['actor'] = df_interact['actor'].str.replace(r'\.0$', '', regex=True)
df_interact['target'] = df_interact['target'].str.replace(r'\.0$', '', regex=True)
df_interact['date'] = df_interact['date'].str.replace(r'\.0$', '', regex=True)
df_interact['date'] = pd.to_datetime(df_interact['date'], format='%Y%m%d%H%M', errors='coerce')
df_interact = df_interact.dropna(subset=['date'])


# In[5]:



def get_strong_ties(user_id, current_time, df_inter, K=5):
    start_time = current_time - timedelta(days=30)
    
    mask = (df_inter['actor'] == user_id) & \
           (df_inter['date'] >= start_time) & \
           (df_inter['date'] < current_time)
    
    recent_interactions = df_inter[mask]
    if recent_interactions.empty:
        return {'TopComment': [], 'TopQuote': [], 'TopRepost': [], 'TopAny': []}
    
    def get_top_k(df_sub, k):
        if df_sub.empty: return []
        return df_sub['target'].value_counts().head(k).index.tolist()
    
    return {
        'TopComment': get_top_k(recent_interactions[recent_interactions['interaction_type'] == 'comment'], K),
        'TopQuote': get_top_k(recent_interactions[recent_interactions['interaction_type'] == 'quote'], K),
        'TopRepost': get_top_k(recent_interactions[recent_interactions['interaction_type'] == 'repost'], K),
        'TopAny': get_top_k(recent_interactions, K)
    }


# In[7]:



start_time = time.time()
results = []
unique_user_pool = set()

for idx, row in df_exp.iterrows():
    sender_id = str(row['sender_id'])
    recipient_id = str(row['recipient_id'])
    current_time = row['created_at']
    
    unique_user_pool.add(sender_id)
    unique_user_pool.add(recipient_id)
    
    strong_ties = get_strong_ties(recipient_id, current_time, df_interact, K=5)
    weak_ties = weak_dict.get(recipient_id, {'SN_Out': '', 'SN_In': '', 'SN_Mut': ''})
    
    row_dict = row.to_dict()
    row_dict['SN_In_List'] = weak_ties.get('SN_In', '')
    row_dict['SN_Out_List'] = weak_ties.get('SN_Out', '')
    row_dict['SN_Mut_List'] = weak_ties.get('SN_Mut', '')
    row_dict['SN_TopComment_List'] = ','.join(strong_ties['TopComment'])
    row_dict['SN_TopQuote_List'] = ','.join(strong_ties['TopQuote'])
    row_dict['SN_TopRepost_List'] = ','.join(strong_ties['TopRepost'])
    row_dict['SN_TopAny_List'] = ','.join(strong_ties['TopAny'])
    
    for key in ['SN_In_List', 'SN_Out_List', 'SN_Mut_List', 'SN_TopComment_List', 'SN_TopQuote_List', 'SN_TopRepost_List', 'SN_TopAny_List']:
        if row_dict[key]:
            unique_user_pool.update([n.strip() for n in str(row_dict[key]).split(',') if n.strip()])
            
    results.append(row_dict)
    
    if (idx + 1) % 1000 == 0:
        print(f"completed {idx + 1}/12000 rows... time cost: {time.time() - start_time:.2f} 秒")


# In[9]:


unique_user_pool.discard('')
unique_user_pool.discard('nan')
unique_user_pool.discard('None')

df_exp_with_sn = pd.DataFrame(results)
df_exp_with_sn.to_csv('experiment_12000_with_SN_lists.csv', index=False)

df_pool = pd.DataFrame({'user_id': list(unique_user_pool)})
df_pool.to_csv('user_pool_12k.csv', index=False)


# In[ ]:




