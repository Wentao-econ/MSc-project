#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from collections import Counter
import time

print("📥 加载 10 万级用户池...")
df_pool = pd.read_csv('user_pool_12k.csv', dtype=str)
full_user_pool = set(df_pool['user_id'].str.replace(r'\.0$', '', regex=True))

follower_counts = Counter()
followee_counts = Counter()

print(f"⏳ 扫描巨大边表统计粉丝/关注数 (大概需要几分钟)...")
start_time = time.time()
chunk_size = 1000000

# 遍历原始边表文件
for chunk in pd.read_csv('followers.csv.gz', compression='gzip', header=None, chunksize=chunk_size, dtype=str):
    chunk.columns = ['source', 'target']
    
    sources = chunk.loc[chunk['source'].isin(full_user_pool), 'source']
    followee_counts.update(sources)
    
    targets = chunk.loc[chunk['target'].isin(full_user_pool), 'target']
    follower_counts.update(targets)

df_p_stats = pd.DataFrame({'user_id': list(full_user_pool)})
df_p_stats['FollowerNum'] = df_p_stats['user_id'].map(follower_counts).fillna(0)
df_p_stats['FolloweeNum'] = df_p_stats['user_id'].map(followee_counts).fillna(0)
df_p_stats.to_csv('follow_stats_12k.csv', index=False)
print(f"✅ 粉丝数据提取完毕！耗时: {time.time() - start_time:.2f} 秒。已保存为 'follow_stats_12k.csv'")


# In[1]:


import pandas as pd
import time
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings('ignore')

print("📥 1. 启动【分块吞吐】极速读取底层数据 (绝对不卡内存)...")
start_load = time.time()

# 🔴 救命核心：chunksize=50000，把巨型文件切成小块吃，彻底解决卡死！
chunk_iter = pd.read_csv('historical_posts_dl_features_final.csv', dtype={'author_id': str}, chunksize=50000, low_memory=False)
df_hm_list = []

for i, chunk in enumerate(chunk_iter):
    chunk['author_id'] = chunk['author_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    chunk['created_at'] = pd.to_datetime(chunk['created_at'], errors='coerce')
    chunk = chunk.dropna(subset=['created_at'])
    df_hm_list.append(chunk)
    gc.collect() # 强制清理内存垃圾
    print(f"  -> 已极速读取并清理第 {i+1} 块数据...")

df_hm = pd.concat(df_hm_list, ignore_index=True)
del df_hm_list
gc.collect()
hm_cols = [c for c in df_hm.columns if c.startswith('base_')]

print("  -> 正在读取其他小型配置表...")
df_post = pd.read_csv('user_post_counts_12k.csv', dtype={'user_id': str})
df_post['user_id'] = df_post['user_id'].str.replace(r'\.0$', '', regex=True)
post_count_dict = df_post.set_index('user_id')['total_post_num'].to_dict()
max_post_num = max(post_count_dict.values()) if post_count_dict else 1

df_follow = pd.read_csv('follow_stats_12k.csv', dtype={'user_id': str})
df_follow['user_id'] = df_follow['user_id'].str.replace(r'\.0$', '', regex=True)
follower_dict = df_follow.set_index('user_id')['FollowerNum'].to_dict()
followee_dict = df_follow.set_index('user_id')['FolloweeNum'].to_dict()

df_interact = pd.read_csv('whitelist_interactions_history.csv', dtype=str)
for col in ['actor', 'target', 'date']:
    df_interact[col] = df_interact[col].str.replace(r'\.0$', '', regex=True)
df_interact['date'] = pd.to_datetime(df_interact['date'], errors='coerce')
df_interact = df_interact.dropna(subset=['date'])

df_exp = pd.read_csv('experiment_12000_with_SN_lists.csv', dtype=str)
df_exp['created_at'] = pd.to_datetime(df_exp['created_at'], errors='coerce')
df_exp['label'] = df_exp['label'].astype(int)
df_exp = df_exp.dropna(subset=['created_at'])

print(f"✅ 所有底层数据加载完毕！耗时: {time.time() - start_load:.2f} 秒。")


# In[3]:


# ==========================================
# 2. 建立哈希索引，启动 M2 极速时光机
# ==========================================
print("\n🚀 2. 建立哈希索引，启动 M2 极速时光机...")
hm_dict = dict(tuple(df_hm.groupby('author_id')))
interact_actor_dict = dict(tuple(df_interact.groupby('actor')))
interact_target_dict = dict(tuple(df_interact.groupby('target')))

def get_user_features(user_id, current_time, prefix='U-HM_R_'):
    if pd.isna(user_id) or not str(user_id).strip(): return pd.Series(dtype='float64')
    p_prefix, ha_prefix = prefix.replace('HM', 'P'), prefix.replace('HM', 'HA')
    post_num = post_count_dict.get(user_id, 0)
    
    p_features = pd.Series({
        f"{p_prefix}FollowerNum": follower_dict.get(user_id, 0),
        f"{p_prefix}FolloweeNum": followee_dict.get(user_id, 0),
        f"{p_prefix}PostNum": post_num,
        f"{p_prefix}SpreadActivity": post_num / max_post_num if max_post_num > 0 else 0
    })
    
    user_as_actor = interact_actor_dict.get(user_id, pd.DataFrame())
    hist_active = user_as_actor[user_as_actor['date'] < current_time] if not user_as_actor.empty else pd.DataFrame()
    total_active = len(hist_active)
    repost_reply_count = len(hist_active[hist_active['interaction_type'].isin(['repost', 'comment'])]) if total_active > 0 else 0
    avg_interval = (hist_active['date'].sort_values().diff().dt.total_seconds() / 86400).mean() if total_active > 1 else 0.0
    
    user_as_target = interact_target_dict.get(user_id, pd.DataFrame())
    hist_target = user_as_target[user_as_target['date'] < current_time] if not user_as_target.empty else pd.DataFrame()
    denominator = post_num if post_num > 0 else 1
    
    ha_features = pd.Series({
        f"{ha_prefix}PostNum": total_active,
        f"{ha_prefix}RepostPercent": (repost_reply_count / total_active) if total_active > 0 else 0.0,
        f"{ha_prefix}AverageInterval": avg_interval,
        f"{ha_prefix}RepostedRate": len(hist_target[hist_target['interaction_type'] == 'repost']) / denominator if not hist_target.empty else 0.0,
        f"{ha_prefix}QuotedRate": len(hist_target[hist_target['interaction_type'] == 'quote']) / denominator if not hist_target.empty else 0.0,
        f"{ha_prefix}CommentedRate": len(hist_target[hist_target['interaction_type'] == 'comment']) / denominator if not hist_target.empty else 0.0
    })
    
    user_hm = hm_dict.get(user_id, pd.DataFrame())
    df_user_hist = user_hm[user_hm['created_at'] < current_time].sort_values('created_at', ascending=False).head(50) if not user_hm.empty else pd.DataFrame()
    
    if df_user_hist.empty:
        hm_features = pd.Series([0.0]*len(hm_cols), index=[c.replace('base_', prefix) for c in hm_cols])
    else:
        hm_features = df_user_hist[hm_cols].mean()
        hm_features.index = [c.replace('base_', prefix) for c in hm_features.index]
        
    return pd.concat([p_features, ha_features, hm_features])

def get_sn_features(sn_list_str, current_time, prefix='SN_Mut-HM_'):
    if pd.isna(sn_list_str) or not str(sn_list_str).strip(): return pd.Series(dtype='float64')
    neighbors = [n.strip() for n in str(sn_list_str).split(',') if n.strip()]
    if not neighbors: return pd.Series(dtype='float64')
    
    neighbor_feats = [get_user_features(n_id, current_time, prefix='temp_HM_') for n_id in neighbors]
    neighbor_feats = [nf for nf in neighbor_feats if not nf.empty and not nf.isna().all()]
    
    if not neighbor_feats: return pd.Series(dtype='float64')
    sn_mean = pd.DataFrame(neighbor_feats).mean()
    sn_mean.index = [c.replace('temp_HM_', prefix).replace('temp_P_', prefix.replace('HM', 'P')).replace('temp_HA_', prefix.replace('HM', 'HA')) for c in sn_mean.index]
    return sn_mean

start_time = time.time()
final_results = []

# 这里带有绿色的 tqdm 进度条，你会直观看到速度！
for idx, row in tqdm(df_exp.iterrows(), total=len(df_exp), desc="M2 全速狂飙中"):
    T, R, S = row['created_at'], row['recipient_id'], row['sender_id']
    row_dict = {'post_id': row['post_id'], 'created_at': T, 'label': row['label']}
    
    row_dict.update(get_user_features(R, T, prefix='U-HM_R_').to_dict())
    row_dict.update(get_user_features(S, T, prefix='U-HM_S_').to_dict())
    row_dict.update(get_sn_features(row.get('SN_In_List', ''), T, prefix='SN_In-HM_').to_dict())
    row_dict.update(get_sn_features(row.get('SN_Out_List', ''), T, prefix='SN_Out-HM_').to_dict())
    row_dict.update(get_sn_features(row.get('SN_Mut_List', ''), T, prefix='SN_Mut-HM_').to_dict())
    row_dict.update(get_sn_features(row.get('SN_TopComment_List', ''), T, prefix='SN_TopComment-HM_').to_dict())
    row_dict.update(get_sn_features(row.get('SN_TopQuote_List', ''), T, prefix='SN_TopQuote-HM_').to_dict())
    row_dict.update(get_sn_features(row.get('SN_TopRepost_List', ''), T, prefix='SN_TopRepost-HM_').to_dict())
    row_dict.update(get_sn_features(row.get('SN_TopAny_List', ''), T, prefix='SN_TopAny-HM_').to_dict())
    
    final_results.append(row_dict)

df_assembled = pd.DataFrame(final_results)
print(f"✅ 拼装彻底完成！耗时: {time.time() - start_time:.2f} 秒。")


# In[5]:


# ==========================================
# 3. 合并 M 特征 & 切分数据
# ==========================================
print("\n🧩 3. 安全合并 M 特征并切分...")
df_m = pd.read_csv('features_M_full_v2.csv', dtype={'post_id': str}, low_memory=False) 
df_m['post_id'] = df_m['post_id'].astype(str).str.replace(r'\.0$', '', regex=True)
exclude_m = ['post_id', 'author_id', 'timestamp', 'hashtag', 'text', 'created_at']
m_cols = [c for c in df_m.columns if c not in exclude_m]

df_m_clean = df_m[['post_id'] + m_cols].drop_duplicates(subset=['post_id']).copy()
df_final_1to5 = pd.merge(df_assembled, df_m_clean, on='post_id', how='left')

meta_cols = ['post_id', 'created_at', 'sender_id', 'recipient_id', 'label']
for col in df_final_1to5.columns:
    if col not in meta_cols:
        df_final_1to5[col] = pd.to_numeric(df_final_1to5[col], errors='coerce').fillna(0)

df_pos_only = df_final_1to5[df_final_1to5['label'] == 1]
df_neg_only = df_final_1to5[df_final_1to5['label'] == 0].groupby('post_id').head(1)
df_final_1to1 = pd.concat([df_pos_only, df_neg_only]).reset_index(drop=True)

df_final_1to1 = df_final_1to1.sort_values('created_at').reset_index(drop=True)
df_final_1to5 = df_final_1to5.sort_values('created_at').reset_index(drop=True)

split_idx_1to1 = int(len(df_final_1to1) * 0.8)
split_idx_1to5 = int(len(df_final_1to5) * 0.8)

df_final_1to1.iloc[:split_idx_1to1].to_pickle('dataset_1to1_Time_ID.pkl')
df_final_1to5.iloc[:split_idx_1to5].to_pickle('dataset_1to5_Time_ID.pkl')
df_final_1to1.iloc[split_idx_1to1:].to_pickle('dataset_1to1_Time_OOD.pkl')
df_final_1to5.iloc[split_idx_1to5:].to_pickle('dataset_1to5_Time_OOD.pkl')

print("\n🎉 全部大功告成！拿去给导师汇报吧！")


# In[7]:


import pandas as pd

print("📥 正在将 Pickle 转换为 CSV 方便预览...")

# 我们刚刚生成的 4 个终极数据集
files = [
    'dataset_1to1_Time_ID',
    'dataset_1to5_Time_ID',
    'dataset_1to1_Time_OOD',
    'dataset_1to5_Time_OOD'
]

for file in files:
    print(f"⏳ 正在转换 {file}.pkl -> {file}.csv ...")
    df = pd.read_pickle(f"{file}.pkl")
    df.to_csv(f"{file}.csv", index=False)
    
print("✅ 全部转换完成！快去你的文件夹里双击打开看看吧！")


# In[ ]:




