#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🚀 开始运行【18组全维度细粒度】消融实验 (1:1 & 1:5)...")

datasets = {
    "1:1": {"ID": "dataset_1to1_Time_ID.pkl", "OOD": "dataset_1to1_Time_OOD.pkl"},
    "1:5": {"ID": "dataset_1to5_Time_ID.pkl", "OOD": "dataset_1to5_Time_OOD.pkl"}
}

def get_feature_sets(df):
    all_cols = set(df.columns)
    meta = {'post_id', 'created_at', 'label', 'sender_id', 'recipient_id'}
    
    u_cols = {c for c in all_cols if c.startswith('U-')}
    
    # 7种单独的 SN 定义
    sn_in = {c for c in all_cols if c.startswith('SN_In-')}
    sn_out = {c for c in all_cols if c.startswith('SN_Out-')}
    sn_mut = {c for c in all_cols if c.startswith('SN_Mut-')}
    sn_tc = {c for c in all_cols if c.startswith('SN_TopComment-')}
    sn_tq = {c for c in all_cols if c.startswith('SN_TopQuote-')}
    sn_tr = {c for c in all_cols if c.startswith('SN_TopRepost-')}
    sn_ta = {c for c in all_cols if c.startswith('SN_TopAny-')}
    
    sn_all = sn_in | sn_out | sn_mut | sn_tc | sn_tq | sn_tr | sn_ta
    m_cols = all_cols - meta - u_cols - sn_all
    
    # 🎯 精心编排的 18 组实验矩阵
    return {
        # 基础基线组 (2组)
        "01. M Only": list(m_cols),
        "02. M + U": list(m_cols | u_cols),
        
        # 纯 SN 单变量对照组 (7组：仅 M + 单独一种 SN，不加 U)
        "03. M + SN(In)": list(m_cols | sn_in),
        "04. M + SN(Out)": list(m_cols | sn_out),
        "05. M + SN(Mut)": list(m_cols | sn_mut),
        "06. M + SN(TopComment)": list(m_cols | sn_tc),
        "07. M + SN(TopQuote)": list(m_cols | sn_tq),
        "08. M + SN(TopRepost)": list(m_cols | sn_tr),
        "09. M + SN(TopAny)": list(m_cols | sn_ta),
        
        # 🌟 核心拆解组 (7组：M + U + 单独一种 SN，精准对比哪种定义最强)
        "10. M + U + SN(In)": list(m_cols | u_cols | sn_in),
        "11. M + U + SN(Out)": list(m_cols | u_cols | sn_out),
        "12. M + U + SN(Mut)": list(m_cols | u_cols | sn_mut),
        "13. M + U + SN(TopComment)": list(m_cols | u_cols | sn_tc),
        "14. M + U + SN(TopQuote)": list(m_cols | u_cols | sn_tq),
        "15. M + U + SN(TopRepost)": list(m_cols | u_cols | sn_tr),
        "16. M + U + SN(TopAny)": list(m_cols | u_cols | sn_ta),
        
        # 综合大一统组 (2组)
        "17. M + SN(All)": list(m_cols | sn_all),
        "18. M + U + SN(Full)": list(m_cols | u_cols | sn_all)
    }

results_list = []

for ratio, paths in datasets.items():
    print(f"\n🚀 正在处理 {ratio} 数据集的 18 组全矩阵消融...")
    try:
        df_id = pd.read_pickle(paths["ID"])
        df_ood = pd.read_pickle(paths["OOD"])
    except FileNotFoundError:
        print(f"⚠️ 找不到 {ratio} 的数据集文件，跳过...")
        continue
    
    feature_configs = get_feature_sets(df_id)
    train_idx, test_idx = train_test_split(df_id.index, test_size=0.2, random_state=42, stratify=df_id['label'])
    train_df, test_df = df_id.loc[train_idx], df_id.loc[test_idx]
    
    y_train = train_df['label'].astype(int)
    y_test_id = test_df['label'].astype(int)
    y_test_ood = df_ood['label'].astype(int)
    
    scale = len(y_train[y_train==0]) / len(y_train[y_train==1]) if "1:5" in ratio else 1
    
    for config_name, features in feature_configs.items():
        features = [f for f in features if f in df_id.columns]
        if not features: continue
            
        X_train = train_df[features]
        X_test_id = test_df[features]
        X_test_ood = df_ood[features]
        
        clf = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, 
                            scale_pos_weight=scale, random_state=42, n_jobs=-1, eval_metric='logloss')
        clf.fit(X_train, y_train)
        
        prob_train = clf.predict_proba(X_train)[:, 1]
        prob_id = clf.predict_proba(X_test_id)[:, 1]
        prob_ood = clf.predict_proba(X_test_ood)[:, 1]
        
        # 动态阈值优化 F1
        best_thresh, best_f1 = 0.5, 0
        for th in np.arange(0.05, 0.95, 0.02):
            f1_tmp = f1_score(y_train, (prob_train >= th).astype(int), zero_division=0)
            if f1_tmp > best_f1:
                best_f1 = f1_tmp
                best_thresh = th
                
        pred_id = (prob_id >= best_thresh).astype(int)
        pred_ood = (prob_ood >= best_thresh).astype(int)
        
        for eval_type, y_true, y_pred, y_prob in [("ID_Test", y_test_id, pred_id, prob_id), 
                                                  ("OOD_Test", y_test_ood, pred_ood, prob_ood)]:
            results_list.append({
                "Ratio": ratio,
                "Eval_Type": eval_type,
                "Config": config_name,
                "Best_Thresh": round(best_thresh, 2),
                "AUC": roc_auc_score(y_true, y_prob),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred)
            })
        print(f"  -> {config_name} 训练完成。")

if results_list:
    df_res = pd.DataFrame(results_list)
    output_file = "XGBoost_18_Ablation_Results.xlsx"
    df_res.to_excel(output_file, index=False)

    print("\n" + "="*80)
    print("🏆 【18 组全矩阵消融实验完成】")
    print("="*80)
    print(df_res.groupby(['Ratio', 'Eval_Type', 'Config']).mean().round(4).to_string())
    print(f"\n✅ 所有详细结果已保存至 '{output_file}'！")
else:
    print("⚠️ 没有任何结果生成。")


# In[3]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🚀 开始运行【17组纯结构化特征 (无M)】消融实验 (1:1 & 1:5)...")

datasets = {
    "1:1": {"ID": "dataset_1to1_Time_ID.pkl", "OOD": "dataset_1to1_Time_OOD.pkl"},
    "1:5": {"ID": "dataset_1to5_Time_ID.pkl", "OOD": "dataset_1to5_Time_OOD.pkl"}
}

def get_feature_sets_no_m(df):
    all_cols = set(df.columns)
    
    # 提取 U 特征
    u_cols = {c for c in all_cols if c.startswith('U-')}
    
    # 提取 7 种单独的 SN 特征
    sn_in = {c for c in all_cols if c.startswith('SN_In-')}
    sn_out = {c for c in all_cols if c.startswith('SN_Out-')}
    sn_mut = {c for c in all_cols if c.startswith('SN_Mut-')}
    sn_tc = {c for c in all_cols if c.startswith('SN_TopComment-')}
    sn_tq = {c for c in all_cols if c.startswith('SN_TopQuote-')}
    sn_tr = {c for c in all_cols if c.startswith('SN_TopRepost-')}
    sn_ta = {c for c in all_cols if c.startswith('SN_TopAny-')}
    
    # 汇总所有 SN 特征
    sn_all = sn_in | sn_out | sn_mut | sn_tc | sn_tq | sn_tr | sn_ta
    
    # 🎯 重新编排的 17 组无 M 实验矩阵
    return {
        # 组别 1: 纯个体画像
        "01. U Only": list(u_cols),
        
        # 组别 2-9: 纯社交网络 (8组)
        "02. SN(In) Only": list(sn_in),
        "03. SN(Out) Only": list(sn_out),
        "04. SN(Mut) Only": list(sn_mut),
        "05. SN(TopComment) Only": list(sn_tc),
        "06. SN(TopQuote) Only": list(sn_tq),
        "07. SN(TopRepost) Only": list(sn_tr),
        "08. SN(TopAny) Only": list(sn_ta),
        "09. SN(All) Only": list(sn_all),
        
        # 组别 10-17: 个体画像 + 社交网络 (8组)
        "10. U + SN(In)": list(u_cols | sn_in),
        "11. U + SN(Out)": list(u_cols | sn_out),
        "12. U + SN(Mut)": list(u_cols | sn_mut),
        "13. U + SN(TopComment)": list(u_cols | sn_tc),
        "14. U + SN(TopQuote)": list(u_cols | sn_tq),
        "15. U + SN(TopRepost)": list(u_cols | sn_tr),
        "16. U + SN(TopAny)": list(u_cols | sn_ta),
        "17. U + SN(All)": list(u_cols | sn_all)
    }

results_list = []

for ratio, paths in datasets.items():
    print(f"\n🚀 正在处理 {ratio} 数据集的 17 组无 M 消融...")
    try:
        df_id = pd.read_pickle(paths["ID"])
        df_ood = pd.read_pickle(paths["OOD"])
    except FileNotFoundError:
        print(f"⚠️ 找不到 {ratio} 的数据集文件，跳过...")
        continue
    
    feature_configs = get_feature_sets_no_m(df_id)
    train_idx, test_idx = train_test_split(df_id.index, test_size=0.2, random_state=42, stratify=df_id['label'])
    train_df, test_df = df_id.loc[train_idx], df_id.loc[test_idx]
    
    y_train = train_df['label'].astype(int)
    y_test_id = test_df['label'].astype(int)
    y_test_ood = df_ood['label'].astype(int)
    
    scale = len(y_train[y_train==0]) / len(y_train[y_train==1]) if "1:5" in ratio else 1
    
    for config_name, features in feature_configs.items():
        features = [f for f in features if f in df_id.columns]
        if not features: continue
            
        X_train = train_df[features]
        X_test_id = test_df[features]
        X_test_ood = df_ood[features]
        
        clf = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, 
                            scale_pos_weight=scale, random_state=42, n_jobs=-1, eval_metric='logloss')
        clf.fit(X_train, y_train)
        
        prob_train = clf.predict_proba(X_train)[:, 1]
        prob_id = clf.predict_proba(X_test_id)[:, 1]
        prob_ood = clf.predict_proba(X_test_ood)[:, 1]
        
        # 动态寻找最优 F1 阈值
        best_thresh, best_f1 = 0.5, 0
        for th in np.arange(0.05, 0.95, 0.02):
            f1_tmp = f1_score(y_train, (prob_train >= th).astype(int), zero_division=0)
            if f1_tmp > best_f1:
                best_f1 = f1_tmp
                best_thresh = th
                
        pred_id = (prob_id >= best_thresh).astype(int)
        pred_ood = (prob_ood >= best_thresh).astype(int)
        
        for eval_type, y_true, y_pred, y_prob in [("ID_Test", y_test_id, pred_id, prob_id), 
                                                  ("OOD_Test", y_test_ood, pred_ood, prob_ood)]:
            results_list.append({
                "Ratio": ratio,
                "Eval_Type": eval_type,
                "Config": config_name,
                "Best_Thresh": round(best_thresh, 2),
                "AUC": roc_auc_score(y_true, y_prob),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred)
            })
        print(f"  -> {config_name} 训练完成。")

if results_list:
    df_res = pd.DataFrame(results_list)
    output_file = "XGBoost_17_Ablation_Results_NoM.xlsx"
    df_res.to_excel(output_file, index=False)

    print("\n" + "="*80)
    print("🏆 【17 组无 M 消融实验完成】")
    print("="*80)
    print(df_res.groupby(['Ratio', 'Eval_Type', 'Config'])[['AUC', 'F1', 'Precision', 'Recall']].mean().round(4).to_string())
    print(f"\n✅ 所有详细结果已保存至 '{output_file}'！")
else:
    print("⚠️ 没有任何结果生成。")


# In[ ]:




