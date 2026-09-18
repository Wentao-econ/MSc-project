#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_FILE_LOCKS"] = "1" 

import pandas as pd
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# In[ ]:


df_h = pd.read_pickle('df_step4_backup.pkl')
texts = df_h['text'].fillna('').astype(str).tolist()
total_rows = len(texts)
del df_h
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128


# In[ ]:


def extract_features_raw(model_name, texts, batch_size, task_name):
    
    # 🧨 local_files_only=True：
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    model.to(device)
    model.eval() 
    id2label = model.config.id2label
    
    all_results = []
    
    pbar = tqdm(range(0, len(texts), batch_size), desc=f"handling {task_name}", file=sys.stdout, ncols=90)
    
    for i in pbar:
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            
        for prob_array in probs:
            res_dict = {id2label[j]: float(prob) for j, prob in enumerate(prob_array)}
            all_results.append(res_dict)
            
    del model, tokenizer, inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    return all_results


irony_raw = extract_features_raw("cardiffnlp/twitter-roberta-base-irony", texts, BATCH_SIZE, "Irony")
irony_scores = [d['irony'] if 'irony' in d else (1.0 - d.get('non_irony', 0.0)) for d in irony_raw]
pd.to_pickle(irony_scores, 'scores_irony.pkl')


# In[ ]:


import os
import sys
import pandas as pd
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_FILE_LOCKS"] = "1" 


df_h = pd.read_pickle('df_step4_backup.pkl')
texts = df_h['text'].fillna('').astype(str).tolist()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128


def extract_features_raw(model_name, texts, batch_size, task_name):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        
    model.to(device)
    model.eval() 
    id2label = model.config.id2label
    
    all_results = []
    pbar = tqdm(range(0, len(texts), batch_size), desc=f"handling {task_name}", file=sys.stdout, ncols=90)
    
    for i in pbar:
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits) if "toxic" in model_name else torch.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()
            
        for prob_array in probs:
            res_dict = {id2label[j]: float(prob) for j, prob in enumerate(prob_array)}
            all_results.append(res_dict)
            
    del model, tokenizer, inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return all_results


if not os.path.exists('scores_hate.pkl'):
    hate_raw = extract_features_raw("unitary/toxic-bert", texts, BATCH_SIZE, "Hate Speech")
    hate_scores = {'base_hs_hateful': [], 'base_hs_aggressive': [], 'base_hs_targeted': [], 'base_offensive': [], 'base_hs_count': []}

    for scores in hate_raw:
        hate_scores['base_hs_hateful'].append(scores.get('toxic', 0.0))
        hate_scores['base_hs_aggressive'].append(scores.get('severe_toxic', 0.0))
        hate_scores['base_hs_targeted'].append(max(scores.get('insult', 0.0), scores.get('identity_hate', 0.0)))
        hate_scores['base_offensive'].append(scores.get('obscene', 0.0))
        hate_scores['base_hs_count'].append(sum(1 for v in scores.values() if v > 0.5))

    pd.to_pickle(hate_scores, 'scores_hate.pkl')


if not os.path.exists('scores_emo.pkl'):
    emo_raw = extract_features_raw("j-hartmann/emotion-english-distilroberta-base", texts, BATCH_SIZE, "Emotion")
    emo_scores = {
        'base_emo_disgust': [], 'base_emo_anger': [], 'base_emo_joy': [], 'base_emo_fear': [], 
        'base_emo_surprise': [], 'base_emo_sadness': [], 'base_emo_others': [], 'base_emo_overall': []
    }
    emotion_mapping = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6}

    for scores in emo_raw:
        emo_scores['base_emo_disgust'].append(scores.get('disgust', 0.0))
        emo_scores['base_emo_anger'].append(scores.get('anger', 0.0))
        emo_scores['base_emo_joy'].append(scores.get('joy', 0.0))
        emo_scores['base_emo_fear'].append(scores.get('fear', 0.0))
        emo_scores['base_emo_surprise'].append(scores.get('surprise', 0.0))
        emo_scores['base_emo_sadness'].append(scores.get('sadness', 0.0))
        emo_scores['base_emo_others'].append(scores.get('neutral', 0.0))
        
        top_emotion = max(scores, key=scores.get)
        emo_scores['base_emo_overall'].append(emotion_mapping.get(top_emotion, 6))

    pd.to_pickle(emo_scores, 'scores_emo.pkl')



# In[ ]:


import pandas as pd
import os


df_h = pd.read_pickle('df_step4_backup.pkl')

irony_probs = pd.read_pickle('scores_irony.pkl')
df_hate = pd.DataFrame(pd.read_pickle('scores_hate.pkl'))
df_emo = pd.DataFrame(pd.read_pickle('scores_emo.pkl'))


df_h['base_irony'] = [1 if p >= 0.5 else 0 for p in irony_probs]

df_h = pd.concat([df_h, df_hate, df_emo], axis=1)

df_features_h = df_h.drop(columns=['text'])

df_features_h.to_pickle('historical_posts_dl_features_final.pkl')

df_features_h.to_csv('historical_posts_dl_features_final.csv', index=False)



# In[ ]:




