#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import joblib

# ==========================================
# 0. OFFLINE ENVIRONMENT SETUP 
# ==========================================
# Force HuggingFace to completely disconnect from the internet and only read local cache
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1' 

import pandas as pd
import numpy as np
import time
import torch
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import textstat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD DATA & HARDWARE CHECK
# ==========================================
print("Loading target posts...")
df_m = pd.read_csv('target_posts_M.csv')
df_m['post_id'] = df_m['post_id'].astype(str).str.replace(r'\.0$', '', regex=True)
df_m['text'] = df_m['text'].fillna('').astype(str)

device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print("🚀 GPU detected! RTX 4090 is ready to fire.")
else:
    print("🐌 No GPU detected. Running on CPU.")

# ==========================================
# 2. FAST HEURISTIC FEATURES (Readability, etc.)
# ==========================================
print("\nExtracting fast heuristic and readability features...")
start_time = time.time()
analyzer = SentimentIntensityAnalyzer()

def extract_fast_features(text):
    if not text.strip():
        return [0] * 19 # 19 fast features in total
    
    text_len = len(text)
    word_count = len(text.split())
    emoji_count = emoji.emoji_count(text)
    
    # Readability scores
    try:
        flesch = textstat.flesch_reading_ease(text)
        fog = textstat.gunning_fog(text)
        dale_chall = textstat.dale_chall_readability_score(text)
        coleman = textstat.coleman_liau_index(text)
        kincaid = textstat.flesch_kincaid_grade(text)
        lix = textstat.lix(text)
        rix = textstat.rix(text)
        ari = textstat.automated_readability_index(text)
        smog = textstat.smog_index(text)
    except Exception:
        flesch, fog, dale_chall = 0.0, 0.0, 0.0
        coleman, kincaid, lix, rix, ari, smog = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    vs = analyzer.polarity_scores(text)
    if vs['compound'] >= 0.05: overall_sent = 1
    elif vs['compound'] <= -0.05: overall_sent = -1
    else: overall_sent = 0
        
    tb = TextBlob(text)
    
    return [text_len, word_count, emoji_count, 
            flesch, fog, dale_chall, coleman, kincaid, lix, rix, ari, smog,
            vs['neg'], vs['neu'], vs['pos'], vs['compound'], overall_sent, 
            tb.sentiment.subjectivity, tb.sentiment.polarity]

fast_cols = ['M-text_len', 'M-word_count', 'M-emoji', 
             'M-FleschReadingEase', 'M-GunningFogIndex', 'M-DaleChallIndex', 
             'M-Coleman-Liau', 'M-Kincaid', 'M-LIX', 'M-RIX', 'M-ARI', 'M-SMOGIndex',
             'M-neg', 'M-neu', 'M-pos', 'M-compound', 'M-sentiment_overall', 
             'M-subjectivity', 'M-polarity']

fast_features = df_m['text'].apply(lambda x: pd.Series(extract_fast_features(x)))
fast_features.columns = fast_cols
df_m = pd.concat([df_m, fast_features], axis=1)
print(f"Fast features extracted in {time.time() - start_time:.2f} seconds.")

# ==========================================
# 3. TOPIC MODELING (LDA)
# ==========================================
print("\nTraining LDA topic model...")
start_time = time.time()
vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2, max_features=5000)
tf = vectorizer.fit_transform(df_m['text'])

lda = LatentDirichletAllocation(n_components=20, random_state=42, n_jobs=-1)
topic_distributions = lda.fit_transform(tf)

# Assign topic distributions to dataframe
topic_cols = [f'M-topic_{i}' for i in range(20)]
df_topics = pd.DataFrame(topic_distributions, columns=topic_cols)
df_topics['M-topic_overall'] = df_topics[topic_cols].idxmax(axis=1).apply(lambda x: int(x.split('_')[1]))
df_topics['M-topic_count'] = (df_topics[topic_cols] >= 0.5).sum(axis=1)

df_m = pd.concat([df_m.reset_index(drop=True), df_topics.reset_index(drop=True)], axis=1)

# Extract top words for each topic so you can label them in your paper!
feature_names = vectorizer.get_feature_names_out()
topic_words = {}
for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[:-16:-1]
    topic_words[f'Topic_{topic_idx}'] = [feature_names[i] for i in top_indices]

pd.DataFrame(topic_words).to_csv('lda_topics_top_words.csv', index=False)
print("Saved top words for each topic to 'lda_topics_top_words.csv'. (Use this for your paper!)")

# Save models for future SN-HM extraction
joblib.dump(vectorizer, 'lda_vectorizer.pkl')
joblib.dump(lda, 'lda_model.pkl')
print("Saved fitted LDA models for consistent feature extraction on SN-HM posts.")
print(f"LDA modeling completed in {time.time() - start_time:.2f} seconds.")

# ==========================================
# 4. DEEP LEARNING FEATURES (GPU Accelerated)
# ==========================================
print("\nLoading Deep Learning models (Instantly from local cache in OFFLINE mode)...")

irony_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony", device=device, truncation=True, max_length=512, local_files_only=True)
emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=device, truncation=True, max_length=512, local_files_only=True)
hate_pipe = pipeline("text-classification", model="unitary/toxic-bert", top_k=None, device=device, truncation=True, max_length=512, local_files_only=True)

print("Running text through DL pipelines...")
start_time = time.time()
texts = df_m['text'].tolist()
BATCH_SIZE = 256

# 4a. Irony
irony_results = irony_pipe(texts, batch_size=BATCH_SIZE)
df_m['M-irony'] = [1 if res['label'] == 'irony' else 0 for res in irony_results]

# 4b. Hate / Toxic / Offensive
hate_results = hate_pipe(texts, batch_size=BATCH_SIZE)
hs_hateful, hs_aggressive, hs_targeted, offensive, hs_count = [], [], [], [], []
for res_list in hate_results:
    scores = {d['label']: d['score'] for d in res_list}
    hs_hateful.append(scores.get('toxic', 0.0))
    hs_aggressive.append(scores.get('severe_toxic', 0.0))
    hs_targeted.append(max(scores.get('insult', 0.0), scores.get('identity_hate', 0.0)))
    offensive.append(scores.get('obscene', 0.0))
    # Count how many toxic labels have a score > 0.5
    count_over_thresh = sum(1 for v in scores.values() if v > 0.5)
    hs_count.append(count_over_thresh)

df_m['M-hs_hateful'] = hs_hateful
df_m['M-hs_aggressive'] = hs_aggressive
df_m['M-hs_targeted'] = hs_targeted
df_m['M-offensive'] = offensive
df_m['M-hs_count'] = hs_count

# 4c. Emotions
emotion_results = emotion_pipe(texts, batch_size=BATCH_SIZE)
emo_disgust, emo_anger, emo_joy, emo_fear, emo_surprise, emo_sadness, emo_others = [], [], [], [], [], [], []
emo_overall = []

for res_list in emotion_results:
    scores = {d['label']: d['score'] for d in res_list}
    emo_disgust.append(scores.get('disgust', 0.0))
    emo_anger.append(scores.get('anger', 0.0))
    emo_joy.append(scores.get('joy', 0.0))
    emo_fear.append(scores.get('fear', 0.0))
    emo_surprise.append(scores.get('surprise', 0.0))
    emo_sadness.append(scores.get('sadness', 0.0))
    emo_others.append(scores.get('neutral', 0.0)) # Mapping neutral to 'others'
    
    # Get the emotion with the highest probability
    top_emotion = max(scores, key=scores.get)
    emo_overall.append(top_emotion)

df_m['M-emo_disgust'] = emo_disgust
df_m['M-emo_anger'] = emo_anger
df_m['M-emo_joy'] = emo_joy
df_m['M-emo_fear'] = emo_fear
df_m['M-emo_surprise'] = emo_surprise
df_m['M-emo_sadness'] = emo_sadness
df_m['M-emo_others'] = emo_others
df_m['M-emo_overall'] = emo_overall

print(f"Deep learning extraction completed in {time.time() - start_time:.2f} seconds.")

# ==========================================
# 5. SAVE FINAL DATA
# ==========================================
feature_cols = ['post_id'] + [c for c in df_m.columns if c.startswith('M-')]
df_features_m = df_m[feature_cols]

df_features_m.to_csv('features_M_full_v2.csv', index=False)
print(f"\n✅ All done! Saved updated features to 'features_M_full_v2.csv'.")


# In[2]:


import pandas as pd

df = pd.read_csv('features_M_full_v2.csv')

emotion_mapping = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'sadness': 4,
    'surprise': 5,
    'neutral': 6
}


df['M-emo_overall'] = df['M-emo_overall'].map(emotion_mapping)


df.to_csv('features_M_full_v2.csv', index=False)


# In[ ]:




