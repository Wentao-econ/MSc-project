#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tarfile
import json
import re
import pandas as pd
import time


# In[43]:


# ==========================================
# Configuration and Parameters
# ==========================================
FILE_PATH = 'user_posts.tar.gz'
QUOTA_PER_TAG = 10000

# To prevent infinite running if some tags never reach the quota, 
# we set a maximum number of users to scan (e.g., 200,000).
# You can increase this if your computer is fast and you want to search deeper.
MAX_USERS_TO_SCAN = 200000  

# Define our curated list of 10 target hashtags
TARGET_HASHTAGS = {
    '#writing', '#books', '#pokemon', '#ukraine', '#gaza', 
    '#ai', '#birds', '#music', '#anime', '#poetry'
}

# Regex to extract hashtags
hashtag_pattern = re.compile(r'#\w+')

# Dictionary to keep track of how many posts we have collected for each tag
hashtag_counts = {tag: 0 for tag in TARGET_HASHTAGS}

# List to store the extracted valid posts (M)
extracted_posts = []

print(f"Starting extraction. Target quota: {QUOTA_PER_TAG} posts per hashtag...")
start_time = time.time()


# In[45]:


# ==========================================
# Single-Pass Scanning for Original Posts
# ==========================================
processed_users = 0
all_quotas_met = False

with tarfile.open(FILE_PATH, 'r:gz') as tar:
    for member in tar:
        if member.isfile():
            f = tar.extractfile(member)
            if f is not None:
                for line in f:
                    try:
                        post_data = json.loads(line.decode('utf-8'))
                        
                        # CRITICAL: We only want original posts to serve as our candidate M.
                        # If 'repost_from' is not null, this is a repost, so we skip it.
                        if post_data.get('repost_from') is not None:
                            continue
                            
                        text = post_data.get('text', '')
                        
                        if text:
                            # Extract all hashtags from the text, convert to lowercase
                            tags_in_text = set(tag.lower() for tag in hashtag_pattern.findall(text))
                            
                            # Find intersection with our target hashtags
                            matched_targets = tags_in_text.intersection(TARGET_HASHTAGS)
                            
                            for tag in matched_targets:
                                # If we haven't reached the quota for this specific hashtag
                                if hashtag_counts[tag] < QUOTA_PER_TAG:
                                    
                                    # Handle potential missing post_id by creating a composite ID fallback
                                    post_id = post_data.get('post_id')
                                    author_id = post_data.get('user_id')
                                    timestamp = post_data.get('date', '')
                                    
                                    if not post_id:
                                        post_id = f"{author_id}_{timestamp}"
                                        
                                    extracted_posts.append({
                                        'post_id': post_id,
                                        'author_id': author_id,
                                        'hashtag': tag,
                                        'timestamp': timestamp,
                                        'text': text # Save text for NLP processing later
                                    })
                                    
                                    hashtag_counts[tag] += 1
                                    
                    except Exception:
                        continue
            
            processed_users += 1
            
            # Print progress
            if processed_users % 10000 == 0:
                print(f"Scanned {processed_users} users...")
            
            # Check if all quotas are fulfilled
            if all(count >= QUOTA_PER_TAG for count in hashtag_counts.values()):
                print("\nSUCCESS: All hashtag quotas have been met! Stopping scan early.")
                all_quotas_met = True
                break
                
    # Break out of outer loop if quotas are met or max users reached
    if all_quotas_met or processed_users >= MAX_USERS_TO_SCAN:
        pass

end_time = time.time()
print(f"\nExtraction stopped. Total time elapsed: {end_time - start_time:.2f} seconds.")


# In[47]:


# ==========================================
# Reporting & Shortfall Alert
# ==========================================
print("\n=== Collection Status Report ===")
shortfall_detected = False

for tag, count in hashtag_counts.items():
    if count < QUOTA_PER_TAG:
        print(f"⚠️ WARNING: {tag} only collected {count}/{QUOTA_PER_TAG} posts.")
        shortfall_detected = True
    else:
        print(f"✅ SUCCESS: {tag} collected {count}/{QUOTA_PER_TAG} posts.")

if shortfall_detected:
    print("\nACTION REQUIRED: Consider replacing the hashtags with warnings above, or increase MAX_USERS_TO_SCAN.")
else:
    print("\nPerfect! All quotas met. Proceeding to save the dataset.")


# In[49]:


# ==========================================
# Save to CSV
# ==========================================
df_m = pd.DataFrame(extracted_posts)
df_m.to_csv('target_posts_M.csv', index=False)
print(f"\nSaved {len(df_m)} candidate posts to 'target_posts_M.csv'.")


# In[ ]:




