#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tarfile
import json
import re
from collections import Counter
import time


# In[9]:


# ==========================================
# Configuration and Parameters
# ==========================================
FILE_PATH = 'user_posts.tar.gz'

# Limit the number of users to process to save time during the sampling phase.
# 20,0000 is a reasonable sample size to identify globally trending hashtags.
MAX_USERS_TO_PROCESS = 200000 

# Compile a regex pattern to extract hashtags (e.g., #AI, #Pokemon).
# \w+ matches Unicode word characters (letters, digits, underscores).
hashtag_pattern = re.compile(r'#\w+')
hashtag_counter = Counter()

print(f"Starting to process the dataset. Maximum users to process: {MAX_USERS_TO_PROCESS}...")
start_time = time.time()


# In[11]:


# ==========================================
# Read and Count Hashtags
# ==========================================
processed_users = 0

with tarfile.open(FILE_PATH, 'r:gz') as tar:
    for member in tar:
        # Ensure the extracted member is a file
        if member.isfile():
            f = tar.extractfile(member)
            if f is not None:
                # Read each line (historical post) of the current user
                for line in f:
                    try:
                        # Parse the JSON line; this automatically decodes \uXXXX sequences
                        post_data = json.loads(line.decode('utf-8'))
                        text = post_data.get('text', '')
                        
                        if text:
                            # Extract all hashtags and convert to lowercase for consistency
                            tags = hashtag_pattern.findall(text)
                            tags = [tag.lower() for tag in tags]
                            hashtag_counter.update(tags)
                            
                    except Exception:
                        # Silently ignore malformed JSON lines
                        continue
            
            processed_users += 1
            
            # Print progress every 2,000 users
            if processed_users % 2000 == 0:
                print(f"Processed {processed_users} users...")
                
            # Terminate early to save time once the threshold is reached
            if processed_users >= MAX_USERS_TO_PROCESS:
                break

end_time = time.time()
print(f"\nProcessing complete! Total time elapsed: {end_time - start_time:.2f} seconds.")


# In[21]:


# ==========================================
# Display and Save Results
# ==========================================
print("\n=== Top 150 Most Frequent Hashtags ===")
top_150_tags = hashtag_counter.most_common(150)

for rank, (tag, count) in enumerate(top_150_tags, 1):
    print(f"Top {rank}: {tag} (Frequency: {count})")

# Isolate the top 10 for future target post extraction
top_10_list = [tag for tag, count in hashtag_counter.most_common(10)]
print(f"\nTop 10 Hashtags for downstream tracking: \n{top_10_list}")


# In[ ]:




