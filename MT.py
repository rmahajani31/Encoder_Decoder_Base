#!/usr/bin/env python
# coding: utf-8

# ## Initialization Code

# In[1]:




import torch
import numpy as np
import spacy
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from nltk.tokenize import RegexpTokenizer
import time
import math
from tqdm import tqdm
import os
import pickle


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[4]:


nlp = spacy.load("en_core_web_lg")


# ## Below is the code to generate data once

# In[8]:


dataset = load_dataset("europarl_bilingual", lang1="en", lang2="fr")


# In[9]:


def process_europarl_corpus(dataset):
    en_fr_sents = []
    for i,sent in enumerate(dataset["train"]["translation"]):
        en_fr_sents.append((sent["en"], sent["fr"]))
    # if i%1000==0:
    #   print(f"Currently on iteration {i}")
    return en_fr_sents


# In[10]:


en_fr_sents = process_europarl_corpus(dataset)


# In[11]:


en_fr_df = pd.DataFrame(en_fr_sents, columns=["en", "fr"])
en_fr_df = en_fr_df.loc[:1000000]


# In[12]:


en_fr_df.head()


# In[13]:


def preprocess(text, nltk_tokenizer):
    #cur_sent = " ".join(["cls"] + nltk_tokenizer.tokenize(text.lower()) + ["sep"])
    cur_sent = " ".join(["cls"] + [x for x in text.split() if x !=""] + ["sep"])
    return cur_sent


# In[14]:


nltk_tokenizer = RegexpTokenizer(r"[\w\d'\s]+")
en_fr_df["en"] = en_fr_df["en"].apply(lambda x: preprocess(x, nltk_tokenizer))
en_fr_df["fr"] = en_fr_df["fr"].apply(lambda x: preprocess(x, nltk_tokenizer))


# In[40]:


def generate_mappings(en_fr_df, col_name, start_batch=1, batch_size=10000):
  sents = list(set(en_fr_df[col_name].tolist()))
  sents.sort()
  latest_token_index = 0
  sent_to_tokens = {}
  token_to_index = {}
  token_vector_list = []
  seen_tokens = set()
  cls_doc = nlp("cls", disable=["parser","ner"])
  seen_tokens.add(cls_doc[0].text)
  token_to_index[cls_doc[0].text] = latest_token_index
  latest_token_index += 1
  token_vector_list.append(torch.Tensor(cls_doc[0].vector))
  sep_doc = nlp("sep", disable=["parser","ner"])
  seen_tokens.add(sep_doc[0].text)
  token_to_index[sep_doc[0].text] = latest_token_index
  latest_token_index += 1
  token_vector_list.append(torch.Tensor(sep_doc[0].vector))
  start_time = time.time()
  num_sents = len(sents)
  num_batches = math.ceil(num_sents/batch_size)
  if start_batch > 1:
    with open(f"./data/{col_name}_sent_to_tokens_{start_batch-1}.pkl", "rb") as f:
      sent_to_tokens = pickle.load(f)
    with open(f"./data/{col_name}_token_to_index_{start_batch-1}.pkl", "rb") as f:
      token_to_index = pickle.load(f)
    with open(f"./data/{col_name}_token_vectors_{start_batch-1}.pkl", "rb") as f:
      token_vectors = pickle.load(f)
  print(f"Number of sentences: {num_sents}")
  print(f"Number of batches: {num_batches}")
  for batch in range(start_batch,num_batches+1):
    batch_start_index = (batch-1)*batch_size
    docs = nlp.pipe(sents[batch_start_index:min(batch*batch_size, num_sents)], disable=["parser", "ner"])
    for i,doc in enumerate(docs):
      cur_sent_tokens = []
      for token in doc:
        cur_sent_tokens.append(token.text)
        if token.text not in seen_tokens:
          seen_tokens.add(token.text)
          token_to_index[token.text] = latest_token_index
          latest_token_index += 1
          token_vector_list.append(torch.Tensor(token.vector))
        sent_to_tokens[sents[batch_start_index+i]] = cur_sent_tokens
    with open(f"./data/{col_name}_sent_to_tokens_{batch}.pkl", "wb") as f:
      pickle.dump(sent_to_tokens, f)
    with open(f"./data/{col_name}_token_to_index_{batch}.pkl", "wb") as f:
      pickle.dump(token_to_index, f)
    with open(f"./data/{col_name}_token_vectors_{batch}.pkl", "wb") as f:
      pickle.dump(token_vector_list, f)
    end_time = time.time()
    total_time = end_time-start_time
    print(f"Total time to finish batch {batch} is: {total_time/60} minutes")
    start_time = time.time()
  token_vectors = torch.stack(token_vector_list)
  index_to_token = {i:token for token,i in token_to_index.items()}
  return sent_to_tokens,token_to_index,index_to_token,token_vectors


# In[ ]:


fr_sent_to_tokens,fr_token_to_index,fr_index_to_token,fr_token_vectors = generate_mappings(en_fr_df, "fr", start_batch=6, batch_size=100000)
en_sent_to_tokens,en_token_to_index,en_index_to_token,en_token_vectors = generate_mappings(en_fr_df, "en", batch_size=100000)


# In[25]:


fr_token_vectors.size(),len(fr_token_to_index),len(fr_sent_to_tokens),en_token_vectors.size(),len(en_token_to_index),len(en_sent_to_tokens)


# In[18]:


# code to check if the batched data has been created accurately
col_names = ["en", "fr"]
final_batch = 10
for col_name in col_names:
  with open(f"./data/{col_name}_sent_to_tokens_{final_batch}.pkl", "rb") as f:
    sent_to_tokens = pickle.load(f)
  print(len(sent_to_tokens.keys()))


# In[ ]:


# code to aggregate all the batch pickle files


# In[26]:


en_fr_df.to_csv("./data/en_fr_df.csv")
with open("./data/fr_sent_to_tokens.pkl", "wb") as f:
  pickle.dump(fr_sent_to_tokens, f)
with open("./data/fr_token_to_index.pkl", "wb") as f:
  pickle.dump(fr_token_to_index, f)
with open("./data/fr_index_to_token.pkl", "wb") as f:
  pickle.dump(fr_index_to_token, f)
with open("./data/fr_token_vectors.pkl", "wb") as f:
  pickle.dump(fr_token_vectors, f)
with open("./data/en_sent_to_tokens.pkl", "wb") as f:
  pickle.dump(en_sent_to_tokens, f)
with open("./data/en_token_to_index.pkl", "wb") as f:
  pickle.dump(en_token_to_index, f)
with open("./data/en_index_to_token.pkl", "wb") as f:
  pickle.dump(en_index_to_token, f)
with open("./data/en_token_vectors.pkl", "wb") as f:
  pickle.dump(en_token_vectors, f)


