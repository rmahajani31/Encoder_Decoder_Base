#!/usr/bin/env python
# coding: utf-8

# ## Initialization Code


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


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# ## Below is the main code which needs to be executed to train the encoder decoder architecture

# In[5]:


en_fr_df = pd.read_csv("./data/en_fr_df.csv")


# In[6]:


class MTDataset(Dataset):
  def __init__(self,df,max_len,batch):
    self.df = df
    self.max_len = max_len
    with open(f"./data/fr_sent_to_tokens_{batch}.pkl", "rb") as f:
      self.fr_sent_to_tokens = pickle.load(f)
    with open(f"./data/fr_token_to_index.pkl_{batch}", "rb") as f:
      self.fr_token_to_index = pickle.load(f)
    with open(f"./data/fr_index_to_token.pkl_{batch}", "rb") as f:
      self.fr_index_to_token = pickle.load(f)
    with open(f"./data/fr_token_vectors.pkl_{batch}", "rb") as f:
      self.fr_token_vectors = pickle.load(f)
    with open(f"./data/en_sent_to_tokens.pkl_{batch}", "rb") as f:
      self.en_sent_to_tokens = pickle.load(f)
    with open(f"./data/en_token_to_index.pkl_{batch}", "rb") as f:
      self.en_token_to_index = pickle.load(f)
    with open(f"./data/en_index_to_token.pkl_{batch}", "rb") as f:
      self.en_index_to_token = pickle.load(f)
    with open(f"./data/en_token_vectors.pkl_{batch}", "rb") as f:
      self.en_token_vectors = pickle.load(f)
  def __len__(self):
    return len(self.df)
  def __getitem__(self, idx):
    cur_en_sent = self.df.loc[idx,"en"]
    cur_fr_sent = self.df.loc[idx,"fr"]
    cur_en_tokens = self.en_sent_to_tokens[cur_en_sent]
    cur_fr_tokens = self.fr_sent_to_tokens[cur_fr_sent]
    if (len(cur_en_tokens))>self.max_len or (len(cur_fr_tokens))>self.max_len:
      raise Exception("The input or target sentence is more than max len tokens")
    inputs = torch.stack([torch.Tensor(self.en_token_vectors[self.en_token_to_index[token]]) for token in cur_en_tokens])
    targets = torch.stack([torch.Tensor(self.fr_token_vectors[self.fr_token_to_index[token]]) for token in cur_fr_tokens[:-1]])
    labels = torch.LongTensor([self.fr_token_to_index[token] for token in cur_fr_tokens[1:]])
    return {
        "inputs":inputs,
        "targets": targets,
        "input_seq_len": torch.LongTensor([len(inputs)]),
        "target_seq_len": torch.LongTensor([len(targets)]),
        "labels": labels
    }


# In[7]:


def collate_fn(batch):
  ignore_index = -1
  inputs = nn.utils.rnn.pad_sequence([batch[i]["inputs"] for i in range(len(batch))], batch_first=True)
  input_seq_lens = torch.stack([batch[i]["input_seq_len"] for i in range(len(batch))]).squeeze()
  targets = nn.utils.rnn.pad_sequence([batch[i]["targets"] for i in range(len(batch))], batch_first=True)
  target_seq_lens = torch.stack([batch[i]["target_seq_len"] for i in range(len(batch))]).squeeze()
  labels = nn.utils.rnn.pad_sequence([batch[i]["labels"] for i in range(len(batch))], batch_first=True, padding_value=ignore_index)
  return inputs,targets,input_seq_lens,target_seq_lens,labels


# In[8]:


def auto_reg_collate_fn(batch):
  max_len=512
  ignore_index = -1
  inputs = nn.utils.rnn.pad_sequence([batch[i]["inputs"] for i in range(len(batch))], batch_first=True)
  input_seq_lens = torch.stack([batch[i]["input_seq_len"] for i in range(len(batch))]).squeeze()
  targets = torch.stack([batch[i]["targets"][0] for i in range(len(batch))])
  labels = nn.utils.rnn.pad_sequence([batch[i]["labels"] for i in range(len(batch))], batch_first=True, padding_value=ignore_index)
  ignore_index_vals = torch.full((1,max_len-1-labels.size()[-1]), ignore_index).expand(len(batch), -1)
  labels = torch.cat([labels, ignore_index_vals], dim=-1)
  return inputs,targets,input_seq_lens,labels


# In[9]:

print("Creating validation and test dataloaders...")
max_len = 512
dataset_batch_size = 200000
batch_size = 16
num_batches = 10
num_train_batches = num_batches-2
valid_batch = num_train_batches+1
valid_dataset = MTDataset(en_fr_df.loc[(valid_batch-1)*dataset_batch_size:min(valid_batch*dataset_batch_size,len(en_fr_df))].reset_index(drop=True),max_len,valid_batch)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=auto_reg_collate_fn)
# test_batch = num_train_batches+2
# test_dataset = MTDataset(en_fr_df.loc[(test_batch-1)*dataset_batch_size:min(test_batch*dataset_batch_size,len(en_fr_df))].reset_index(drop=True),max_len,test_batch)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=auto_reg_collate_fn)


# In[ ]:

print("Computing vocab size...")
vocab = set()
for batch in range(1,num_batches+1):
  with open(f"./data/fr_token_to_index.pkl_{batch}", "rb") as f:
    cur_fr_token_to_index = pickle.load(f)
  vocab = vocab.union(set(cur_fr_token_to_index.keys()))
vocab_size = len(vocab)
print(f"Vocab size is: {vocab_size}")



class Encoder(nn.Module):
  def __init__(self, emb_dim, enc_hidden_dim, train_emb=False):
    super().__init__()
    self.GRU = nn.GRU(emb_dim, enc_hidden_dim)
    self.layernorm_layer = nn.LayerNorm(enc_hidden_dim)
  def forward(self, inputs, input_seq_lens):
    inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_seq_lens.cpu(), batch_first=True, enforce_sorted=False)
    outputs, hidden = self.GRU(inputs)
    hidden = self.layernorm_layer(hidden)
    return outputs, hidden


# In[13]:


class Decoder(nn.Module):
  def __init__(self, emb_dim, enc_hidden_dim, dec_hidden_dim, vocab_size, train_emb=False):
    super().__init__()
    self.GRU = nn.GRU(emb_dim+enc_hidden_dim, dec_hidden_dim)
    self.layernorm_layer = nn.LayerNorm(dec_hidden_dim)
    self.dense_layer = nn.Linear(dec_hidden_dim, vocab_size)
    self.softmax_layer = nn.Softmax(dim=-1)
  def forward(self, context_vector, init_hidden_state, targets, target_seq_lens):
    targets_with_context = torch.cat([targets, context_vector.unsqueeze(dim=1).expand(-1, targets.size()[1], -1)], dim=-1)
    targets_with_context = nn.utils.rnn.pack_padded_sequence(targets_with_context, target_seq_lens.cpu(), batch_first=True, enforce_sorted=False)
    outputs, hidden = self.GRU(targets_with_context, init_hidden_state)
    outputs, seq_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
    outputs = self.layernorm_layer(outputs)
    x = self.dense_layer(outputs)
    x = self.softmax_layer(x)
    return hidden,x



emb_dim = 300
enc_hidden_dim = 128
dec_hidden_dim = 128
enc = Encoder(emb_dim, enc_hidden_dim)
dec = Decoder(emb_dim, enc_hidden_dim, dec_hidden_dim, vocab_size)
enc.to(device)
dec.to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
encoder_optimizer = optim.Adam(enc.parameters(), lr=1e-3)
decoder_optimizer = optim.Adam(dec.parameters(), lr=1e-3)
print(enc.parameters)
print(dec.parameters)


# In[16]:

print("Starting training loop...")
if os.path.exists("./log.txt"):
    os.remove("./log.txt")
teacher_forcing = True
epochs = 10
print(f"number of batches: {dataset_batch_size/batch_size}")
train_losses = []
valid_losses = []
min_valid_epoch_loss = float("inf")
for epoch_num in range(epochs):
  print(f"EPOCH {epoch_num}")
  running_loss = 0
  epoch_start_time = time.time()
  enc.train()
  dec.train()
  total_train_batches = 0
  for train_batch_num in range(1,num_train_batches+1):
    cur_train_ds = MTDataset(en_fr_df.loc[(train_batch_num-1)*dataset_batch_size:min(train_batch_num*dataset_batch_size,len(en_fr_df))].reset_index(drop=True),max_len,train_batch_num)
    train_dataloader = DataLoader(cur_train_ds, batch_size=batch_size, collate_fn=collate_fn)
    batch_start_time = time.time()
    cur_batch_start_time = time.time()
    for batch_num,batch in enumerate(tqdm(train_dataloader)):
      inputs,targets,input_seq_lens,target_seq_lens,labels = batch
      inputs = inputs.to(device)
      targets = targets.to(device)
      input_seq_lens = input_seq_lens.to(device)
      target_seq_lens = target_seq_lens.to(device)
      labels = labels.to(device)
      _, context = enc(inputs, input_seq_lens)
      hidden_state,decoder_output = dec(context.squeeze(dim=0), context, targets, target_seq_lens)
      decoder_output = decoder_output.permute(0,-1,1)
      loss = loss_fn(decoder_output, labels)
      loss.backward()
      encoder_optimizer.step()
      decoder_optimizer.step()
      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()
      running_loss += loss
      total_train_batches += 1
      if batch_num%500==0:
          batch_end_time = time.time()
          batch_total_time = batch_end_time-batch_start_time
          # print(f"BATCH_END,epoch:{epoch_num},batch:{batch_num},loss:{loss},time:{batch_total_time/60}")
          # print(decoder_output.size(),labels.size())
          with open("./log.txt", "a+") as f:
            f.write(f"BATCH_END,epoch:{epoch_num},batch:{batch_num},loss:{loss},time:{batch_total_time/60}\n")
          batch_start_time = time.time()
      #print(f"Finished batch {batch_num} in time {(time.time()-cur_batch_start_time)/60} minutes")
      #cur_batch_start_time = time.time()
  train_epoch_loss = running_loss/total_train_batches
  train_losses.append(train_epoch_loss)
  epoch_end_time = time.time()
  epoch_total_time = epoch_end_time-epoch_start_time
  print(f"EPOCH_END,epoch:{epoch_num},loss:{train_epoch_loss},time:{epoch_total_time/60}")
  with open("./log.txt", "a+") as f:
    f.write(f"EPOCH_END,epoch:{epoch_num},loss:{train_epoch_loss},time:{epoch_total_time/60}\n")
  with torch.no_grad():
    enc.eval()
    dec.eval()
    print("Running Validation...")
    running_loss = 0
    epoch_start_time = time.time()
    for batch_num,batch in enumerate(tqdm(valid_dataloader)):
      inputs,targets,input_seq_lens,labels = batch
      inputs = inputs.to(device)
      targets = targets.to(device)
      input_seq_lens = input_seq_lens.to(device)
      labels = labels.to(device)
      _, context = enc(inputs, input_seq_lens)
      decoder_final_outputs = []
      for decoder_start_index in range(len(targets)):
        num_decoder_output_tokens = 1
        cur_predicted_token = None
        cur_context_vector = context[:,decoder_start_index,:].unsqueeze(dim=1)
        decoder_start_vector = targets[decoder_start_index].unsqueeze(0).unsqueeze(0)
        cur_input_vector = decoder_start_vector
        prev_hidden_state = cur_context_vector
        decoder_sequence_outputs = []
        while cur_predicted_token!="sep" and num_decoder_output_tokens<max_len:
          hidden_state,decoder_output = dec(cur_context_vector.squeeze(dim=0), prev_hidden_state, cur_input_vector, torch.LongTensor([1]))
          cur_predicted_token_index = int(torch.argmax(decoder_output[0,0,:]))
          cur_predicted_token = fr_index_to_token[cur_predicted_token_index]
          prev_hidden_state = hidden_state
          cur_input_vector = fr_token_vectors[cur_predicted_token_index].unsqueeze(0).unsqueeze(0)
          cur_input_vector.to(device)
          num_decoder_output_tokens += 1
          decoder_sequence_outputs.append(decoder_output)
        decoder_sequence_output = torch.cat(decoder_sequence_outputs, dim=1)
        pad_matrix = torch.full((1,max_len-num_decoder_output_tokens,decoder_sequence_output.size()[-1]), 0)
        decoder_sequence_output = torch.cat([decoder_sequence_output,pad_matrix], dim=1)
        decoder_final_outputs.append(decoder_sequence_output)
      decoder_final_output = torch.cat(decoder_final_outputs, dim=0)
      decoder_final_output = decoder_final_output.permute(0,-1,1)
      running_loss += loss_fn(decoder_final_output, labels)
    valid_epoch_loss = running_loss/len(valid_dataloader)
    valid_losses.append(valid_epoch_loss)
    if valid_epoch_loss < min_valid_epoch_loss:
      min_valid_epoch_loss = valid_epoch_loss
      print(f"Saving model after training epoch {epoch_num} with loss {min_valid_epoch_loss}...")
      with open("./log.txt", "a+") as f:
        f.write(f"Saving model after training epoch {epoch_num} with loss {min_valid_epoch_loss}...\n")
      torch.save(enc.state_dict(), f"./best_enc_{epoch_num}_{min_valid_epoch_loss}.bin")
      torch.save(dec.state_dict(), f"./best_dec_{epoch_num}_{min_valid_epoch_loss}.bin")
    epoch_end_time = time.time()
    epoch_total_time = epoch_end_time-epoch_start_time
    print(f"VALID_EPOCH_END,loss:{valid_epoch_loss},time:{epoch_total_time/60}")
    with open("./log.txt", "a+") as f:
      f.write(f"VALID_EPOCH_END,loss:{valid_epoch_loss},time:{epoch_total_time/60}\n")
with open("./data/train_losses.pkl", "wb") as f:
  pickle.dump(train_losses, f)
with open("./data/valid_losses.pkl", "wb") as f:
  pickle.dump(valid_losses, f)



