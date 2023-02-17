#!/usr/bin/env python
# coding: utf-8

# In[21]:


from transformers import AutoTokenizer

model_ckpt = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


# In[22]:


import torch
from transformers import AutoModelForSequenceClassification
num_labels=3

model = (AutoModelForSequenceClassification
    .from_pretrained(model_ckpt, num_labels=num_labels))


# In[23]:


import pandas as pd
import sklearn.preprocessing as sp
df_kokuritu=pd.read_table('univ/kokuritu',header=None)
df_siritu=pd.read_table('univ/siritu',header=None)
df_kouritu=pd.read_table('univ/kouritu',header=None)
df_kokuritu['type']=0
df_siritu['type']=1
df_kouritu['type']=2
df_concat=pd.concat([df_kokuritu,df_siritu,df_kouritu])
df_concat.columns=['name','type']



# In[24]:


X=df_concat.name.to_numpy()
y=df_concat.type.to_numpy().reshape(-1,1) 


# In[25]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[26]:


X_token=tokenizer(X.tolist(),padding=True,truncation=True)['input_ids']


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X_token, y, random_state=0,stratify=y)


# In[28]:


X_train=torch.tensor(X_train)
X_test=torch.tensor(X_test)
y_train=torch.tensor(y_train)
y_test=torch.tensor(y_test)


# In[29]:


from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X,y):
        self.X=X
        self.y=y
        self.len = len(X)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]


# In[30]:


from torch.utils.data import ConcatDataset


# In[31]:


y_train=y_train.squeeze(1)
y_test=y_test.squeeze(1)


# In[32]:


train=MyDataset(X_train,y_train)
test=MyDataset(X_test,y_test)


# In[33]:


train=ConcatDataset([train,test])


# In[34]:


batch_size=4


# In[35]:


from torch.utils.data import DataLoader

# 学習用Dataloader
train_dataloader = DataLoader(
    train, 
    batch_size=batch_size, 
    shuffle=True,
    drop_last=True,
    pin_memory=True)
test_dataloader = DataLoader(
    test, 
    batch_size=batch_size, 
    shuffle=True,
    drop_last=True,
    pin_memory=True
)


# In[36]:


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


model =model.to(device)
print(model)


# In[37]:


import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# In[38]:


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X).logits
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# In[39]:


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).logits
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# In[40]:


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# In[46]:


torch.save(model,'10.pt')


# In[41]:


