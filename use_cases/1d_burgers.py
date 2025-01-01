#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, default_collate
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from utils import MatReader
from pathlib import Path


# In[2]:

import sys
import os
sys.path.append(os.path.abspath(".."))

from xno.models import XNO
from xno.data.datasets import Burgers1dTimeDataset
from xno.utils import count_model_params
from xno.training import AdamW
from xno.training.incremental import IncrementalXNOTrainer
from xno.data.transforms.data_processors import IncrementalDataProcessor
from xno import LpLoss, H1Loss


# In[3]:


# Define the custom Dataset
class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}


# # Loading Burgers 1D dataset

# ## Settings

# ### Data Settings

# In[4]:


ntrain = 1000
ntest = 100
sub = 2**3 #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h


# ### Model and Trainer Settings

# In[5]:


data_path = 'data/burgers_data_R10.mat'
batch_size = 20
dataset_resolution = 1024

# XNO (model) 
max_modes = (16, )
n_modes = (16, )
in_channels = 1
out_channels = 1
n_layers = 4
hidden_channels = 64
transformation = "fno"
kwargs = {
    "wavelet_level": 6, 
    "wavelet_size": [dataset_resolution], "wavelet_filter": ['db6']
} if transformation.lower() == "wno" else {}
conv_non_linearity = F.gelu
mlp_non_linearity = F.gelu

# AdamW (optimizer) 
learning_rate = 1e-3
weight_decay = 1e-4
# CosineAnnealingLR (scheduler) 
step_size = 50
gamma = 0.5

# IncrementalDataProcessor (data_transform) 
dataset_resolution = dataset_resolution
dataset_indices = [2]

# IncrementalXNOTrainer (trainer) 
n_epochs = 5 # 500
save_every = 5
save_testing = True
save_dir = f"save/1d_lorenz/{transformation.lower()}/"


# In[6]:


dataloader = MatReader(data_path)
x_data = dataloader.read_field('a')[:,::sub]
y_data = dataloader.read_field('u')[:,::sub]

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]

x_train = x_train.reshape(ntrain,s,1)
x_test = x_test.reshape(ntest,s,1)


# In[ ]:


print("*** Data shape after importing from raw dataset ***")
print(f"X_Train Shape: {x_train.shape}")
print(f"Y_Train Shape: {y_train.shape}")
print(f"X_Test Shape: {x_test.shape}")
print(f"Y_Test Shape: {y_test.shape}")


# In[8]:


x_train = x_train.permute(0, 2, 1)
y_train = y_train.unsqueeze(1)
x_test = x_test.permute(0, 2, 1)
y_test = y_test.unsqueeze(1)


# In[ ]:


print("*** Data shape after reshaping based on [Batch, Channel, D1, D2, ...] ***")
print(f"X_Train Shape: {x_train.shape}")
print(f"Y_Train Shape: {y_train.shape}")
print(f"X_Test Shape: {x_test.shape}")
print(f"Y_Test Shape: {y_test.shape}")


# In[10]:


train_loader = DictDataset(x_train, y_train)
test_loader = DictDataset(x_test, y_test)


# In[11]:


train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True)
test_loader = {
    dataset_resolution: test_loader
}


# In[ ]:


print("*** One batch of the Train Loader ***")
batch = next(iter(train_loader))
print(f"Loader Type: {type(train_loader)}\nBatch Type: { type(batch)}\nBatch['x'].shape: {batch['x'].shape}\nBatch['y'].shape: {batch['y'].shape}")


# In[ ]:


print("*** One batch of the Test Loader ***")
batch = next(iter(test_loader[dataset_resolution]))
print(f"Loader Type: {type(test_loader[dataset_resolution])}\nBatch Type: { type(batch)}\nBatch['x'].shape: {batch['x'].shape}\nBatch['y'].shape: {batch['y'].shape}")


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"*** Device: {device} ***")


# In[ ]:


model = XNO(
    max_n_modes=max_modes,
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    in_channels=in_channels,
    out_channels=out_channels,
    transformation=transformation,
    transformation_kwargs=kwargs,
    conv_non_linearity=conv_non_linearity, 
    mlp_non_linearity=mlp_non_linearity,
    n_layers=n_layers
)
model = model.to(device)
n_params = count_model_params(model)


# In[16]:


optimizer = AdamW(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=step_size, # default=30
    gamma=gamma # default=0.1
)


# In[ ]:


data_transform = IncrementalDataProcessor(
    in_normalizer=None,
    out_normalizer=None,
    device=device,
    subsampling_rates=[2, 1],
    dataset_resolution=dataset_resolution,
    dataset_indices=dataset_indices,
    verbose=True,
)

data_transform = data_transform.to(device)


# In[ ]:


l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}
print("\n### N PARAMS ###\n", n_params)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print("\n### INCREMENTAL RESOLUTION + GRADIENT EXPLAINED ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()


# In[19]:


# Finally pass all of these to the Trainer
trainer = IncrementalXNOTrainer(
    model=model,
    n_epochs=n_epochs,
    data_processor=data_transform,
    device=device,
    verbose=True,
    incremental_loss_gap=False,
    incremental_grad=True,
    incremental_grad_eps=0.9999,
    incremental_loss_eps = 0.001,
    incremental_buffer=5,
    incremental_max_iter=1,
    incremental_grad_max_iter=2,
)


# In[ ]:


trainer.train(
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
    save_every=save_every,
    save_testing=save_testing, 
    save_dir=save_dir
)

