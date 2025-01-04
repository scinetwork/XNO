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
import numpy as np
from timeit import default_timer

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
from xno.data.transforms.normalizers import UnitGaussianNormalizer

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
ntest = 200

scheduler_step = 100
scheduler_gamma = 0.5

sub = 1
S = 64 // sub
T_in = 10
T = 10

# ### Model and Trainer Settings

# In[5]:
data_path = 'data/NavierStokes_V1e-5_N1200_T20.mat'
data_name = '3d_navier_stoke'
batch_size = 5
dataset_resolution = 64

# XNO (model) 
max_modes = (16, 16, 16)
n_modes = (8, 8, 8)
in_channels = 10
out_channels = 1
n_layers = 4
hidden_channels = 20
transformation = "wno"
kwargs = {
    "wavelet_level": 2, 
    "wavelet_size": [10, dataset_resolution, dataset_resolution], "wavelet_filter": ['db6']
} if transformation.lower() == "wno" else {}

conv_non_linearity = None
mlp_non_linearity = None

match transformation.lower():
    case "fno" | "hno":
        conv_non_linearity = F.gelu
        mlp_non_linearity = F.gelu
    case "wno":
        conv_non_linearity = F.gelu
        mlp_non_linearity = F.gelu
    case "lno":
        conv_non_linearity = torch.sin
        mlp_non_linearity = torch.tanh

# AdamW (optimizer) 
learning_rate = 1e-3
weight_decay = 1e-4
# CosineAnnealingLR (scheduler) 
step_size = 100 if transformation.lower() == "lno" else 100
gamma = 0.5

# IncrementalDataProcessor (data_transform) 
dataset_resolution = dataset_resolution
dataset_indices = [2]

# IncrementalXNOTrainer (trainer) 
n_epochs = 500 # 500
save_every = 50
save_testing = True
save_dir = f"save/{data_name}/{transformation.lower()}/"


# Open the file at the start of the script
output_file = open(f"{data_name}_{transformation.lower()}.txt", "w")
sys.stdout = output_file  # Redirect stdout to the file


# In[6]:
# Data is of the shape (number of samples, grid size)
reader = MatReader(data_path)
train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

reader = MatReader(data_path)
test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

a_normalizer = UnitGaussianNormalizer(dim=[0, 1, 2, 3])
a_normalizer.fit(train_a)
train_a = a_normalizer.transform(train_a)
test_a = a_normalizer.transform(test_a)

y_normalizer = UnitGaussianNormalizer(dim=[0, 1, 2, 3]) 
y_normalizer.fit(train_u)  
train_u = y_normalizer.transform(train_u) 

train_a = train_a.reshape(ntrain,S,S,1,T_in).repeat([1,1,1,T,1])
test_a = test_a.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])

# In[ ]:
print("\n=== Data shape after importing from raw dataset ===\n")
print(f"X_Train Shape: {train_a.shape}")
print(f"Y_Train Shape: {train_u.shape}")
print(f"X_Test Shape: {test_a.shape}")
print(f"Y_Test Shape: {test_u.shape}")


# In[8]:
x_train = train_a.permute(0, 4, 3, 2, 1)
y_train = train_u.unsqueeze(1)
y_train = y_train.permute(0, 1, 4, 3, 2)
x_test = test_a.permute(0, 4, 3, 2, 1)
y_test = test_u.unsqueeze(1)
y_test = y_test.permute(0, 1, 4, 3, 2)


# In[ ]:
print("\n=== Data shape after reshaping based on [Batch, Channel, D1, D2, ...] ===\n")
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
print("\n=== One batch of the Train Loader ===\n")
batch = next(iter(train_loader))
print(f"Loader Type: {type(train_loader)}\nBatch Type: { type(batch)}\nBatch['x'].shape: {batch['x'].shape}\nBatch['y'].shape: {batch['y'].shape}")


# In[ ]:
print("\n=== One batch of the Test Loader ===\n")
batch = next(iter(test_loader[dataset_resolution]))
print(f"Loader Type: {type(test_loader[dataset_resolution])}\nBatch Type: { type(batch)}\nBatch['x'].shape: {batch['x'].shape}\nBatch['y'].shape: {batch['y'].shape}")


# In[ ]:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=== Device: {device} ===\n")


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


l2loss = LpLoss(d=2, p=2,)
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


mess = trainer.train(
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

print(mess)

# %%

# At the end of the script
sys.stdout = sys.__stdout__  # Restore original stdout
output_file.close()  # Close the file
# %%
