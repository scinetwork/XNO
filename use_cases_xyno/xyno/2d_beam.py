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

# In[2]:
import sys
import os
sys.path.append(os.path.abspath(".."))

from xno.models import XYNO
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

ntrain = 200
ntest = 10

h = 40           # total grid size divided by the subsampling rate
grid_range = 1
in_channel = 2   # (a(x), x) for this case
# ### Model and Trainer Settings

# In[5]:


data_path = 'data/data_beam.mat'
data_name = 'data_beam'
batch_size = 10
dataset_resolution = 50

# XNO (model) 
max_modes = (8, 8, )
n_modes = (8, 8, )
in_channels = 1
out_channels = 1
n_layers = 2
hidden_channels = 16
transformation = "wno"
kwargs = {
    "wavelet_level": 4, 
    "wavelet_size": [dataset_resolution, dataset_resolution], "wavelet_filter": ['db6']
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
step_size = 50 if transformation.lower() == "lno" else 50
gamma = 0.5

# IncrementalDataProcessor (data_transform) 
dataset_resolution = dataset_resolution
dataset_indices = [2]

# IncrementalXNOTrainer (trainer) 
n_epochs = 100 # 500
save_every = 50
save_testing = True
save_dir = f"save/{data_name}/{transformation.lower()}/"


# Open the file at the start of the script
# output_file = open(f"{data_name}_{transformation.lower()}.txt", "w")
# sys.stdout = output_file  # Redirect stdout to the file


# In[6]:

# Data is of the shape (number of samples, grid size)
reader = MatReader(data_path)
x_train = reader.read_field('f_train')[:ntrain,::,::]
y_train = reader.read_field('u_train')[:ntrain,::,::]

x_test = reader.read_field('f_vali')[:ntest,::,::]
y_test = reader.read_field('u_vali')[:ntest,::,::]


# x_normalizer = UnitGaussianNormalizer(dim=[0, 1, 2])
# x_normalizer.fit(x_train)
# x_train = x_normalizer.transform(x_train)
# x_test = x_normalizer.transform(x_test)

# y_normalizer = UnitGaussianNormalizer(dim=[0, 1, 2])
# y_normalizer.fit(y_train)
# y_train = y_normalizer.transform(y_train)

# In[ ]:


print("\n=== Data shape after importing from raw dataset ===\n")
print(f"X_Train Shape: {x_train.shape}")
print(f"Y_Train Shape: {y_train.shape}")
print(f"X_Test Shape: {x_test.shape}")
print(f"Y_Test Shape: {y_test.shape}")


# In[8]:

x_train = x_train.unsqueeze(1)
y_train = y_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)
y_test = y_test.unsqueeze(1)


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


model = XYNO(
    max_n_modes=max_modes,
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    in_channels=in_channels,
    out_channels=out_channels,
    mix_mode="parallel",
    parallel_kernels=['fno', 'wno', 'lno'],
    transformation_kwargs=kwargs,
    conv_non_linearity=conv_non_linearity, 
    mlp_non_linearity=mlp_non_linearity,
    n_layers=n_layers, 
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
    disable_incremental=True,
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
    # save_every=save_every,
    # save_testing=save_testing, 
    # save_dir=save_dir
)

print(mess)

# %%

# At the end of the script
sys.stdout = sys.__stdout__  # Restore original stdout
output_file.close()  # Close the file
# %%
