
import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, default_collate
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from utils import MatReader
from pathlib import Path
from xno.models import XNO, XYNO
from xno.data.datasets import Burgers1dTimeDataset
from xno.utils import count_model_params
from xno.training import AdamW
from xno.training.incremental import IncrementalXNOTrainer
from xno.data.transforms.data_processors import IncrementalDataProcessor
from xno import LpLoss, H1Loss
from data_loader import *

import argparse
parser = argparse.ArgumentParser(description="Accept arguments in Python.")

# parser.add_argument('--config', type=str, help="All required arguments to run a training job")

parser.add_argument('--data_path', type=str, help="Input data path")
parser.add_argument('--dataset', type=str, help="Name of the dataset")
parser.add_argument('--mix_mode', type=str, help="How to mix different kernels, Parallel or Peure.")
parser.add_argument('--scenario', type=str, help="Variation of kernels, participate in parallel convolution in each layer.")
parser.add_argument('--parallel_kernels', nargs='+', help="Variation of kernels, participate in parallel convolution in each layer.")
parser.add_argument('--pure_kernels_order', nargs='+', help="The order of individual convolution in each layer.")
parser.add_argument('--save_out', type=lambda x: x.lower() == 'true', default=True, help="Name of the dataset")


args = parser.parse_args()
data_path = args.data_path
dataset = args.dataset.lower()
mix_mode = args.mix_mode.lower()
scenario = args.scenario.lower()
parallel_kernels = [kernel.lower() for kernel in args.parallel_kernels]
pure_kernels_order = [kernel.lower() for kernel in args.pure_kernels_order]
save_out = args.save_out # save terminal output as a text file

batch_size = 16
dataset_resolution = 1024
# =======================================
# DIMENTIONALITY SENSITIVE CONFIGS 
max_modes = (16, )
n_modes = (16, )
kwargs = {
    "wavelet_level": 6, 
    "wavelet_size": [dataset_resolution], "wavelet_filter": ['db6']
} 
dataset_indices = [2]
# =======================================
in_channels = 1
out_channels = 1
n_layers = 4
hidden_channels = 64
# AdamW (optimizer) 
learning_rate = 1e-3
weight_decay = 1e-4
# CosineAnnealingLR (scheduler) 
step_size = 100
gamma = 0.5
# IncrementalDataProcessor (data_transform) 
dataset_resolution = dataset_resolution
# IncrementalXNOTrainer (trainer) 
n_epochs = 500 # 500
save_every = 50
save_testing = True
save_dir = f"save/{dataset}/{mix_mode}/{scenario}"

if save_out:
    # Open the file at the start of the script
    output_file = open(f"save/{dataset}/{dataset}_{mix_mode}_{scenario}.txt", "w")
    sys.stdout = output_file  # Redirect stdout to the file


train_loader, test_loader = burger_1d(
    data_path=data_path, 
    batch_size=batch_size, 
    resolution=dataset_resolution
    )


print("\n=== One batch of the Train Loader ===\n")
batch = next(iter(train_loader))
print(f"Loader Type: {type(train_loader)}\nBatch Type: { type(batch)}\nBatch['x'].shape: {batch['x'].shape}\nBatch['y'].shape: {batch['y'].shape}")

print("\n=== One batch of the Test Loader ===\n")
batch = next(iter(test_loader[dataset_resolution]))
print(f"Loader Type: {type(test_loader[dataset_resolution])}\nBatch Type: { type(batch)}\nBatch['x'].shape: {batch['x'].shape}\nBatch['y'].shape: {batch['y'].shape}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=== Device: {device} ===\n")

model = XYNO(
    max_n_modes=max_modes,
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    in_channels=in_channels,
    out_channels=out_channels,
    mix_mode=mix_mode,
    parallel_kernels=parallel_kernels,
    pure_kernels_order=pure_kernels_order,
    transformation_kwargs=kwargs,
    # conv_non_linearity=conv_non_linearity, 
    # mlp_non_linearity=mlp_non_linearity,
    n_layers=n_layers
)
model = model.to(device)
n_params = count_model_params(model)

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
    disable_incremental=True
)

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
