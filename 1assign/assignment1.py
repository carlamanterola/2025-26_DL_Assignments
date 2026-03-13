import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
 
import optuna
from optuna.samplers import TPESampler

# SEED FOR REPRODUCIBILITY
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")