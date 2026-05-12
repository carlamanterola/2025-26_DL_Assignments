import random

import numpy as np
import torch

from datasets import load_dataset
from tasks import language


# REPRODUCIBILITY
# ===========================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
 
CONFIG = {
    'seed':       SEED,
    'device':     torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'epochs':     60,
    'batch_size': 16,
    'embed_dim':  16,
    'hidden_dim': 32,
    'dropout':    0.3,
    'lr':         5e-3,
}
 
print(f'PyTorch {torch.__version__}  |  device: {CONFIG["device"]}')


# ===========================================================================
# TASK 1: LANGUAGE - MOVIE REVIEW CLASSIFICATION
# ===========================================================================
print("=" * 80)
print("TASK 1: LANGUAGE - MOVIE REVIEW CLASSIFICATION")
print("=" * 80)

ds = load_dataset('imdb')
texts  = ds['train']['text']
labels = ds['train']['label']
 
assert len(texts) > 0, 'Load your IMDb data into `texts` and `labels` before running.'
 
predict = language.run(texts, labels, CONFIG)
 
# Example inference after training
predict('what a stunning and beautiful film this was')
predict('what a dreadful and boring film this was')




# ===========================================================================
# TASK 2: TIME-SERIES
# ===========================================================================
print("=" * 80)
print("TASK 2: TIME-SERIES - ")
print("=" * 80)
