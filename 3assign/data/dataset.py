import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 0
UNK_IDX = 1


def encode(tokens, word2idx):
    return torch.tensor(
        [word2idx.get(w, UNK_IDX) for w in tokens],
        dtype=torch.long
    )


class ReviewDataset(Dataset):
    def __init__(self, idx_list, encoded, labels):
        self.samples = [(encoded[i], float(labels[i])) for i in idx_list]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        seq, lbl = self.samples[i]
        return seq, torch.tensor(lbl)


def collate_fn(batch):
    seqs, lbls = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    return padded, torch.stack(lbls)