import torch.nn as nn

from data.dataset import PAD_IDX


class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_IDX
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))
        _, (h_n, _) = self.lstm(embedded)
        last_hidden = self.dropout(h_n[-1])    # many-to-one: only h_T
        return self.fc(last_hidden).squeeze(1) # (batch,) raw logit