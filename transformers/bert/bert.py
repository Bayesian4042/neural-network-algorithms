import torch
from torch import nn


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_segments, max_sequence_length, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.segment_embeddings = nn.Embedding(n_segments, hidden_size)
        self.position_embeddings = nn.Embedding(max_sequence_length, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.pos_input = torch.tensor([i for i in range(max_sequence_length)],)
    
    def forward(self, x, segment_ids):
        token_embeddings = self.token_embeddings(x)
        segment_embeddings = self.segment_embeddings(segment_ids)
        position_embeddings = self.position_embeddings(self.pos_input)
        return token_embeddings + segment_embeddings + position_embeddings