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


class Bert(nn.Module):
    def __init__(self, vocab_size, embed_size, n_segments, max_sequence_length, n_layers, attn_heads, dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, embed_size, n_segments, max_sequence_length, dropout)
        self.encoder = nn.TransformerEncoderLayer(embed_size, attn_heads, embed_size * 4)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_layers)
    
    def forward(self, x, segment_ids):
        x = self.embeddings(x, segment_ids)
        return self.encoder_block(x)

def main():
    VOCAB_SIZE = 30000
    N_SEGMENTS = 3
    MAX_LEN = 512

    # Architecture
    EMBEDDING_DIM = 768
    N_LAYERS = 12
    ATTN_HEADS = 12
    DROPOUT = 0.1

    # Training
    BATCH_SIZE = 32

    sample_seq = torch.randint(high=VOCAB_SIZE, size=[MAX_LEN,])
    sample_seg = torch.randint(high=N_SEGMENTS, size=[MAX_LEN,])

    embeddings = BertEmbeddings(VOCAB_SIZE, EMBEDDING_DIM, N_SEGMENTS, MAX_LEN, DROPOUT)
    embeddings_tensor = embeddings(sample_seq, sample_seg)
    print(embeddings_tensor.shape)

    bert = Bert(VOCAB_SIZE, EMBEDDING_DIM, N_SEGMENTS, MAX_LEN, N_LAYERS, ATTN_HEADS, DROPOUT, ATTN_HEADS)
    bert_out = bert(sample_seq, sample_seg)
    print(bert_out.shape)