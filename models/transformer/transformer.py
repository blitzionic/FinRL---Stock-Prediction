import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim



class Transformer(nn.Module):
    def __init__(self, feature_size, num_layers, num_heads, forward_expansion, dropout, max_len):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(1, feature_size) 
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, feature_size))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_size,
                nhead=num_heads,
                dim_feedforward=forward_expansion * feature_size,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(feature_size, 1)  # output layer

    def forward(self, src):
        src = self.embedding(src)  
        # ensure src is [batch_size, seq_length, feature_size]
        src = src + self.pos_embedding[:, :src.size(1)]  
        out = self.encoder(src)
        return self.fc_out(out[:, -1, :])  # Output from the last timestep
