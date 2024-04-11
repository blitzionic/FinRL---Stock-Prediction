import torch
import torch.nn as nn
import math
from PositionalEncoding import PositionalEncoding

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Transformer(nn.Module):
    def __init__(self, feature_size=200, num_layers=2, dropout=0.1, nhead=10):
        """
            feature_size (int): Embedding dimension (d_model)
            num_layers (int): Number of encoder layers
            dropout (float): Dropout rate
            nhead (int): Number of attention heads
        """
        super().__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(feature_size, 1)
        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1)  
        mask = mask.float().masked_fill(mask == 0, float('-inf'))  
        mask = mask.masked_fill(mask == 1, float(0.0))  
        return mask
    
'''Test Transformer'''   

def test_transformer():
    
    feature_size = 200
    num_layers = 2
    dropout = 0.1
    nhead = 10
    batch_size = 4
    seq_len = 20

    model = Transformer(feature_size, num_layers, dropout, nhead)

    input_tensor = torch.randn(batch_size, seq_len, feature_size)

    output_tensor = model(input_tensor)

    assert output_tensor.shape == (batch_size, seq_len, 1), "Output shape is incorrect"

    assert not torch.allclose(output_tensor, torch.zeros_like(output_tensor)), "Output is all zeros"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable parameters found"

    print("Test passed")

if __name__ == "__main__":
    test_transformer()