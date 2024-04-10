from torch import nn

import encoder



class TransformerEmbedding(nn.Module):
    """
    Combines raw input features with positional encoding.
    """
    def __init__(self, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        pos_emb = self.pos_emb(x)
        return self.drop_out(x + pos_emb)