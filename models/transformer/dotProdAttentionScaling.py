import math
from torch import nn


class ScaleDotProductAttention(nn.Module):


    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):

        batch_size, head, length, d_tensor = k.size()


        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # apply masking
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # pass them softmax to make [0, 1] range
        score = self.softmax(score)

        v = score @ v

        return v, score
    
