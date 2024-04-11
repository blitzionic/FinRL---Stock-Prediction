import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
      super().__init__()
      self.dropout = nn.Dropout(p=dropout)
      
      pe = torch.zeros(max_len, d_model)
      position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)
      pe = pe.unsqueeze(0)
      self.register_buffer('pe', pe)

    def forward(self, x):
      x = x + self.pe[:, :x.size(1)]
      return self.dropout(x)
    

'''Test Positional Encoder'''   

"""
if __name__ == "__main__":
  # Set the parameters
  d_model = 512
  dropout = 0.1
  max_len = 100
  batch_size = 4
  seq_len = 20

  pos_encoding = PositionalEncoding(d_model, dropout, max_len)

  input_tensor = torch.randn(batch_size, seq_len, d_model)

  output_tensor = pos_encoding(input_tensor)

  print("Input Tensor:")
  print(input_tensor)
  print("\nOutput Tensor:")
  print(output_tensor)
"""