import torch
import torch.nn as nn

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, feature_size=200, hidden_size=200, num_layers=2, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.model_type = 'LSTM'
        self.lstm = nn.LSTM(input_size=feature_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.decoder = nn.Linear(hidden_size, 1)

    def forward(self, src):
        # Ensuring the LSTM weights are contiguous in memory
        self.lstm.flatten_parameters()
        # LSTM output: (batch, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(src)
        # Decoding the output at each time step
        output = self.decoder(output)  # Apply linear layer to each time step
        return output
    
'''Test LSTM Model'''   

def test_lstm():
    feature_size = 200
    hidden_size = 200
    num_layers = 2
    dropout = 0.1
    batch_size = 4
    seq_len = 20

    model = LSTMModel(feature_size, hidden_size, num_layers, dropout)
    model.eval()  # Evaluate mode, if the model contains specific layers like dropout

    input_tensor = torch.randn(batch_size, seq_len, feature_size)
    output_tensor = model(input_tensor)

    assert output_tensor.shape == (batch_size, seq_len, 1), f"Output shape is incorrect: {output_tensor.shape}"

    print("Test passed")

if __name__ == "__main__":
    test_lstm()
