import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    A LSTM model class for time series prediction.

    Attributes:
        model_type (str): Type of the model.
        lstm (LSTM): Long Short-Term Memory layer.
        decoder (Linear): Linear layer to decode the LSTM outputs to desired output size.

    Parameters:
        feature_size (int): The number of features in the input data.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        dropout (float): Fraction of neurons affected by Dropout.
    """
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
        """
        Defines the forward pass of the LSTM model.

        Parameters:
            src (Tensor): Input tensor of shape (batch_size, sequence_length, feature_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, 1) after decoding.
        """
        self.lstm.flatten_parameters()  # Ensuring the LSTM weights are contiguous in memory
        output, _ = self.lstm(src)      # LSTM output: (batch, seq_len, hidden_size)
        output = self.decoder(output)   # Apply linear layer to each time step
        return output
    
'''Test LSTM Model'''   

def test_lstm():
    feature_size = 200
    hidden_size = 200
    num_layers = 2
    dropout = 0.1
    batch_size = 4
    seq_len = 20

    # Initialize the model
    model = LSTMModel(feature_size, hidden_size, num_layers, dropout)
    model.eval() 

    # Create a dummy input tensor of appropriate shape
    input_tensor = torch.randn(batch_size, seq_len, feature_size)
    output_tensor = model(input_tensor)

    # Assert the shape of the output tensor
    assert output_tensor.shape == (batch_size, seq_len, 1), f"Output shape is incorrect: {output_tensor.shape}"

    print("Test passed")

if __name__ == "__main__":
    test_lstm()
