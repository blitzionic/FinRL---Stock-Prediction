from torch import nn

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden=2048, drop_prob=0.2):
        """
        Position-wise feedforward network for Transformer model.

        Args:
            d_model (int): The number of expected features in the input (required).
            hidden (int): The number of features in the hidden layer (default=2048).
            drop_prob (float): Dropout probability (default=0.2).
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
