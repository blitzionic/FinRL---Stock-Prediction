import torch
import torch.nn as nn
import torch.optim as optim
from lstm import LSTMModel
from utils.data_prep import load_and_preprocess_data

def train_model(model, data_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for seq, labels in data_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if epoch % 2 == 1:
            print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')

if __name__ == "__main__":
    data_loader = load_and_preprocess_data('data/processed/stock_data_processed.csv')
    model = LSTMModel(input_size=4, hidden_layer_size=100, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, data_loader, criterion, optimizer)
