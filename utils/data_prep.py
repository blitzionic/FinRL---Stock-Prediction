import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import yfinance as yf
from datetime import timedelta, datetime
# from sklearn.preprocessing import MinMaxScaler
# import itertools

def fetch_data():
    stock_list = pd.read_csv("ind_nifty500list.csv")
    stock_list = stock_list["Symbol"].to_list()
    stock_list = [i+".NS" for i in stock_list]
    #Adding .NS to represent national stock
    delta = timedelta(days=-300)# I need 300 days history data
    today = datetime.now()

    data = yf.download(stock_list, today+delta)

    data = yf.download(stock_list, today+delta)
    data.columns = pd.MultiIndex.from_tuples([i[::-1] for i in data.columns])
    #Modifying the dataframe index so that we can eaisly extract the data in seprate file.
    save_location = "stock_data"
    for i in stock_list:
        try:
            TEMP = data[i].copy(deep=True)
            TEMP = TEMP.dropna()
            TEMP.to_csv(save_location+"/"+i+".csv")
        except:
            print("Unaable to load data for {}".format(i))

    # aapl_df_yf = yf.download(tickers = "aapl", start='2024-01-01', end='2024-01-31')
    # aapl_df_yf.head()

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Select features and target
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    target = df['Close'].values.reshape(-1, 1)

    # Normalize features and target
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target)

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    target_tensor = torch.tensor(target_scaled, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(features_tensor, target_tensor)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    return data_loader

def main():
    fetch_data()
    
if __name__ == "__main__":
    main()