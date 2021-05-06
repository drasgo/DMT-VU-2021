import torch
from torch import nn


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, inps, labels):
        """Initialization"""
        self.labels = labels
        self.inps = inps

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inps)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        x = self.inps[index]
        y = int(self.labels[index])
        return x, y


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device="cpu"):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, data):
        data = self.lin1(data)
        data = torch.tanh(data)
        data = self.lin2(data)
        return data


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lstm_layers=1, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.dropout = nn.Dropout(0.5)
        self.lstm1 = torch.nn.LSTM(input_size=15, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.relu = torch.nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features = int(hidden_size))
        self.lin1 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, data):
        # Initialization of hidden and cell states of LSTM
        data = data.to(torch.float32).view(1, data.shape[0], -1)
        data, _ = self.lstm1(data)
        data = self.dropout(data)
        data = data.contiguous().view(-1, self.hidden_size)
        data = self.relu(data)
        return self.lin1(data)
