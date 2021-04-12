import torch
from torch import nn


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, data):
        data = self.lin1(data)
        data = torch.tanh(data)
        data = self.lin2(data)
        return self.softmax(data)

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lstm_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.dropout = nn.Dropout(0.5)
        self.emb1 = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.lstm1 = torch.nn.LSTM(hidden_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.relu = torch.nn.ReLU()
        # self.global_max_pool = torch.nn.AdaptiveMaxPool2d
        self.lin1 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, data):
        # Initialization of hidden and cell states of LSTM
        h_weight = torch.zeros((self.input_size, data.size(0), self.hidden_size))
        c_matrix = torch.zeros((self.input_size, data.size(0), self.hidden_size))
        torch.nn.init.xavier_normal_(h_weight)
        torch.nn.init.xavier_normal_(c_matrix)

        data = self.emb1(data)
        # data = data.permute(1, 0, 2)
        data, _ = self.lstm1(data, (h_weight, c_matrix))
        # data = data.permute(1, 0, 2)
        data = self.dropout(data)
        data = self.lin1(self.relu(data))
        data = self.dropout(data)

        return torch.sigmoid(data)
