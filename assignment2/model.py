import torch


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
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size)
        self.lin_hid = torch.nn.Linear(hidden_size, int(hidden_size/2))
        self.lin2 = torch.nn.Linear(int(hidden_size/2), output_size)
        self.softmax = torch.nn.Softmax(1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        data = self.lin1(data)
        data = torch.relu(data)
        data = self.lin_hid(data)
        data = self.dropout(data)
        data = torch.tanh(data)
        data = self.lin2(data)
        return self.softmax(data)