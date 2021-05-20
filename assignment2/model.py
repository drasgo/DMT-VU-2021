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
        self.lin2 = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, data):
        data = self.lin1(data)
        data = torch.tanh(data)
        data = self.lin2(data)
        return self.softmax(data)
