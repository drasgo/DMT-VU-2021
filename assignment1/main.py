import json
import time
from typing import Tuple
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MLP, Dataset
from models import LSTM
import torch
RESULT_FILE = "results.json"
NN_TYPE = {
    "mlp": MLP,
    "lstm": LSTM
}


def prepare_dataset(train_data, train_pred, batch_size):
    trainset = Dataset(train_data, train_pred)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader


def prepare_datasets(inputs:list, outputs:list, validation_percentage: float=0.15, testing_percentage: float=0.15) -> Tuple[list, list, list]:
    # input_data, output_data = divide_batches(input_data, output_data)
    train_data_tensors = [torch.tensor(vector) for vector in inputs]
    train_target_tensors = [torch.tensor(vector) for vector in outputs]
    return train_data_tensors, train_target_tensors, []


def network_validation(network, criterion, validation_dataset, device):
    epoch_validation_loss = []
    for batch_idx, (inputs, labels) in tqdm(enumerate(validation_dataset)):
        # inputs = inputs.view(-1,1,5)
        outs = network(inputs.to(device))
        loss = criterion(outs, labels.to(device))
        loss.backward()
        epoch_validation_loss.append(loss.item())
    return epoch_validation_loss


def network_training(network, train_dataset, epochs, device, validation_dataset=None, learning_rate=0.001, weight_decay=0.001):
    total_iterations = 0
    train_losses = []
    validation_losses = []
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=0.001)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_train_loss = []

        for batch_idx, (inputs, labels) in tqdm(enumerate(train_dataset)):
            optimizer.zero_grad()
            outs = network(inputs.to(device))
            loss = criterion(outs, labels.to(device))
            loss.backward()
            optimizer.step()

            total_iterations += 1
            epoch_train_loss.append(loss.item())
            running_loss += loss.item()
            if batch_idx % 1000 == 999:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, batch_idx + 1, running_loss / 1000)
                )
                running_loss = 0.0

        train_losses.append(epoch_train_loss)
        if validation_dataset is not None:
            validation_loss = network_validation(network, criterion, validation_dataset, device)
            validation_losses.append(validation_loss)
    return network, train_losses, validation_losses


def network_testing(network, test_dataset, batches, device):
    corr = 0
    tot = 0
    counter = 0
    with torch.no_grad():
        for data, labels in test_dataset:
            counter += 1
            outs = network(data)
            _, predicted = torch.max(outs.data, 1)
            tot += labels.size(0)
            corr += (predicted == labels.to(device)).sum().item()
    acc = 100 * corr / tot
    print("Accuracy of the network on the %d test data: %d %%" % (counter * batches, acc))
    return acc, corr, tot


def save_results(name: str, training_losses: list, validation_losses:list,
                 test_accuracy: float, test_correct: float, test_total: float,
                 delta_time: float, batches: int, epochs: int, learning_rate: float, weight_decay: float) -> None:
    res = {
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "accuracy": test_accuracy,
        "correct": test_correct,
        "total": test_total,
        "batches": batches,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "delta_training_time": delta_time
    }
    with open(name + RESULT_FILE, "w") as fp:
        json.dump(res, fp)


def run_network(network, train_dataset, test_dataset, epochs, batches, device, name, validation_dataset=None, learning_rate=0.001, weight_decay: float=0.001):
    print("pre training")
    initial_train_time = time.time()
    network, training_losses, validation_losses = network_training(network, train_dataset, epochs, device, validation_dataset, learning_rate, weight_decay)
    delta_train_time = time.time() - initial_train_time
    print("Training time: " + str(delta_train_time))
    print("post training, pre testing")
    # input()
    accuracy, correct, total = network_testing(network, test_dataset, batches, device)
    print("post testing, pre saving results")
    save_results(name, training_losses, validation_losses, accuracy, correct, total, delta_train_time)
    print("post saving results")


if __name__ == '__main__':
    nn_type = "MLP"
    val_percentage = test_percentage = 0.15
    epch = 10
    btch = 3
    lrate = 0.001
    wdecay = 0.001
    dev = "gpu"

    input_size = 0
    hidden_size = 0
    output_size = 0

    input_data, output_data = zip([], [])
    train, validation, test = prepare_datasets(input_data, output_data, val_percentage, test_percentage)
    net = NN_TYPE[nn_type]
    net = net(input_size, hidden_size, output_size)
    run_network(network=net, train_dataset=train, validation_dataset=validation, test_dataset=test,
                epochs=epch, batches=btch, device=dev, name=nn_type, learning_rate=lrate, weight_decay=wdecay)
