import json
import os
import time
from typing import Tuple

import math
import pickle
import numpy
from torch.utils.data import DataLoader
from tqdm import tqdm

from assignment1.plots import plot_results
from models import MLP, Dataset
from models import LSTM
import torch

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT_FILE = "results.json"
NN_TYPE = {
    "mlp": MLP,
    "lstm": LSTM
}


def prepare_dataset(train_data, train_pred, batch_size):
    """Given the input and the label data, create a dataloader to automatically retrieve them later on."""
    train_data = numpy.array(train_data)
    train_pred = numpy.array(train_pred)
    trainset = Dataset(train_data, train_pred)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader


def prepare_datasets(inputs:list, labels:list, batch: int, validation_percentage: float=0.15,
                     testing_percentage: float=0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Divide the data for training, validation and testing, and creating the dataloader for them"""
    train_data = inputs[int((validation_percentage + testing_percentage) * len(inputs)):]
    train_labels = labels[int((validation_percentage + testing_percentage) * len(labels)):]
    print("divided train data")

    validation_data = inputs[:int(validation_percentage * len(inputs))]
    validation_labels = labels[: int(validation_percentage * len(labels))]
    print("divided validation data")

    testing_data = inputs[int(validation_percentage * len(inputs)): int(validation_percentage * len(inputs)) + int(
        testing_percentage * len(inputs))]
    testing_labels = labels[int(validation_percentage * len(labels)): int(validation_percentage * len(labels)) + int(
        testing_percentage * len(labels))]

    train_dataset = prepare_dataset(train_data, train_labels, batch)
    validation_dataset = prepare_dataset(validation_data, validation_labels, batch)
    test_dataset = prepare_dataset(testing_data, testing_labels, batch)
    return train_dataset, validation_dataset, test_dataset


def network_validation(network, criterion, validation_dataset, device):
    """Validation phase, executed after each epoch of the training phase to see the improvements of the network."""
    epoch_validation_loss = []
    for inputs, labels in tqdm(validation_dataset):
        if nn_type == "lstm":
            inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.long).to(device)
        else:
            inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.float32).to(device)
        labels = labels.reshape([labels.shape[0], 1]).to(torch.float32).to(device)
        outs = network(inputs)
        loss = criterion(outs, labels)
        epoch_validation_loss.append(loss.item())
    return epoch_validation_loss


def network_training(network, train_dataset: DataLoader, epochs, device, validation_dataset=None, learning_rate=0.001, weight_decay=0.001):
    """
    Training the network with Adam optimizer for MLP and RMSprop for LSTM. Both have MSE loss technique.
    After every batch we validate with the validation dataset.
    """
    total_iterations = 0
    train_losses = []
    validation_losses = []
    criterion = torch.nn.MSELoss()
    if nn_type == "lstm":
        optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_train_loss = []

        for batch_idx, (inputs, labels) in tqdm(enumerate(train_dataset)):
            if nn_type == "lstm":
                inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.long).to(device)
            else:
                inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.float32).to(device)

            labels = labels.reshape([labels.shape[0], 1]).to(torch.float32).to(device)
            optimizer.zero_grad()
            outs = network(inputs)

            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            total_iterations += 1
            epoch_train_loss.append(loss.item())
            running_loss += loss.item()

        train_losses.append(epoch_train_loss)

        if validation_dataset is not None:
            validation_loss = network_validation(network, criterion, validation_dataset, device)
            validation_losses.append(validation_loss)

    return network, train_losses, validation_losses


def network_testing(network, test_dataset, device):
    """Test the network with the testing dataset, and computing the average error and the error distances."""
    tot = 0
    error = []
    counter = 0
    error_distances = {}
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataset):
            if nn_type == "lstm":
                inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.long).to(device)
            else:
                inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.float32).to(device)
            labels = labels.reshape([labels.shape[0], 1]).to(torch.float32).to(device)
            counter += 1
            outs = network(inputs)
            preds = outs.tolist()
            labs = labels.tolist()

            for pred, lab in zip(preds, labs):
                error.append(abs(round(pred[0]) - lab[0]))
                if abs(round(pred[0]) - lab[0]) not in error_distances:
                    error_distances[abs(round(pred[0]) - lab[0])] = 1

                else:
                    error_distances[abs(round(pred[0]) - lab[0])] += 1

    error = sum(error) / len(error)
    return error, tot, error_distances


def save_results(name: str, training_losses: list, validation_losses:list,
                 test_error: float, test_total: float, error_distances: dict,
                 delta_time: float, batches: int, epochs: int, learning_rate: float, weight_decay: float) -> None:
    """
    Save the results obtained from training, validation and testing (includind the losses, the error distances,
    the other information of the model) in a json file
    """
    res = {
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "average_test_error": test_error,
        "total": test_total,
        "batches": batches,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "delta_training_time": delta_time,
        "error_distances": error_distances
    }
    with open(name + RESULT_FILE, "w") as fp:
        json.dump(res, fp)


def run_network(network, train_dataset, test_dataset, epochs, batches, device, name, validation_dataset=None, learning_rate=0.001, weight_decay: float=0.001):
    print("pre training")
    initial_train_time = time.time()
    network.to(device)

    network, training_losses, validation_losses = network_training(network, train_dataset, epochs, device, validation_dataset, learning_rate, weight_decay)
    delta_train_time = time.time() - initial_train_time
    print("Training time: " + str(delta_train_time))
    print("post training, pre testing")

    error, total, error_distances = network_testing(network, test_dataset, device)
    print("post testing, pre saving results")

    save_results(name=name, training_losses=training_losses, validation_losses=validation_losses,
                 test_error=error, test_total=total, delta_time=delta_train_time, batches=batches, epochs=epochs,
                 learning_rate=learning_rate, weight_decay=weight_decay, error_distances=error_distances)

    plot_results(training_losses, validation_losses, error, error_distances, long_name)
    print("post saving results")


if __name__ == '__main__':
    nn_type = "lstm"
    long_name = "Long-Short Term Memory"

    val_percentage = test_percentage = 0.15
    epch = 10
    btch = 3
    lrate = 0.005
    wdecay = 0.001

    input_size = 15
    hidden_size = 50
    output_size = 1
    final_input = []
    output_data = []

    with open("input.pkl", "rb") as fp:
        input_data = pickle.load(fp)

    with open("output.pkl", "rb") as fp:
        output_data = pickle.load(fp)

    # Prepare the dataset
    average_2 = []
    average_4 = []
    # attributes at index 3 and 5 (we are removing the timestamp at position 1) are averaged out throughout the
    # whole dataset cause, when they are missing (nan), we can substitute them with their mean value
    for single_input in input_data:
        single_input = single_input.tolist()
        for single_day in single_input:
            single_day.pop(0)
            if not math.isnan(single_day[2]):
                average_2.append(single_day[2])
            if not math.isnan(single_day[4]):
                average_4.append(single_day[4])

    average_2 = sum(average_2) / len(average_2)
    average_4 = sum(average_4) / len(average_4)

    # Creating the actual dataset, removing the Timestamp object (pos 0) and updating the attributes (pos 3 and pos 5)
    # with their average throughout the dataset, found above
    for single_input in input_data:
        single_input = single_input.tolist()

        for single_day in single_input:
            single_day.pop(0)

            for index in range(len(single_day)):
                if math.isnan(single_day[index]):
                    if index == 2:
                        single_day[index] = average_2

                    elif index == 4:
                        single_day[index] = average_4
                    else:
                        single_day[index] = 0

        final_input.append(single_input)

    # Divide the dataset in training, validation and testing dataloaders
    train, validation, test = prepare_datasets(inputs=final_input, labels=output_data,
                                               validation_percentage=val_percentage,
                                               testing_percentage=test_percentage, batch=btch)
    net = NN_TYPE[nn_type.lower()]
    net = net(input_size, hidden_size, output_size, device=dev)
    run_network(network=net, train_dataset=train, validation_dataset=validation, test_dataset=test,
                epochs=epch, batches=btch, device=dev, name=nn_type, learning_rate=lrate, weight_decay=wdecay)
