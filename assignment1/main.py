import json
import time
from typing import Tuple
from torch.nn import functional
from models import MLP
from models import LSTM
import torch
RESULT_FILE = "results.json"
EPOCHS = 1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
NN_TYPE = {
    "mlp": MLP,
    "lstm": LSTM
}


def prepare_tensors(input_data:list, output_data:list) -> Tuple[list, list]:
    # input_data, output_data = divide_batches(input_data, output_data)
    train_data_tensors = [torch.tensor(vector) for vector in input_data]
    train_target_tensors = [torch.tensor(vector) for vector in output_data]

    # Changing structure of target from (batch_size) to (batch_size, 1)
    # for e in range(len(train_target_tensors)):
    #     train_target_tensors[e] = train_target_tensors[e].view(-1, 1)
    return train_data_tensors, train_target_tensors

def train_network(input_batches, target_batches, net):
    """
    :param target_batches:
    :param input_batches:
    :param net:

    """
    total_iterations = 0
    total_losses = []
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        for batch_idx, (inputs, labels) in enumerate(zip(input_batches, target_batches)):
            optimizer.zero_grad()
            outputs = net(inputs)
            print("output shape :" + str(outputs.shape))
            print("labels shape:  " + str(labels.shape))
            # loss = criterion(outputs, labels)
            loss = functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            # input()


            total_iterations += 1
            total_losses.append(loss.item())
            if len(input_batches) % ((batch_idx + 1) * 10) == 0:
                print("Epoch n°: " + str(epoch) + ", batches computed: " + str(batch_idx) + ", current loss: " + str(loss.item()))
    return net, total_losses


def test_network(test_data, test_target, net) -> Tuple[float, float, float]:
    """
    :param net:
    """
    corr = 0
    tot = 0
    counter = 0
    with torch.no_grad():
        for inp_data, target in zip(test_data, test_target):
            counter += 1
            outputs = net(inp_data)
            _, predicted = torch.max(outputs.data, 1)
            tot += target.size(0)
            corr += (predicted == target).sum().item()
    acc = 100 * corr / tot
    print("Accuracy of the network on the %d test images: %d %%" % (counter, acc))
    return acc, corr, tot


def save_results(name: str, losses: list, test_accuracy: float, test_correct: float, test_total: float, delta_time: float) -> None:
    res = {
        "losses": losses,
        "accuracy": test_accuracy,
        "correct": test_correct,
        "total": test_total,
        "delta_training_time": delta_time
    }
    with open(name + RESULT_FILE, "w") as fp:
        json.dump(res, fp)


def run_network(input_data, target_data,
                # validation_input, validation_target,
                test_input, test_target, net, name):
    print("pre training")
    initial_train_time = time.time()
    net, losses = train_network(input_data, target_data,
                                # validation_input, validation_target,
                                net)
    delta_train_time = time.time() - initial_train_time
    print("Training time: " + str(delta_train_time))
    print("post training, pre testing")
    # input()
    accuracy, correct, total = test_network(test_input, test_target, net)
    print("post testing, pre saving results")
    save_results(name, losses, accuracy, correct, total, delta_train_time)
    print("post saving results")


if __name__ == '__main__':
    nn_type = "MLP"

    data = []
    network = NN_TYPE[nn_type]
