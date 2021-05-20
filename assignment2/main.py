import json
import os
import pprint
from typing import Tuple, List

import pandas
import torch
import numpy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MLP
from model import Dataset

# TRAIN_SET = "/content/drive/MyDrive/train_set.csv"
TRAIN_SET = "datasets/train_set.csv"
TEST_SET = "/content/drive/MyDrive/test_set.csv"
MODEL = "network.ptk"
# device = "cpu"
device = "cuda" if torch.cuda.is_available() is True else "cpu"



def retrieve_network():
    mod = MLP(input_size, hidden_size, output_size).to(device)

    if os.path.exists(MODEL):
        mod.load_state_dict(torch.load(MODEL))
        mod.eval()

    return mod


def save_model(network):
    torch.save(network.state_dict(), MODEL)


def prepare_input(dataset, training: bool=True):
    if training:
        dataset = dataset.drop(["gross_bookings_usd", "position", "click_bool", "booking_bool"], axis=1)

    dataset['prop_location_score'] = dataset[['prop_location_score1', 'prop_location_score2']].mean(axis=1)
    dataset["competitor"] = 0

    dataset.loc[((dataset["comp1_rate"] == -1) & (dataset["comp1_inv"] == 0)) | (
                (dataset["comp2_rate"] == -1) & (dataset["comp2_inv"] == 0)) | \
                ((dataset["comp3_rate"] == -1) & (dataset["comp3_inv"] == 0)) | (
                (dataset["comp4_rate"] == -1) & (dataset["comp4_inv"] == 0)) | \
                ((dataset["comp5_rate"] == -1) & (dataset["comp5_inv"] == 0)) | (
                (dataset["comp6_rate"] == -1) & (dataset["comp6_inv"] == 0)) | \
                ((dataset["comp7_rate"] == -1) & (dataset["comp7_inv"] == 0)) | (
                (dataset["comp8_rate"] == -1) & (dataset["comp8_inv"] == 0)), "competitor"] = 1

    # pca = PCA(n_components=5)
    # dataset = pca.fit(dataset)

    dataset = dataset.drop(["srch_id",
                                "date_time",
                                "site_id",
                                "visitor_location_country_id",
                                "prop_country_id",
                                "prop_id",
                                "prop_brand_bool",
                                "prop_location_score1",
                                "prop_location_score2",
                                "prop_log_historical_price",
                                "srch_destination_id",
                                "srch_booking_window",
                                "srch_saturday_night_bool",
                                "srch_query_affinity_score",
                                "orig_destination_distance",
                                "random_bool",
                                "comp1_rate",
                                "comp1_inv",
                                "comp1_rate_percent_diff",
                                "comp2_rate",
                                "comp2_inv",
                                "comp2_rate_percent_diff",
                                "comp3_rate",
                                "comp3_inv",
                                "comp3_rate_percent_diff",
                                "comp4_rate",
                                "comp4_inv",
                                "comp4_rate_percent_diff",
                                "comp5_rate",
                                "comp5_inv",
                                "comp5_rate_percent_diff",
                                "comp6_rate",
                                "comp6_inv",
                                "comp6_rate_percent_diff",
                                "comp7_rate",
                                "comp7_inv",
                                "comp7_rate_percent_diff",
                                "comp8_rate",
                                "comp8_inv",
                                "comp8_rate_percent_diff",
                                ], axis=1)

    print(list(dataset.columns))

    dataset = dataset.fillna(0)
    dataset = StandardScaler().fit_transform(dataset)

    return dataset


def prepare_label(clicked, booked):
    labels = []
    for cl, book in zip(clicked, booked):
        if book == 1:
            labels.append(2)
        elif cl == 1:
            labels.append(1)
        else:
            labels.append(0)
    return numpy.array(labels)


def create_loader(train_data, train_pred):
    trainset = Dataset(train_data, train_pred)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader


def prepare_dataset(dataset, training: bool=True) -> pandas.DataFrame:
    pandas.set_option('display.max_columns', None)

    if not training:
        return prepare_input(dataset)

    dataset = dataset.drop(dataset[(dataset["click_bool"] == 0) & (dataset["booking_bool"] == 0)].sample(frac=.7).index)
    output_data = prepare_label(dataset["click_bool"].to_numpy(), dataset["booking_bool"].to_numpy())
    input_data = prepare_input(dataset)

    print(input_data[:3])

    train_data = input_data[int((validation_percentage + testing_percentage) * len(input_data)):]
    train_labels = output_data[int((validation_percentage + testing_percentage) * len(output_data)):]

    train_data = torch.tensor(train_data, device=device)
    train_labels= torch.tensor(train_labels, device=device)
    train_loader = create_loader(train_data, train_labels)
    print("divided train data")

    validation_data = input_data[:int(validation_percentage * len(input_data))]
    validation_labels = output_data[: int(validation_percentage * len(output_data))]

    validation_data = torch.tensor(validation_data, device=device)
    validation_labels= torch.tensor(validation_labels, device=device)
    validation_loader = create_loader(validation_data, validation_labels)
    print("divided validation data")

    testing_data = input_data[int(validation_percentage * len(input_data)): int(validation_percentage * len(input_data)) + int(
        testing_percentage * len(input_data))]
    testing_labels = output_data[int(validation_percentage * len(output_data)): int(validation_percentage * len(output_data)) + int(
        testing_percentage * len(output_data))]

    testing_data = torch.tensor(testing_data, device=device)
    testing_labels = torch.tensor(testing_labels, device=device)
    testing_loader = create_loader(testing_data, testing_labels)
    print("divided test data")

    return train_loader, validation_loader, testing_loader


def network_training_testing():
    train_file = pandas.read_csv(TRAIN_SET)
    train_loader, validation_loader, testing_loader = prepare_dataset(train_file)
    nn = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    nn, training_losses, validation_losses = train_network(nn, train_loader, validation_loader)
    metric = test_network(testing_loader, nn)
    save_model(nn)

    with open("results.json", "w")as fp:
        json.dump({
        "trainin_loss": training_losses,
        "validation_loss": validation_losses,
        "metrics": metric,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
    }, fp)

    return nn


def metrics(pred_flat, labels_flat) -> dict:
    """Function to various metrics of our predictions vs labels"""
    print(json.dumps(classification_report(labels_flat, pred_flat, output_dict=True)))
    print("\n**** Classification report")
    print(classification_report(labels_flat, pred_flat))

    print("\n***Confusion matrix")
    pprint.pprint(confusion_matrix(pred_flat, labels_flat))

    info = {
        "classification_report": json.dumps(classification_report(labels_flat, pred_flat)),
        # "confusion_matrix": json.dumps(confusion_matrix(pred_flat, labels_flat))
    }
    return info


def network_validation(network, criterion, validation_dataset) -> Tuple[float, float]:
    """Validation phase, executed after each epoch of the training phase to see the improvements of the network."""
    epoch_validation_loss = []
    epoch_validation_accuracy = []
    corr = 0
    tot = 0
    for batch_idx, (inputs, labels) in tqdm(enumerate(validation_dataset)):
        inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.float32).to(device)

        with torch.no_grad():
            outs = network(inputs.to(device))
        loss = criterion(outs, labels.to(device))
        _, predicted = torch.max(outs.data, 1)

        tot += labels.size(0)
        try:
            corr += (predicted == labels.cpu()).sum().item()
        except Exception as exc:
            print(outs)
            print(predicted)
            print(labels)
            print(exc)
            input()
        epoch_validation_accuracy.append(corr/tot)
        epoch_validation_loss.append(loss.item())

    return sum(epoch_validation_loss) / len(epoch_validation_loss), sum(epoch_validation_accuracy) / len(epoch_validation_accuracy)


def train_network(
    network: MLP, trainloader: DataLoader, validation_loader: DataLoader) -> Tuple[MLP, list, List[Tuple[float, float]]]:
    total_iterations = 0
    total_losses = []
    validation_losses = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)

     # optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):  # loop over the dataset multiple times
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        running_loss = 0.0
        network.train()

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.float32).to(device)

            optimizer.zero_grad()
            outputs = network(inputs.to(device))
            loss = criterion(outputs.to(device), labels.to(device))

            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)

            loss.backward()
            optimizer.step()

            total_iterations += 1
            total_losses.append(loss.item())
            running_loss += loss.item()
            if batch_idx % 2000 == 1999:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, batch_idx + 1, running_loss / 2000)
                )
                running_loss = 0.0
        validation_losses += network_validation(network, criterion, validation_loader)

    return network, total_losses, validation_losses


def test_network(testloader: DataLoader, network: MLP) -> dict:
    corr = 0
    tot = 0
    counter = 0
    predictions, true_labels = [], []
    network.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.float32).to(device)

            counter += 1
            outputs = network(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            tot += labels.size(0)
            corr += (predicted == labels).sum().item()

            predictions += predicted
            true_labels += labels.to("cpu")

    res = metrics(predictions, true_labels)
    acc = 100 * corr / tot
    print("Accuracy of the network on the %d test images: %d %%" % (counter, acc))

    return res


def deploy_phase(nn):
    dataset = pandas.read_csv(TEST_SET)
    dataset = prepare_dataset(dataset, False)
    result = pandas.DataFrame(columns=["srch_id", "prop_id"])
    temp_results = {}
    prev_index = -1

    for _, row in dataset.iterrows():
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            output = nn(model)

        if row["srch_id"] != prev_index and prev_index != -1:
            for r in sorted(temp_results, key=temp_results.get, reverse=True):
                # add results in new dataframe
                result["srch_id"] = prev_index
                result["prop_id"] = r

            prev_index = row["srch_id"]
            temp_results = {}

        temp_results["prop_id"] = output
    result.to_csv('out.csv')


if __name__ == "__main__":
    train_flag = True
    batch_size = 10
    testing_percentage = 0.15
    validation_percentage = 0.15
    input_size = 12
    hidden_size = 50
    output_size = 3
    learning_rate = 0.001
    weight_decay = 0.0001
    epochs = 1

    if train_flag:
        model = network_training_testing()
        quit()
    else:
        model = retrieve_network()

    deploy_phase(model)