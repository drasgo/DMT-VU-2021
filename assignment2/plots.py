import json
from matplotlib import pyplot as plt
import numpy

def plot_results(training_losses, validation_losses, error, name):
    mean_training_loss = numpy.array(training_losses)
    mean_validation_loss = numpy.array(validation_losses).reshape(10,-1)
    validation_losses = mean_validation_loss.tolist()
    mean_training_loss = mean_training_loss.mean(1)
    mean_validation_loss = mean_validation_loss.mean(1)

    import pprint
    pprint.pprint(error)
    plt.plot(training_losses[0])
    plt.suptitle(f"{name}")
    plt.title(f"Training Loss of 1st epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.xlabel("Batch number")
    plt.show()

    plt.plot(mean_training_loss)
    plt.suptitle(f"{name}")
    plt.title(f"Training Loss average over 10 epochs")
    plt.ylabel("MSE Loss")
    plt.xlabel("Batch number")
    plt.show()

    plt.suptitle(f"{name}")
    plt.plot(validation_losses[0])
    plt.title(f"Validation Loss of 1st epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.xlabel("Batch number")
    plt.show()

    plt.plot(mean_validation_loss)
    plt.suptitle(f"{name}")
    plt.title(f"Validation Loss average over 10 epochs")
    plt.ylabel("MSE Loss")
    plt.xlabel("Batch number")
    plt.show()


if __name__ == '__main__':
    file_name = "results.json"
    name = "MLP Classifier"

    with open(file_name, "r") as fp:
        result = json.load(fp)

    plot_results(
        training_losses=result["training_loss"],
        validation_losses=result["validation_loss"],
        error=result["metrics"],
        name=name
    )