import json
from matplotlib import pyplot as plt
import numpy

def plot_results(training_losses, validation_losses, error, error_distances, name):
    mean_training_loss = numpy.array(training_losses)
    mean_validation_loss = numpy.array(validation_losses)
    mean_training_loss = mean_training_loss.mean(1)
    mean_validation_loss = mean_validation_loss.mean(1)

    plt.plot(training_losses[0])
    plt.suptitle(f"{name}")
    plt.title(f"Training Loss of 1st epoch")
    plt.ylabel("MSE Loss")
    plt.xlabel("Batch number")
    plt.show()

    plt.plot(mean_training_loss)
    plt.suptitle(f"{name}")
    plt.title(f"Training Loss average over 10 epochs")
    plt.ylabel("MSE Loss")
    plt.xlabel("Batch number")
    plt.show()

    plt.plot(mean_validation_loss)
    plt.suptitle(f"{name}")
    plt.title(f"Validation Loss average over 10 epochs")
    plt.ylabel("MSE Loss")
    plt.xlabel("Batch number")
    plt.show()

    plt.bar(list(error_distances.keys()), height=list(error_distances.values()))
    plt.suptitle(f"{name}")
    plt.title("Testing error distances (i.e. |predicted - label|)")
    plt.annotate(f"Average error distance: {str(round(error, 2))}", (4.5, 80),
                 horizontalalignment='right', verticalalignment='top')
    plt.ylabel("Number of Occurrences")
    plt.xlabel("Distance")
    plt.show()


if __name__ == '__main__':
    file_name = "lstmresults.json"
    name = "Long-Short Term Memory"

    with open(file_name, "r") as fp:
        result = json.load(fp)

    plot_results(
        training_losses=result["training_losses"],
        validation_losses=result["validation_losses"],
        error=result["average_test_error"],
        error_distances=result["error_distances"],
        name=name
    )